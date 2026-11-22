import torch
import torch.nn as nn
import torch.nn.functional as F


class MCNN(nn.Module):
    """MCNN from Çimen & Bolat, Table 2, for 305x305 inputs.

    Three 5x5/32/stride-2 convs, maxpool, two 3x3/64 convs, maxpool, a 2x2/128
    conv, BN, ReLU, FC-256, FC-10. The paper only mentions one ReLU (after conv6).
    We add a ReLU after every conv, assuming the others were left out of the text.
    Output sizes: 305 -> 153 -> 77 -> 39 -> 19 -> 19 -> 19 -> 9 -> 9, giving the
    paper's 9x9x128 feature map and about 2.8M parameters.
    """

    def __init__(self, n_classes=10):
        super().__init__()

        def conv(i, o, k, s=1, p=0):
            return [nn.Conv2d(i, o, k, s, p), nn.ReLU(inplace=True)]

        self.features = nn.Sequential(
            *conv(1, 32, 5, 2, 2),
            *conv(32, 32, 5, 2, 2),
            *conv(32, 32, 5, 2, 2),
            nn.MaxPool2d(2),
            *conv(32, 64, 3, 1, 1),
            *conv(64, 64, 3, 1, 1),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 2, padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(9 * 9 * 128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, n_classes),
        )

    def forward(self, x, sizes=None):
        return self.head(self.features(x))



class MCNNTable2(nn.Module):
    """MCNN with the pooling of the design table (Table 2 of the MCNN paper draft).

    Differences from MCNN above: both max pools are 3x3 with stride 2 (padding 1), giving the
    table's 20x20 and 10x10 maps, and conv6 (2x2) has no padding, giving 10 -> 9. conv1-3 use
    padding 2: the table lists padding 1 for conv2-3 but output sizes 77 and 39, which need
    padding 2. A ReLU follows every convolution, as in MCNN above (the table lists one ReLU,
    after conv6). Output sizes: 305 -> 153 -> 77 -> 39 -> 20 -> 20 -> 20 -> 10 -> 9;
    same parameter count as MCNN (2,797,706).
    """

    def __init__(self, n_classes=10):
        super().__init__()

        def conv(i, o, k, s=1, p=0):
            return [nn.Conv2d(i, o, k, s, p), nn.ReLU(inplace=True)]

        self.features = nn.Sequential(
            *conv(1, 32, 5, 2, 2),
            *conv(32, 32, 5, 2, 2),
            *conv(32, 32, 5, 2, 2),
            nn.MaxPool2d(3, 2, 1),
            *conv(32, 64, 3, 1, 1),
            *conv(64, 64, 3, 1, 1),
            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(64, 128, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(9 * 9 * 128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, n_classes),
        )

    def forward(self, x, sizes=None):
        return self.head(self.features(x))



class MCNNVariant(nn.Module):
    """MCNNTable2 with optional changes, for the ablation (sonar_cnn.scene.classifiers):

    bn:  BatchNorm after every convolution (MCNNTable2 has it after conv6 only);
    gap: global average pooling over the last map, then FC-256 and FC-n, instead of flattening
         the 9x9x128 map into FC-256 (which holds ~2.65 M of MCNN's 2.80 M parameters);
    canvas: input size (305 as in MCNN; smaller canvases need gap=True or a matching FC size).
    """

    def __init__(self, n_classes=10, bn=False, gap=False, canvas=305):
        super().__init__()

        def conv(i, o, k, s=1, p=0, norm=bn):
            return [nn.Conv2d(i, o, k, s, p)] + ([nn.BatchNorm2d(o)] if norm else []) + [nn.ReLU(inplace=True)]

        self.features = nn.Sequential(
            *conv(1, 32, 5, 2, 2),
            *conv(32, 32, 5, 2, 2),
            *conv(32, 32, 5, 2, 2),
            nn.MaxPool2d(3, 2, 1),
            *conv(32, 64, 3, 1, 1),
            *conv(64, 64, 3, 1, 1),
            nn.MaxPool2d(3, 2, 1),
            *conv(64, 128, 2, norm=True),
        )
        if gap:
            self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 256),
                                      nn.ReLU(inplace=True), nn.Linear(256, n_classes))
        else:
            with torch.no_grad():
                n = self.features(torch.zeros(1, 1, canvas, canvas)).numel()
            self.head = nn.Sequential(nn.Flatten(), nn.Linear(n, 256), nn.ReLU(inplace=True),
                                      nn.Linear(256, n_classes))

    def forward(self, x, sizes=None):
        return self.head(self.features(x))


def size_mask(sizes, H, W):
    """(B,1,H,W) float mask, 1 inside each sample's real (h, w) top-left region."""
    rows = torch.arange(H, device=sizes.device)[None, :, None] < sizes[:, 0, None, None]
    cols = torch.arange(W, device=sizes.device)[None, None, :] < sizes[:, 1, None, None]
    return (rows & cols)[:, None].float()


class MaskedBatchNorm2d(nn.BatchNorm2d):
    """BatchNorm whose training statistics only use the real (masked-in) pixels."""

    def forward(self, x, mask):
        if not self.training:
            return super().forward(x) * mask
        n = mask.sum()
        mean = (x * mask).sum((0, 2, 3)) / n
        var = (((x - mean[None, :, None, None]) ** 2) * mask).sum((0, 2, 3)) / n
        with torch.no_grad():
            self.running_mean.lerp_(mean, self.momentum)
            self.running_var.lerp_(var * n / (n - 1), self.momentum)
        y = (x - mean[None, :, None, None]) / torch.sqrt(var[None, :, None, None] + self.eps)
        return (y * self.weight[None, :, None, None] + self.bias[None, :, None, None]) * mask


class VarCNN(nn.Module):
    """A fully convolutional MCNN variant that takes images at their original size.

    A batch is padded to its largest image (bottom/right) only so it can be
    stacked into one tensor. After every layer the padded area is set back to
    zero, BN statistics ignore it, and the final pooling (mean + max) is taken
    over real pixels only. So in eval mode each sample's output is exactly what
    you'd get running that image on its own (tests/test_varcnn.py checks this).

    Architecture: MCNN's conv stack with stride 1 on conv2 (so 42 px images
    still have a spatial extent at the end) and masked global pooling instead of
    the 2.65M-parameter FC layer.
    """

    # (in, out, kernel, stride, pad) or "M" for 2x2 maxpool
    CFG = [
        (1, 32, 5, 2, 2),
        (32, 32, 5, 1, 2),
        (32, 32, 5, 2, 2),
        "M",
        (32, 64, 3, 1, 1),
        (64, 64, 3, 1, 1),
        "M",
        (64, 128, 3, 1, 1),
    ]

    def __init__(self, n_classes=10, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for c in self.CFG:
            if c == "M":
                continue
            i, o, k, s, p = c
            self.convs.append(nn.Conv2d(i, o, k, s, p))
            self.bns.append(MaskedBatchNorm2d(o))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * 128, n_classes))

    def forward(self, x, sizes):
        sizes = sizes.to(x.device)
        li = 0
        for c in self.CFG:
            if c == "M":
                # Valid outputs j < h//2 only read real inputs. But with odd h the
                # cell at j = h//2 pools a real pixel with padding, so it is nonzero
                # and the next conv would read it as padding. Zero it out again.
                x = F.max_pool2d(x, 2)
                sizes = sizes // 2
                x = x * size_mask(sizes, x.shape[2], x.shape[3])
                continue
            _, _, k, s, p = c
            x = self.convs[li](x)
            sizes = (sizes + 2 * p - k) // s + 1
            mask = size_mask(sizes, x.shape[2], x.shape[3])
            x = F.relu(self.bns[li](x, mask))  # bn output is already masked; relu(0)=0
            li += 1
        mask = size_mask(sizes, x.shape[2], x.shape[3])
        mean = (x * mask).sum((2, 3)) / mask.sum((2, 3))
        mx = x.masked_fill(mask == 0, float("-inf")).amax((2, 3))
        return self.head(torch.cat([mean, mx], 1))


MODELS = {"mcnn": MCNN, "mcnn_t2": MCNNTable2, "varcnn": VarCNN}
