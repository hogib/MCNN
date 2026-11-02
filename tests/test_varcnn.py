import torch

from sonar_cnn.data import collate
from sonar_cnn.models import MCNN, VarCNN


def test_varcnn_batch_matches_single_image():
    """Padding inside a batch must not change any sample's output (eval mode)."""
    torch.manual_seed(0)
    model = VarCNN()
    # Train for a step so the BN running stats aren't the defaults.
    model.train()
    shapes = [(42, 43), (235, 266), (100, 61), (57, 200)]
    imgs = [torch.rand(1, h, w) for h, w in shapes]
    x, sizes, _ = collate([(im, im.shape[1:], 0) for im in imgs])
    model(x, sizes).sum().backward()
    model.eval()
    with torch.no_grad():
        batched = model(x, sizes)
        for i, im in enumerate(imgs):
            single = model(im[None], torch.tensor([im.shape[1:]]))
            torch.testing.assert_close(batched[i], single[0], rtol=1e-4, atol=1e-5)


def test_mcnn_shapes_and_params():
    m = MCNN()
    assert m(torch.zeros(2, 1, 305, 305)).shape == (2, 10)
    n = sum(p.numel() for p in m.parameters())
    assert 2.7e6 < n < 2.9e6, n
