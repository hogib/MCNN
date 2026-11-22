"""MAC and parameter counts with forward hooks, the way MCNN's 234.7 M was counted
(sonar_cnn.scene.mcnn_cv / M5 §3.5): conv = output elements x (cin / groups) x kh x kw,
linear = in x out per sample; BN, activations and pooling are not counted."""

import torch
import torch.nn as nn


def count_macs(model, x):
    total = [0]

    def conv_hook(m, inp, out):
        kh, kw = m.kernel_size
        total[0] += out.numel() // out.shape[0] * (m.in_channels // m.groups) * kh * kw

    def lin_hook(m, inp, out):
        total[0] += m.in_features * m.out_features

    hs = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hs.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hs.append(m.register_forward_hook(lin_hook))
    model.eval()
    with torch.no_grad():
        model(x)
    for h in hs:
        h.remove()
    return total[0]


def count_params(model):
    return sum(p.numel() for p in model.parameters())
