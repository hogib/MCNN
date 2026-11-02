"""Build report/sonar_cnn_report.pdf from results/*.json, the models and the dataset.

  uv run --group report python report/build_report.py

CPU only; regenerates all figures into report/figures/.
"""

import json
import os
from datetime import date
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image as RLImage,
    KeepTogether,
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from sonar_cnn.data import DATA_ROOT, load_images
from sonar_cnn.dedup import greedy_keep
from sonar_cnn.dupes import load_thumbs
from sonar_cnn.models import MCNN, VarCNN

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "report"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Palette (validated, light mode): series 1 blue, series 2 orange; neutrals for text/context.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, CONTEXT = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#cfcec8"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 8.5, "axes.titlecolor": INK, "savefig.dpi": 300,
})


# ---------------------------------------------------------------- data gathering
def res(name):
    p = RES / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def acc_str(r, key="acc"):
    a = np.array([x[key] for x in r["runs"]]) * 100
    return f"{a.mean():.2f} ± {a.std():.2f}"


def acc_ms(r):
    a = np.array([x["acc"] for x in r["runs"]]) * 100
    return a.mean(), a.std()


images, labels, classes = load_images()
hw = np.array([im.shape for im in images])

EXPERIMENTS = [
    # (result name, model, input description, split label)
    ("mcnn_pad_tr80", "MCNN", "pad (paper setup)", "random"),
    ("mcnn_pad_tr90", "MCNN", "pad, 90/10 split", "random"),
    ("mcnn_silhouette_tr80", "MCNN", "silhouette (no content)", "random"),
    ("mcnn_randsize-pad_tr80", "MCNN", "randsize-pad", "random"),
    ("mcnn_randsize-pad-constant_tr80", "MCNN", "randsize-pad-constant", "random"),
    ("varcnn_native_tr80", "VarCNN", "native", "random"),
    ("varcnn_native-constant_tr80", "VarCNN", "native-constant (no content)", "random"),
    ("varcnn_randsize_tr80", "VarCNN", "randsize", "random"),
    ("varcnn_randsize-constant_tr80", "VarCNN", "randsize-constant", "random"),
    ("mcnn_pad_tr80_group", "MCNN", "pad (paper setup)", "group"),
    ("mcnn_randsize-pad_tr80_group", "MCNN", "randsize-pad", "group"),
    ("varcnn_native_tr80_group", "VarCNN", "native", "group"),
    ("varcnn_randsize_tr80_group", "VarCNN", "randsize", "group"),
    ("varcnn_randsize_tr80_dedup_t90", "VarCNN", "randsize", "random, dedup t=0.9"),
    ("mcnn_randsize-pad_tr80_dedup_t90", "MCNN", "randsize-pad", "random, dedup t=0.9"),
]


def mcnn_layers():
    m = MCNN().eval()
    rows, x = [], torch.zeros(1, 1, 305, 305)
    names = {torch.nn.Conv2d: "Conv", torch.nn.MaxPool2d: "MaxPool", torch.nn.BatchNorm2d: "BatchNorm",
             torch.nn.ReLU: None}
    ci = 0
    with torch.no_grad():
        for mod in m.features:
            x = mod(x)
            kind = names[type(mod)]
            if kind is None:
                continue
            if kind == "Conv":
                ci += 1
                pad = mod.padding if isinstance(mod.padding, str) else mod.padding[0]
                cfg = f"{mod.kernel_size[0]}x{mod.kernel_size[0]}, {mod.out_channels}, s{mod.stride[0]}, p{pad}"
                kind = f"conv{ci} + ReLU" if ci < 6 else "conv6"
            elif kind == "MaxPool":
                cfg = "2x2, s2"
            else:
                cfg = "128 ch"
                kind = "BatchNorm + ReLU"
            n = sum(p.numel() for p in mod.parameters())
            rows.append([kind, cfg, f"{x.shape[1]} x {x.shape[2]} x {x.shape[3]}", f"{n:,}"])
        rows.append(["flatten", "", f"{x.numel():,}", "0"])
        for mod, name in [(m.head[1], "fc1 + ReLU"), (m.head[3], "fc2 (softmax)")]:
            rows.append([name, f"{mod.in_features:,} -> {mod.out_features}", f"{mod.out_features}",
                         f"{sum(p.numel() for p in mod.parameters()):,}"])
    total = sum(p.numel() for p in m.parameters())
    fc1 = sum(p.numel() for p in m.head[1].parameters())
    return rows, total, fc1


def varcnn_layers():
    m = VarCNN()
    rows, li = [], 0
    sizes = np.array([[42, 43], [235, 266]])
    for c in VarCNN.CFG:
        if c == "M":
            sizes = sizes // 2
            rows.append(["MaxPool + re-mask", "2x2, s2", f"{sizes[0][0]}x{sizes[0][1]}", f"{sizes[1][0]}x{sizes[1][1]}", "0"])
            continue
        i, o, k, s, p = c
        sizes = (sizes + 2 * p - k) // s + 1
        n = sum(q.numel() for q in m.convs[li].parameters()) + sum(q.numel() for q in m.bns[li].parameters())
        li += 1
        rows.append([f"conv{li} + masked BN + ReLU", f"{k}x{k}, {o}, s{s}, p{p}",
                     f"{sizes[0][0]}x{sizes[0][1]}", f"{sizes[1][0]}x{sizes[1][1]}", f"{n:,}"])
    rows.append(["masked global mean + max", "over real pixels", "256", "256", "0"])
    rows.append(["dropout 0.3, fc", "256 -> 10", "10", "10", f"{sum(q.numel() for q in m.head.parameters()):,}"])
    return rows, sum(q.numel() for q in m.parameters())


# ---------------------------------------------------------------- figures
def fig_sizes():
    fig, axes = plt.subplots(2, 5, figsize=(7.2, 3.6), sharex=True, sharey=True)
    for ci, (ax, c) in enumerate(zip(axes.ravel(), classes)):
        ax.scatter(hw[:, 1], hw[:, 0], s=2, color=CONTEXT, linewidths=0, rasterized=True)
        s = hw[labels == ci]
        ax.scatter(s[:, 1], s[:, 0], s=3, color=BLUE, linewidths=0, rasterized=True)
        ax.set_title(f"{c}\nn = {len(s)}", pad=3, fontsize=7.5, linespacing=1.1)
        ax.set_xlim(0, 280), ax.set_ylim(0, 250)
        ax.grid(color=GRID, linewidth=0.5), ax.set_axisbelow(True)
        ax.tick_params(length=2, labelsize=7)
    for ax in axes[1]:
        ax.set_xlabel("width (px)")
    for ax in axes[:, 0]:
        ax.set_ylabel("height (px)")
    fig.tight_layout(pad=0.4, w_pad=0.3, h_pad=0.6)
    p = FIG / "sizes_by_class.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def fig_splits():
    rows = [
        ("MCNN, pad (paper)", "mcnn_pad_tr80", "mcnn_pad_tr80_group"),
        ("VarCNN, native", "varcnn_native_tr80", "varcnn_native_tr80_group"),
        ("VarCNN, randsize", "varcnn_randsize_tr80", "varcnn_randsize_tr80_group"),
        ("MCNN, randsize-pad", "mcnn_randsize-pad_tr80", "mcnn_randsize-pad_tr80_group"),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 2.3))
    for yi, (label, a, b) in enumerate(rows):
        y = len(rows) - 1 - yi
        (ma, sa), (mb, sb) = acc_ms(res(a)), acc_ms(res(b))
        ax.plot([mb, ma], [y, y], color=CONTEXT, linewidth=2, zorder=1, solid_capstyle="round")
        for m_, s_, col, dy, ha, off in [(ma, sa, BLUE, 0, "left", 0.35), (mb, sb, ORANGE, 0, "right", -0.35)]:
            ax.errorbar(m_, y, xerr=s_, fmt="o", ms=6, color=col, ecolor=col, elinewidth=1, capsize=0,
                        markeredgecolor="#fcfcfb", markeredgewidth=1.2, zorder=3)
            ax.text(m_ + off, y + 0.22, f"{m_:.1f}", ha=ha, va="bottom", fontsize=7.5, color=INK2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], color=INK)
    ax.set_xlim(86, 101)
    ax.set_ylim(-0.6, len(rows) - 0.3)
    ax.set_xlabel("test accuracy (%), mean ± std over 3 seeds")
    ax.grid(axis="x", color=GRID, linewidth=0.5), ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    h1 = ax.scatter([], [], color=BLUE, s=30, label="random 80/20 split")
    h2 = ax.scatter([], [], color=ORANGE, s=30, label="group split (no near-duplicates across)")
    ax.legend(handles=[h1, h2], loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, frameon=False,
              fontsize=7.5, handletextpad=0.3, columnspacing=1.5)
    fig.tight_layout(pad=0.3)
    p = FIG / "random_vs_group.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def fig_pairs():
    import csv

    manifest = ROOT / "data/dedup_t90/manifest.csv"
    rows = [r for r in csv.DictReader(open(manifest)) if r["kept"] == "0"]
    picked = []
    for lo, hi in [(0.90, 0.91), (0.92, 0.93), (0.95, 0.97)]:
        sel = [r for r in rows if lo <= float(r["similarity"]) < hi]
        picked += sel[:: max(1, len(sel) // 3)][:3]
    fig, axes = plt.subplots(3, 6, figsize=(7.2, 3.6))
    for k, r in enumerate(picked):
        row, col = divmod(k, 3)
        for j, name in enumerate([r["file"], r["duplicate_of"]]):
            ax = axes[row, 2 * col + j]
            ax.imshow(Image.open(DATA_ROOT / r["class"] / name).convert("L"), cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_title(name.removesuffix(".png"), fontsize=6.5, color=INK2, pad=2)
        axes[row, 2 * col].text(1.05, -0.12, f"r = {float(r['similarity']):.3f}", transform=axes[row, 2 * col].transAxes,
                                ha="center", va="top", fontsize=7, color=INK)
    fig.tight_layout(pad=0.3, h_pad=1.2)
    p = FIG / "duplicate_pairs.png"
    fig.savefig(p)
    plt.close(fig)
    return p


# ---------------------------------------------------------------- PDF building blocks
ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=ss["Normal"], fontName="Helvetica", fontSize=9.5, leading=13.2,
                      textColor=colors.HexColor(INK), spaceAfter=5)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8, leading=10.5, textColor=colors.HexColor(INK2))
CELL = ParagraphStyle("cell", parent=BODY, fontSize=8, leading=10, spaceAfter=0)
CELLB = ParagraphStyle("cellb", parent=CELL, fontName="Helvetica-Bold")
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=14, leading=18,
                    spaceBefore=10, spaceAfter=6, textColor=colors.HexColor(INK), keepWithNext=1)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=11, leading=14,
                    spaceBefore=8, spaceAfter=4, textColor=colors.HexColor(INK), keepWithNext=1)
TITLE = ParagraphStyle("title", parent=H1, fontSize=19, leading=24, spaceAfter=4, alignment=TA_LEFT)
CAPTION = ParagraphStyle("cap", parent=SMALL, spaceBefore=3, spaceAfter=10)


def P(t, st=BODY):
    return Paragraph(t, st)


def bullets(items):
    return ListFlowable([ListItem(P(t), leftIndent=10) for t in items], bulletType="bullet", start="•",
                        leftIndent=12, bulletFontSize=8)


def table(rows, widths, header=True, align_right=(), bold_rows=()):
    data = [[P(str(c), CELLB if (header and i == 0) or i in bold_rows else CELL) for c in r]
            for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1 if header else 0)
    st = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, -1), (-1, -1), 0.6, colors.HexColor(MUTED)),
        ("LINEBELOW", (0, 1), (-1, -2), 0.3, colors.HexColor(GRID)),
    ]
    if header:
        st += [("LINEABOVE", (0, 0), (-1, 0), 0.6, colors.HexColor(MUTED)),
               ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor(MUTED)),
               ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f3f2ef"))]
    t.setStyle(TableStyle(st))
    return t


def figure(path, width_mm, caption):
    im = Image.open(path)
    h = width_mm * im.height / im.width
    return KeepTogether([RLImage(str(path), width=width_mm * mm, height=h * mm), P(caption, CAPTION)])


def on_page(c, doc):
    c.saveState()
    c.setFont("Helvetica", 7.5)
    c.setFillColor(colors.HexColor(MUTED))
    c.drawString(18 * mm, 10 * mm, "MCNN reimplementation and leakage analysis")
    c.drawRightString(A4[0] - 18 * mm, 10 * mm, f"{doc.page}")
    c.restoreState()


# ---------------------------------------------------------------- content
def build():
    f_sizes, f_splits, f_pairs = fig_sizes(), fig_splits(), fig_pairs()
    mrows, mtotal, mfc1 = mcnn_layers()
    vrows, vtotal = varcnn_layers()
    dupes = json.loads((RES / "dupes.json").read_text())
    X, tl, _, _ = load_thumbs()
    assert (tl == labels).all()

    s = []
    s += [P("MCNN on forward-looking sonar: reimplementation, a variable-size CNN, and how much of 99.9% is leakage", TITLE),
          P(f"Report generated {date.today().isoformat()} from <font face='Courier'>results/*.json</font> in the "
            "sonar-cnn repository.", SMALL), Spacer(1, 6)]

    # 1 Summary
    g = {k: acc_ms(res(k))[0] for k in ["mcnn_pad_tr90", "mcnn_pad_tr80_group", "mcnn_randsize-pad_tr80_group",
                                        "varcnn_randsize_tr80_group", "mcnn_silhouette_tr80"]}
    sb = res("size_baseline_tr80")
    s += [P("1. Summary", H1), bullets([
        f"<b>The paper's number reproduces.</b> MCNN as described in Table 2 of Çimen &amp; Bolat reaches "
        f"{g['mcnn_pad_tr90']:.2f}% on a random 90/10 split, matching the 99.87% stated in the paper's text "
        "(the 99.98% in the abstract appears to be a typo).",
        f"<b>Crop size alone nearly solves the task.</b> A random forest given only each image's height and width "
        f"(no pixels) gets {acc_str(sb)}%. Padding to 305x305 exposes that size to the network: MCNN trained on "
        f"silhouettes with all sonar content removed still gets {g['mcnn_silhouette_tr80']:.1f}%.",
        "<b>Random splits are dominated by near-duplicate frames.</b> A 1-nearest-neighbour lookup on 32x32 "
        f"thumbnails gets {100 * dupes['splits'][0]['nn1_acc']:.2f}% on a random split; "
        f"{100 * dupes['splits'][0]['frac_best_sim_gt_0.90']:.0f}% of test images have a training image with "
        "correlation above 0.9.",
        f"<b>Honest numbers are lower.</b> On a split by pseudo-sequence groups, MCNN drops to "
        f"{g['mcnn_pad_tr80_group']:.1f}% with padding and {g['mcnn_randsize-pad_tr80_group']:.1f}% once size "
        f"information is removed. The proposed VarCNN (fully convolutional, {vtotal / 1e3:.0f}k parameters, no "
        f"padding effect) holds at {g['varcnn_randsize_tr80_group']:.1f}%.",
    ])]

    # 2 Dataset
    s += [P("2. Dataset", H1),
          P("Valdenegro-Toro marine debris forward-looking sonar dataset (ARIS Explorer 3000, water tank), folder "
            "<font face='Courier'>md_fls_dataset/data/watertank-cropped</font> of "
            "<font face='Courier'>mvaldenegro/marine-debris-fls-datasets</font> at commit 4a77998. 2,364 PNG images "
            "in 10 classes; per-class counts match the paper's Table 1 exactly. Images are stored as RGBA with "
            "R=G=B and constant alpha, so one channel is used. Each image is a hand-cropped box around one object, "
            "and the box size depends strongly on the class (Table 1, Figure 1). File numbers do not follow "
            "recording order: consecutive file numbers are no more similar than random pairs of the same class "
            f"(median thumbnail correlation {dupes['consecutive_median']:.2f} vs {dupes['random_same_class_median']:.2f})."),
          ]
    rows = [["Class", "Images", "Mean h x w (px)", "Height range", "Width range"]]
    for ci, c in enumerate(classes):
        h = hw[labels == ci]
        rows.append([c, len(h), f"{h[:, 0].mean():.0f} x {h[:, 1].mean():.0f}", f"{h[:, 0].min()}-{h[:, 0].max()}",
                     f"{h[:, 1].min()}-{h[:, 1].max()}"])
    rows.append(["total", len(labels), "", f"{hw[:, 0].min()}-{hw[:, 0].max()}", f"{hw[:, 1].min()}-{hw[:, 1].max()}"])
    s += [KeepTogether([P("Table 1. Images per class and crop size.", SMALL), table(rows, [38, 20, 35, 30, 30], bold_rows=(len(rows) - 1,))]),
          Spacer(1, 8),
          figure(f_sizes, 170, "Figure 1. Crop height vs width for each class (blue) against all images (grey). "
                 "Standing and shampoo bottles are long and flat, drink cartons small and square, propellers and "
                 "hooks large: size and aspect ratio alone separate most classes.")]

    # 3 Architectures
    s += [P("3. Architectures", H1), P("3.1 MCNN (paper replication)", H2),
          P(f"Built from Table 2 of the paper for 305x305 zero-padded input. {mtotal:,} parameters, of which "
            f"{mfc1:,} ({100 * mfc1 / mtotal:.0f}%) are in fc1. Two departures from the text, both because the "
            "paper is ambiguous: a ReLU after every convolution (the text mentions only one, after conv6), and "
            "'same' padding on the 2x2 conv6 so the output is the paper's 9x9x128. conv1-3 use padding 2: the paper "
            "lists padding 1 for conv2 and conv3, which would give 76 and 38 instead of its stated 77 and 39."),
          KeepTogether([P("Table 2. MCNN layers (input 1 x 305 x 305).", SMALL),
                        table([["Layer", "Config (k, ch, stride, pad)", "Output (C x H x W)", "Params"]] + mrows +
                              [["total", "", "", f"{mtotal:,}"]], [42, 50, 42, 30], bold_rows=(len(mrows) + 1,))]),
          P("3.2 VarCNN (variable-size input)", H2),
          P("A fully convolutional variant that takes each image at its original size. Batches are padded to the "
            "largest image only so they can be stacked into one tensor. After every layer the padded area is set "
            "back to zero (including after max-pooling, where an odd size would otherwise leave a real value in "
            "the padded area), BatchNorm statistics are computed over real pixels only, and the final pooling "
            "(global mean and global max, concatenated) uses real pixels only. In eval mode each sample's output "
            "is therefore identical to running that image alone; <font face='Courier'>tests/test_varcnn.py</font> "
            "checks this to 1e-5. Compared with MCNN, conv2 has stride 1 so that the smallest images (42 px) still "
            f"have spatial extent at the end, and the 2.65M-parameter fc1 is replaced by global pooling: "
            f"{vtotal:,} parameters in total, {mtotal / vtotal:.0f}x fewer than MCNN."),
          KeepTogether([P("Table 3. VarCNN layers, with output sizes for the smallest and largest images in the dataset.", SMALL),
                        table([["Layer", "Config (k, ch, stride, pad)", "42 x 43 in", "235 x 266 in", "Params"]] + vrows +
                              [["total", "", "", "", f"{vtotal:,}"]], [52, 42, 22, 24, 24], bold_rows=(len(vrows) + 1,))]),
          ]

    # 4 Protocol
    s += [P("4. Training and evaluation", H1),
          KeepTogether([P("Table 4. Training configuration (identical for all experiments).", SMALL), table([
              ["Setting", "Value"],
              ["Optimizer", "AdamW, lr 1e-3, weight decay 1e-4, cosine decay to 0 over all steps"],
              ["Batch size / epochs", "32 / 30 (fixed-size inputs), 50 (randsize inputs)"],
              ["Loss", "cross-entropy, no class weighting"],
              ["Augmentation", "none, except the random resize of randsize inputs"],
              ["Split", "stratified 80/20 (one run at 90/10); seeds 0, 1, 2"],
              ["Model selection", "none: no validation set, accuracy at the final epoch (as in the paper)"],
              ["Hardware", "NVIDIA RTX 3050 Laptop (4 GB), PyTorch 2.15.0 + CUDA 13.2, Python 3.12"],
          ], [40, 124])]), Spacer(1, 8),
          KeepTogether([P("Table 5. Input modes. 'Probe' modes replace all sonar content with 1s, so the model can only use the crop's size and shape.", SMALL), table([
              ["Input", "What the model sees"],
              ["pad", "image centered on a 305x305 zero canvas (the paper's setup)"],
              ["silhouette (probe)", "as pad, content replaced by 1s"],
              ["native", "image at its original size (VarCNN only)"],
              ["native-constant (probe)", "as native, content replaced by 1s"],
              ["randsize", "image resized to random h, w, each uniform in [48, 160] and independent of class; new size every epoch for training, one fixed seeded size per test image"],
              ["randsize-pad", "as randsize, then centered on a 305x305 zero canvas (for MCNN)"],
              ["...-constant (probe)", "as above, content replaced by 1s"],
          ], [40, 124])]), Spacer(1, 8),
          P("<b>Splits.</b> <i>Random</i>: stratified by class, the paper's protocol. <i>Group</i>: images are "
            "linked when their 32x32 mean-centered thumbnails have correlation 0.85 or more, and connected chains "
            "(single linkage) within a class form a group, an approximation of a stretch of one recording. 358 "
            "groups result. Whole groups are assigned to the test set in random order until it holds about 20% of "
            "each class, so no test image has a training image above 0.85 similarity."),
          ]

    # 5 Results
    rows = [["Model", "Input", "Split", "Accuracy (%)", "Macro-F1 (%)", "Errors per seed"]]
    for name, model, inp, split in EXPERIMENTS:
        r = res(name)
        if r is None:
            continue
        errs = "/".join(str(x["errors"]) for x in r["runs"]) + f" of {r['runs'][0]['n_test']}"
        rows.append([model, inp, split, acc_str(r), acc_str(r, "macro_f1"), errs])
    sb90 = res("size_baseline_tr90")
    rows.append(["Random forest", "(h, w) only", "random", acc_str(sb), acc_str(sb, "macro_f1"), ""])
    rows.append(["Random forest", "(h, w) only, 90/10", "random", acc_str(sb90), acc_str(sb90, "macro_f1"), ""])
    s += [P("5. Results", H1),
          KeepTogether([P("Table 6. All experiments: mean ± std over 3 seeds. 'Errors per seed' counts misclassified test images.", SMALL),
                        table(rows, [24, 40, 28, 25, 25, 32])]),
          P("5.1 Image size leaks the class", H2),
          P("The probes isolate what the models can learn from crop size and shape alone. With padding (the "
            f"paper's setup), MCNN reaches {acc_str(res('mcnn_silhouette_tr80'))}% with no sonar content at all, "
            f"close to the {acc_str(sb)}% of a random forest on (h, w). Removing padding does not remove the cue: "
            f"VarCNN on constant images of the original size still reaches {acc_str(res('varcnn_native-constant_tr80'))}%, "
            "because convolutions see the image boundary. Resizing every image to a random size independent of its "
            "class removes it completely: both randsize probes fall to 19.03%, the share of the largest class "
            "(bottle). With real content, both models still score above 99% on a random split, so on random splits "
            "the size cue is available but not needed."),
          P("5.2 Near-duplicates inflate random splits", H2),
          ]
    d = dupes["splits"]
    s += [KeepTogether([P("Table 7. Nearest training image for each test image, using 32x32 thumbnails (no size information, no learning). Mean over 3 splits.", SMALL),
                        table([["Split", "1-NN accuracy", "Median best similarity", "Best > 0.90", "Best > 0.95"]] +
                              [[r["split"], f"{100 * r['nn1_acc']:.2f} ± {100 * r['nn1_acc_std']:.2f}", f"{r['median_best_sim']:.3f}",
                                f"{100 * r['frac_best_sim_gt_0.90']:.1f}%", f"{100 * r['frac_best_sim_gt_0.95']:.1f}%"] for r in d],
                              [30, 32, 38, 30, 30])]),
          Spacer(1, 4),
          P("On a random split, a nearest-neighbour lookup nearly matches the CNNs, because most test images have "
            "an almost identical frame in training (Figure 3). A split by file-number blocks behaves like a random "
            "split, since file numbers are shuffled. The group split removes every pair above 0.85 and is the only "
            "split that lowers accuracy (Figure 2). There, MCNN depends on the shortcuts: it loses 2.6 points with "
            "padding and 9 points once size is also removed, with 35 to 61 errors per seed. VarCNN loses 3 "
            "to 4 points and has the lowest spread of any model on the group split."),
          figure(f_splits, 160, "Figure 2. Accuracy on a random split vs the group split for the four setups run on both. "
                 "Dots are means over 3 seeds, whiskers ± 1 std. Note the axis starts at 86%."),
          figure(f_pairs, 170, "Figure 3. Removed near-duplicate pairs at three similarity levels (thumbnail "
                 "correlation r). Even pairs just above 0.90 are the same object in the same pose, frames apart."),
          P("5.3 Deduplicated dataset", H2),
          ]
    short = {"drink-carton": "carton", "shampoo-bottle": "shamp.", "standing-bottle": "stand.", "propeller": "propel."}
    rows = [["Threshold", "Total"] + [short.get(c, c) for c in classes],
            ["original", len(labels)] + [int((labels == i).sum()) for i in range(len(classes))]]
    for t in (0.95, 0.9, 0.85):
        keep, _ = greedy_keep(X, labels, t, 0)
        rows.append([f"{t:.2f}", int(keep.sum())] + [int((keep & (labels == i)).sum()) for i in range(len(classes))])
    s += [P("Within each class, images are visited in a seeded random order and kept only if their thumbnail "
            "correlation with every image already kept is below a threshold. At 0.9 this keeps 1,172 of 2,364 "
            "images (<font face='Courier'>data/dedup_t90/</font>, with a manifest recording what each removed "
            "image duplicates)."),
          KeepTogether([P("Table 8. Images kept per class at each threshold (carton = drink-carton, propel. = propeller, shamp./stand. = shampoo/standing bottle).", SMALL),
                        table(rows, [17, 12] + [14.5] * 10)]), Spacer(1, 4),
          P(f"On the 0.9 dataset with a random split, VarCNN (randsize) scores {acc_str(res('varcnn_randsize_tr80_dedup_t90'))}% "
            f"and MCNN (randsize-pad) {acc_str(res('mcnn_randsize-pad_tr80_dedup_t90'))}%, barely below the full dataset. "
            "Greedy deduplication at 0.9 leaves frames just under the threshold, which still come from the same "
            "stretch of a recording, so every test image keeps a close relative in training. It gives a smaller, "
            "less redundant dataset but not a leak-free evaluation; the group split remains the honest test."),
          ]

    # 6 Paper
    s += [KeepTogether([P("6. Implications for the paper", H1), bullets([
        "The headline 99.98% (abstract, Tables 3 and 5) is inconsistent with the 99.87% in the Results text and "
        "with Table 4. This replication gets 99.86 ± 0.20% at 90/10, supporting 99.87%.",
        "Table 3's standard deviations are impossible: with ~237 test images, one error in one of 10 runs already "
        "gives 99.96%, and a mean of 99.98% allows a standard deviation of at most ~0.06, not 0.58.",
        "Random image-level splits put near-duplicate frames on both sides. A split by recording sequence (or the "
        "group split here) is needed; accuracy then drops to about 97% for MCNN as published.",
        "Zero-padding hand-cropped images to 305x305 exposes the crop size, which alone predicts the class with "
        "about 98% accuracy. Any claim about learning from sonar returns needs a control for this.",
        f"The 'fewer parameters' claim does not hold: {100 * mfc1 / mtotal:.0f}% of MCNN's {mtotal / 1e6:.1f}M "
        f"parameters are in fc1, and a fully convolutional network with {vtotal / 1e3:.0f}k parameters is more "
        "accurate once shortcuts are removed.",
        "Other issues from the review: Table 5 lists MobileNet at 96.7% while the text calls it the 99.7% previous "
        "best; Table 6 gives SqueezeNet ~112M parameters (it has ~1.2M); Table 5 compares against results on "
        "different data and splits; no optimizer, learning rate, epochs or validation set are reported.",
    ])])]

    # 7 Limitations
    s += [P("7. Limitations and next steps", H1), bullets([
        "The groups only approximate recording sequences. The full watertank release includes the original "
        "sequences and would allow a true split by sequence.",
        "3 seeds per setting and no validation set; reported accuracy is at the final epoch with fixed "
        "hyperparameters, untuned for either model.",
        "No per-class metrics yet (the paper's Table 4 equivalent); this needs a rerun of the main settings.",
        "Random resizing distorts aspect ratio; it removes the size cue but also changes object shape. A "
        "scale-only augmentation with size jitter at test time would be a gentler alternative.",
    ])]

    # Appendix
    s += [P("Appendix: reproducing", H1), P(
        "<font face='Courier' size=8>./run_all.sh &nbsp; ./run_randsize.sh &nbsp; ./run_group.sh<br/>"
        "uv run python -m sonar_cnn.dupes<br/>uv run python -m sonar_cnn.dedup --threshold 0.9 &nbsp; ./run_dedup.sh<br/>"
        "uv run pytest<br/>uv run --group report python report/build_report.py</font>")]

    doc = SimpleDocTemplate(str(OUT / "sonar_cnn_report.pdf"), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm, title="MCNN sonar reimplementation report",
                            author="Oguzb")
    doc.build(s, onFirstPage=on_page, onLaterPages=on_page)
    print("wrote", OUT / "sonar_cnn_report.pdf")


if __name__ == "__main__":
    build()
