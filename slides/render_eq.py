"""Equation and mixed-prose images for build_pptx.py.

Rebuilt to the contract the builder expects: each entry comes back as
(path, width_px, height_px) at EQ_DPI, so picture() can size it in inches.

matplotlib's mathtext does the typesetting, with the `cm` font set -- that is
what the already-published decks in this series use, so new decks set their
equations in the same Computer Modern.

A line may be pure maths ($...$ all the way through) or prose carrying inline
maths; matplotlib handles both in one call, since it treats the $...$ spans of
a string as maths and the rest as text. The text face is chosen per string:
Malgun Gothic where there is Korean, Arial otherwise, matching the body text
each deck sets around it.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

EQ_DPI = 300

matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["axes.unicode_minus"] = False


def _has_hangul(s):
    return any("가" <= c <= "힣" for c in s)


def render(text, path, fontsize=15, color="black"):
    """Render one line to a transparent PNG; returns (path, w_px, h_px)."""
    matplotlib.rcParams["font.family"] = (
        "Malgun Gothic" if _has_hangul(text) else "Arial")
    fig = plt.figure(figsize=(0.02, 0.02))
    fig.text(0, 0, text, fontsize=fontsize, color=color)
    fig.savefig(path, dpi=EQ_DPI, bbox_inches="tight", pad_inches=0.02,
                transparent=True)
    plt.close(fig)
    with Image.open(path) as im:
        w, h = im.size
    return path, w, h


def _render_map(items, outdir, fontsize):
    os.makedirs(outdir, exist_ok=True)
    out = {}
    for key, text in items.items():
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(key))
        out[key] = render(text, os.path.join(outdir, safe + ".png"), fontsize)
    return out


def render_all(eqs, outdir, fontsize=15):
    """Display equations, one image each."""
    return _render_map(eqs, outdir, fontsize)


def render_mixed_all(lines, outdir, fontsize=13):
    """Prose lines that carry inline maths, one image each."""
    return _render_map(lines, outdir, fontsize)


if __name__ == "__main__":
    p, w, h = render(r"$K = \mathrm{lqr}(A,\ B,\ Q,\ R)"
                     r"\qquad\Longrightarrow\qquad \dot x = (A - BK)x$",
                     "eq_selftest.png")
    print(p, w, h)
