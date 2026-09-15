"""Show the extremum detection on the raw encoder trace.

Everything downstream -- period, envelope, sigma, omega_n -- is computed from
the extremum sequence, so this is the one step worth looking at directly. If the
detection is wrong here, every number after it is wrong and nothing else will
say so.

Maxima and minima are marked separately, because that distinction is what makes
adjacent extrema HALF a period apart rather than a full one.

    python plot_extrema_detection.py                 # every log in ../data
    python plot_extrema_detection.py --zoom 6 12     # zoom window in seconds
"""
import argparse
import os

import matplotlib
import numpy as np

from _common import (DATA_DIR, ENC_RES_DEG, extrema_of, log_paths, split_runs)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--glob", default="encoder_log_*.csv")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--zoom", nargs=2, type=float, metavar=("T0", "T1"),
                    default=None, help="seconds to show in the zoom panel")
    ap.add_argument("--out", default="extrema_detection.png")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = log_paths(args.glob, args.data_dir)
    if not paths:
        raise SystemExit(f"no logs matching {args.glob} in {args.data_dir}")

    fig, axes = plt.subplots(len(paths), 2, figsize=(16, 4.2 * len(paths)),
                             squeeze=False,
                             gridspec_kw={"width_ratios": [2.2, 1]})

    for row, path in enumerate(paths):
        t, theta, idx, eq = extrema_of(path)
        deg = np.rad2deg(theta - eq)
        ext_t, ext_deg = t[idx], deg[idx]
        is_max = ext_deg > 0

        amp = np.abs(theta[idx] - eq)
        # keep only the splits that actually start a usable decay, the same
        # filter the identification uses; a log that begins at rest otherwise
        # shows a "release" marker where nothing was released
        runs = [(a, b) for a, b in split_runs(t[idx], amp)
                if (amp[a:b] > 2 * np.deg2rad(ENC_RES_DEG)).sum() >= 8
                and amp[a:b][-1] < amp[a:b][0]]

        name = os.path.basename(path)
        for col, ax in enumerate(axes[row]):
            ax.plot(t, deg, lw=0.6, color="#5a6070", label="encoder angle", zorder=1)
            ax.plot(ext_t[is_max], ext_deg[is_max], "^", ms=5, color="#c0392b",
                    label=f"maxima ({int(is_max.sum())})", zorder=3)
            ax.plot(ext_t[~is_max], ext_deg[~is_max], "v", ms=5, color="#2774B8",
                    label=f"minima ({int((~is_max).sum())})", zorder=3)
            ax.axhline(0, color="k", lw=0.8, ls=":", zorder=2)
            for a, b in runs:
                ax.axvline(t[idx][a], color="#3D8C54", lw=1.2, ls="--", zorder=2)

            ax.set_xlabel("Time (s)")
            ax.grid(alpha=0.3)
            if col == 0:
                ax.set_ylabel("Angle from rest (deg)")
                ax.set_title(f"{name}\n{len(idx)} extrema, {len(runs)} decay run(s); "
                             f"rest at {np.rad2deg(eq):+.2f} deg   "
                             f"(dashed green = release)")
                ax.legend(fontsize=8, loc="upper right")
            else:
                if args.zoom:
                    lo, hi = args.zoom
                else:
                    # centre on the biggest swing, not on the first detected
                    # extremum -- in a log that starts at rest the first one is
                    # the resting point and the window would show nothing
                    peak_i = int(np.argmax(np.abs(ext_deg)))
                    lo = ext_t[peak_i] - 0.3
                    hi = lo + 4.0
                ax.set_xlim(lo, hi)
                win = (t >= lo) & (t <= hi)
                if win.any():
                    span = np.abs(deg[win]).max()
                    ax.set_ylim(-1.25 * span, 1.35 * span)
                m = (ext_t >= lo) & (ext_t <= hi)
                if m.sum() >= 3:
                    # annotate the gap closest to the median, so the label is
                    # representative rather than a release transient
                    tm = ext_t[m]
                    gaps = np.diff(tm)
                    j = int(np.argmin(np.abs(gaps - np.median(gaps))))
                    t0, t1 = tm[j], tm[j + 1]
                    y = np.abs(ext_deg[m]).max() * 0.80
                    ax.annotate("", xy=(t0, y), xytext=(t1, y),
                                arrowprops=dict(arrowstyle="<->", color="#E97820", lw=1.8))
                    ax.text((t0 + t1) / 2, y * 1.08,
                            f"adjacent extrema = T/2 = {t1 - t0:.3f} s",
                            ha="center", color="#E97820", fontsize=9, fontweight="bold")
                ax.set_title("zoom: maxima and minima alternate")

        # amplitudes must alternate in sign, or the half-period claim is wrong
        flips = int(np.sum(np.diff(np.sign(ext_deg)) != 0))
        print(f"{name}: {len(idx)} extrema, {int(is_max.sum())} max / "
              f"{int((~is_max).sum())} min, {flips}/{len(idx)-1} sign changes, "
              f"{len(runs)} decay run(s), rest {np.rad2deg(eq):+.2f} deg")
        print(f"    encoder resolution {ENC_RES_DEG:.3f} deg; "
              f"median adjacent gap {np.median(np.diff(ext_t)):.4f} s")

    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nfigure -> {os.path.abspath(args.out)}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
