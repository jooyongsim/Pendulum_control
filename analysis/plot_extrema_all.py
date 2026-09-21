"""Draw the extremum detection for EVERY measured log and EVERY decay run.

`plot_extrema_detection.py` draws one overview per log plus a single zoom. This
script covers the measurements exhaustively instead:

    extrema_full_<log>.png       the whole 60 s trace of each log, releases marked
    extrema_run_<n>.png          one figure per decay run: the half period, and
                                 both envelope fits overlaid for comparison
    extrema_run_<n>_viscous.png  the viscous-only model on its own, with residuals
    extrema_run_<n>_joint.png    the same page for viscous + Coulomb (--select joint)
    extrema_all_runs.png         every run side by side, releases aligned

The marks are the same everywhere: red up triangles are maxima, blue down
triangles are minima, the dashed green line is a release. Maxima and minima are
kept apart on purpose -- that distinction is what makes ADJACENT extrema half a
period apart rather than a whole one, and every number downstream (T, omega_n,
sigma, the envelope) rests on it.

By default the figures are both saved and opened in matplotlib windows, so they
can be panned and zoomed rather than only looked at as PNGs.

    python plot_extrema_all.py                      # save + open 9 windows
    python plot_extrema_all.py --no-show            # save only, no windows
    python plot_extrema_all.py --select viscous     # only the viscous-only pages
    python plot_extrema_all.py --select runs --no-save   # just look, write nothing
    python plot_extrema_all.py --select viscous,joint    # one page per model
    python plot_extrema_all.py --out-dir figs_extrema
"""
import argparse
import os

import matplotlib
import numpy as np

from _common import DATA_DIR, ENC_RES_DEG, extrema_of, log_paths, split_runs
from pendulum_model_id import envelope_decay, single_mechanism_fits

MAX_C, MIN_C, REL_C, TRACE_C, ANN_C = "#c0392b", "#2774B8", "#3D8C54", "#5a6070", "#E97820"
VISC_C, BOTH_C = "#8E44AD", "#117A65"


def envelope_fits(t_ext, amp_rad):
    """Fit both envelopes to one run, exactly as the identification does.

    Only extrema above twice the encoder resolution are used -- below that the
    "amplitude" is quantisation noise, and the identification drops them too.

        viscous only      A = A0 exp(-sigma t)
        viscous + Coulomb A = (A0 + c/sigma) exp(-sigma t) - c/sigma

    Both R^2 values returned here are measured on A, so they can be compared
    with each other; the viscous-only fit itself is still made in log space,
    where it is linear.
    """
    usable = amp_rad > 2 * np.deg2rad(ENC_RES_DEG)
    if usable.sum() < 8:
        return None
    t0 = t_ext[usable][0]
    tau = t_ext[usable] - t0
    A = amp_rad[usable]

    both = envelope_decay(tau, A)
    visc = single_mechanism_fits(tau, A)["viscous"]

    def curve(kind, x):
        if kind == "viscous":
            return visc["A0"] * np.exp(-visc["sigma"] * x)
        c_over_s = both["coulomb"] / both["sigma"]
        return (both["A0"] + c_over_s) * np.exp(-both["sigma"] * x) - c_over_s

    return dict(t0=t0, tau=tau, A=A, both=both, visc=visc, curve=curve,
                t_end=float(tau[-1]))


def runs_of(t, idx, amp, min_extrema=8):
    """The splits that are a real decay: enough extrema, and actually decaying.

    A log that starts at rest also splits at its first sample; that is not a
    release and must not be drawn as one.
    """
    out = []
    for a, b in split_runs(t[idx], amp):
        usable = (amp[a:b] > 2 * np.deg2rad(ENC_RES_DEG)).sum()
        if usable >= min_extrema and amp[a:b][-1] < amp[a:b][0]:
            out.append((a, b))
    return out


def draw(ax, t, deg, ext_t, ext_deg, releases=(), legend=True, marker=5):
    is_max = ext_deg > 0
    ax.plot(t, deg, lw=0.6, color=TRACE_C, label="encoder angle", zorder=1)
    ax.plot(ext_t[is_max], ext_deg[is_max], "^", ms=marker, color=MAX_C,
            label=f"maxima ({int(is_max.sum())})", zorder=3)
    ax.plot(ext_t[~is_max], ext_deg[~is_max], "v", ms=marker, color=MIN_C,
            label=f"minima ({int((~is_max).sum())})", zorder=3)
    ax.axhline(0, color="k", lw=0.8, ls=":", zorder=2)
    for i, tr in enumerate(releases):
        ax.axvline(tr, color=REL_C, lw=1.3, ls="--", zorder=2,
                   label="release" if (legend and i == 0) else None)
    ax.grid(alpha=0.3)
    if legend:
        ax.legend(fontsize=8, loc="upper right", ncol=2)


def draw_envelopes(ax, f, mirrored=True, lw=1.6, labels=True, t_ref=0.0):
    """Overlay the two fitted envelopes on a trace.

    `t_ref` is the zero of the axis being drawn on: 0 for absolute log time,
    the release instant for a plot of time-since-release.
    """
    x = np.linspace(0, f["t_end"], 400)
    for kind, color, style, lab in (
            ("viscous", VISC_C, "--", "viscous only"),
            ("both", BOTH_C, "-", "viscous + Coulomb")):
        y = np.rad2deg(f["curve"](kind, x))
        ax.plot(f["t0"] - t_ref + x, y, style, lw=lw, color=color, zorder=4,
                label=lab if labels else None)
        if mirrored:
            ax.plot(f["t0"] - t_ref + x, -y, style, lw=lw, color=color, zorder=4)


def single_model_figure(plt, k, r, kind):
    """One run, one damping model, on its own page.

    Overlaying both fits answers "which is better"; this answers "where does
    THIS model go wrong", which is the question the residual panel settles.
    """
    f = r["fit"]
    color = VISC_C if kind == "viscous" else BOTH_C
    style = "--" if kind == "viscous" else "-"
    if kind == "viscous":
        p = f["visc"]
        label = f"viscous only:  $\\sigma$ = {p['sigma']:.4f} 1/s"
        r2 = p["r2_on_A"]
        note = ("A pure exponential must keep the SAME fraction of its "
                "amplitude every cycle.")
    else:
        p = f["both"]
        label = (f"viscous + Coulomb:  $\\sigma$ = {p['sigma']:.4f} 1/s,  "
                 f"c = {np.rad2deg(p['coulomb']):.2f}°/s")
        r2 = p["r2"]
        note = "Coulomb adds a constant loss per cycle on top of the exponential."

    fig = plt.figure(figsize=(15, 8.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1])
    ax_tr = fig.add_subplot(gs[0, :])
    ax_lin = fig.add_subplot(gs[1, 0])
    ax_log = fig.add_subplot(gs[1, 1])
    ax_res = fig.add_subplot(gs[1, 2])

    model_name = "viscous only" if kind == "viscous" else "viscous + Coulomb"
    fig.suptitle(f"run {k}: {r['name']} — release {r['amp0']:.1f}°,  "
                 f"{model_name} model,  $R^2_A$ = {r2:.4f}", fontsize=12)

    pad = 0.6
    lo, hi = r["t0"] - pad, r["t1"] + pad
    win = (r["t"] >= lo) & (r["t"] <= hi)
    draw(ax_tr, r["t"][win], r["deg"][win], r["ext_t"], r["ext_deg"],
         releases=[r["t0"]], legend=False)
    x = np.linspace(0, f["t_end"], 400)
    y = np.rad2deg(f["curve"](kind, x))
    ax_tr.plot(f["t0"] + x, y, style, lw=1.9, color=color, zorder=4, label=label)
    ax_tr.plot(f["t0"] + x, -y, style, lw=1.9, color=color, zorder=4)
    span = max(np.abs(r["deg"][win]).max(), np.abs(y).max())
    ax_tr.set(xlim=(lo, hi), ylim=(-1.15 * span, 1.3 * span),
              xlabel="Time (s)", ylabel="Angle from rest (deg)")
    ax_tr.legend(fontsize=9, loc="upper right", ncol=2)
    ax_tr.set_title("trace with this model's envelope only", fontsize=10)

    A_deg = np.rad2deg(f["A"])
    fit_deg = np.rad2deg(f["curve"](kind, f["tau"]))
    for ax, logscale in ((ax_lin, False), (ax_log, True)):
        ax.plot(f["tau"], A_deg, "o", ms=4.5, color="#34495E",
                label="measured", zorder=3)
        ax.plot(x, y, style, lw=1.9, color=color, label="fit")
        ax.set(xlabel="Time since first fitted extremum (s)",
               ylabel="Amplitude (deg)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        if logscale:
            ax.set_yscale("log")
            ax.set_title("log axis: an exponential is a straight line here",
                         fontsize=10)
        else:
            ax.set_title("linear axis", fontsize=10)

    resid = A_deg - fit_deg
    ax_res.axhline(0, color="k", lw=0.9)
    ax_res.plot(f["tau"], resid, "o-", ms=4, lw=1.0, color=color)
    ax_res.set(xlabel="Time since first fitted extremum (s)",
               ylabel="measured − fit (deg)")
    ax_res.grid(alpha=0.3)
    ax_res.set_title(f"residuals: max |e| = {np.abs(resid).max():.2f}°, "
                     f"RMS = {np.sqrt((resid ** 2).mean()):.2f}°", fontsize=10)

    fig.text(0.5, 0.005, note, ha="center", fontsize=9, color="0.35")
    fig.tight_layout(rect=(0, 0.02, 1, 0.955))
    return fig


def annotate_half_period(ax, ext_t, ext_deg, lo, hi):
    """Mark one representative adjacent gap = T/2 inside [lo, hi]."""
    m = (ext_t >= lo) & (ext_t <= hi)
    if m.sum() < 3:
        return None
    tm, gaps = ext_t[m], np.diff(ext_t[m])
    j = int(np.argmin(np.abs(gaps - np.median(gaps))))
    t0, t1 = tm[j], tm[j + 1]
    y = np.abs(ext_deg[m]).max() * 0.80
    ax.annotate("", xy=(t0, y), xytext=(t1, y),
                arrowprops=dict(arrowstyle="<->", color=ANN_C, lw=1.8))
    ax.text((t0 + t1) / 2, y * 1.08, f"T/2 = {t1 - t0:.3f} s", ha="center",
            color=ANN_C, fontsize=9, fontweight="bold")
    return t1 - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--glob", default="encoder_log_*.csv")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--no-show", action="store_true",
                    help="write the PNGs without opening any window")
    ap.add_argument("--select", default="all",
                    help="which figures to draw: any comma-separated mix of "
                         "full, runs, viscous, joint, summary (default: all)")
    ap.add_argument("--no-save", action="store_true",
                    help="only look at the figures; write nothing")
    args = ap.parse_args()

    KINDS = {"full", "runs", "viscous", "joint", "summary"}
    # "joint" is off by default: the run figures already carry both envelopes,
    # so a page per run for the joint model alone is only worth making on ask
    wanted = (KINDS - {"joint"}) if args.select == "all" else {
        s.strip() for s in args.select.split(",")}
    unknown = wanted - KINDS
    if unknown:
        raise SystemExit(f"--select: unknown {sorted(unknown)}; "
                         f"pick from {', '.join(sorted(KINDS))}")
    if args.no_save and args.no_show:
        raise SystemExit("--no-save with --no-show would produce nothing")

    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = log_paths(args.glob, args.data_dir)
    if not paths:
        raise SystemExit(f"no logs matching {args.glob} in {args.data_dir}")
    if not args.no_save:
        os.makedirs(args.out_dir, exist_ok=True)

    written, all_runs, shown = [], [], []

    def finish(fig, filename):
        """Save unless told not to, and keep the figure open if it is to be shown.

        Closing every figure as it was built is what made --no-show the only
        working mode: plt.show() had nothing left to display.
        """
        if not args.no_save:
            out = os.path.join(args.out_dir, filename)
            fig.savefig(out, dpi=150, bbox_inches="tight")
            written.append(out)
        if args.no_show:
            plt.close(fig)
        else:
            shown.append(fig)

    # ---- one full-trace figure per log ------------------------------------
    for path in paths:
        t, theta, idx, eq = extrema_of(path)
        deg = np.rad2deg(theta - eq)
        ext_t, ext_deg = t[idx], deg[idx]
        amp = np.abs(theta[idx] - eq)
        runs = runs_of(t, idx, amp)
        name = os.path.basename(path)

        if "full" in wanted:
            fig, ax = plt.subplots(figsize=(15, 4.6))
            draw(ax, t, deg, ext_t, ext_deg, releases=[t[idx][a] for a, _ in runs])
            ax.set(xlabel="Time (s)", ylabel="Angle from rest (deg)",
                   xlim=(t[0], t[-1]))
            ax.set_title(f"{name} — {len(idx)} extrema, {len(runs)} decay run(s), "
                         f"rest at {np.rad2deg(eq):+.2f}°")
            fig.tight_layout()
            finish(fig, f"extrema_full_{name.replace('.csv', '')}.png")

        flips = int(np.sum(np.diff(np.sign(ext_deg)) != 0))
        print(f"{name}: {len(idx)} extrema ({int((ext_deg > 0).sum())} max / "
              f"{int((ext_deg <= 0).sum())} min), {flips}/{len(idx) - 1} sign changes, "
              f"{len(runs)} decay run(s), rest {np.rad2deg(eq):+.2f}°")

        for a, b in runs:
            all_runs.append(dict(name=name, t=t, deg=deg, idx=idx,
                                 ext_t=ext_t[a:b], ext_deg=ext_deg[a:b],
                                 t0=t[idx][a], t1=t[idx][b - 1],
                                 amp0=np.rad2deg(amp[a]), n=b - a,
                                 fit=envelope_fits(t[idx][a:b], amp[a:b])))

    # ---- one figure per decay run ----------------------------------------
    for k, r in enumerate(all_runs, start=1):
        gaps = np.diff(r["ext_t"])
        print(f"  run {k}: release {r['amp0']:6.2f} deg, {r['n']:3d} extrema, "
              f"t = {r['t0']:5.2f}..{r['t1']:5.2f} s, median gap "
              f"{np.median(gaps):.4f} s -> T = {2 * np.median(gaps):.4f} s")
        if r["fit"]:
            v, b = r["fit"]["visc"], r["fit"]["both"]
            print(f"          viscous only     : sigma={v['sigma']:.4f} 1/s"
                  f"{'':21}R2(A)={v['r2_on_A']:.4f}")
            print(f"          viscous + Coulomb: sigma={b['sigma']:.4f} 1/s, "
                  f"c={np.rad2deg(b['coulomb']):.3f} deg/s    "
                  f"R2(A)={b['r2']:.4f}")
        # each model on its own page, so one can be read without the other
        if r["fit"]:
            for kind, token, suffix in (("viscous", "viscous", "viscous"),
                                        ("both", "joint", "joint")):
                if token in wanted:
                    finish(single_model_figure(plt, k, r, kind),
                           f"extrema_run_{k}_{suffix}.png")

        if "runs" not in wanted:
            continue
        pad = 0.6
        lo, hi = r["t0"] - pad, r["t1"] + pad
        win = (r["t"] >= lo) & (r["t"] <= hi)
        fig, axes2 = plt.subplots(2, 2, figsize=(15, 8.4),
                                  gridspec_kw={"width_ratios": [2.0, 1]})
        axes = axes2[0]
        for col, ax in enumerate(axes):
            draw(ax, r["t"][win], r["deg"][win], r["ext_t"], r["ext_deg"],
                 releases=[r["t0"]], legend=(col == 0))
            span = np.abs(r["deg"][win]).max()
            ax.set(xlabel="Time (s)", ylim=(-1.2 * span, 1.35 * span))
        med_T = 2 * np.median(np.diff(r["ext_t"]))
        # the run identity goes in the suptitle: two long titles side by side
        # on the top row ran into each other
        fig.suptitle(f"run {k}: {r['name']} — release {r['amp0']:.1f}°, "
                     f"{r['n']} extrema over {r['t1'] - r['t0']:.1f} s, "
                     f"median T = {med_T:.3f} s", fontsize=12)
        axes[0].set(xlim=(lo, hi), ylabel="Angle from rest (deg)")
        axes[0].set_title("trace, extrema and the two fitted envelopes",
                          fontsize=10)
        if r["fit"]:
            draw_envelopes(axes[0], r["fit"], mirrored=True)
            axes[0].legend(fontsize=8, loc="upper right", ncol=3)
        zlo = r["t0"] - 0.2
        zhi = zlo + 3.2
        axes[1].set_xlim(zlo, zhi)
        zwin = (r["t"] >= zlo) & (r["t"] <= zhi)
        zspan = np.abs(r["deg"][zwin]).max()
        axes[1].set_ylim(-1.2 * zspan, 1.35 * zspan)
        half = annotate_half_period(axes[1], r["ext_t"], r["ext_deg"], zlo, zhi)
        # the gap shown is the one just after release, where the swing is
        # widest; a pendulum's period grows with amplitude, so quote it against
        # the run median rather than letting it stand as "the" period
        zoom_title = "zoom: maxima and minima alternate"
        if half:
            zoom_title += (f"\nT here = {2 * half:.3f} s "
                           f"(run median {med_T:.3f} s)")
        axes[1].set_title(zoom_title, fontsize=10)

        # bottom row: the amplitude sequence and the two fitted envelopes, on a
        # linear axis and a log one. Viscous-only decay is a straight line in
        # log A, so the log panel is where its failure is easiest to see.
        f = r["fit"]
        for col, ax in enumerate(axes2[1]):
            if not f:
                ax.set_axis_off()
                continue
            A_deg = np.rad2deg(f["A"])
            ax.plot(f["tau"], A_deg, "o", ms=4.5, color="#34495E",
                    label="measured amplitude", zorder=3)
            x = np.linspace(0, f["t_end"], 400)
            ax.plot(x, np.rad2deg(f["curve"]("viscous", x)), "--", lw=1.8,
                    color=VISC_C,
                    label=(f"viscous only: $\\sigma$={f['visc']['sigma']:.4f} 1/s"
                           f"   $R^2_A$={f['visc']['r2_on_A']:.3f}"))
            ax.plot(x, np.rad2deg(f["curve"]("both", x)), "-", lw=2.0,
                    color=BOTH_C,
                    label=(f"viscous + Coulomb: $\\sigma$={f['both']['sigma']:.4f} 1/s,"
                           f" c={np.rad2deg(f['both']['coulomb']):.2f}°/s"
                           f"   $R^2_A$={f['both']['r2']:.3f}"))
            ax.set(xlabel="Time since first fitted extremum (s)",
                   ylabel="Amplitude (deg)")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8, loc="upper right")
            if col == 0:
                ax.set_title("envelope fits (linear axis)", fontsize=10)
            else:
                ax.set_yscale("log")
                ax.set_title("log axis: pure viscous decay would be straight",
                             fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        finish(fig, f"extrema_run_{k}.png")

    # ---- every run on one page, amplitude-normalised ----------------------
    if all_runs and "summary" in wanted:
        fig, axes = plt.subplots(len(all_runs), 1, figsize=(13, 3.2 * len(all_runs)),
                                 squeeze=False, sharex=True)
        for k, (r, ax) in enumerate(zip(all_runs, axes[:, 0]), start=1):
            rel_t = r["t"] - r["t0"]
            win = (rel_t >= -0.4) & (r["t"] <= r["t1"] + 0.4)
            draw(ax, rel_t[win], r["deg"][win], r["ext_t"] - r["t0"], r["ext_deg"],
                 releases=[0.0], legend=False)
            if r["fit"]:
                draw_envelopes(ax, r["fit"], t_ref=r["t0"], lw=1.3)
            if k == 1:
                ax.legend(fontsize=8, loc="upper right", ncol=3)
            ax.set_ylabel("deg")
            ax.set_title(f"run {k}: {r['name']}, release {r['amp0']:.1f}°, "
                         f"{r['n']} extrema", fontsize=10)
        axes[-1, 0].set_xlabel("Time since release (s)")
        fig.tight_layout()
        finish(fig, "extrema_all_runs.png")

    if written:
        print(f"\n{len(written)} figures written:")
        for w in written:
            print("  " + os.path.abspath(w))
    if shown:
        # ASCII only: this console is cp949, where an em dash raises
        print(f"\nopening {len(shown)} window(s) - close them all to exit"
              f"  [backend: {matplotlib.get_backend()}]")
        plt.show()


if __name__ == "__main__":
    main()
