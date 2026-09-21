"""Draw what `check_pid_gain_design.py` prints.

Same numbers, same source: the plant constants come from the simulator itself
and every curve is produced by the functions the check script uses, so a figure
cannot disagree with the table next to it in pid_gain_design.md.

    pid_design_placement.png   where the poles and the PID zero end up (§3, §5)
    pid_design_integrator.png  what alpha buys and what it costs (§4.1)
    pid_design_zero.png        the overshoot is the zero's doing, not the placement (§5)
    pid_design_saturation.png  the command against the +-3 wn^2 limit (§6)

Figures are saved AND opened by default.

    python plot_pid_gain_design.py                       # save + 4 windows
    python plot_pid_gain_design.py --no-show             # save only
    python plot_pid_gain_design.py --select integrator   # one of them
    python plot_pid_gain_design.py --select zero --no-save
"""
import argparse
import os

import matplotlib
import numpy as np

from check_pid_gain_design import (AL, C, FC, J, K, REF, WD, ZD, gains,
                                   linear_step, m, scenario, simulate)
from check_pid_gain_design import KEY_CASES as m_cases

PLANT_C, DES_C, ZERO_C, SAT_C = "#c0392b", "#2774B8", "#8E44AD", "#E97820"
ALPHAS = (1.0, 2.0, 4.0, 8.0, 12.0, 20.0)
CROSS = K / (2 * ZD * WD)          # alpha where the zero meets the integrator pole


def fig_placement(plt):
    """Open-loop poles, placed closed-loop poles, and the zero the PID adds."""
    kp, ki, kd = gains(AL)
    ol = np.roots([J, C, K])                     # hanging pendulum: stable pair
    cl = np.roots([J, C + kd, K + kp, ki])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.2),
                                  gridspec_kw={"width_ratios": [1, 1.15]})

    ax.axvspan(0, 4, color="#f7d9d9", zorder=0)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.axvline(0, color="0.5", lw=0.8)
    ax.plot(ol.real, ol.imag, "x", ms=13, mew=2.5, color=PLANT_C,
            label=f"plant (open loop), $\\zeta$={C / (2 * np.sqrt(K)):.4f}")
    ax.plot(cl.real, cl.imag, "o", ms=9, color=DES_C, label="closed loop, placed")
    ax.plot([-ki / kp], [0], "s", ms=9, mfc="none", mew=2, color=ZERO_C,
            label=f"PID zero  $-K_i/K_p$ = {-ki / kp:.2f}")
    for x, y, txt in ((-ZD * WD, WD * np.sqrt(1 - ZD ** 2), f"$-{ZD * WD:.1f}+{WD * np.sqrt(1 - ZD ** 2):.2f}j$"),
                      (-AL, 0.0, f"$-\\alpha = {-AL:.0f}$")):
        ax.annotate(txt, (x, y), textcoords="offset points", xytext=(8, 8),
                    fontsize=9, color=DES_C)
    ax.set(xlabel="Re $s$ [1/s]", ylabel="Im $s$ [1/s]", xlim=(-14, 4),
           ylim=(-11, 11), title="the design moves the plant's slow, barely\n"
                                 "damped pair to a fast, damped one")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")

    r = simulate(AL, T=1.5)
    ax2.axhline(r["ref_deg"], color="0.4", lw=1.0, ls="--", label="reference 20°")
    ax2.axhspan(0.98 * r["ref_deg"], 1.02 * r["ref_deg"], color="0.9", zorder=0)
    ax2.plot(r["t"], r["theta_deg"], color=DES_C, lw=1.8)
    pk = int(np.argmax(r["theta_deg"]))
    ax2.annotate(f"overshoot {r['overshoot']:.1f} %",
                 (r["t"][pk], r["theta_deg"][pk]), textcoords="offset points",
                 xytext=(14, -4), fontsize=9, color=DES_C,
                 arrowprops=dict(arrowstyle="->", color=DES_C))
    ax2.axvline(r["settle"], color="0.5", lw=1.0, ls=":")
    ax2.text(r["settle"], 3, f"  2 % settle {r['settle']:.2f} s", fontsize=9, color="0.35")
    ax2.set(xlabel="t [s]", ylabel=r"$\theta$ [deg]", xlim=(0, 1.5),
            title=f"step to 20°:  $K_p$={kp:.1f}  $K_i$={ki:.0f}  $K_d$={kd:.2f}")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig, "pid_design_placement.png"


def fig_integrator(plt):
    """What alpha buys (speed, zero steady-state error) and costs (overshoot)."""
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.4))

    # (a) step responses for a range of alpha
    ax = axes[0, 0]
    cmap = plt.get_cmap("viridis")
    for i, a in enumerate(ALPHAS):
        r = simulate(a, T=3.0)
        ax.plot(r["t"], r["theta_deg"], lw=1.5, color=cmap(i / (len(ALPHAS) - 1)),
                label=f"$\\alpha$={a:g}")
    ax.axhline(20.0, color="0.4", lw=1.0, ls="--")
    ax.set(xlabel="t [s]", ylabel=r"$\theta$ [deg]", xlim=(0, 3),
           title="step response vs the integrator pole")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    # (b) the trade-off curve
    ax = axes[0, 1]
    # one run per alpha, not one per metric: this loop is the slow part
    sweep = np.linspace(0.5, 20.0, 25)
    swept = [simulate(a, T=4.0, dt=5e-4) for a in sweep]
    ov = [r["overshoot"] for r in swept]
    ts = [r["settle"] for r in swept]
    ax.plot(sweep, ov, "-", color=DES_C, lw=1.8, label="overshoot")
    ax.axvline(CROSS, color=ZERO_C, ls="--", lw=1.4)
    ax.annotate(f"zero = pole\n$\\alpha$ = {CROSS:.2f}", (CROSS, 20),
                textcoords="offset points", xytext=(8, 0), fontsize=9, color=ZERO_C)
    ax.axvline(AL, color="0.45", ls=":", lw=1.4)
    ax.annotate(f"current\n$\\alpha$={AL:g}", (AL, 2.0), textcoords="offset points",
                xytext=(6, 0), fontsize=9, color="0.35")
    ax.set(xlabel=r"$\alpha$ [rad/s]", ylabel="overshoot [%]",
           title="what $\\alpha$ buys and what it costs")
    a2 = ax.twinx()
    a2.grid(False)
    a2.plot(sweep, ts, "--", color="tab:red", lw=1.6, label="2 % settling")
    a2.set_ylabel("2 % settling time [s]", color="tab:red")
    a2.set_ylim(0.3, 3.0)      # alpha below ~1.5 settles in seconds; without a
                               # cap it stretches the axis and flattens the rest
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
    ax.grid(alpha=0.3)

    # (c) zero and integrator pole vs alpha -- why (b) bends where it does
    ax = axes[1, 0]
    zero = [gains(a)[1] / gains(a)[0] for a in sweep]
    ax.plot(sweep, sweep, "-", color=DES_C, lw=1.8, label=r"integrator pole $\alpha$")
    ax.plot(sweep, zero, "-", color=ZERO_C, lw=1.8, label=r"zero $K_i/K_p$")
    ax.plot([CROSS], [CROSS], "o", ms=9, color="k", zorder=5)
    ax.annotate(f"they meet at {CROSS:.2f}", (CROSS, CROSS),
                textcoords="offset points", xytext=(10, -14), fontsize=9)
    ax.fill_between(sweep, sweep, zero, where=np.asarray(zero) > sweep,
                    color=ZERO_C, alpha=0.12, label="zero faster: near cancellation")
    ax.fill_between(sweep, sweep, zero, where=np.asarray(zero) < sweep,
                    color=PLANT_C, alpha=0.12, label="zero slower: overshoot grows")
    ax.set(xlabel=r"$\alpha$ [rad/s]", ylabel="distance from origin [rad/s]",
           xlim=(0.5, 20), ylim=(0, 21),
           title="the zero does not keep up with the pole")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    # (d) what the integrator is actually for
    ax = axes[1, 1]
    for a, ls in ((0.0, "--"), (2.0, "-"), (8.0, "-")):
        for fc, alpha_v in ((0.0, 1.0), (FC, 0.45)):
            r = simulate(a, T=6.0, dt=5e-4, fc=fc)
            lab = (f"$\\alpha$={a:g}" + (" + dry friction" if fc else "")) \
                if (fc == 0.0 or a == 0.0) else None
            ax.plot(r["t"], r["theta_deg"], ls, lw=1.6, alpha=alpha_v,
                    color={0.0: PLANT_C, 2.0: "tab:green", 8.0: DES_C}[a], label=lab)
    ax.axhline(20.0, color="0.4", lw=1.0, ls="--")
    ax.annotate("PD alone stops 7.8° short:\nonly an error produces torque",
                (3.0, 12.4), fontsize=9, color=PLANT_C)
    ax.set(xlabel="t [s]", ylabel=r"$\theta$ [deg]", xlim=(0, 6), ylim=(0, 26),
           title="without the integrator there is a standing error")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    return fig, "pid_design_integrator.png"


def fig_zero(plt):
    """The overshoot comes from the PID zero, not from the pole placement."""
    kp, ki, kd = gains(AL)
    den = [J, C + kd, K + kp, ki]
    y_z = linear_step([kp, ki], den)
    y_n = linear_step([ki], den)
    t = np.arange(len(y_z)) * 1e-4
    wd_d = WD * np.sqrt(1 - ZD ** 2)
    y_2nd = 1 - np.exp(-ZD * WD * t) * (np.cos(wd_d * t) + ZD * WD / wd_d * np.sin(wd_d * t))
    r = simulate(AL, T=3.0)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.0),
                                  gridspec_kw={"width_ratios": [1.25, 1]})
    ax.axhline(1.0, color="0.4", lw=1.0, ls="--")
    ax.plot(t, y_z / y_z[-1], color=DES_C, lw=1.9,
            label=f"ref through $K_p$ and $K_i$ — {100 * (y_z.max() / y_z[-1] - 1):.1f} %")
    ax.plot(t, y_n / y_n[-1], color="tab:green", lw=1.7,
            label=f"ref through $K_i$ only (no zero) — {100 * (y_n.max() / y_n[-1] - 1):.1f} %")
    ax.plot(t, y_2nd, ":", color="0.45", lw=1.6,
            label=f"the placed pair alone — "
                  f"{100 * np.exp(-np.pi * ZD / np.sqrt(1 - ZD ** 2)):.1f} %")
    ax.plot(r["t"], r["theta_deg"] / r["ref_deg"], "--", color=PLANT_C, lw=1.4,
            label=f"nonlinear plant (the simulator) — {r['overshoot']:.1f} %")
    ax.set(xlabel="t [s]", ylabel="normalised response", xlim=(0, 1.5),
           ylim=(0, 1.35), title="same poles, four numerators")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    bars = ["$K_p$+$K_i$\n(code)", "$K_i$ only", "2nd-order\nformula", "nonlinear\nplant"]
    vals = [100 * (y_z.max() / y_z[-1] - 1), 100 * (y_n.max() / y_n[-1] - 1),
            100 * np.exp(-np.pi * ZD / np.sqrt(1 - ZD ** 2)), r["overshoot"]]
    ax2.bar(bars, vals, color=[DES_C, "tab:green", "0.6", PLANT_C])
    for i, v in enumerate(vals):
        ax2.text(i, v + 0.6, f"{v:.1f}%", ha="center", fontsize=9)
    ax2.set(ylabel="overshoot [%]", ylim=(0, 24),
            title=f"the zero at {-ki / kp:.2f} explains all of it")
    ax2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return fig, "pid_design_zero.png"


def fig_saturation(plt):
    """How much of the +-3 wn^2 authority the design actually uses."""
    kp, _, _ = gains(AL)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.0))

    # What the KEYBOARD can actually ask for -- not arbitrary large steps. The
    # reference only ever moves REF_STEP at a time, so a 45 deg jump is not a
    # scenario this simulator can be put in.
    show = [(0, DES_C, "-"), (1, "tab:green", "-"), (3, SAT_C, "-"),
            (4, "tab:brown", "-"), (5, "0.45", "--")]
    for idx, color, ls in show:
        name, kw = m_cases[idx]
        r = scenario(T=2.0, **kw)
        ax.plot(r["t"], r["u"], ls, lw=1.6, color=color,
                label=f"{name.split(' (')[0]} — peak {r['peak']:.0f}")
    ax.axhline(m.TAU_MAX, color=PLANT_C, ls=":", lw=1.4)
    ax.axhline(-m.TAU_MAX, color=PLANT_C, ls=":", lw=1.4,
               label=f"$\\pm 3\\omega_n^2$ = {m.TAU_MAX:.0f}")
    ax.set(xlabel="t [s]", ylabel=r"$u = \tau/J$ [rad/s$^2$]", xlim=(0, 2),
           ylim=(-1.15 * m.TAU_MAX, 1.15 * m.TAU_MAX),
           title="nothing the keyboard can ask for reaches the limit\n"
                 "(the dashed 45° jump, which it cannot ask for, does)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    ang = np.linspace(0, 90, 200)
    ax2.plot(ang, K * np.sin(np.deg2rad(ang)), color=DES_C, lw=1.9,
             label=r"holding torque $K\sin\theta$")
    ax2.plot(ang, K * np.deg2rad(ang), "--", color="0.55", lw=1.4,
             label=r"linearised $K\theta$")
    ax2.axhline(m.TAU_MAX, color=PLANT_C, ls=":", lw=1.4,
                label=f"limit {m.TAU_MAX:.0f}")
    ax2.axhline(kp * REF, color=SAT_C, ls="-.", lw=1.4,
                label=f"20° step kick $K_p e$ = {kp * REF:.0f}")
    thr = np.rad2deg(m.TAU_MAX / kp)
    note = ("a single step saturates only once\n"
            f"the error reaches {thr:.0f}° — and the keys\n"
            f"move the reference {np.rad2deg(m.REF_STEP):.0f}° at a time")
    ax2.annotate(note, (44, 0.78 * m.TAU_MAX), fontsize=8.5, color="0.3")
    for deg in (20, 45, 90):
        need = K * np.sin(np.deg2rad(deg))
        ax2.plot([deg], [need], "o", ms=6, color=DES_C)
        ax2.annotate(f"{100 * need / m.TAU_MAX:.0f} % of limit", (deg, need),
                     textcoords="offset points", xytext=(-8, 10), fontsize=8,
                     ha="right", color=DES_C)
    ax2.set(xlabel=r"$\theta$ [deg]", ylabel=r"[rad/s$^2$]", xlim=(0, 92),
            ylim=(0, 1.15 * m.TAU_MAX),
            title=r"$3\omega_n^2$ is 3x the torque that holds the rod horizontal")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig, "pid_design_saturation.png"


FIGURES = {"placement": fig_placement, "integrator": fig_integrator,
           "zero": fig_zero, "saturation": fig_saturation}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--select", default="all",
                    help="comma-separated: " + ", ".join(FIGURES))
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    wanted = set(FIGURES) if args.select == "all" else {
        s.strip() for s in args.select.split(",")}
    unknown = wanted - set(FIGURES)
    if unknown:
        raise SystemExit(f"--select: unknown {sorted(unknown)}; "
                         f"pick from {', '.join(sorted(FIGURES))}")
    if args.no_save and args.no_show:
        raise SystemExit("--no-save with --no-show would produce nothing")
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not args.no_save:
        os.makedirs(args.out_dir, exist_ok=True)
    shown = []
    for name in FIGURES:                       # keep the documented order
        if name not in wanted:
            continue
        fig, filename = FIGURES[name](plt)
        if not args.no_save:
            out = os.path.join(args.out_dir, filename)
            fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
            print(f"  wrote {os.path.abspath(out)}")
        if args.no_show:
            plt.close(fig)
        else:
            shown.append(fig)
    if shown:
        print(f"\nopening {len(shown)} window(s) - close them all to exit"
              f"  [backend: {matplotlib.get_backend()}]")
        plt.show()


if __name__ == "__main__":
    main()
