r"""Why viscous damping decays exponentially and Coulomb damping decays linearly.

Both take energy out of the swing, but they take it out in different amounts,
and that difference is visible in the shape of the amplitude envelope.

VISCOUS (air, oil, eddy currents)
    torque   tau = -c * theta_dot          proportional to speed
    A big swing is also a fast swing, so a big swing loses proportionally more.
    Energy per cycle:  dE/dt = -c <theta_dot^2>, and E = 1/2 k A^2, so

        dA/dt = -sigma A        ->      A(t) = A0 exp(-sigma t)

    Each cycle keeps the same FRACTION of the previous amplitude. On a log axis
    that is a straight line, and the motion never quite stops.

COULOMB (dry / bearing friction)
    torque   tau = -F * sign(theta_dot)    constant magnitude, direction only
    The torque is the same whether the swing is big or small, so every swing
    costs the same amount of energy. Over a quarter cycle the pendulum travels
    A, so the work is F*A; over a full cycle 4*F*A. With E = 1/2 k A^2:

        k A dA = -4 F A  per cycle   ->   dA = -4F/k per cycle, independent of A

    Each cycle loses the same ABSOLUTE amount. That is a straight line on a
    LINEAR axis, and it reaches zero in finite time -- the pendulum stops dead,
    anywhere inside the stiction band, not necessarily at zero.

Together:  dA/dt = -(sigma A + c_coulomb)

Run this to see both laws drawn against the measured envelope.
"""
import argparse
import os

import matplotlib
import numpy as np

from _common import envelope_decay, longest_run, single_mechanism_r2


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="damping_mechanisms.png")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t, A, _, path = longest_run()
    env = envelope_decay(t, A)
    sigma, c = env["sigma"], env["coulomb"]
    r2v, r2l = single_mechanism_r2(t, A)

    A0 = np.rad2deg(A[0])
    deg = np.rad2deg(A)
    tt = np.linspace(0, t[-1], 400)

    # Each curve must be drawn with ITS OWN best fit, not with the joint one:
    # the viscous-only sigma and the Coulomb-only slope are both larger, because
    # each has to account for the whole decay by itself.
    pv = np.polyfit(t, np.log(A), 1)          # log A linear in t  -> viscous only
    sigma_v = -pv[0]
    pl = np.polyfit(t, A, 1)                  # A linear in t      -> Coulomb only
    rate_c = np.rad2deg(-pl[0])

    visc = np.rad2deg(np.exp(np.polyval(pv, tt)))
    coul = np.clip(np.rad2deg(np.polyval(pl, tt)), 1e-3, None)
    both = np.rad2deg((A[0] + c / sigma) * np.exp(-sigma * tt) - c / sigma)

    print(f"run: {os.path.basename(path)}, {len(t)} extrema, "
          f"{A0:.1f} -> {deg[-1]:.1f} deg over {t[-1]:.1f} s")
    print(f"  joint fit  : sigma = {sigma:.4f} 1/s, Coulomb = {np.rad2deg(c):.3f} deg/s")
    print(f"  viscous only fit : sigma = {sigma_v:.4f} 1/s  (inflated: it must explain all the decay)")
    print(f"  Coulomb only fit : rate  = {rate_c:.3f} deg/s")
    print(f"  R^2 : viscous only {r2v:.4f}   Coulomb only {r2l:.4f}   joint {env['r2']:.4f}")
    print()
    print("  amplitude ratio between successive same-sign extrema:")
    print("    pure viscous keeps this CONSTANT; pure Coulomb makes it fall")
    same = deg[::2]
    ratios = same[1:] / same[:-1]
    drops = same[:-1] - same[1:]
    for i in range(0, min(len(ratios), 6)):
        print(f"      {same[i]:6.1f} -> {same[i+1]:6.1f} deg   "
              f"ratio {ratios[i]:.3f}   drop {drops[i]:5.2f} deg")
    print(f"    ratio: first {ratios[0]:.3f} -> last {ratios[-1]:.3f}  "
          f"({'falls' if ratios[-1] < ratios[0] else 'constant'})")
    print(f"    drop : first {drops[0]:5.2f} -> last {drops[-1]:5.2f} deg")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

    for a, logy in ((ax[0], False), (ax[1], True)):
        a.plot(t, deg, "o", ms=4, color="#203864", label="measured extrema", zorder=4)
        a.plot(tt, visc, "--", lw=1.8, color="#3D8C54",
               label=f"viscous only: $\\sigma$={sigma_v:.3f} 1/s   R²={r2v:.3f}")
        a.plot(tt, coul, ":", lw=2.0, color="#E97820",
               label=f"Coulomb only: {rate_c:.2f} deg/s   R²={r2l:.3f}")
        a.plot(tt, both, "-", lw=1.8, color="#c0392b",
               label=f"viscous + Coulomb   R²={env['r2']:.3f}")
        a.set_xlabel("Time since release (s)")
        a.grid(alpha=0.3, which="both")
        a.legend(fontsize=8)
        if logy:
            a.set_yscale("log")
            # the Coulomb-only line dives to zero in finite time; clamp the axis
            # to the measured range so the comparison stays readable
            a.set_ylim(0.6 * deg.min(), 1.6 * deg.max())
            a.set_ylabel("Amplitude (deg, log)")
            a.set_title("log axis: viscous would be a STRAIGHT line")
        else:
            a.set_ylabel("Amplitude (deg)")
            a.set_title("linear axis: Coulomb would be a STRAIGHT line")

    # per-cycle signature
    n = np.arange(len(ratios))
    ax[2].plot(n, ratios, "o-", ms=4, color="#6D237C", label="amplitude ratio $A_{k+1}/A_k$")
    ax[2].axhline(np.exp(-sigma_v * 2 * np.pi / 7.508), color="#3D8C54", ls="--",
                  label=f"pure viscous would hold this constant "
                        f"({np.exp(-sigma_v * 2 * np.pi / 7.508):.3f})")
    ax[2].set_xlabel("Cycle index")
    ax[2].set_ylabel("Ratio of successive same-sign extrema")
    ax[2].set_title("the tell: the ratio is not constant")
    ax[2].grid(alpha=0.3)
    ax[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nfigure -> {os.path.abspath(args.out)}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
