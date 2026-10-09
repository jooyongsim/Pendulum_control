"""Design a PID from the MEASURED wn and zeta, for both operating points.

The free-swing identification gives two numbers and one derived length:

    wn    = 7.5080 rad/s     zeta = 0.010629      L_eff = g/wn^2 = 173.97 mm

Everything below is a closed-form function of those. Nothing is hand tuned.

Two plants, one formula
-----------------------
With u the pivot acceleration [m/s^2], the linearised pendulum is

    hanging  (theta from hanging):   theta'' = -wn^2 theta - 2 sigma theta' - u/L
    upright  (theta from upright):   theta'' = +wn^2 theta - 2 sigma theta' - u/L

Only the sign of the gravity term differs, so one gain formula covers both -- a
sign flag is all that changes. The hanging case is a servo problem (track a
reference), the upright case is a stabilisation problem (hold zero), and the
same measured numbers serve both.

What this script answers
------------------------
1. what gains the measurement implies, for each operating point
2. whether the placement is exact (closed-loop poles vs requested)
3. how the design behaves on the REAL loop: 100 Hz, 0.3 deg encoder,
   +-6 m/s^2, dry friction, derivative filter
4. how much the answer moves when the measurement itself is wrong --
   the identification says zeta is the poorly determined number, so this is
   the question that matters for trusting the design

    python pid_from_measurement.py                 # tables + figure
    python pid_from_measurement.py --no-show       # write the figure only
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np

from _common import PROJECT  # noqa: F401  (puts the project on sys.path)
import pendulum_sim as ps

G = ps.G

# --- design specification --------------------------------------------------
# dominant pair, then the integrator pole. wd is chosen in section 4 from the
# actuator limit, not picked by taste.
SPEC = dict(hanging=dict(wd=12.0, zd=0.7, wi=4.0),
            upright=dict(wd=12.0, zd=0.8, wi=4.0))
TAU_D = 0.01          # s, derivative filter = one control period at 100 Hz
LOOP = dict(control_hz=100.0, encoder_deg=0.3, accel_max=6.0)


def place_pid(wn, zeta, wd, zd, wi, upright):
    """Closed-form PID gains for either operating point.

    Closed loop is third order (the integrator adds a state):

        s^3 + (2 sigma + kd/L) s^2 + (kp/L -+ wn^2) s + ki/L = 0
                                            ^ minus upright, plus hanging

    Matching (s^2 + 2 zd wd s + wd^2)(s + wi):

        kd = L (2 zd wd + wi - 2 sigma)
        kp = L (wd^2 + 2 zd wd wi -+ wn^2)
        ki = L wd^2 wi
    """
    sigma = zeta * wn
    L = G / wn ** 2
    grav = wn ** 2 if upright else -wn ** 2
    kd = L * (2 * zd * wd + wi - 2 * sigma)
    kp = L * (wd ** 2 + 2 * zd * wd * wi + grav)
    ki = L * wd ** 2 * wi
    return kp, ki, kd


def closed_loop_poles(wn, zeta, kp, ki, kd, upright):
    sigma, L = zeta * wn, G / wn ** 2
    grav = wn ** 2 if upright else -wn ** 2
    return np.roots([1.0, 2 * sigma + kd / L, kp / L - grav, ki / L])


def requested_poles(wd, zd, wi):
    return np.append(np.roots([1.0, 2 * zd * wd, wd ** 2]), -wi)


class MeasuredPID(ps.Controller):
    """PID on the angle, derivative on the measurement, conditional integration.

    ps.PID does the same thing; this one is written out so the doc can point at
    every line, and so the hanging case can carry a non-zero reference.
    """

    def __init__(self, kp, ki, kd, ref=0.0, u_max=np.inf, tau_d=TAU_D,
                 upright=True):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.ref, self.u_max, self.tau_d = ref, u_max, tau_d
        # simulate() measures theta from UPRIGHT. Working near hanging needs
        # two changes, and both are easy to miss:
        #   offset: the error is the deviation from pi, not from 0
        #   sign  : the input enters as -(u/L) cos(theta), and cos(pi) = -1,
        #           so the same command pushes the other way
        self.offset = 0.0 if upright else np.pi
        self.sign = 1.0 if upright else -1.0
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.d_state = 0.0
        self.prev = None

    def __call__(self, s, dt):
        theta = s.theta - self.offset
        e = theta - self.ref
        if self.prev is None:
            self.prev = theta
        raw = (theta - self.prev) / dt            # derivative on measurement
        self.prev = theta
        beta = dt / (self.tau_d + dt)
        self.d_state += beta * (raw - self.d_state)
        u = self.sign * (self.kp * e + self.ki * self.integral
                         + self.kd * self.d_state)
        if abs(u) <= self.u_max:                  # anti-windup
            self.integral += e * dt
        return float(np.clip(u, -self.u_max, self.u_max))


def run(p, ctrl, theta0_deg, duration=8.0, **over):
    kw = dict(LOOP)
    kw.update(over)
    cfg = ps.SimConfig(duration=duration, theta0_deg=theta0_deg, **kw)
    log = ps.simulate(p, ctrl, cfg)
    tail = int(1.0 / cfg.dt_plant)
    ref = getattr(ctrl, "ref", 0.0)
    err = np.abs(log["theta_deg"][-tail:] - np.rad2deg(ref))
    return dict(log=log, err=err.max(), mean=log["theta_deg"][-tail:].mean(),
                u_peak=np.abs(log["u"]).max(), rotor=np.abs(log["alpha_deg"]).max(),
                sat=100.0 * log["saturated"].mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--out", default="pid_from_measurement.png")
    args = ap.parse_args()
    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")

    p = ps.PendulumParams()
    wn, zeta, L = p.wn, p.zeta, p.L_eff

    print("=" * 78)
    print("1. WHAT WAS MEASURED")
    print("=" * 78)
    print(f"  wn      = {wn:.4f} rad/s      (extrema spacing, T0 = {2*np.pi/wn:.4f} s)")
    print(f"  zeta    = {zeta:.6f}          (envelope fit)")
    print(f"  sigma   = {p.sigma:.4f} 1/s       = zeta*wn")
    print(f"  L_eff   = {L*1000:.2f} mm         = g/wn^2, derived")
    print(f"  Coulomb = {p.coulomb_dps:.3f} deg/s      = {p.coulomb_accel:.4f} rad/s^2")
    print(f"  input gain -1/L_eff = {p.b:.3f} (rad/s^2) per (m/s^2)")

    print()
    print("=" * 78)
    print("2. GAINS: ONE FORMULA, TWO OPERATING POINTS")
    print("=" * 78)
    print("     kd = L (2 zd wd + wi - 2 sigma)")
    print("     kp = L (wd^2 + 2 zd wd wi -+ wn^2)      - upright, + hanging")
    print("     ki = L wd^2 wi")
    print()
    print(f"{'':10}{'wd':>6}{'zd':>6}{'wi':>6}{'kp':>10}{'ki':>10}{'kd':>9}"
          f"   placement")
    gains = {}
    for name, upright in (("hanging", False), ("upright", True)):
        s = SPEC[name]
        kp, ki, kd = place_pid(wn, zeta, s["wd"], s["zd"], s["wi"], upright)
        gains[name] = (kp, ki, kd)
        got = np.sort_complex(closed_loop_poles(wn, zeta, kp, ki, kd, upright))
        want = np.sort_complex(requested_poles(s["wd"], s["zd"], s["wi"]))
        ok = np.allclose(got, want, atol=1e-9)
        print(f"{name:10}{s['wd']:>6.1f}{s['zd']:>6.2f}{s['wi']:>6.1f}"
              f"{kp:>10.3f}{ki:>10.2f}{kd:>9.4f}   {'exact' if ok else 'MISMATCH'}")
        assert ok, f"{name}: placement failed"
    print(f"  cross-check with pendulum_sim.place_pid (upright): "
          f"{np.allclose(place_pid(wn, zeta, 12.0, 0.8, 4.0, True), ps.place_pid(p, 12.0, 0.8, 4.0))}")
    print(f"  gravity's share of kp: upright {100*L*wn**2/gains['upright'][0]:.0f} %,"
          f" hanging {-100*L*wn**2/gains['hanging'][0]:.0f} %"
          f"   (the two designs differ by 2 L wn^2 = {2*L*wn**2:.2f})")

    print()
    print("=" * 78)
    print("3. ON THE REAL LOOP  (100 Hz, 0.3 deg encoder, +-6 m/s^2, dry friction)")
    print("=" * 78)
    print(f"{'case':<34}{'residual':>11}{'peak u':>9}{'sat':>7}{'rotor':>9}")
    kp, ki, kd = gains["upright"]
    up = run(p, MeasuredPID(kp, ki, kd, u_max=6.0), theta0_deg=5.0)
    print(f"{'upright, 5 deg tilt':<34}{up['err']:>10.3f}d{up['u_peak']:>9.2f}"
          f"{up['sat']:>6.0f}%{up['rotor']:>8.0f}d")
    kp_h, ki_h, kd_h = gains["hanging"]
    ref = np.deg2rad(20.0)
    hang = run(p, MeasuredPID(kp_h, ki_h, kd_h, ref=ref, u_max=6.0, upright=False),
               theta0_deg=180.0, duration=6.0)
    hang_end = hang["mean"] - 180.0
    print(f"{'hanging, 20 deg ref, PID':<34}{hang_end - 20.0:>+10.3f}d"
          f"{hang['u_peak']:>9.2f}{hang['sat']:>6.0f}%{hang['rotor']:>8.0f}d")
    nopi = run(p, MeasuredPID(kp_h, 0.0, kd_h, ref=ref, u_max=6.0, upright=False),
               theta0_deg=180.0, duration=6.0)
    nopi_end = nopi["mean"] - 180.0
    print(f"{'hanging, 20 deg ref, PD only':<34}{nopi_end - 20.0:>+10.3f}d"
          f"{nopi['u_peak']:>9.2f}{nopi['sat']:>6.0f}%{nopi['rotor']:>8.0f}d")
    print(f"  the PD row is gravity plus dry friction: it stops {20.0-nopi_end:.2f} deg")
    print(f"  short of the reference, and only the integrator closes that gap.")
    print(f"  the rotor column is physics, not a bug: holding the rod {hang_end:.0f} deg")
    print(f"  off hanging needs a steady pivot acceleration u = g sin(ref) ="
          f" {G*np.sin(ref):.2f} m/s^2,")
    print("  which integrates twice into arm angle. A rotary rig cannot hold a")
    print("  tilted reference forever -- that case belongs to a torque-driven pivot.")

    print()
    print("=" * 78)
    print("4. HOW FAST CAN wd BE?  (upright, 5 deg tilt)")
    print("=" * 78)
    print(f"{'wd':>6}{'kp':>9}{'ki':>10}{'kd':>8}{'residual':>11}{'peak u':>9}{'sat':>7}")
    sweep = []
    for wd in (6.0, 9.0, 12.0, 16.0, 20.0):
        kp_w, ki_w, kd_w = place_pid(wn, zeta, wd, 0.8, wd / 3.0, True)
        r = run(p, MeasuredPID(kp_w, ki_w, kd_w, u_max=6.0), theta0_deg=5.0)
        sweep.append((wd, kp_w, ki_w, kd_w, r))
        print(f"{wd:>6.0f}{kp_w:>9.2f}{ki_w:>10.2f}{kd_w:>8.3f}{r['err']:>10.3f}d"
              f"{r['u_peak']:>9.2f}{r['sat']:>6.0f}%")
    print("  the +-6 m/s^2 limit, not the measurement, is what caps wd.")

    print()
    print("=" * 78)
    print("5. WHAT IF THE MEASUREMENT IS WRONG?")
    print("=" * 78)
    print("  design with the measured values, then run against a DIFFERENT truth")
    print(f"{'true plant':<28}{'residual':>11}{'peak u':>9}{'poles move to':>34}")
    kp, ki, kd = gains["upright"]
    for label, dwn, dz in (("as measured", 1.00, 1.0), ("wn +5 %", 1.05, 1.0),
                           ("wn -5 %", 0.95, 1.0), ("wn +10 %", 1.10, 1.0),
                           ("zeta x3", 1.00, 3.0), ("zeta x1/3", 1.00, 1 / 3)):
        pt = ps.PendulumParams(wn=wn * dwn, zeta=zeta * dz,
                               coulomb_dps=p.coulomb_dps)
        r = run(pt, MeasuredPID(kp, ki, kd, u_max=6.0), theta0_deg=5.0,
                duration=4.0)
        cl = np.sort(closed_loop_poles(pt.wn, pt.zeta, kp, ki, kd, True).real)
        print(f"{label:<28}{r['err']:>10.3f}d{r['u_peak']:>9.2f}"
              f"   {np.array2string(cl, precision=2)}")
    print("  zeta moves nothing: it enters kd only as -2 sigma = -0.16 out of 23.")
    print("  That is the good news -- zeta is the number the identification")
    print("  determines worst (per-run spread of 8x), and the design ignores it.")

    # ---------------------------------------------------------------- figure
    fig, ax = plt.subplots(2, 2, figsize=(13.5, 8.2))
    fig.suptitle(f"PID from the measured model: $\\omega_n$ = {wn:.4f} rad/s, "
                 f"$\\zeta$ = {zeta:.6f}, $L_{{eff}}$ = {L*1000:.1f} mm",
                 fontsize=12)

    a = ax[0, 0]
    t = up["log"]["t"]
    a.axhspan(-0.5, 0.5, color="0.9", zorder=0)
    a.plot(t, up["log"]["theta_deg"], color="#2774B8", lw=1.6)
    a.set(xlim=(0, 3), xlabel="t [s]", ylabel=r"$\theta$ [deg]",
          title=f"upright: 5° tilt, residual {up['err']:.2f}°")
    a.grid(alpha=0.3)

    a = ax[0, 1]
    a.axhline(20.0, color="0.4", ls="--", lw=1.0, label="reference 20°")
    a.plot(hang["log"]["t"], hang["log"]["theta_deg"] - 180.0, color="#117A65",
           lw=1.6, label=f"PID  -> {hang_end:.2f} deg")
    a.plot(nopi["log"]["t"], nopi["log"]["theta_deg"] - 180.0, "--", color="#c0392b",
           lw=1.5, label=f"PD only -> {nopi_end:.2f} deg")
    a.set(xlim=(0, 6), xlabel="t [s]", ylabel=r"$\theta$ from hanging [deg]",
          title="hanging: the integrator against dry friction")
    a.legend(fontsize=8)
    a.grid(alpha=0.3)

    a = ax[1, 0]
    wds = [s[0] for s in sweep]
    a.plot(wds, [s[4]["u_peak"] for s in sweep], "o-", color="#c0392b",
           label="peak |u|")
    a.axhline(6.0, color="r", ls=":", lw=1.2, label="±6 m/s² limit")
    a.set(xlabel=r"$\omega_d$ [rad/s]", ylabel=r"peak $|u|$ [m/s$^2$]",
          title="the actuator sets the ceiling")
    a.legend(fontsize=8)
    a.grid(alpha=0.3)

    a = ax[1, 1]
    # Sensitivity is read off the POLES, not off the residual: with dry
    # friction the residual is the width of the stiction band (0.01-0.15 deg)
    # and jitters with wherever the rod happens to stick, which hides the
    # trend instead of showing it. The poles are deterministic.
    err_pct = np.linspace(-15, 15, 121)
    dom_wn = [np.sort(closed_loop_poles(wn * (1 + e / 100), zeta,
                                        kp, ki, kd, True).real)[0]
              for e in err_pct]
    zmult = np.linspace(0.2, 5.0, 121)
    dom_z = [np.sort(closed_loop_poles(wn, zeta * z, kp, ki, kd, True).real)[0]
             for z in zmult]
    a.plot(err_pct, dom_wn, color="#8E44AD", lw=2.0, label=r"$\omega_n$ error [%]")
    a.axhline(np.sort(closed_loop_poles(wn, zeta, kp, ki, kd, True).real)[0],
              color="0.5", ls=":", lw=1.2)
    a.axvline(0, color="0.7", lw=0.9)
    a.set(xlabel=r"$\omega_n$ error [%]   /   $\zeta$ multiplier (top)",
          ylabel="fastest closed-loop pole [1/s]",
          title="sensitivity: the poles, not the residual")
    a2 = a.twiny()
    a2.plot(zmult, dom_z, color="#E97820", lw=2.0, label=r"$\zeta$ multiplier")
    a2.set_xlabel(r"$\zeta$ multiplier", color="#E97820")
    a2.tick_params(axis="x", colors="#E97820")
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower left")
    a.grid(alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nfigure -> {args.out}")

    print()
    print("=" * 78)
    print("6. GAINS TO PUT ON THE RIG")
    print("=" * 78)
    for name in ("upright", "hanging"):
        kp_n, ki_n, kd_n = gains[name]
        print(f"  {name:8}  kp = {kp_n:8.3f}   ki = {ki_n:8.3f}   kd = {kd_n:7.4f}"
              f"   [(m/s^2) per rad]")
    print(f"  loop      100 Hz, derivative filter {TAU_D*1000:.0f} ms,"
          f" |u| <= {LOOP['accel_max']:.0f} m/s^2")
    print(f"  rotor     u -> arm acceleration: divide by r (measure it);"
          f" kp, kd do not depend on r")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
