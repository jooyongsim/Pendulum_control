"""Design a PID balancing controller from the IDENTIFIED coefficients, and check
it in simulation.

Everything the design needs comes out of the free-swing fit:

    wn    = 7.508 rad/s      natural frequency
    zeta  = 0.010629         damping ratio     (sigma = zeta*wn = 0.0798 1/s)
    L_eff = g/wn^2 = 174 mm  effective length -- and with it the input gain

Linearised about UPRIGHT (theta = 0 there), with u the pivot acceleration,

    theta'' = wn^2 theta - 2 sigma theta' - u/L_eff

which is open-loop unstable: poles at +7.429 and -7.588 1/s. A PID on theta
closes that to third order, and place_pid() solves for the gains in closed form.
No hand tuning happens anywhere in this script.

The simulations then ask whether the design survives what actually breaks
balancing controllers: 100 Hz sampling, a 0.3 deg encoder, +-6 m/s^2 of
authority, dry friction, a rig that is not level, and a rotor with finite travel.

    python design_pid.py                 # tables + figure
    python design_pid.py --no-show       # just write design_pid.png
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np

from _common import PROJECT  # noqa: F401  (puts the project on sys.path)
import pendulum_sim as ps

# --- specification ---------------------------------------------------------
WC = 12.0        # rad/s  closed-loop natural frequency of the dominant pair
ZC = 0.8         # -      its damping ratio
W_INT = 4.0      # rad/s  integrator pole, 3x slower than the pair
POLES4 = [-10 + 10j, -10 - 10j, -2 + 1j, -2 - 1j]   # for the cascade
TILT_DEG = 2.0   # how far out of level the rig is, in the disturbance test
TAU_D = 0.01     # s, derivative filter -- one control period. Section 6 shows why.

REALISTIC = dict(control_hz=100.0, encoder_deg=0.3, accel_max=6.0)
IDEAL = dict(control_hz=1000.0, encoder_deg=0.0, accel_max=np.inf)


def run(p, ctrl, duration=8.0, theta0_deg=5.0, tilt_deg=0.0, **over):
    kw = dict(REALISTIC)
    kw.update(over)
    cfg = ps.SimConfig(duration=duration, theta0_deg=theta0_deg,
                       disturb_accel=p.wn ** 2 * np.sin(np.deg2rad(tilt_deg)), **kw)
    return ps.simulate(p, ctrl, cfg), cfg


def metrics(log, cfg):
    t, th = log["t"], np.abs(log["theta_deg"])
    inside = th < 0.5
    outside = np.flatnonzero(~inside)
    settle = (t[outside[-1] + 1] if len(outside) else 0.0) if inside[-1] else np.nan
    tail = int(1.0 / cfg.dt_plant)
    return dict(settle=settle,
                err=np.abs(log["theta_deg"][-tail:]).max(),
                mean=log["theta_deg"][-tail:].mean(),
                u_peak=np.abs(log["u"]).max(),
                rotor=np.abs(log["alpha_deg"]).max(),
                rotor_end=log["alpha_deg"][-1],
                sat=100.0 * log["saturated"].mean())


def fmt_settle(v):
    return "   --  " if not np.isfinite(v) else f"{v:>6.2f}s"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    p = ps.PendulumParams()
    L = p.L_eff

    # ---------------------------------------------------------------- 1 ----
    print("=" * 78)
    print("1. THE PLANT THE DESIGN STARTS FROM (nothing here is new)")
    print("=" * 78)
    print(p.summary())
    ol = np.roots([1.0, 2 * p.sigma, -p.wn ** 2])
    print(f"open-loop poles about upright : {ol[1]:+.4f}, {ol[0]:+.4f} 1/s")
    print(f"  the unstable one doubles an error in {np.log(2) / ol[1] * 1000:.0f} ms"
          f" -- that is the whole time budget")

    # ---------------------------------------------------------------- 2 ----
    print()
    print("=" * 78)
    print(f"2. DESIGN  (dominant pair wn_des={WC:.0f} rad/s, zeta_des={ZC},"
          f" integrator pole {W_INT:.0f} rad/s)")
    print("=" * 78)
    print("   u = kp e + ki int(e) + kd e'      ->   s^3 + (2s + kd/L) s^2"
          " + (kp/L - wn^2) s + ki/L")
    print()
    print("     kd = L (2 zd wd + wi - 2 sigma)")
    print("     kp = L (wd^2 + 2 zd wd wi + wn^2)")
    print(f"     ki = L wd^2 wi                        L = L_eff = {L:.5f} m")
    kp0, ki0, kd0 = ps.place_pid(p, WC, ZC, 0.0)
    kp1, ki1, kd1 = ps.place_pid(p, WC, ZC, W_INT)
    print()
    print(f"{'':22}{'kp':>10}{'ki':>10}{'kd':>10}      [(m/s^2) per rad]")
    print(f"{'PD   (wi = 0)':22}{kp0:>10.3f}{ki0:>10.3f}{kd0:>10.3f}")
    print(f"{'PID':22}{kp1:>10.3f}{ki1:>10.3f}{kd1:>10.3f}")
    print()
    got = ps.closed_loop_poles_pid(p, kp1, ki1, kd1)
    print(f"   closed-loop poles : {np.array2string(got, precision=3)}")
    print(f"   requested         : {-ZC * WC:.3f} +- {WC * np.sqrt(1 - ZC ** 2):.3f}j,"
          f" {-W_INT:.3f}")
    k1, k2 = ps.place_poles(p, WC, ZC)
    print(f"   cross-check: place_poles() -> kp={-k1:.3f}, kd={-k2:.3f}:"
          f" the same PD law, opposite sign convention")

    # ---------------------------------------------------------------- 3 ----
    print()
    print("=" * 78)
    print("3. RESPONSE to a 5 deg initial tilt")
    print("=" * 78)
    print(f"{'controller / loop':<42}{'settle':>8}{'|th| end':>10}{'peak u':>9}"
          f"{'sat':>6}{'rotor':>9}")
    rows = {}
    for key, label, gains, loop in (
            ("pd_ideal", "PD    ideal loop", (kp0, ki0, kd0), IDEAL),
            ("pid_ideal", "PID   ideal loop", (kp1, ki1, kd1), IDEAL),
            ("pd", "PD    100 Hz, 0.3 deg, 6 m/s^2", (kp0, ki0, kd0), REALISTIC),
            ("pid", "PID   100 Hz, 0.3 deg, 6 m/s^2", (kp1, ki1, kd1), REALISTIC)):
        ctrl = ps.PID(*gains, u_max=loop["accel_max"], tau_d=TAU_D)
        log, cfg = run(p, ctrl, **loop)
        m = metrics(log, cfg)
        rows[key] = (log, m)
        print(f"{label:<42}{fmt_settle(m['settle'])}{m['err']:>9.3f}d"
              f"{m['u_peak']:>9.2f}{m['sat']:>5.0f}%{m['rotor']:>8.0f}d")
    print("   the encoder and the 100 Hz loop cost almost nothing here."
          " The rotor column is the problem -- section 6.")

    # ---------------------------------------------------------------- 4 ----
    print()
    print("=" * 78)
    print("4. HOW FAST CAN THE DOMINANT PAIR BE?  (100 Hz loop, 5 deg start)")
    print("=" * 78)
    print(f"{'wn_des':>7}{'kp':>9}{'kd':>8}{'ki':>9}{'settle':>9}{'peak u':>9}"
          f"{'sat':>6}  verdict")
    sweep = []
    for wc in (6.0, 9.0, 12.0, 16.0, 20.0, 25.0):
        kp, ki, kd = ps.place_pid(p, wc, ZC, wc / 3.0)
        log, cfg = run(p, ps.PID(kp, ki, kd, u_max=6.0, tau_d=TAU_D))
        m = metrics(log, cfg)
        verdict = ("ok" if np.isfinite(m["settle"]) else
                   ("limit cycle" if log["settled"] else "FALLS OVER"))
        sweep.append((wc, m, verdict))
        print(f"{wc:>7.0f}{kp:>9.2f}{kd:>8.2f}{ki:>9.2f}{fmt_settle(m['settle'])}"
              f"{m['u_peak']:>9.2f}{m['sat']:>5.0f}%  {verdict}")
    first_sat = next((w for w, m, _ in sweep if m["u_peak"] >= 5.999), None)
    print(f"   +-6 m/s^2 sets the ceiling: the command first hits the limit at"
          f" wn_des = {first_sat:.0f} rad/s,")
    print(f"   and by 25 rad/s the loop spends {sweep[-1][1]['sat']:.0f} % of its time"
          f" saturated -- it is then a")
    print(f"   bang-bang controller, not the one that was designed, and settling gets"
          f" WORSE ({sweep[-1][1]['settle']:.1f} s).")
    print(f"   12 rad/s leaves {100*(1 - sweep[2][1]['u_peak']/6):.0f} % of the"
          f" authority in reserve for disturbances.")

    # ---------------------------------------------------------------- 5 ----
    print()
    print("=" * 78)
    print(f"5. A CONSTANT BIAS: the rig is {TILT_DEG:.0f} deg out of level")
    print("=" * 78)
    d_acc = p.wn ** 2 * np.sin(np.deg2rad(TILT_DEG))
    print(f"   it enters as wn^2 sin(phi) = {d_acc:.3f} rad/s^2 of constant torque.")
    print(f"   PD theory: standing error = wn^2 phi / wn_des^2 ="
          f" {np.rad2deg(p.wn ** 2 * np.deg2rad(TILT_DEG) / WC ** 2):.3f} deg")
    print()
    print(f"{'controller':<42}{'theta end':>11}{'rotor max':>11}{'rotor end':>11}")
    dist = {}
    for key, label, ctrl in (
            ("pd", "PD  (no integrator)", ps.PID(kp0, ki0, kd0, u_max=6.0, tau_d=TAU_D)),
            ("pid", "PID", ps.PID(kp1, ki1, kd1, u_max=6.0, tau_d=TAU_D))):
        log, cfg = run(p, ctrl, duration=15.0, tilt_deg=TILT_DEG)
        m = metrics(log, cfg)
        dist[key] = (log, m)
        print(f"{label:<42}{m['mean']:>10.3f}d{m['rotor']:>10.0f}d{m['rotor_end']:>10.0f}d")
    print()
    print("   The I term does what it promises -- and that is the trap. With the rig")
    print("   tilted, theta = 0 is NOT an equilibrium: standing there needs a constant")
    print(f"   u = L*d = {L * d_acc:.3f} m/s^2 forever, and u integrates twice into the")
    print("   rotor. True vertical is at theta = -phi, where u = 0. Driving the")
    print("   measured angle to zero means fighting gravity until the arm runs out of")
    print("   travel. Neither PD nor PID can see that, because neither one watches the rotor.")

    # ---------------------------------------------------------------- 6 ----
    print()
    print("=" * 78)
    print("6. THE ROTOR: closing the outer loop")
    print("=" * 78)
    print("   theta_setpoint = -(k_alpha alpha + k_alpha_dot alpha')")
    print("   Expanded, with the derivative taken on the MEASUREMENT, that IS")
    print("   4-state feedback:")
    print("     u = kp theta + kd theta' + kp k_alpha alpha + kp k_alpha_dot alpha'")
    print("   (differentiate the ERROR instead and the moving setpoint feeds alpha''")
    print("    = u/r straight back into u -- an algebraic loop of gain"
          " kd k_alpha_dot / r = 1.77.")
    print("    Above unity the loop feeds itself and never balances:)")
    kk = ps.place_poles4(p, POLES4)
    print(f"{'   derivative taken on':<42}{'|th| end':>10}{'rotor max':>11}")
    for flag, lab in ((False, "the error (textbook PID)"), (True, "the measurement")):
        c = ps.CascadePID(ps.PID(-kk[0], 0.0, -kk[1], u_max=6.0, tau_d=TAU_D,
                                 d_on_measurement=flag),
                          kk[2] / kk[0], kk[3] / kk[0], tilt_max_deg=5.0)
        lg, cfg = run(p, c, duration=15.0)
        m = metrics(lg, cfg)
        print(f"{'   ' + lab:<42}{m['err']:>9.2f}d{m['rotor']:>10.1f}d")
    print()
    print("   So the outer gains are not free: they move all four closed-loop poles.")
    print("   Sweeping the usual quasi-static choice k_alpha = wo^2 r/g, zeta_o = 0.9:")
    print()
    print(f"{'   outer wn':<20}{'k_alpha':>10}{'k_a_dot':>10}{'|th| end':>10}{'rotor':>9}"
          f"   closed-loop Re(poles)")
    A, B = ps.state_space(p)
    for wo in (0.5, 1.0, 2.0, 3.0, 4.0):
        ka = wo ** 2 * p.arm_radius_m / ps.G
        kad = 2 * 0.9 * wo * p.arm_radius_m / ps.G
        c = ps.CascadePID(ps.PID(-kk[0], 0.0, -kk[1], u_max=6.0, tau_d=TAU_D),
                          ka, kad, tilt_max_deg=5.0)
        lg, cfg = run(p, c, duration=15.0)
        m = metrics(lg, cfg)
        K = np.array([[kk[0], kk[1], kk[0] * ka, kk[0] * kad]])
        re = np.sort(np.linalg.eigvals(A - B @ K).real)
        flag = "" if lg["settled"] else "  FALLS OVER"
        print(f"{'   ' + f'{wo:.1f} rad/s':<20}{ka:>10.5f}{kad:>10.5f}{m['err']:>9.2f}d"
              f"{m['rotor']:>8.1f}d   {np.array2string(re, precision=2)}{flag}")
    print(f"{'   placed':<20}{kk[2]/kk[0]:>10.5f}{kk[3]/kk[0]:>10.5f}"
          f"{'':>10}{'':>9}   {np.array2string(np.sort(np.array(POLES4).real), precision=2)}")
    print()
    print("   Past about 2 rad/s a pole pair crosses into the right half plane and the")
    print("   pendulum falls: asking the rotor to come home faster means leaning harder")
    print("   the wrong way first. The quasi-static rule of thumb (outer loop ~6x slower")
    print("   than the inner one, 2 rad/s here) lands just inside that boundary with no")
    print("   margin left, which is the argument for placing all four poles instead.")
    print()
    print("   The derivative filter is the other design choice, and it is not free:")
    print(f"{'   tau_d [ms]':<20}{'|th| end':>10}{'rotor':>9}{'peak u':>9}")
    for tau in (0.0, 0.005, 0.01, 0.02, 0.04):
        c, _ = ps.cascade_from_4state(p, POLES4, u_max=6.0, tau_d=tau)
        lg, cfg = run(p, c, duration=15.0)
        m = metrics(lg, cfg)
        note = ("  <- hunts" if m["err"] > 2.0 else
                ("  <- chosen" if abs(tau - TAU_D) < 1e-9 else ""))
        print(f"{'   ' + f'{tau*1000:.0f}':<20}{m['err']:>9.2f}d{m['rotor']:>8.1f}d"
              f"{m['u_peak']:>9.2f}{note}")
    print("   The filter costs phase where the loop has none to spare: 20 ms turns a")
    print("   0.55 deg settle into a +-3.6 deg hunt. One control period (10 ms) is")
    print("   enough to tame the 30 deg/s of quantisation noise on the difference.")
    print()
    casc, g = ps.cascade_from_4state(p, POLES4, w_int=0.0, u_max=6.0, tau_d=TAU_D)
    print(f"   cascade_from_4state({POLES4}):")
    print(f"     kp={g[0]:.3f}  ki={g[1]:.3f}  kd={g[2]:.3f}"
          f"   k_alpha={g[3]:.5f}  k_alpha_dot={g[4]:.5f}")
    print()
    print(f"{'controller':<42}{'|th| end':>10}{'peak u':>9}{'rotor max':>11}{'rotor end':>11}")
    casc_log, cfg = run(p, casc, duration=15.0)
    mc = metrics(casc_log, cfg)
    sf_log, cfg = run(p, ps.StateFeedback(*ps.place_poles4(p, POLES4), u_max=6.0),
                      duration=15.0)
    ms = metrics(sf_log, cfg)
    pid_long, cfg = run(p, ps.PID(kp1, ki1, kd1, u_max=6.0, tau_d=TAU_D), duration=15.0)
    mp = metrics(pid_long, cfg)
    for label, m in (("PID on the angle only", mp),
                     ("cascade PID (this design)", mc),
                     ("4-state feedback, exact theta'  (reference)", ms)):
        print(f"{label:<42}{m['err']:>9.3f}d{m['u_peak']:>9.2f}"
              f"{m['rotor']:>10.1f}d{m['rotor_end']:>10.1f}d")
    print()
    print("   The cascade matches the 4-state reference while using only what the")
    print("   hardware has: a quantised angle, differenced and filtered. The reference")
    print("   row is handed the exact velocity.")

    # ---------------------------------------------------------------- 7 ----
    print()
    print("=" * 78)
    print(f"7. THE SAME {TILT_DEG:.0f} deg TILT, with the outer loop closed")
    print("=" * 78)
    print(f"{'controller':<42}{'theta end':>11}{'rotor max':>11}{'rotor end':>11}")
    casc_t, cfg = run(p, casc, duration=25.0, tilt_deg=TILT_DEG)
    mct = metrics(casc_t, cfg)
    print(f"{'PID on the angle only':<42}{dist['pid'][1]['mean']:>10.3f}d"
          f"{dist['pid'][1]['rotor']:>10.0f}d{dist['pid'][1]['rotor_end']:>10.0f}d")
    print(f"{'cascade PID':<42}{mct['mean']:>10.3f}d{mct['rotor']:>10.1f}d"
          f"{mct['rotor_end']:>10.1f}d")
    print()
    print(f"   The cascade settles at theta = {mct['mean']:+.2f} deg -- true vertical,"
          f" not measured zero --")
    print("   and parks the rotor instead of running it away. The parking angle is")
    print(f"   roughly phi / k_alpha = {TILT_DEG / g[3]:.0f} deg, so with +-90 deg of travel")
    print(f"   the rig has to be level to about 2 deg. Level the rig; do not ask the")
    print("   controller to hold a bias it can only hold by accelerating forever.")

    # ---------------------------------------------------------------- fig --
    fig, ax = plt.subplots(2, 3, figsize=(15.5, 8.0))
    fig.suptitle("PID designed from the identified model  "
                 rf"($\omega_n$ = {p.wn:.3f} rad/s, $\zeta$ = {p.zeta:.5f}, "
                 rf"$L_{{eff}}$ = {L * 1000:.1f} mm)  —  100 Hz, 0.3° encoder, ±6 m/s²",
                 fontsize=12)

    a = ax[0, 0]
    for key, c, lab in (("pd", "tab:orange", "PD"), ("pid", "tab:blue", "PID"),):
        a.plot(rows[key][0]["t"], rows[key][0]["theta_deg"], color=c, label=lab)
    a.plot(casc_log["t"], casc_log["theta_deg"], color="tab:green", label="cascade PID")
    a.axhspan(-0.5, 0.5, color="0.9", zorder=0)
    a.set(title="5° initial tilt", xlabel="t [s]", ylabel=r"$\theta$ [deg]", xlim=(0, 3))
    a.legend(fontsize=8); a.grid(alpha=0.3)

    a = ax[0, 1]
    for key, c, lab in (("pd", "tab:orange", "PD"), ("pid", "tab:blue", "PID")):
        a.plot(rows[key][0]["t"], rows[key][0]["u"], color=c, label=lab)
    a.plot(casc_log["t"], casc_log["u"], color="tab:green", label="cascade PID")
    a.axhline(6, color="r", ls=":", lw=1); a.axhline(-6, color="r", ls=":", lw=1,
                                                     label="saturation")
    a.set(title="command", xlabel="t [s]", ylabel=r"$u$ [m/s$^2$]", xlim=(0, 3))
    a.legend(fontsize=8); a.grid(alpha=0.3)

    a = ax[0, 2]
    a.plot(pid_long["t"], pid_long["alpha_deg"], color="tab:blue", label="PID only")
    a.plot(casc_log["t"], casc_log["alpha_deg"], color="tab:green", label="cascade PID")
    a.plot(sf_log["t"], sf_log["alpha_deg"], color="0.5", ls="--",
           label="4-state (exact $\\dot\\theta$)")
    a.axhline(90, color="r", ls=":", lw=1); a.axhline(-90, color="r", ls=":", lw=1,
                                                      label="±90° travel")
    a.set(title="rotor angle: what angle-only feedback cannot see",
          xlabel="t [s]", ylabel=r"$\alpha$ [deg]")
    a.legend(fontsize=8); a.grid(alpha=0.3)

    a = ax[1, 0]
    a.plot(dist["pd"][0]["t"], dist["pd"][0]["theta_deg"], color="tab:orange", label="PD")
    a.plot(dist["pid"][0]["t"], dist["pid"][0]["theta_deg"], color="tab:blue", label="PID")
    a.plot(casc_t["t"], casc_t["theta_deg"], color="tab:green", label="cascade PID")
    a.axhline(0, color="k", lw=0.8)
    a.axhline(-TILT_DEG, color="k", ls="--", lw=1, label="true vertical")
    a.set(title=f"rig {TILT_DEG:.0f}° out of level", xlabel="t [s]",
          ylabel=r"$\theta$ [deg]", ylim=(-4, 5.5), xlim=(0, 15))
    a.legend(fontsize=8); a.grid(alpha=0.3)

    a = ax[1, 1]
    a.plot(dist["pd"][0]["t"], dist["pd"][0]["alpha_deg"], color="tab:orange", label="PD")
    a.plot(dist["pid"][0]["t"], dist["pid"][0]["alpha_deg"], color="tab:blue", label="PID")
    a.plot(casc_t["t"], casc_t["alpha_deg"], color="tab:green", label="cascade PID")
    a.axhline(90, color="r", ls=":", lw=1); a.axhline(-90, color="r", ls=":", lw=1)
    a.set(title="…and its rotor: zero angle error costs unbounded travel",
          xlabel="t [s]", ylabel=r"$\alpha$ [deg]", yscale="symlog", xlim=(0, 15))
    a.legend(fontsize=8); a.grid(alpha=0.3)

    a = ax[1, 2]
    wcs = [s_[0] for s_ in sweep]
    a.plot(wcs, [s_[1]["u_peak"] for s_ in sweep], "o-", color="tab:red",
           label="peak |u|")
    a.axhline(6, color="r", ls=":", lw=1, label="±6 m/s² limit")
    a.set(xlabel=r"$\omega_{n,des}$ [rad/s]", ylabel=r"peak $|u|$ [m/s$^2$]",
          title="how fast the design can be asked to be")
    a2 = a.twinx()
    a2.plot(wcs, [s_[1]["settle"] for s_ in sweep], "s--", color="tab:blue",
            label="settling time")
    a2.set_ylabel("settling time to 0.5° [s]", color="tab:blue")
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    a.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("design_pid.png", dpi=140)
    print("\nwrote design_pid.png")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
