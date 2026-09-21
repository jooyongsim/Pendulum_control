"""Regenerate every number quoted in pid_gain_design.md.

The doc argues from numbers; this reproduces them from the same constants the
simulator uses, so the two cannot drift apart. Run it after touching the gains,
the plant parameters or the doc.

    python check_pid_gain_design.py
"""
import importlib.util
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "pidrt", os.path.join(HERE, "pid_pendulum_realtime.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

J, K, C, WN = m.J, m.K, m.C, m.WN
WD, ZD, AL = m.WN_DES, m.ZETA_DES, m.ALPHA
REF = np.deg2rad(20.0)
FC = np.deg2rad(2.705) * WN ** 2 * (2 * np.pi / WN) / 4.0   # rotary_model.py's dry friction


def gains(alpha):
    kd = J * (2 * ZD * WD + alpha) - C
    kp = J * (WD ** 2 + 2 * alpha * ZD * WD) - K
    ki = J * alpha * WD ** 2
    return kp, ki, kd


def simulate(alpha, T=6.0, dt=2e-4, fc=0.0, ref=REF):
    """The script's own loop, with alpha and dry friction as knobs.

    Returns the metrics AND the trajectories, so the plotting script can draw
    exactly what is being measured here.
    """
    kp, ki, kd = gains(alpha)
    x = np.zeros(3)
    n = int(T / dt)
    th, u = np.empty(n), np.empty(n)
    for i in range(n):
        a, w, ie = x
        e = ref - a
        tau_raw = kp * e + ki * ie - kd * w
        die = e if abs(tau_raw) < m.TAU_MAX else 0.0
        tau = float(np.clip(tau_raw, -m.TAU_MAX, m.TAU_MAX))
        dw = (tau - C * w - K * np.sin(a) - fc * np.tanh(w / 1e-3)) / J
        x = x + dt * np.array([w, dw, die])
        th[i], u[i] = x[0], tau
    out = np.abs(th - ref) > 0.02 * ref
    settle = (np.flatnonzero(out)[-1] + 1) * dt if out.any() and not out[-1] else np.nan
    return dict(overshoot=100 * (th.max() / ref - 1), settle=settle,
                final=np.rad2deg(th[-1]), t=np.arange(n) * dt,
                theta_deg=np.rad2deg(th), u=u, ref_deg=np.rad2deg(ref))


def scenario(ref_deg, theta0_deg=0.0, events=(), T=4.0, dt=2e-4, fc=0.0):
    """Drive the loop the way the KEYBOARD does, not with an arbitrary step.

    `events` is a list of (time, "ref"|"kick", amount) — `ref` moves the
    reference by that much (the up/down keys move it 5 deg at a time), `kick`
    adds to omega (the left/right keys add 200 deg/s). This matters for the
    saturation question: the UI can never command a large step, it can only
    nudge the reference, so asking "does a 0 -> 45 deg step saturate" answers a
    question the simulator is never actually asked.
    """
    kp, ki, kd = gains(AL)
    x = np.array([np.deg2rad(theta0_deg), 0.0, 0.0])
    ref = np.deg2rad(ref_deg)
    ev = sorted(events)
    n = int(T / dt)
    u, th = np.empty(n), np.empty(n)
    for i in range(n):
        t = i * dt
        while ev and t >= ev[0][0]:
            _, kind, val = ev.pop(0)
            if kind == "ref":
                ref += val
            else:
                x[1] += val
        a, w, ie = x
        e = ref - a
        raw = kp * e + ki * ie - kd * w
        die = e if abs(raw) < m.TAU_MAX else 0.0
        tau = float(np.clip(raw, -m.TAU_MAX, m.TAU_MAX))
        dw = (tau - C * w - K * np.sin(a) - fc * np.tanh(w / 1e-3)) / J
        x = x + dt * np.array([w, dw, die])
        u[i], th[i] = tau, x[0]
    return dict(t=np.arange(n) * dt, u=u, theta_deg=np.rad2deg(th),
                peak=np.abs(u).max(),
                sat=100.0 * np.mean(np.abs(u) >= m.TAU_MAX - 1e-9))


# The keyboard scenarios, in one place so the doc, the check and the figures
# all quote the same runs. REF_STEP and KICK are the simulator's own constants.
KEY_CASES = [
    ("initial step 0 -> 20 deg (--check)", dict(ref_deg=20.0)),
    ("up arrow: 20 -> 25 deg (one UI step)",
     dict(ref_deg=20.0, theta0_deg=20.0, events=[(0.5, "ref", m.REF_STEP)])),
    ("up arrow held, 4 steps 0.1 s apart",
     dict(ref_deg=20.0, theta0_deg=20.0,
          events=[(0.5 + 0.1 * k, "ref", m.REF_STEP) for k in range(4)])),
    ("kick +200 deg/s (left/right key)",
     dict(ref_deg=20.0, theta0_deg=20.0, events=[(0.5, "kick", m.KICK)])),
    ("kick +400 deg/s (twice the key)",
     dict(ref_deg=20.0, theta0_deg=20.0, events=[(0.5, "kick", 2 * m.KICK)])),
    ("artificial jump 0 -> 45 deg", dict(ref_deg=45.0)),
    ("artificial jump 0 -> 90 deg", dict(ref_deg=90.0)),
]


def linear_step(num, den, T=3.0, dt=1e-4):
    """Step response of num/den in controllable canonical form."""
    den = np.asarray(den, float)
    n = len(den) - 1
    A = np.zeros((n, n))
    A[0, :] = -den[1:] / den[0]
    A[1:, :-1] = np.eye(n - 1)
    B = np.zeros(n)
    B[0] = 1.0
    b = np.concatenate([np.zeros(n + 1 - len(num)), np.asarray(num, float)]) / den[0]
    Cm = b[1:] - b[0] * den[1:] / den[0]
    x, y = np.zeros(n), []
    for _ in range(int(T / dt)):
        x = x + dt * (A @ x + B)
        y.append(float(Cm @ x + b[0]))
    return np.asarray(y)


def main():
    print("=" * 72)
    print("section 1 and 3: plant, gains, placement")
    print("=" * 72)
    kp, ki, kd = gains(AL)
    print(f"  K = wn^2 = {K:.3f} 1/s^2      C = 2 zeta wn = {C:.4f} 1/s")
    print(f"  Kp = {kp:.3f}   Ki = {ki:.1f}   Kd = {kd:.4f}")
    poles = np.roots([J, C + kd, K + kp, ki])
    want = np.append(np.roots([1.0, 2 * ZD * WD, WD ** 2]), -AL)
    print(f"  closed-loop poles : {np.round(np.sort_complex(poles), 3)}")
    print(f"  requested         : {np.round(np.sort_complex(want), 3)}")
    assert np.allclose(np.sort_complex(poles), np.sort_complex(want), atol=1e-9)
    print("  -> placement exact")
    print(f"  gravity share of Kp: {100 * K / kp:.0f} %   damping share of Kd: {100 * C / kd:.1f} %")

    print()
    print("=" * 72)
    print("section 4.1: the integrator pole")
    print("=" * 72)
    print(f"  1/alpha = {1 / AL:.3f} s")
    print(f"{'alpha':>7}{'Ki':>9}{'zero':>9}{'overshoot':>11}{'settle 2%':>11}{'final':>10}")
    for a in (0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 20.0):
        kp_a, ki_a, _ = gains(a)
        r = simulate(a)
        z = -ki_a / kp_a if ki_a else float("nan")
        print(f"{a:>7.1f}{ki_a:>9.1f}{z:>9.2f}{r['overshoot']:>10.1f}%"
              f"{r['settle']:>10.2f}s{r['final']:>9.3f}d")

    cross = K / (2 * ZD * WD)
    print(f"\n  zero meets the integrator pole at alpha = k/(2 zeta_des wn_des) = {cross:.3f}")
    kp_c, ki_c, _ = gains(cross)
    assert abs(ki_c / kp_c - cross) < 1e-9, "crossover algebra is wrong"
    print(f"{'alpha':>7}{'int pole':>10}{'zero':>9}{'zero/pole':>11}")
    for a in (1.0, 2.0, cross, 4.0, 8.0, 20.0):
        kp_a, ki_a, _ = gains(a)
        print(f"{a:>7.2f}{-a:>10.2f}{-ki_a / kp_a:>9.2f}{(ki_a / kp_a) / a:>11.2f}")

    print("\n  steady state with and without the integrator (6 s, 20 deg reference):")
    for fc, tag in ((0.0, "viscous only"), (FC, f"+ dry friction Fc={FC:.3f}")):
        row = "   ".join(f"alpha={a:<4.1f} {simulate(a, fc=fc)['final']:7.3f} deg"
                         for a in (0.0, 2.0, 8.0))
        print(f"    {tag:<28} {row}")

    print()
    print("=" * 72)
    print("section 5: the PID zero, not the placement, causes the overshoot")
    print("=" * 72)
    den = [J, C + kd, K + kp, ki]
    y_zero = linear_step([kp, ki], den)
    y_nozero = linear_step([ki], den)
    print(f"  zero at s = {-ki / kp:.2f} rad/s   (poles at {-ZD * WD:.1f}+-j..., {-AL:.1f})")
    print(f"  linear, ref through Kp and Ki : {100 * (y_zero.max() / y_zero[-1] - 1):.1f} %")
    print(f"  linear, ref through Ki only   : {100 * (y_nozero.max() / y_nozero[-1] - 1):.1f} %")
    print(f"  second-order formula          : "
          f"{100 * np.exp(-np.pi * ZD / np.sqrt(1 - ZD ** 2)):.1f} %")
    print(f"  nonlinear plant (--check)     : {simulate(AL)['overshoot']:.1f} %")

    print()
    print("=" * 72)
    print("section 6: saturation")
    print("=" * 72)
    print(f"  |u| <= 3 wn^2 = {m.TAU_MAX:.1f} rad/s^2")
    for deg in (20, 45, 90):
        need = K * np.sin(np.deg2rad(deg))
        print(f"    holding {deg:2d} deg: {need:6.2f}  ({100 * need / m.TAU_MAX:4.1f} % of the limit)")
    kick = kp * REF
    print(f"    20 deg step kick Kp*e: {kick:6.2f}  ({100 * kick / m.TAU_MAX:4.1f} %)")
    print(f"  a single step saturates once Kp*e reaches the limit, i.e. at"
          f" e = {np.rad2deg(m.TAU_MAX / kp):.1f} deg")
    print(f"  but the keys move the reference {np.rad2deg(m.REF_STEP):.0f} deg at a time:")
    print(f"{'    scenario':<48}{'peak |u|':>10}{'saturated':>11}")
    for name, kw in KEY_CASES:
        r = scenario(**kw)
        print(f"    {name:<44}{r['peak']:>10.1f}{r['sat']:>10.1f}%")
    print(f"  linearisation error at 20 deg: "
          f"{100 * (K * REF - K * np.sin(REF)) / (K * np.sin(REF)):+.1f} %")
    print(f"  dry friction vs viscous at w = 1 rad/s: {FC / C:.1f}x")
    print("\nall numbers in pid_gain_design.md reproduced")


if __name__ == "__main__":
    main()
