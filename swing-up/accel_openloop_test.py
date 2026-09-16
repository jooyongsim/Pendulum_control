"""Open-loop acceleration test with the pendulum HANGING (SwingupController.ino).

Run this before any balancing attempt. The arm gets a small acceleration doublet
(+a for T, -a for 2T, +a for T) that ideally returns it to where it started
(peak excursion ~14 deg at the defaults), then everything is analysed:

  1. loop timing   : achieved control period and jitter, dropped replies
  2. step loss     : does the rotor come back to 0?
  3. coupling sign : hanging model  theta_h'' = -wn^2 sin - 2 zeta wn theta_h' - b cos(theta_h) phi''
                     (Lec01: the hanging rod swings opposite to the arm acceleration).
                     If the data needs b < 0, flip the sign of the upright controller.
  4. b = 3r/2l     : fitted from the swing, then r = 2 l b / 3 (compare with the ruler)

Before running: motor power ON, pendulum hanging perfectly still, board reset while it
hung still (encoder 0 = hanging), encoder cable slack, nothing within +-30 deg of the arm.

    python3 swing-up/accel_openloop_test.py                 # 2000 pps^2, 0.25 s pulses
    python3 swing-up/accel_openloop_test.py --accel 3000

Ctrl+C at any time sends HARD_STOP.
"""
import argparse
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from control_comms import ControlComms, StatusCode, DebugLevel, find_board_port  # noqa: E402

# SwingupController.ino
CMD_SET_HOME, CMD_SET_ACCEL, CMD_HARD_STOP, CMD_QUERY, CMD_RESET_SAFETY, CMD_ZERO_VEL = 0, 1, 2, 3, 4, 5
OBS_PEND, OBS_ROTOR, OBS_MEAS_PPS, OBS_TARGET_PPS, OBS_ACCEL, OBS_L6474, OBS_AGE = range(7)
STEP_RAD = 2 * math.pi / (200 * 16)          # 1/16 microstep
BAUD = 500000

# identified pendulum (encoder_log_20260915_152806.csv)
WN, ZETA = 7.499, 0.01172
L = 3 * 9.81 / (2 * WN ** 2)                 # 0.262 m
R_RULER = 0.12

ABORT_ROTOR_DEG = 45.0
ABORT_PEND_DEG = 60.0


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def profile(t, a, T):
    """+a on [0,T), -a on [T,3T), +a on [3T,4T), then 0."""
    if t < T:
        return a
    if t < 3 * T:
        return -a
    if t < 4 * T:
        return a
    return 0.0


def record(ctrl, accel_pps2, pulse_s, rate_hz, settle_s):
    period = 1.0 / rate_hz
    duration = 4 * pulse_s + settle_s
    rows, dropped = [], 0
    t0 = time.perf_counter()
    k = 0
    aborted = None
    zeroed = False
    while True:
        t = time.perf_counter() - t0
        if t >= duration:
            break
        a = profile(t, accel_pps2, pulse_s)
        tq = time.perf_counter()
        if t >= 4 * pulse_s and not zeroed:
            # The MCU integrates acceleration on its own clock, so the doublet does not cancel to
            # exactly zero velocity; a residual above the 30 pps floor keeps the arm creeping.
            r = ctrl.step(CMD_ZERO_VEL, [0.0])
            zeroed = True
        else:
            r = ctrl.step(CMD_SET_ACCEL, [a])
        rtt = (time.perf_counter() - tq) * 1000
        if r is None:
            dropped += 1
        else:
            st, mcu_ms, latched, obs = r
            rows.append((t, a, rtt, st, int(latched), obs[OBS_PEND], obs[OBS_ROTOR],
                         obs[OBS_MEAS_PPS], obs[OBS_TARGET_PPS], obs[OBS_ACCEL], obs[OBS_AGE]))
            if abs(obs[OBS_ROTOR]) > ABORT_ROTOR_DEG:
                aborted = f"rotor {obs[OBS_ROTOR]:.1f} deg"
            elif abs(wrap_deg(obs[OBS_PEND])) > ABORT_PEND_DEG:
                aborted = f"pendulum {wrap_deg(obs[OBS_PEND]):.1f} deg"
            elif latched:
                aborted = f"firmware latched, status {st}"
            if aborted:
                break
        k += 1
        deadline = t0 + k * period
        while time.perf_counter() < deadline:
            time.sleep(max(0.0, min(0.002, deadline - time.perf_counter())))
    return np.array(rows), dropped, aborted


def simulate_hanging(t, phidd, b, th0):
    """theta_h'' = -wn^2 sin th - 2 zeta wn th' - b cos(th) phi'',  RK4 on the sample grid (ZOH)."""
    th, w = th0, 0.0
    out = np.empty(len(t))
    for i in range(len(t)):
        out[i] = th
        if i + 1 == len(t):
            break
        h, u = t[i + 1] - t[i], phidd[i]
        f = lambda th_, w_: (w_, -WN ** 2 * math.sin(th_) - 2 * ZETA * WN * w_ - b * math.cos(th_) * u)
        k1 = f(th, w); k2 = f(th + h / 2 * k1[0], w + h / 2 * k1[1])
        k3 = f(th + h / 2 * k2[0], w + h / 2 * k2[1]); k4 = f(th + h * k3[0], w + h * k3[1])
        th += h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        w += h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    return out


def analyse(d, dropped, aborted, accel_pps2, pulse_s, rate_hz, out_png, show):
    t, a_cmd, rtt = d[:, 0], d[:, 1], d[:, 2]
    pend = np.array([wrap_deg(x) for x in d[:, 5]])
    rotor, meas_pps = d[:, 6], d[:, 7]

    print("\n" + "=" * 70)
    print("1. LOOP TIMING")
    dt = np.diff(t) * 1000
    print(f"   target {1000/rate_hz:.1f} ms | achieved median {np.median(dt):.2f} ms, "
          f"p95 {np.percentile(dt, 95):.2f} ms, max {dt.max():.2f} ms")
    print(f"   round trip median {np.median(rtt):.2f} ms, max {rtt.max():.2f} ms | dropped replies {dropped}")

    print("2. STEP LOSS (rotor should return to ~0)")
    end = rotor[t > 4 * pulse_s + 0.3]
    final = float(end.mean()) if end.size else float(rotor[-1])
    print(f"   peak |rotor| {np.abs(rotor).max():.2f} deg (ideal {accel_pps2*STEP_RAD*pulse_s**2*180/math.pi:.1f}) "
          f"| final {final:+.2f} deg")
    print("   note: the rotor angle is step-count based; the 30 pps dead zone near each velocity")
    print("         reversal also leaves a small offset, so a few tenths of a degree is normal.")

    print("3-4. COUPLING SIGN AND b = 3r/2l")
    phidd = a_cmd * STEP_RAD                                   # rad/s^2, as commanded
    th = np.deg2rad(pend)
    th0 = float(np.median(th[t < 0.02])) if np.any(t < 0.02) else float(th[0])
    fit_mask = t < 4 * pulse_s + 1.5
    best = None
    for b in np.linspace(-1.5, 1.5, 301):
        sim = simulate_hanging(t[fit_mask], phidd[fit_mask], b, th0)
        rms = float(np.sqrt(np.mean((sim - th[fit_mask]) ** 2)))
        if best is None or rms < best[1]:
            best = (b, rms, sim)
    b, rms, sim = best
    b_ruler = 3 * R_RULER / (2 * L)
    print(f"   best fit b = {b:+.3f}  (RMS {math.degrees(rms):.2f} deg)")
    if abs(b) < 0.05:
        print("   -> the pendulum barely responded; increase --accel and retry.")
    elif b > 0:
        print("   -> SAME as the Lec01 hanging model (rod swings opposite to +arm accel).")
        print("      Upright controller: use B_U = +3r/2l, i.e. the simulator sign as is.")
    else:
        print("   -> OPPOSITE to the Lec01 convention for this encoder/motor wiring.")
        print("      Upright controller: flip the sign (B_U = -3r/2l, or negate the command).")
    print(f"   r from fit = 2 l |b| / 3 = {2*L*abs(b)/3*1000:.0f} mm   "
          f"(ruler 120 mm -> b = {b_ruler:.3f}; slide 140 mm -> {3*0.14/(2*L):.3f})")
    if aborted:
        print(f"\n   !! run aborted early: {aborted}")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    ax[0].plot(t, a_cmd, "k", lw=1.2); ax[0].set_ylabel("accel cmd [pps^2]")
    ax[1].plot(t, rotor, "b", lw=1.2); ax[1].axhline(0, color="0.5", lw=0.7); ax[1].set_ylabel("rotor [deg]")
    ax[2].plot(t, meas_pps, "g", lw=1.0, label="measured"); ax[2].plot(t, d[:, 8], "g--", lw=1.0, label="target")
    ax[2].axhspan(-30, 30, color="r", alpha=0.08, label="30 pps dead zone"); ax[2].set_ylabel("step rate [pps]")
    ax[2].legend(fontsize=8)
    ax[3].plot(t, pend, "r", lw=1.2, label="measured pendulum")
    ax[3].plot(t[fit_mask], np.rad2deg(sim), "k--", lw=1.0, label=f"model, fitted b={b:+.3f}")
    ax[3].set_ylabel("pendulum from hanging [deg]"); ax[3].set_xlabel("t [s]"); ax[3].legend(fontsize=8)
    for x in ax:
        x.grid(alpha=.3)
    fig.suptitle(f"Open-loop acceleration test  (+-{accel_pps2:.0f} pps^2, T={pulse_s}s, {rate_hz:.0f} Hz)")
    fig.tight_layout(); fig.savefig(out_png, dpi=120)
    print(f"\nfigure -> {out_png}")
    if show:
        plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--accel", type=float, default=2000.0, help="pps^2, capped at 24000")
    ap.add_argument("--pulse", type=float, default=0.25, help="s")
    ap.add_argument("--rate", type=float, default=100.0, help="Hz")
    ap.add_argument("--settle", type=float, default=2.5, help="s of logging after the doublet")
    ap.add_argument("--port", default=None)
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--yes", action="store_true", help="skip the Enter confirmation")
    args = ap.parse_args()
    accel = float(np.clip(args.accel, -24000, 24000))

    port = find_board_port(args.port)
    ctrl = ControlComms(timeout=0.2, debug_level=DebugLevel.DEBUG_ERROR)
    if ctrl.connect(port, BAUD) is not StatusCode.OK:
        raise SystemExit(f"could not open {port}")
    time.sleep(2.0)
    try:
        ctrl.step(CMD_HARD_STOP, [0.0])
        ctrl.step(CMD_SET_HOME, [0.0])          # rotor here = 0, clears latches
        ctrl.step(CMD_RESET_SAFETY, [0.0])
        r = ctrl.step(CMD_QUERY, [0.0])
        if r is None or len(r[3]) != 7:
            raise SystemExit("no/odd reply -- is SwingupController.ino (7 observations) on the board?")

        print("checking the pendulum hangs still (1 s) ...")
        samples = []
        t_end = time.perf_counter() + 1.0
        while time.perf_counter() < t_end:
            q = ctrl.step(CMD_QUERY, [0.0])
            if q:
                samples.append(wrap_deg(q[3][OBS_PEND]))
            time.sleep(0.01)
        samples = np.array(samples)
        print(f"   pendulum {samples.mean():+.2f} deg, spread {np.ptp(samples):.2f} deg, L6474 0x{int(r[3][OBS_L6474]):04X}")
        if np.ptp(samples) > 1.0:
            raise SystemExit("pendulum is moving -- let it hang still and rerun")
        if abs(samples.mean()) > 5.0:
            print("   WARNING: encoder zero is not at hanging. Reset the board while the pendulum hangs still.")

        prompt = (f"\nArm will accelerate +-{accel:.0f} pps^2 (peak ~{accel*STEP_RAD*args.pulse**2*180/math.pi:.0f} deg). "
                  "Hands clear, press Enter ... ")
        if args.yes:
            print(prompt + "(--yes)")
        else:
            input(prompt)
        data, dropped, aborted = record(ctrl, accel, args.pulse, args.rate, args.settle)
    except KeyboardInterrupt:
        print("\ninterrupted")
        data, dropped, aborted = np.empty((0, 11)), 0, "Ctrl+C"
    finally:
        for cmd in (CMD_SET_ACCEL, CMD_ZERO_VEL, CMD_HARD_STOP):
            try:
                ctrl.step(cmd, [0.0])
            except Exception:
                pass
        ctrl.close()

    if len(data) < 10:
        raise SystemExit(f"too little data ({len(data)} samples){' -- ' + aborted if aborted else ''}")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv = ROOT / "data" / f"accel_openloop_{stamp}.csv"
    np.savetxt(csv, data, delimiter=",", comments="",
               header="t_s,accel_cmd_pps2,rtt_ms,status,latched,pendulum_deg,rotor_deg,"
                      "measured_pps,target_pps,fw_accel_pps2,cmd_age_ms")
    print(f"CSV -> {csv}")
    analyse(data, dropped, aborted, accel, args.pulse, args.rate,
            Path(__file__).with_name(f"accel_openloop_{stamp}.png"), not args.no_show)


if __name__ == "__main__":
    main()
