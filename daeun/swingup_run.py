"""Start / tune / log the on-board swing-up controller (SwingUpController.ino).

The control loop lives on the MCU at 500 Hz; this script only sets the mode,
optionally overrides gains, and records what happens.

    python swingup_run.py --port COM40                 # swing up from hanging
    python swingup_run.py --port COM40 --mode balance  # hand it the pendulum instead
    python swingup_run.py --port COM40 --pump 0.3 --k-rotor 2.0

Observation from the firmware:
    [theta_deg (from upright), rotor_deg, v_cmd_pps, mode, u_accel, theta_dot_dps, status]
"""

import argparse
import csv
import math
import sys
import time

import numpy as np
from datetime import datetime
from pathlib import Path

UPSTREAM = Path(__file__).resolve().parents[1]   # the repository root

CMD_SET_HOME, CMD_SET_STEP_MODE = 0, 3
CMD_HARD_STOP, CMD_QUERY, CMD_RESET_SAFETY = 5, 6, 7
CMD_SET_MODE, CMD_SET_GAIN, CMD_SET_HANG_REF, CMD_SET_ACCEL = 10, 11, 12, 13
CMD_RESET_GAINS = 14
STEP_MODE_16 = 4
MODES = {"idle": 0, "balance": 1, "swingup": 2}
GAIN_IDS = {"k1": 0, "k2": 1, "k3": 2, "k4": 3, "pump": 4, "k_rotor": 5,
            "k_rotor_rate": 6, "catch_deg": 7, "catch_dps": 8, "guard_deg": 9,
            "catch_arm_deg": 10, "bias_tau": 11, "phiref_tau": 12, "upright_offset": 13,
            "tau_d": 14, "bias_deadband_deg": 15, "ke": 16, "rotor_limit_deg": 17, "obs_pole": 18}
SPIN_MARGIN_S = 0.003


def tilt_from_hanging(theta_deg):
    """The firmware reports tilt from upright; hanging is +-180."""
    return (theta_deg + 180.0 + 180.0) % 360.0 - 180.0


def calibrate_vertical(step, wn, accel=1.5, seg=0.2, watch=3.5, rate=100.0):
    """Measure the true vertical: doublet, then the centre of the free swing.

    The rod rests anywhere inside its stiction band, so "hanging" is not exactly
    vertical and the balancer would park the arm ~64 deg away per degree of error.
    A freely swinging pendulum, though, is symmetric about gravity, so the centre
    of the swing IS the vertical, whatever the encoder zero happens to be.

    The arm drive is a zero-mean doublet (+a, -a, -a, +a over 4*seg), so the arm
    ends where it started.
    """
    period, T0 = 1.0 / rate, 2 * math.pi / wn
    rows = []
    t0 = time.perf_counter()
    while True:
        t = time.perf_counter() - t0
        if t >= 4 * seg + watch:
            break
        if t < seg:
            step(CMD_SET_ACCEL, [accel, 0.0])
        elif t < 3 * seg:
            step(CMD_SET_ACCEL, [-accel, 0.0])
        elif t < 4 * seg:
            step(CMD_SET_ACCEL, [accel, 0.0])
        else:
            r = step(CMD_SET_ACCEL, [0.0, 0.0]) if t < 4 * seg + 2 * period else step(CMD_QUERY, [0.0, 0.0])
            if r:
                rows.append((t, tilt_from_hanging(r[3][0])))
            time.sleep(max(0.0, period - 0.002))
            continue
        time.sleep(max(0.0, period - 0.002))
    step(CMD_SET_ACCEL, [0.0, 0.0])

    if len(rows) < 40:
        return None, 0, 0.0
    t = np.array([r[0] for r in rows])
    x = np.array([r[1] for r in rows])
    mids, a = [], t[0] + 0.2
    while a + T0 / 2 <= t[-1]:
        w = (t >= a) & (t < a + T0 / 2)
        if w.sum() > 8 and x[w].max() - x[w].min() > 1.0:
            mids.append((x[w].max() + x[w].min()) / 2)
        a += T0 / 4
    if len(mids) < 3:
        return None, len(mids), float(np.ptp(x))
    return float(np.median(mids)), len(mids), float(np.ptp(x))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", required=True)
    ap.add_argument("--mode", choices=list(MODES), default="swingup")
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--rate", type=float, default=100.0, help="logging rate [Hz]")
    for name in GAIN_IDS:
        ap.add_argument(f"--{name.replace('_', '-')}", type=float, default=None,
                        help="override this firmware gain")
    ap.add_argument("--calibrate", action="store_true",
                    help="measure the true vertical from a free swing before starting")
    ap.add_argument("--calib-accel", type=float, default=1.5, help="doublet strength [m/s^2]")
    ap.add_argument("--no-hang-ref", action="store_true",
                    help="skip pinning the reference to the hanging rod")
    ap.add_argument("--out")
    a = ap.parse_args()

    sys.path.insert(0, str(UPSTREAM))
    from control_comms import ControlComms, DebugLevel, StatusCode, find_board_port

    ctrl = ControlComms(timeout=0.05, debug_level=DebugLevel.DEBUG_ERROR)
    port = find_board_port(a.port)
    if ctrl.connect(port, 500000) is not StatusCode.OK:
        raise SystemExit(f"could not open {port}")

    def step(cmd, action):
        return ctrl.step(cmd, action)

    first = step(CMD_QUERY, [0.0, 0.0])
    if first is None:
        raise SystemExit("no reply; is SwingUpController.ino uploaded?")
    if len(first[3]) != 7:
        raise SystemExit(f"the board returned {len(first[3])} observations, expected 7 — "
                         "that is the old PendulumController firmware, not SwingUpController")

    step(CMD_HARD_STOP, [0.0, 0.0])
    # Gains live in RAM and survive between runs, so a run could silently inherit
    # whatever the last one set. Start from the compiled defaults every time.
    step(CMD_RESET_GAINS, [0.0, 0.0])
    step(CMD_SET_STEP_MODE, [STEP_MODE_16, 0.0])
    step(CMD_RESET_SAFETY, [0.0, 0.0])
    step(CMD_SET_HOME, [0.0, 0.0])

    # The encoder counts from power-up, and a rod that went over the top in an
    # earlier run leaves that zero half a turn out -- the board would then try to
    # balance at the BOTTOM. Pin the reference to the rod as it hangs now.
    if not a.no_hang_ref:
        settled = []
        t_wait = time.perf_counter() + 2.0
        while time.perf_counter() < t_wait:
            r = step(CMD_QUERY, [0.0, 0.0])
            if r:
                # hanging reads +-180, which wraps: compare tilt from hanging,
                # or a still rod looks like it is swinging 360 deg
                settled.append(tilt_from_hanging(r[3][0]))
            time.sleep(0.02)
        if settled and max(settled) - min(settled) > 1.5:
            raise SystemExit("the pendulum is still moving: let it hang still, then rerun "
                             "(or pass --no-hang-ref)")
        step(CMD_SET_HANG_REF, [0.0, 0.0])
        r = step(CMD_QUERY, [0.0, 0.0])
        print(f"hanging reference set; pendulum now reads {r[3][0]:+.1f} deg from upright"
              if r else "hanging reference set")

    if a.calibrate:
        print("calibrating the vertical (doublet + free swing, ~6 s) — keep clear")
        centre, n, span = calibrate_vertical(step, wn=7.521, accel=a.calib_accel)
        if centre is None:
            raise SystemExit(f"calibration swing too small ({n} windows, span {span:.1f} deg): "
                             "raise --calib-accel or check the arm")
        if abs(centre) > 5.0:
            raise SystemExit(f"measured vertical {centre:+.2f} deg is outside the stiction band; "
                             "check the hanging reference")
        print(f"  true vertical is {centre:+.2f} deg from the hanging reference "
              f"({n} half-period windows, swing {span:.1f} deg)")
        a.upright_offset = centre
        while True:
            r = step(CMD_QUERY, [0.0, 0.0])
            if r and abs(tilt_from_hanging(r[3][0])) < 3.0 and abs(r[3][5]) < 20.0:
                break
            time.sleep(0.05)

    for name, idx in GAIN_IDS.items():
        value = getattr(a, name)
        if value is not None:
            step(CMD_SET_GAIN, [float(idx), float(value)])
            print(f"  gain {name} = {value}")
    print("  (all other gains: compiled defaults)")

    out = Path(a.out) if a.out else (UPSTREAM / "data")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"swingup_{datetime.now():%Y%m%d_%H%M%S}.csv"
    print(f"logging -> {path}")
    if a.mode == "swingup":
        print("the arm will start moving by itself — keep clear. Ctrl+C stops it.")
    else:
        print("hold the pendulum near upright; the board catches it. Ctrl+C stops.")

    step(CMD_SET_MODE, [float(MODES[a.mode]), 0.0])
    period = 1.0 / a.rate
    t0 = time.perf_counter()
    rows, best = 0, 180.0
    try:
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "mcu_ms", "status", "terminated", "theta_deg", "rotor_deg",
                        "v_cmd_pps", "mode", "u_accel", "theta_dot_dps", "l6474_status"])
            while time.perf_counter() - t0 < a.duration:
                tick = time.perf_counter()
                resp = step(CMD_QUERY, [0.0, 0.0])
                if resp is None:
                    continue
                status, mcu_ms, terminated, obs = resp
                w.writerow([f"{tick - t0:.4f}", mcu_ms, status, terminated,
                            *[f"{v:.3f}" for v in obs]])
                rows += 1
                best = min(best, abs(obs[0]))
                if rows % max(1, int(a.rate // 2)) == 0:
                    print(f"\r t={tick - t0:5.1f}s  theta {obs[0]:+7.2f} deg  "
                          f"rotor {obs[1]:+7.1f} deg  cmd {obs[2]:+6.0f} pps  "
                          f"{'BAL' if int(obs[3]) == 3 else 'swing'}  "
                          f"offset {obs[6]:+5.2f} deg  closest {best:5.2f} deg", end="", flush=True)
                if terminated:
                    print("\nfirmware latched a fault; stopping")
                    break
                deadline = tick + period
                while True:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        break
                    if remaining > SPIN_MARGIN_S:
                        time.sleep(remaining - SPIN_MARGIN_S)
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        step(CMD_SET_MODE, [0.0, 0.0])
        step(CMD_HARD_STOP, [0.0, 0.0])
        ctrl.close()
    print(f"\n{rows} samples, closest approach to upright {best:.2f} deg -> {path}")


if __name__ == "__main__":
    main()
