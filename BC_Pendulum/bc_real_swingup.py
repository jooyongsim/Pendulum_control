"""Run a behavior-cloned swing-up policy on the real STEVAL-EDUKIT01 (SwingupController firmware).

Same start-up and guards as Pendulum_control/swing-up/swingup_control.py (driver check, arm homing,
wait still, SET_HOME, upright calibration from a small free swing, arm limit, stuck-arm and firmware
latch stops, CSV log) -- only the control law is the trained network:

    obs = [cos th, sin th, th', phi, phi']     th = tilt from the calibrated vertical (0 = upright)
    a   = policy(obs) in [-1, 1]  ->  u = a * 47.1 rad/s^2, recomputed every 20 ms (50 Hz) as in training

Start with the pendulum HANGING still (board reset while it hung). Ctrl+C stops the motor.

    python3 bc_real_swingup.py                                   # DAgger weights, 30 s
    python3 bc_real_swingup.py --model bc_models/rotary_bc_bc_dart.npz --duration 20
"""
import argparse
import math
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# inside the repo: Pendulum_control/BC_Pendulum -> ../swing-up;  outside: ~/rotary/BC_Pendulum -> ../Pendulum_control/swing-up
SWING_UP = next(p for p in (HERE.parent / "swing-up", HERE.parent / "Pendulum_control" / "swing-up") if p.is_dir())
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SWING_UP))
import balance_control as bc            # noqa: E402  serial link, estimator, constants
import swingup_control as sc            # noqa: E402  home_arm, calibrate_centre, wait_still
from bc_rotary_swingup import Policy, obs   # noqa: E402

ACCEL_MAX = bc.ACCEL_MAX_PPS2 * bc.STEP_RAD      # 47.12 rad/s^2
POLICY_EVERY = 2                                 # 100 Hz loop -> 50 Hz policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(HERE / "bc_models" / "rotary_bc_dagger.npz"))
    ap.add_argument("--duration", type=float, default=30.0, help="s")
    ap.add_argument("--arm-limit", type=float, default=140.0, help="deg")
    ap.add_argument("--calibrate", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--upright-offset", type=float, default=0.0, help="deg, with --no-calibrate")
    ap.add_argument("--port", default=None)
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    policy = Policy.load(args.model)
    period = 1.0 / 100.0
    print(f"policy {Path(args.model).name}: {len(policy.layers)} layers, 50 Hz, u = a x {ACCEL_MAX:.1f} rad/s^2")

    link = bc.Link(args.port)
    rows, dropped, reason = [], 0, "duration"
    try:
        link.step(bc.CMD_HARD_STOP)
        link.step(bc.CMD_RESET_SAFETY)
        time.sleep(0.1)
        link.step(bc.CMD_RESET_SAFETY)
        q = link.step(bc.CMD_QUERY)
        if q is None or len(q[3]) != 7:
            raise SystemExit("no/odd reply -- is SwingupController.ino on the board?")
        drv = int(q[3][5])
        if drv in (0xFFFF, 0x0000) or not all(drv & m for m in (0x0200, 0x0800, 0x1000)):
            raise SystemExit(f"L6474 driver not healthy (0x{drv:04X}): check motor power, reset the board")
        if abs(q[3][bc.OBS_ROTOR]) > 2.0:
            print(f"arm is at {q[3][bc.OBS_ROTOR]:+.1f} deg -- returning it before homing ...")
            print(f"   arm now {sc.home_arm(link, period):+.2f} deg")
        if not sc.wait_still(link):
            raise SystemExit("pendulum did not settle within 60 s")
        link.step(bc.CMD_SET_HOME)
        q = link.step(bc.CMD_QUERY)
        hang = math.degrees(bc.wrap_rad(math.radians(q[3][bc.OBS_PEND])))
        if abs(hang) > 5:
            raise SystemExit(f"pendulum reads {hang:+.1f} deg from hanging: let it hang still and reset the board")
        print(f"driver OK (0x{drv:04X}), pendulum {hang:+.1f} deg from hanging")
        offset = args.upright_offset
        if args.calibrate:
            c, n = sc.calibrate_centre(link, period)
            if c is None:
                raise SystemExit(f"calibration swing too small ({n} windows) -- rerun")
            offset = c
            print(f"calibrated vertical: {offset:+.2f} deg from the encoder zero ({n} half-period windows)")
            if abs(offset) > 5.0:
                raise SystemExit("offset larger than the stiction band -- check the encoder zero")
            if not sc.wait_still(link):
                raise SystemExit("pendulum did not settle after calibration")
        b_hat = math.radians(offset)
        if not args.yes:
            input(f"The NETWORK will swing the rod over the top (arm limit {args.arm_limit:.0f} deg). "
                  "Clear the area, press Enter ... ")

        est = bc.Estimator(180.0)
        stuck = deque()
        up_at, arm_max = None, 0.0
        t0 = time.perf_counter()
        k, last, u = 0, link.step(bc.CMD_QUERY), 0.0
        while True:
            t = time.perf_counter() - t0
            if t >= args.duration:
                break
            x = est.update(last[3], t) if last else None
            if x is None:
                dropped += 1
            elif k % POLICY_EVERY == 0:
                th = bc.wrap_rad(x[0] - b_hat)
                u = policy(obs((th, x[1], x[2], x[3]))) * ACCEL_MAX
                if up_at is None and abs(math.degrees(th)) < 3.0:
                    up_at = t
                    print(f"   upright at {t:.2f} s")
            pps2 = float(np.clip(u * bc.PPS2_PER_RAD, -bc.ACCEL_MAX_PPS2, bc.ACCEL_MAX_PPS2))
            resp = link.step(bc.CMD_SET_ACCEL, pps2)
            if resp is None:
                dropped += 1
            else:
                last = resp
                st, _, latched, o = resp
                arm_max = max(arm_max, abs(o[bc.OBS_ROTOR]))
                if x is not None:
                    rows.append((t, math.degrees(bc.wrap_rad(x[0] - b_hat)), math.degrees(x[1]),
                                 o[bc.OBS_ROTOR], math.degrees(x[3]), u, pps2, st, int(latched)))
                if latched:
                    reason = f"firmware latched (status {st})"; break
                if abs(o[bc.OBS_ROTOR]) > args.arm_limit:
                    reason = f"arm {o[bc.OBS_ROTOR]:.1f} deg"; break
                if abs(u) > 10.0:                      # a large command must move the arm
                    stuck.append((t, o[bc.OBS_ROTOR]))
                    while stuck and t - stuck[0][0] > 0.3:
                        stuck.popleft()
                    arms = [a for _, a in stuck]
                    if t - stuck[0][0] >= 0.28 and max(arms) - min(arms) < 0.5:
                        reason = "arm not following -- motor driver/power?"; break
                else:
                    stuck.clear()
            k += 1
            deadline = t0 + k * period
            while time.perf_counter() < deadline:
                time.sleep(max(0.0, min(0.002, deadline - time.perf_counter())))
    except KeyboardInterrupt:
        reason = "Ctrl+C"
    finally:
        link.close()

    print(f"\nstopped: {reason} | samples {len(rows)} | dropped {dropped}")
    if not rows:
        return
    d = np.array(rows)
    out = HERE / "data"
    out.mkdir(exist_ok=True)
    csv = out / f"bc_real_{Path(args.model).stem}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    np.savetxt(csv, d, delimiter=",", comments="",
               header="t_s,theta_deg,theta_dot_dps,phi_deg,phi_dot_dps,u_rad_s2,cmd_pps2,status,latched")
    print(f"CSV -> {csv}")
    tail = d[d[:, 0] > d[-1, 0] - 2]
    print(f"upright: {'%.2f s' % up_at if up_at is not None else 'never'} | arm max {arm_max:.1f} deg | "
          f"last 2 s |theta| rms {np.sqrt(np.mean(tail[:, 1] ** 2)):.2f} deg")


if __name__ == "__main__":
    main()
