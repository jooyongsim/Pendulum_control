"""Run the behavior-cloning policy (bc_policy.pt) on the real STEVAL-EDUKIT01 (SwingupController.ino).

Same start-up as swing-up/swingup_control.py (driver check, arm homing, wait for a still rod,
SET_HOME, free-swing calibration of the true vertical), then:

    0.0-0.3 s  the teacher's open-loop kick (+pump, -pump), as in training
    0.3 s -    u = network([theta, theta', phi, phi'])        (--driver student, default)
               u = rotary_model.SwingUp, the teacher itself    (--driver teacher, for comparison)

theta is measured from the calibrated vertical. The student has no memory, so unlike the teacher
it does not adapt the vertical while balancing.
Guards: |phi| > --arm-limit, large command with a motionless arm, firmware latch -> stop. Ctrl+C stops.

    python behavior_cloning/bc_real.py --duration 20
    python behavior_cloning/bc_real.py --driver teacher --duration 20
"""
import argparse
import math
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "swing-up"))
from bc_play import load_policy  # noqa: E402
from bc_train import rm  # noqa: E402
import balance_control as bc  # noqa: E402
import swingup_control as sc  # noqa: E402

UP_DEG = 12.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--driver", choices=("student", "teacher"), default="student")
    ap.add_argument("--model", default=str(HERE / "output" / "bc_policy.pt"))
    ap.add_argument("--duration", type=float, default=20.0, help="s")
    ap.add_argument("--arm-limit", type=float, default=140.0, help="deg")
    ap.add_argument("--calibrate", action=argparse.BooleanOptionalAction, default=True,
                    help="measure the true vertical from a free swing before starting")
    ap.add_argument("--upright-offset", type=float, default=0.0, help="deg, with --no-calibrate")
    ap.add_argument("--port", default=None)
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(1)
    model = load_policy(args.model)
    teacher = rm.SwingUp(rm.place4())                 # kick for the student, whole run for --driver teacher
    pump = teacher.p["pump"]
    print(f"driver: {args.driver}  model: {args.model}")

    rate = rm.CONTROL_HZ
    period = 1.0 / rate
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
            print(f"arm at {q[3][bc.OBS_ROTOR]:+.1f} deg -- returning it before homing ...")
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
        off_rad = math.radians(offset)
        teacher.reset(offset_guess=off_rad)
        if not args.yes:
            input(f"The rod will swing over the top and the arm may move up to the {args.arm_limit:.0f} deg "
                  "limit. Clear the area, press Enter ... ")

        est = bc.Estimator(180.0)                     # theta from the encoder zero; hanging = +-180 deg
        stuck = deque()
        t0 = time.perf_counter()
        k, last = 0, link.step(bc.CMD_QUERY)
        while True:
            t = time.perf_counter() - t0
            if t >= args.duration:
                break
            x = est.update(last[3], t) if last else None
            if x is None:
                dropped += 1; u = 0.0; th = float("nan"); t_inf = 0.0
            else:
                th_raw, w, phi, phid = x
                th = bc.wrap_rad(th_raw - off_rad)
                ti = time.perf_counter()
                if args.driver == "teacher":
                    u = teacher(np.array([th_raw, w, phi, phid]), t)
                elif t < 0.15:
                    u = pump
                elif t < 0.30:
                    u = -pump
                else:
                    u = model.act([th, w, phi, phid])
                t_inf = (time.perf_counter() - ti) * 1000
            pps2 = float(np.clip(u * bc.PPS2_PER_RAD, -bc.ACCEL_MAX_PPS2, bc.ACCEL_MAX_PPS2))
            tq = time.perf_counter()
            resp = link.step(bc.CMD_SET_ACCEL, pps2)
            rtt = (time.perf_counter() - tq) * 1000
            if resp is None:
                dropped += 1
            else:
                last = resp
                st, _, latched, obs = resp
                if x is not None:
                    rows.append((t, math.degrees(th), math.degrees(x[1]), obs[bc.OBS_ROTOR],
                                 math.degrees(x[3]), u, pps2, rtt, t_inf, st, int(latched)))
                if latched:
                    reason = f"firmware latched (status {st})"; break
                if abs(obs[bc.OBS_ROTOR]) > args.arm_limit:
                    reason = f"arm {obs[bc.OBS_ROTOR]:.1f} deg"; break
                if abs(u) > 10.0:                     # same stall check as swingup_control.py
                    stuck.append((t, obs[bc.OBS_ROTOR]))
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
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv = ROOT / "data" / f"bc_real_{args.driver}_{stamp}.csv"
    np.savetxt(csv, d, delimiter=",", comments="",
               header="t_s,theta_deg,theta_dot_dps,phi_deg,phi_dot_dps,u_rad_s2,cmd_pps2,rtt_ms,infer_ms,status,latched")
    print(f"CSV -> {csv}")
    up = np.abs(d[:, 1]) < UP_DEG
    first = d[up, 0][0] if up.any() else None
    dt = np.diff(d[:, 0])
    print(f"loop: median {1000 * np.median(dt):.2f} ms, max {1000 * dt.max():.1f} ms | "
          f"inference median {np.median(d[:, 8]):.2f} ms")
    print(f"first upright (|theta| < {UP_DEG:.0f} deg): {'%.2f s' % first if first is not None else 'never'} | "
          f"time upright {up.sum() / rate:.1f} of {d[-1, 0]:.1f} s | arm max {np.abs(d[:, 3]).max():.1f} deg")
    if up.any():
        tail = d[d[:, 0] > d[-1, 0] - 2]
        if (np.abs(tail[:, 1]) < UP_DEG).all():
            print(f"last 2 s: |theta| rms {np.sqrt(np.mean(tail[:, 1] ** 2)):.2f} deg, arm now {d[-1, 3]:+.1f} deg")


if __name__ == "__main__":
    main()
