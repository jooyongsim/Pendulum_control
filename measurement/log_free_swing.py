"""Record the pendulum's free swing: the measurement behind the identification.

The motor is held in hard stop and only CMD_QUERY is sent, so nothing drives the
pendulum. You pull it aside by hand, let go, and this logs the decay. Everything
downstream (wn, zeta, sigma, rod length) comes from this one CSV.

Why the loop looks the way it does
----------------------------------
sampling grid
    The next deadline is session_start + n*T, not "now + T". Adding T to the
    current time lets the period drift upward every cycle; anchoring to a fixed
    grid keeps the average rate exact even when a serial round trip runs late.

busy-wait tail
    time.sleep() on Windows only resolves to roughly 15 ms, which is most of a
    20 ms period. The last SPIN_MARGIN_S is spun instead of slept.

dropped samples
    A serial timeout drops that one sample and the loop carries on; the run is
    not aborted. The count is reported at the end.

Usage
    python log_free_swing.py                  # 60 s at 50 Hz, port auto-detected
    python log_free_swing.py --seconds 120
    python log_free_swing.py --port COM10 --rate 100
"""

import argparse
import math
import sys
import time
from datetime import datetime
from pathlib import Path

# control_comms.py lives one directory up, in Pendulum_control/
for _candidate in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents):
    if (_candidate / "control_comms.py").exists():
        sys.path.insert(0, str(_candidate))
        break
else:
    raise FileNotFoundError("control_comms.py not found above " + str(Path(__file__).parent))

import pandas as pd  # noqa: E402

from control_comms import (ControlComms, DebugLevel, StatusCode,  # noqa: E402
                           find_board_port)

# Firmware command IDs (PendulumController.ino)
CMD_SET_HOME = 0
CMD_SET_STEP_MODE = 3
CMD_HARD_STOP = 5
CMD_QUERY = 6
CMD_RESET_SAFETY = 7
STEP_MODE_16 = 4

BAUD_RATE = 500000
TIMEOUT = 0.25
SPIN_MARGIN_S = 0.003      # busy-wait tail; Windows sleep() granularity is ~15 ms


def wrapped_error_deg(angle_deg, setpoint_deg=180.0):
    """Error from upright, wrapped to -180..180 deg."""
    return (setpoint_deg - angle_deg + 180.0) % 360.0 - 180.0


def connect(port, baud):
    ctrl = ControlComms(timeout=TIMEOUT, debug_level=DebugLevel.DEBUG_ERROR)
    if ctrl.connect(port, baud) is not StatusCode.OK:
        raise RuntimeError(f"could not open {port}")
    # Known state: motor stopped, encoder readable. No drive command is ever sent.
    ctrl.step(CMD_HARD_STOP, [0.0])
    ctrl.step(CMD_SET_STEP_MODE, [STEP_MODE_16])
    ctrl.step(CMD_SET_HOME, [0.0])
    ctrl.step(CMD_RESET_SAFETY, [0.0])

    resp = ctrl.step(CMD_QUERY, [0.0])
    if resp is None:
        raise RuntimeError("the board is not answering; re-flash PendulumController.ino "
                           "and check the port")
    status, mcu_ms, terminated, obs = resp
    if len(obs) != 4:
        raise RuntimeError(f"the board returned {len(obs)} observations, expected 4 — "
                           "that is different firmware than PendulumController.ino")
    print(f"connected to {port}")
    print(f"  status={status} latched={terminated}  pendulum={obs[0]:.2f} deg  "
          f"rotor={obs[1]:.2f} deg  L6474=0x{int(obs[3]):04X}")
    return ctrl


def record(ctrl, duration, period, progress_every=5.0):
    records, dropped = [], 0
    prev_t = prev_pend = prev_rotor = None
    first_mcu_ms = None
    next_progress = progress_every
    cycle = 0
    t0 = time.perf_counter()

    print(f"\nrecording {duration:.0f} s at {1 / period:.0f} Hz — "
          f"pull the pendulum aside and let go (Ctrl+C stops early, data is kept)")
    try:
        while True:
            if time.perf_counter() - t0 >= duration:
                break

            resp = ctrl.step(CMD_QUERY, [0.0])
            elapsed = time.perf_counter() - t0
            unix = time.time()

            if resp is None:
                dropped += 1
            else:
                status, mcu_ms, terminated, obs = resp
                pend = float(obs[0]) if len(obs) > 0 else math.nan
                rotor = float(obs[1]) if len(obs) > 1 else math.nan
                speed = float(obs[2]) if len(obs) > 2 else math.nan
                l6474 = int(obs[3]) if len(obs) > 3 else -1
                if first_mcu_ms is None:
                    first_mcu_ms = int(mcu_ms)

                dt = math.nan if prev_t is None else elapsed - prev_t
                pend_vel = rotor_vel = math.nan
                if prev_t is not None and dt > 0:
                    # the pendulum angle wraps at 0/360, so difference it wrapped
                    dp = (pend - prev_pend + 180.0) % 360.0 - 180.0
                    pend_vel = dp / dt
                    rotor_vel = (rotor - prev_rotor) / dt

                records.append({
                    "host_unix_s": unix,
                    "host_elapsed_s": elapsed,
                    "host_dt_s": dt,
                    "mcu_timestamp_ms": int(mcu_ms),
                    "mcu_elapsed_s": (int(mcu_ms) - first_mcu_ms) / 1000.0,
                    "response_status": int(status),
                    "terminated": bool(terminated),
                    "pendulum_angle_deg": pend,
                    "pendulum_error_deg": wrapped_error_deg(pend),
                    "pendulum_velocity_dps_est": pend_vel,
                    "rotor_angle_deg": rotor,
                    "rotor_velocity_dps_est": rotor_vel,
                    "motor_speed_pps_observed": speed,
                    "l6474_status_raw": l6474,
                })
                prev_t, prev_pend, prev_rotor = elapsed, pend, rotor

                if terminated:
                    print("firmware reported a safety condition; stopping")
                    break

            if elapsed >= next_progress:
                print(f"  {elapsed:5.1f} s / {duration:.0f} s   "
                      f"samples {len(records)}   dropped {dropped}")
                next_progress += progress_every

            # fixed grid: session_start + n*T, so the period cannot drift
            cycle += 1
            deadline = t0 + cycle * period
            if time.perf_counter() > deadline + period:
                cycle = int((time.perf_counter() - t0) / period) + 1
                deadline = t0 + cycle * period
            while True:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                if remaining > SPIN_MARGIN_S:
                    time.sleep(remaining - SPIN_MARGIN_S)
    except KeyboardInterrupt:
        print("interrupted; keeping what was collected")
    finally:
        try:
            ctrl.step(CMD_HARD_STOP, [0.0])
        except Exception:
            pass
    return records, dropped


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", default=None, help="serial port (default: auto-detect ST-Link)")
    ap.add_argument("--baud", type=int, default=BAUD_RATE)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--rate", type=float, default=50.0, help="sample rate in Hz")
    ap.add_argument("--data-dir", type=Path, default=None,
                    help="default: ../data next to control_comms.py")
    args = ap.parse_args()

    ctrl = connect(find_board_port(args.port), args.baud)
    try:
        records, dropped = record(ctrl, args.seconds, 1.0 / args.rate)
    finally:
        ctrl.close()

    if not records:
        print("no samples recorded")
        return 1

    span = records[-1]["host_elapsed_s"]
    print(f"\ndone: {len(records)} samples over {span:.2f} s "
          f"({len(records) / span:.1f} Hz), {dropped} dropped")

    data_dir = args.data_dir or (Path(sys.path[0]) / "data")
    data_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = data_dir / f"encoder_log_{stamp}.csv"
    pd.DataFrame(records).to_csv(out, index=False)
    print(f"CSV saved: {out.resolve()}")
    print(f"\nnext:  python ../pendulum_model_id.py --glob \"data/{out.name}\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
