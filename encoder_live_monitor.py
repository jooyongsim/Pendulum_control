"""Live pendulum-encoder monitor and CSV logger.

This program only sends CMD_QUERY and one initial CMD_HARD_STOP.  It never
commands motor motion.  Start it with the pendulum hanging down, allow the
zero calibration to finish, lift the pendulum by hand, then press Enter and
release it after the countdown.

Examples
--------
python encoder_live_monitor.py
python encoder_live_monitor.py --port COM5 --duration 20 --countdown 3
python encoder_live_monitor.py --no-wait --no-calibrate
"""

from __future__ import annotations

import argparse
import csv
import math
import queue
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

from control_comms import ControlComms, DebugLevel, StatusCode, find_board_port


CMD_HARD_STOP = 5
CMD_QUERY = 6
ENCODER_DEG_PER_COUNT = 360.0 / 1200.0


def wrap_deg(angle: float) -> float:
    """Map an angle to [-180, 180)."""
    return (angle + 180.0) % 360.0 - 180.0


def circular_mean_deg(values: list[float]) -> float:
    """Mean of wrapped degree measurements, safe near 0/360 degrees."""
    if not values:
        raise ValueError("at least one angle is required")
    s = sum(math.sin(math.radians(v)) for v in values)
    c = sum(math.cos(math.radians(v)) for v in values)
    return math.degrees(math.atan2(s, c)) % 360.0


def query_sample(ctrl: ControlComms) -> dict | None:
    """Return one decoded query sample or None after a serial timeout."""
    host_ns = time.perf_counter_ns()
    response = ctrl.step(CMD_QUERY, [0.0])
    received_ns = time.perf_counter_ns()
    if response is None:
        return None
    status, mcu_ms, terminated, observation = response
    if len(observation) < 4:
        raise RuntimeError(
            f"Firmware returned {len(observation)} observations; expected 4. "
            "Flash PendulumController/PendulumController.ino first."
        )
    return {
        "host_query_ns": host_ns,
        "host_received_ns": received_ns,
        "mcu_ms": mcu_ms,
        "status": status,
        "terminated": terminated,
        "pendulum_raw_deg": observation[0],
        "rotor_deg": observation[1],
        "motor_speed_pps": observation[2],
        "l6474_status": int(observation[3]),
    }


def calibrate_down(ctrl: ControlComms, seconds: float) -> float:
    """Measure the downward encoder reference while the pendulum is still."""
    print(f"Keep the pendulum hanging still for {seconds:.1f} s (zero calibration).")
    values: list[float] = []
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        sample = query_sample(ctrl)
        if sample is not None:
            values.append(sample["pendulum_raw_deg"])
    if len(values) < 3:
        raise RuntimeError("Too few replies during calibration; check port and firmware.")
    zero = circular_mean_deg(values)
    residual = [wrap_deg(v - zero) for v in values]
    print(
        f"Down reference: {zero:.3f} deg from {len(values)} samples; "
        f"peak residual {max(abs(v) for v in residual):.3f} deg."
    )
    return zero


def reader_loop(
    ctrl: ControlComms,
    out: queue.Queue,
    stop_event: threading.Event,
    zero_deg: float,
    start_ns: int,
) -> None:
    """Own all serial access in a background thread."""
    while not stop_event.is_set():
        try:
            sample = query_sample(ctrl)
            if sample is None:
                continue
            sample["time_s"] = (sample["host_received_ns"] - start_ns) * 1e-9
            sample["pendulum_deg"] = wrap_deg(sample["pendulum_raw_deg"] - zero_deg)
            out.put(sample)
        except Exception as exc:  # send errors to the GUI/main thread
            out.put(exc)
            stop_event.set()


def save_csv(rows: list[dict], output: Path, zero_deg: float, port: str, baud: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "time_s", "host_query_ns", "host_received_ns", "mcu_ms", "status",
        "terminated", "pendulum_raw_deg", "pendulum_deg", "rotor_deg",
        "motor_speed_pps", "l6474_status",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        handle.write(f"# down_reference_deg={zero_deg:.9f},port={port},baud={baud}\n")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)


def run(args: argparse.Namespace) -> Path:
    port = find_board_port(args.port)
    ctrl = ControlComms(timeout=args.timeout, debug_level=DebugLevel.DEBUG_ERROR)
    if ctrl.connect(port, args.baud) is not StatusCode.OK:
        raise RuntimeError(f"Could not open {port}")

    rows: list[dict] = []
    stop_event = threading.Event()
    reader: threading.Thread | None = None
    try:
        # Clear any prior motion command. No other motor command is issued.
        ctrl.step(CMD_HARD_STOP, [0.0])
        zero_deg = args.zero_deg
        if zero_deg is None:
            zero_deg = calibrate_down(ctrl, args.calibration_seconds)

        if not args.no_wait:
            input("Lift and hold the pendulum, then press Enter to arm recording... ")
        for remaining in range(args.countdown, 0, -1):
            print(f"Release in {remaining}...", flush=True)
            time.sleep(1.0)
        print("RELEASE / recording")

        sample_queue: queue.Queue = queue.Queue()
        start_ns = time.perf_counter_ns()
        reader = threading.Thread(
            target=reader_loop,
            args=(ctrl, sample_queue, stop_event, zero_deg, start_ns),
            daemon=True,
        )
        reader.start()

        times: deque[float] = deque()
        angles: deque[float] = deque()
        fig, ax = plt.subplots(figsize=(10, 5))
        line, = ax.plot([], [], lw=1.5)
        ax.axhline(0.0, color="black", lw=0.8)
        ax.set(xlabel="Time (s)", ylabel="Angle from down (deg)", title="Pendulum encoder")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        deadline = time.perf_counter() + args.duration

        try:
            while time.perf_counter() < deadline and plt.fignum_exists(fig.number):
                while True:
                    try:
                        item = sample_queue.get_nowait()
                    except queue.Empty:
                        break
                    if isinstance(item, Exception):
                        raise item
                    rows.append(item)
                    times.append(item["time_s"])
                    angles.append(item["pendulum_deg"])
                if times:
                    while times and times[-1] - times[0] > args.window:
                        times.popleft()
                        angles.popleft()
                    line.set_data(times, angles)
                    ax.set_xlim(max(0.0, times[-1] - args.window), max(args.window, times[-1]))
                    lo, hi = min(angles), max(angles)
                    margin = max(5.0, 0.1 * (hi - lo + 1.0))
                    ax.set_ylim(lo - margin, hi + margin)
                plt.pause(args.refresh)
        except KeyboardInterrupt:
            print("Stopping acquisition and saving received samples...")
    finally:
        stop_event.set()
        if reader is not None:
            reader.join(timeout=2.0)
        try:
            ctrl.step(CMD_HARD_STOP, [0.0])
        except Exception:
            pass
        ctrl.close()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(args.output) if args.output else Path("data") / f"pendulum_release_{stamp}.csv"
    if not rows:
        raise RuntimeError("No samples acquired; no CSV was written.")
    save_csv(rows, output, zero_deg, port, args.baud)
    rate = (len(rows) - 1) / (rows[-1]["time_s"] - rows[0]["time_s"]) if len(rows) > 1 else 0.0
    print(f"Saved {len(rows)} samples ({rate:.1f} samples/s) to {output}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", help="serial port; default is ST-Link auto-detection")
    parser.add_argument("--baud", type=int, default=500000)
    parser.add_argument("--timeout", type=float, default=0.2)
    parser.add_argument("--duration", type=float, default=20.0, help="recording seconds")
    parser.add_argument("--window", type=float, default=10.0, help="visible plot seconds")
    parser.add_argument("--refresh", type=float, default=0.04, help="plot refresh seconds")
    parser.add_argument("--calibration-seconds", type=float, default=2.0)
    parser.add_argument("--zero-deg", type=float, help="skip calibration and use this raw down angle")
    parser.add_argument("--countdown", type=int, default=3)
    parser.add_argument("--no-wait", action="store_true", help="do not wait for Enter")
    parser.add_argument("--output", help="CSV path; default is timestamped under data/")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        run(parse_args())
    except KeyboardInterrupt:
        print("Stopped by user.")
