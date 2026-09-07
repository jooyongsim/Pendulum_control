"""Manual keyboard balancing and data logger for the rotary inverted pendulum.

Use the keyboard to manually move the rotary arm while trying to keep the
pendulum upright. Every observation returned by the firmware and every control
input sent to it is logged; data are saved as CSV and plots as PNG when the run
ends.

Controls
    Left  / A : negative motor velocity
    Right / D : positive motor velocity
    Up    / W : increase command magnitude
    Down  / S : decrease command magnitude
    Q         : step move -90 deg (relative; unlimited unless ROTOR_LIMIT_ENABLED)
    E         : step move +90 deg (relative; unlimited unless ROTOR_LIMIT_ENABLED)
    Space     : request hard stop and latch manual E-stop
    R         : clear the manual E-stop latch (firmware safety still applies)
    Esc       : stop the experiment, save data, and generate plots

Safety: start with a small command speed and keep hands clear of the mechanism.

Example:
    python manual_keyboard_balance.py
"""

import argparse
import math
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from pynput import keyboard

from control_comms import ControlComms, DebugLevel, StatusCode, find_board_port

# Connection and experiment settings
BAUD_RATE = 500000
TIMEOUT = 0.25

# Firmware command IDs
CMD_SET_HOME = 0
CMD_MOVE_TO = 1
CMD_MOVE_BY = 2
CMD_SET_STEP_MODE = 3
CMD_SET_VELOCITY = 4
CMD_HARD_STOP = 5
CMD_QUERY = 6
CMD_RESET_SAFETY = 7

STEP_MODE_16 = 4

# Firmware status codes, from host_status() in PendulumController.ino
STATUS_OK = 0
STATUS_MOVING = 1
STATUS_LIMIT = 2
STATUS_DRIVER_FAULT = 3

# Must match PendulumController.ino. set_signed_velocity() turns any command
# below MOTOR_MIN_SPEED_PPS into a hard stop and clips anything above
# MOTOR_MAX_SPEED_PPS, so the host keeps the speed setting inside that band --
# otherwise pressing Down twice would silently leave the motor dead.
MOTOR_MIN_SPEED_PPS = 200.0
MOTOR_MAX_SPEED_PPS = 1000.0

# Manual-control settings
INITIAL_SPEED_PPS = 250.0
SPEED_STEP_PPS = 50.0
CONTROL_PERIOD_S = 0.02     # target logging/query period (~50 Hz)
CONTROL_SIGN = 1.0          # use -1 if left/right direction is reversed

# Step-move settings (Q / E). CMD_MOVE_BY takes a signed angle in degrees.
MOVE_STEP_DEG = 90.0
MOVE_TIMEOUT_S = 8.0        # give up waiting for a step move to report completion
MIN_MOVE_DEG = 0.5          # ignore step moves smaller than this

# Must match ROTOR_LIMIT_ENABLED / ROTOR_SOFT_LIMIT_DEG in PendulumController.ino.
# When the limit is enforced the host keeps a margin, so a step move never parks
# the rotor exactly on the firmware limit where rotor_limit_exceeded_for_command()
# would latch the E-stop on the next velocity command in that direction.
# With it disabled the rotor turns without limit and step moves are not clipped.
ROTOR_LIMIT_ENABLED = False
ROTOR_SOFT_LIMIT_DEG = 90.0
ROTOR_MOVE_MARGIN_DEG = 2.0

DATA_DIR = Path("data")


def wrapped_error_deg(angle_deg: float, setpoint_deg: float = 180.0) -> float:
    return (setpoint_deg - angle_deg + 180.0) % 360.0 - 180.0


def clamp_speed_pps(speed_pps: float) -> float:
    """Keep a command magnitude inside the band the firmware will actually run."""
    return max(MOTOR_MIN_SPEED_PPS, min(MOTOR_MAX_SPEED_PPS, speed_pps))


def clamp_move_deg(delta_deg: float, rotor_deg: Optional[float]) -> float:
    """Clip a relative move so the rotor stops short of the firmware soft limit.

    Returns the achievable delta in degrees, or 0.0 when there is no room left.
    With ROTOR_LIMIT_ENABLED false the move passes through untouched, so Q/E keep
    turning the rotor in the same direction indefinitely.
    """
    if not ROTOR_LIMIT_ENABLED:
        return delta_deg if abs(delta_deg) >= MIN_MOVE_DEG else 0.0
    if rotor_deg is None or math.isnan(rotor_deg):
        rotor_deg = 0.0
    limit = ROTOR_SOFT_LIMIT_DEG - ROTOR_MOVE_MARGIN_DEG
    target = max(-limit, min(limit, rotor_deg + delta_deg))
    delta = target - rotor_deg
    return delta if abs(delta) >= MIN_MOVE_DEG else 0.0


# Windows virtual-key codes for the letter bindings. Matching on key.char alone
# breaks under a non-English IME: with Korean input active the A key reports
# 'ㅁ', not 'a'. The vk comes from the physical key, so it is layout- and
# IME-independent.
LETTER_VK = {"a": 0x41, "d": 0x44, "e": 0x45, "q": 0x51, "r": 0x52, "s": 0x53, "w": 0x57}


def is_key(key, name: str) -> bool:
    """True when `key` is the physical key for `name`, whatever the IME reports."""
    char = getattr(key, "char", None)
    if char and char.lower() == name:
        return True
    return getattr(key, "vk", None) == LETTER_VK[name]


class ManualControlState:
    """Keyboard state shared between the listener thread and the control loop."""

    def __init__(self, initial_speed_pps: float = INITIAL_SPEED_PPS,
                 move_step_deg: float = MOVE_STEP_DEG):
        self._lock = threading.Lock()
        self.left = False
        self.right = False
        self.speed_pps = clamp_speed_pps(initial_speed_pps)
        self.move_step_deg = move_step_deg
        self.stop_requested = False
        self.emergency_stop = False
        self.pending_move_deg: Optional[float] = None

    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            return {
                "left": self.left,
                "right": self.right,
                "speed_pps": self.speed_pps,
                "stop_requested": self.stop_requested,
                "emergency_stop": self.emergency_stop,
            }

    def take_pending_move(self) -> Optional[float]:
        """Pop a requested step move, or None if none is pending."""
        with self._lock:
            delta, self.pending_move_deg = self.pending_move_deg, None
        return delta

    def _request_move(self, delta_deg: float) -> None:
        # Release the motion keys so the control loop does not fight the move.
        with self._lock:
            self.pending_move_deg = delta_deg
            self.left = False
            self.right = False
        print(f"step move {delta_deg:+.0f} deg requested")

    def _set_motion_key(self, name: str, pressed: bool) -> None:
        with self._lock:
            setattr(self, name, pressed)

    def on_press(self, key):
        if key == keyboard.Key.left or is_key(key, "a"):
            self._set_motion_key("left", True)
        elif key == keyboard.Key.right or is_key(key, "d"):
            self._set_motion_key("right", True)
        elif key == keyboard.Key.up or is_key(key, "w"):
            with self._lock:
                self.speed_pps = min(MOTOR_MAX_SPEED_PPS, self.speed_pps + SPEED_STEP_PPS)
                speed_pps = self.speed_pps
            note = "  (firmware maximum)" if speed_pps >= MOTOR_MAX_SPEED_PPS else ""
            print(f"speed = {speed_pps:.0f} pps{note}")
        elif key == keyboard.Key.down or is_key(key, "s"):
            with self._lock:
                self.speed_pps = max(MOTOR_MIN_SPEED_PPS, self.speed_pps - SPEED_STEP_PPS)
                speed_pps = self.speed_pps
            note = "  (firmware minimum; below this the motor stops)" if speed_pps <= MOTOR_MIN_SPEED_PPS else ""
            print(f"speed = {speed_pps:.0f} pps{note}")
        elif is_key(key, "e"):
            self._request_move(self.move_step_deg)
        elif is_key(key, "q"):
            self._request_move(-self.move_step_deg)
        elif key == keyboard.Key.space:
            # Do not use the serial port from this keyboard thread.
            # The main loop sees this latch and issues CMD_HARD_STOP on the next cycle.
            with self._lock:
                self.emergency_stop = True
                self.left = False
                self.right = False
            print("HARD STOP requested")
        elif is_key(key, "r"):
            with self._lock:
                self.emergency_stop = False
                self.left = False
                self.right = False
            print("Manual E-stop latch cleared")
        elif key == keyboard.Key.esc:
            with self._lock:
                self.stop_requested = True
            return False

    def on_release(self, key):
        if key == keyboard.Key.left or is_key(key, "a"):
            self._set_motion_key("left", False)
        elif key == keyboard.Key.right or is_key(key, "d"):
            self._set_motion_key("right", False)


def connect(port: str, baud_rate: int) -> ControlComms:
    """Open the serial link and put the firmware in a known state."""
    ctrl = ControlComms(timeout=TIMEOUT, debug_level=DebugLevel.DEBUG_ERROR)
    if ctrl.connect(port, baud_rate) is not StatusCode.OK:
        raise RuntimeError(f"Could not connect to {port}")

    ctrl.step(CMD_HARD_STOP, [0.0])
    ctrl.step(CMD_SET_STEP_MODE, [STEP_MODE_16])
    ctrl.step(CMD_SET_HOME, [0.0])
    ctrl.step(CMD_RESET_SAFETY, [0.0])

    print("Connected.")
    print("Use Left/Right or A/D to move, Up/Down or W/S to change speed.")
    print("Q / E = step move -90 / +90 deg.")
    print("Space = hard stop latch, R = clear manual latch, Esc = finish/save.")
    return ctrl


def run_experiment(
    ctrl: ControlComms,
    state: ManualControlState,
    control_period_s: float = CONTROL_PERIOD_S,
    control_sign: float = CONTROL_SIGN,
) -> List[Dict[str, object]]:
    """Drive the arm from the keyboard until Esc, returning one record per sample."""
    records: List[Dict[str, object]] = []
    session_start_perf = time.perf_counter()
    prev_host_t: Optional[float] = None
    prev_pendulum: Optional[float] = None
    prev_rotor: Optional[float] = None
    last_sent_velocity: Optional[float] = None
    move_active = False
    move_deadline = 0.0

    listener = keyboard.Listener(on_press=state.on_press, on_release=state.on_release)
    listener.start()

    print("Manual balancing started. Press Esc to finish.")

    try:
        while True:
            loop_start = time.perf_counter()

            keys = state.snapshot()
            left = keys["left"]
            right = keys["right"]
            speed_pps = keys["speed_pps"]
            estop = keys["emergency_stop"]

            if keys["stop_requested"]:
                break

            pending_move = state.take_pending_move()
            if estop:
                # An E-stop cancels any queued or running step move. Clearing
                # last_sent_velocity forces the CMD_HARD_STOP below, which is
                # what actually halts a move that is still running.
                pending_move = None
                if move_active:
                    move_active = False
                    last_sent_velocity = None
            elif pending_move is not None and move_active:
                print("step move ignored: a move is already running")
                pending_move = None

            if estop:
                requested_velocity = 0.0
            elif left and not right:
                requested_velocity = -control_sign * speed_pps
            elif right and not left:
                requested_velocity = control_sign * speed_pps
            else:
                requested_velocity = 0.0

            move_command_deg = 0.0

            if pending_move is not None:
                # move_stepper_by() is ignored while the driver is still running,
                # so stop any velocity command before starting the step move.
                if last_sent_velocity not in (None, 0.0):
                    ctrl.step(CMD_HARD_STOP, [0.0])
                last_sent_velocity = 0.0
                requested_velocity = 0.0

                move_command_deg = clamp_move_deg(control_sign * pending_move, prev_rotor)
                if move_command_deg == 0.0:
                    rotor_now = 0.0 if prev_rotor is None else prev_rotor
                    print(f"step move skipped: rotor at {rotor_now:.1f} deg, no room before the limit")
                    command_sent = CMD_QUERY
                    resp = ctrl.step(CMD_QUERY, [0.0])
                else:
                    command_sent = CMD_MOVE_BY
                    resp = ctrl.step(CMD_MOVE_BY, [move_command_deg])
                    move_deadline = time.perf_counter() + MOVE_TIMEOUT_S
                    print(f"step move {move_command_deg:+.1f} deg")
            elif move_active:
                # Keep sampling at the normal rate while the driver finishes.
                requested_velocity = 0.0
                command_sent = CMD_QUERY
                resp = ctrl.step(CMD_QUERY, [0.0])
            # Send a new velocity only when the requested input changes.
            # Otherwise query the board so observations continue to be sampled.
            elif last_sent_velocity is None or requested_velocity != last_sent_velocity:
                command_sent = CMD_SET_VELOCITY if requested_velocity != 0.0 else CMD_HARD_STOP
                action_sent = requested_velocity if requested_velocity != 0.0 else 0.0
                resp = ctrl.step(command_sent, [action_sent])
                last_sent_velocity = requested_velocity
            else:
                command_sent = CMD_QUERY
                resp = ctrl.step(CMD_QUERY, [0.0])

            host_elapsed = time.perf_counter() - session_start_perf
            host_unix = time.time()

            # Checked outside the reply handling below: if replies stop arriving
            # mid-move, move_active must still time out or it would block the
            # velocity keys for the rest of the run.
            if move_active and time.perf_counter() > move_deadline:
                move_active = False
                ctrl.step(CMD_HARD_STOP, [0.0])
                last_sent_velocity = 0.0
                print("step move timed out; motor stopped")

            if resp is not None:
                status, mcu_timestamp_ms, terminated, obs = resp
                pendulum_deg = float(obs[0]) if len(obs) > 0 else math.nan
                rotor_deg = float(obs[1]) if len(obs) > 1 else math.nan
                motor_speed_pps = float(obs[2]) if len(obs) > 2 else math.nan
                l6474_status = int(obs[3]) if len(obs) > 3 else -1

                dt = math.nan if prev_host_t is None else host_elapsed - prev_host_t
                pendulum_velocity_dps = math.nan
                rotor_velocity_dps = math.nan
                if prev_host_t is not None and dt > 0:
                    dp = (pendulum_deg - prev_pendulum + 180.0) % 360.0 - 180.0
                    pendulum_velocity_dps = dp / dt
                    rotor_velocity_dps = (rotor_deg - prev_rotor) / dt

                records.append({
                    "host_unix_s": host_unix,
                    "host_elapsed_s": host_elapsed,
                    "mcu_timestamp_ms": int(mcu_timestamp_ms),
                    "mcu_elapsed_s": (int(mcu_timestamp_ms) - records[0]["mcu_timestamp_ms"]) / 1000.0 if records else 0.0,
                    "response_status": int(status),
                    "terminated": bool(terminated),
                    "pendulum_angle_deg": pendulum_deg,
                    "pendulum_error_deg": wrapped_error_deg(pendulum_deg),
                    "pendulum_velocity_dps_est": pendulum_velocity_dps,
                    "rotor_angle_deg": rotor_deg,
                    "rotor_velocity_dps_est": rotor_velocity_dps,
                    "motor_speed_pps_observed": motor_speed_pps,
                    "l6474_status_raw": l6474_status,
                    "command_id_sent": int(command_sent),
                    "command_velocity_pps": float(requested_velocity),
                    "move_command_deg": float(move_command_deg),
                    "manual_speed_setting_pps": float(speed_pps),
                    "key_left": bool(left),
                    "key_right": bool(right),
                    "emergency_stop": bool(estop),
                })

                prev_host_t = host_elapsed
                prev_pendulum = pendulum_deg
                prev_rotor = rotor_deg

                # The firmware answers a command after acting on it, so the reply
                # to CMD_MOVE_BY already reports STATUS_MOVING once the move starts.
                if move_active:
                    if status != STATUS_MOVING:
                        move_active = False
                        print(f"step move finished at rotor {rotor_deg:.1f} deg")
                elif command_sent == CMD_MOVE_BY:
                    move_active = status == STATUS_MOVING
                    if not move_active:
                        print("step move refused by the firmware (safety latched or driver busy)")

                if terminated:
                    print("Firmware reported termination/safety condition.")
                    break

            elapsed = time.perf_counter() - loop_start
            if elapsed < control_period_s:
                time.sleep(control_period_s - elapsed)

    finally:
        try:
            ctrl.step(CMD_HARD_STOP, [0.0])
        except Exception:
            pass
        try:
            listener.stop()
        except Exception:
            pass

    print(f"Stopped. Samples collected: {len(records)}")
    return records


def save_records(records: List[Dict[str, object]], data_dir: Path) -> Tuple[pd.DataFrame, Path, Path]:
    """Write all measured observations and control inputs to CSV."""
    df = pd.DataFrame(records)

    data_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = data_dir / f"manual_balance_{stamp}.csv"
    png_path = data_dir / f"manual_balance_{stamp}.png"

    df.to_csv(csv_path, index=False)
    print(f"CSV saved: {csv_path.resolve()}")

    print(df.head())
    print(df.tail())
    print(df.describe(include="all"))
    return df, csv_path, png_path


def plot_experiment(df: pd.DataFrame, png_path: Path, show: bool = True) -> None:
    """Plot the full experiment and save it next to the CSV."""
    if len(df) == 0:
        raise RuntimeError("No samples were recorded.")

    t = df["host_elapsed_s"]

    fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)

    axes[0].plot(t, df["pendulum_angle_deg"], label="pendulum angle")
    axes[0].axhline(180.0, linestyle="--", label="upright setpoint")
    axes[0].set_ylabel("Pendulum (deg)")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(t, df["pendulum_error_deg"], label="pendulum error")
    axes[1].plot(t, df["pendulum_velocity_dps_est"], label="pendulum velocity est.")
    axes[1].set_ylabel("deg / deg/s")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(t, df["rotor_angle_deg"], label="rotor angle")
    axes[2].plot(t, df["rotor_velocity_dps_est"], label="rotor velocity est.")
    axes[2].set_ylabel("deg / deg/s")
    axes[2].legend()
    axes[2].grid(True)

    axes[3].plot(t, df["command_velocity_pps"], label="command velocity")
    axes[3].plot(t, df["motor_speed_pps_observed"], label="observed motor speed")
    axes[3].set_ylabel("pps")
    axes[3].legend()
    axes[3].grid(True)

    axes[4].plot(t, df["l6474_status_raw"], label="L6474 status raw")
    axes[4].plot(t, df["response_status"], label="firmware response status")
    axes[4].set_ylabel("status")
    axes[4].set_xlabel("Host elapsed time (s)")
    axes[4].legend()
    axes[4].grid(True)

    fig.suptitle("Manual inverted-pendulum balancing experiment")
    fig.tight_layout()
    fig.savefig(png_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    print(f"Plot saved: {png_path.resolve()}")


def print_control_changes(df: pd.DataFrame) -> None:
    """Inspect key/control transitions only."""
    changed = (
        df["command_velocity_pps"].ne(df["command_velocity_pps"].shift())
        | df["move_command_deg"].ne(0.0)
    )
    control_changes = df.loc[changed]
    print(control_changes[[
        "host_elapsed_s",
        "command_velocity_pps",
        "move_command_deg",
        "manual_speed_setting_pps",
        "key_left",
        "key_right",
        "pendulum_angle_deg",
        "rotor_angle_deg",
        "motor_speed_pps_observed",
        "l6474_status_raw",
    ]])


def speed_arg(value: str) -> float:
    """argparse type: a command magnitude the firmware will actually act on."""
    speed = float(value)
    if not MOTOR_MIN_SPEED_PPS <= speed <= MOTOR_MAX_SPEED_PPS:
        raise argparse.ArgumentTypeError(
            f"must be between {MOTOR_MIN_SPEED_PPS:.0f} and {MOTOR_MAX_SPEED_PPS:.0f} pps; "
            f"the firmware hard-stops below {MOTOR_MIN_SPEED_PPS:.0f}"
        )
    return speed


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", default=None,
                        help="serial port (default: auto-detect the ST-Link port)")
    parser.add_argument("--baud", type=int, default=BAUD_RATE, help=f"baud rate (default: {BAUD_RATE})")
    parser.add_argument("--speed", type=speed_arg, default=INITIAL_SPEED_PPS,
                        help=f"initial command magnitude in pps, "
                             f"{MOTOR_MIN_SPEED_PPS:.0f}-{MOTOR_MAX_SPEED_PPS:.0f} "
                             f"(default: {INITIAL_SPEED_PPS:.0f})")
    parser.add_argument("--period", type=float, default=CONTROL_PERIOD_S,
                        help=f"target logging/query period in seconds (default: {CONTROL_PERIOD_S})")
    parser.add_argument("--move-step", type=float, default=MOVE_STEP_DEG,
                        help=f"Q/E step-move magnitude in degrees (default: {MOVE_STEP_DEG:.0f})")
    parser.add_argument("--invert", action="store_true", help="reverse the left/right direction")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help=f"directory for CSV and PNG output (default: {DATA_DIR})")
    parser.add_argument("--no-show", action="store_true", help="save the plot without opening a window")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    ctrl = connect(find_board_port(args.port), args.baud)
    try:
        state = ManualControlState(initial_speed_pps=args.speed, move_step_deg=args.move_step)
        records = run_experiment(
            ctrl,
            state,
            control_period_s=args.period,
            control_sign=-CONTROL_SIGN if args.invert else CONTROL_SIGN,
        )
    finally:
        ctrl.close()

    if not records:
        print("No samples were recorded; nothing to save.")
        return 1

    df, _, png_path = save_records(records, args.data_dir)
    plot_experiment(df, png_path, show=not args.no_show)
    print_control_changes(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
