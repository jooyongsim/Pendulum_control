"""Diagnose 'the arrow / A / D keys do nothing' in the manual balancing run.

The two halves are checked separately, because they fail for different reasons:

  1. keyboard : does pynput see the key at all, and what character does it report?
                A Korean IME turns 'a' into 'ㅁ', which silently breaks the A/D/W/S
                bindings while the arrow keys keep working.
  2. link     : what does the board actually report? Once the firmware latches its
                safety flag, every velocity command is dropped until CMD_SET_HOME
                or CMD_RESET_SAFETY clears it.

Examples:
    python diagnose_manual.py --keys            # keyboard only, no serial port needed
    python diagnose_manual.py                   # board status only
    python diagnose_manual.py --jog             # also spin the motor briefly
"""

import argparse
import time

from control_comms import ControlComms, DebugLevel, StatusCode, find_board_port

CMD_SET_VELOCITY = 4
CMD_HARD_STOP = 5
CMD_QUERY = 6
CMD_RESET_SAFETY = 7

STATUS_NAMES = {0: "OK", 1: "MOVING", 2: "LIMIT", 3: "DRIVER_FAULT"}

JOG_PPS = 300.0     # above the firmware's MOTOR_MIN_SPEED_PPS of 200
JOG_S = 0.6


def check_keys(seconds: float) -> None:
    """Print every key event pynput delivers, with the character it reports."""
    from pynput import keyboard

    seen = []
    print(f"\n=== keyboard: press Left/Right/A/D for {seconds:.0f} s (Esc ends early) ===")

    def on_press(key):
        try:
            ch = key.char
        except AttributeError:
            ch = None
        bound = (key == keyboard.Key.left or key == keyboard.Key.right
                 or (ch or "").lower() in ("a", "d", "w", "s", "q", "e", "r"))
        seen.append(bound)
        print(f"  key={key!s:<22} char={ch!r:<8} -> {'bound' if bound else 'NOT bound'}")
        if key == keyboard.Key.esc:
            return False

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join(timeout=seconds)
        listener.stop()

    if not seen:
        print("  no key events at all -> pynput is not receiving input.")
        print("  On Windows this usually means the listener needs the same or higher")
        print("  privileges as the focused window; try running the terminal as admin.")
    elif not any(seen):
        print("  keys arrive, but none match the bindings.")
        print("  If 'char' shows Korean letters, switch the IME back to English (한/영).")
    else:
        print("  at least one bound key was recognised: the keyboard half is fine.")


EXPECTED_NUM_OBS = 4    # NUM_OBS in PendulumController.ino


def describe(resp) -> None:
    if resp is None:
        print("  no reply (serial timeout) -- the firmware did not answer.")
        return
    status, timestamp, terminated, obs = resp
    name = STATUS_NAMES.get(status, str(status))
    print(f"  status={status} ({name})  latched={terminated}  t={timestamp} ms")

    labels = ("pendulum deg", "rotor deg", "speed pps", "L6474 status")
    print("  observation:", ", ".join(
        f"{labels[i] if i < len(labels) else 'obs%d' % i}={v:.2f}" for i, v in enumerate(obs)))

    if len(obs) != EXPECTED_NUM_OBS:
        print(f"\n  >> The board sent {len(obs)} observations, but this project's firmware")
        print(f"     sends {EXPECTED_NUM_OBS}. The board is running DIFFERENT firmware than")
        print("     PendulumController.ino, so commands this sketch added -- CMD_SET_VELOCITY")
        print("     above all -- fall through to 'default:' and are silently ignored.")
        print("     That is why the arrow / A / D keys move nothing while the board still")
        print("     answers every query. Upload PendulumController.ino to fix it.")


def check_link(port: str, baud: int, jog: bool) -> None:
    ctrl = ControlComms(timeout=0.5, debug_level=DebugLevel.DEBUG_ERROR)
    print(f"\n=== link: {port} @ {baud} ===")
    print("  ports seen:", [p[0] for p in ctrl.get_serial_list()] or "none")

    if ctrl.connect(port, baud) is not StatusCode.OK:
        print(f"  could not open {port}.")
        return

    try:
        resp = ctrl.step(CMD_QUERY, [0.0])
        describe(resp)

        if resp is None:
            print("\n  The board is not replying. Re-flash PendulumController.ino and")
            print("  confirm the port; the firmware only answers commands it could parse.")
            return

        _, _, terminated, obs = resp
        rotor = float(obs[1])
        if terminated:
            print("\n  >> Safety is LATCHED. Every velocity command is dropped until this")
            print("     clears, which is exactly 'the keys do nothing'.")
            if abs(rotor) >= 90.0:
                print(f"     The rotor is at {rotor:.1f} deg, outside the +/-90 deg soft limit.")
                print("     Move the arm back toward centre by hand first.")
            print("     Clear it with: ctrl.step(CMD_RESET_SAFETY, [0.0])")
            resp = ctrl.step(CMD_RESET_SAFETY, [0.0])
            print("     after CMD_RESET_SAFETY:")
            describe(resp)

        if jog:
            print(f"\n  jogging {JOG_PPS:.0f} pps for {JOG_S:.1f} s -- keep clear")
            before = rotor
            ctrl.step(CMD_SET_VELOCITY, [JOG_PPS])
            time.sleep(JOG_S)
            resp = ctrl.step(CMD_HARD_STOP, [0.0])
            describe(resp)
            after = float(resp[3][1]) if resp else before
            moved = after - before
            print(f"  rotor {before:.1f} -> {after:.1f} deg (moved {moved:+.1f})")
            if abs(moved) < 0.5:
                print("  >> The motor did not move on a direct command, so the problem is")
                print("     the board or the driver, not the keyboard handling.")
            else:
                print("  >> The motor responds to a direct command; the link is fine.")
    finally:
        try:
            ctrl.step(CMD_HARD_STOP, [0.0])
        except Exception:
            pass
        ctrl.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", help="serial port to test (default: auto-detect)")
    parser.add_argument("--baud", type=int, default=500000)
    parser.add_argument("--keys", action="store_true", help="run the keyboard check")
    parser.add_argument("--seconds", type=float, default=10.0, help="keyboard check duration")
    parser.add_argument("--jog", action="store_true", help="briefly spin the motor")
    args = parser.parse_args()

    if args.keys:
        check_keys(args.seconds)
    if args.port or not args.keys:
        check_link(find_board_port(args.port), args.baud, args.jog)


if __name__ == "__main__":
    main()
