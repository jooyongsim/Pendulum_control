import argparse
import time
import serial


def read_until_done(ser):
    """Read Arduino output until DONE or ERR appears."""
    while True:
        raw = ser.readline()
        if not raw:
            raise TimeoutError("No response from Arduino")

        line = raw.decode("utf-8", errors="replace").strip()
        if not line:
            continue

        print("  MCU:", line)

        if line.startswith("DONE"):
            return line
        if line.startswith("ERR") or line.startswith("FATAL"):
            raise RuntimeError(line)


def send_command(ser, command):
    print()
    print("PC :", command)
    ser.write((command + "\n").encode("utf-8"))
    ser.flush()
    return read_until_done(ser)


def run_absolute_test(ser, angle_deg, repeats, hold_sec):
    """Absolute target sequence: 0 -> +A -> 0 -> -A -> 0."""
    print()
    print("=" * 60)
    print("ABSOLUTE POSITION TEST")
    print(f"Angle   : {angle_deg} deg")
    print(f"Repeats : {repeats}")
    print("=" * 60)

    send_command(ser, "HOME")

    targets = [+angle_deg, 0.0, -angle_deg, 0.0]

    for rep in range(repeats):
        print()
        print("#" * 60)
        print(f"REPEAT {rep + 1}/{repeats}")
        print("#" * 60)

        for target in targets:
            send_command(ser, f"GOTO {target:.3f}")
            time.sleep(hold_sec)
            send_command(ser, "STATUS")


def run_relative_test(ser, angle_deg, repeats, hold_sec):
    """Relative move sequence: +A, -A, -A, +A -> positions 0,+A,0,-A,0."""
    print()
    print("=" * 60)
    print("RELATIVE MOVE TEST")
    print(f"Angle   : {angle_deg} deg")
    print(f"Repeats : {repeats}")
    print("=" * 60)

    send_command(ser, "HOME")

    moves = [+angle_deg, -angle_deg, -angle_deg, +angle_deg]

    for rep in range(repeats):
        print()
        print("#" * 60)
        print(f"REPEAT {rep + 1}/{repeats}")
        print("#" * 60)

        for delta in moves:
            send_command(ser, f"MOVE {delta:.3f}")
            time.sleep(hold_sec)
            send_command(ser, "STATUS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port",
        required=True,
        help="Serial port, e.g. COM6 or /dev/ttyACM0",
    )
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--angle", type=float, default=45.0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--hold", type=float, default=1.0)
    parser.add_argument("--mode", choices=["goto", "move"], default="goto")
    args = parser.parse_args()

    ser = serial.Serial(
        port=args.port,
        baudrate=args.baud,
        timeout=3.0,
        dsrdtr=False,
    )

    try:
        # Some boards reset when the serial port is opened.
        time.sleep(1.5)
        ser.reset_input_buffer()
        ser.reset_output_buffer()

        print(f"Connected to {args.port}")
        send_command(ser, "STATUS")

        if args.mode == "goto":
            run_absolute_test(ser, args.angle, args.repeats, args.hold)
        else:
            run_relative_test(ser, args.angle, args.repeats, args.hold)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        try:
            send_command(ser, "STOP")
        except Exception:
            pass
    finally:
        ser.close()


if __name__ == "__main__":
    main()
