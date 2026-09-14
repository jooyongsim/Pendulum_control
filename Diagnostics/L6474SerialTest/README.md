# L6474 Serial Motor Diagnostic

This folder contains a minimal motor test for the **NUCLEO-F401RE + X-NUCLEO-IHM01A1 + L6474** setup used in this project.

The goal is to test the stepper motor motion path without the pendulum controller, encoder ISR, PID loop, soft limits, or the normal JSON command protocol. A Python script sends simple rotation commands over serial, and the Arduino firmware waits until each move is fully complete before returning `DONE`.

This is useful for diagnosing symptoms such as:

- the motor only twitching instead of completing a move,
- moving less than the requested angle,
- occasionally moving slightly in the wrong direction,
- inconsistent repeated `MOVE_BY` behavior.

## Files

- `L6474SerialTest.ino` — firmware for the Nucleo/L6474 board.
- `motor_serial_test.py` — Python test runner that sends rotation commands.

## Test configuration

The firmware currently uses:

| Parameter | Value |
|---|---:|
| Motor full steps / revolution | 200 |
| Microstep mode | 1/16 |
| Effective pulses / revolution | 3200 |
| TVAL motor current | 800 mA |
| Maximum speed | 500 pps |
| Minimum speed | 100 pps |
| Acceleration | 1000 pps² |
| Deceleration | 1000 pps² |
| Serial baud rate | 115200 |

At 1/16 microstepping:

```text
3200 pulses = 360 deg
  800 pulses = 90 deg
  400 pulses = 45 deg
~  89 pulses = 10 deg
```

The 10-degree command therefore corresponds to about 89 microsteps.

## Arduino setup

Open:

```text
Diagnostics/L6474SerialTest/L6474SerialTest.ino
```

Compile and upload it to the NUCLEO-F401RE using the same STM32 Arduino environment and L6474 library used by the main project.

The firmware uses the same pin mapping as the pendulum controller:

```text
FLAG       D2
STBY/RESET D8
DIR        D7
STEP/PWM   D9
SPI CS     D10
SPI MOSI   D11
SPI MISO   D12
SPI SCK    D13
```

After startup, the board sets its current motor position as Home = 0 degrees.

## Serial commands

The firmware accepts one text command per line.

### Set current position as Home

```text
HOME
```

### Read current status

```text
STATUS
```

### Stop immediately

```text
STOP
```

### Absolute position command

```text
GOTO 45
GOTO 0
GOTO -45
```

`GOTO` is absolute relative to Home. For example, if the motor is currently at `+45 deg`, then `GOTO -45` requires a physical movement of 90 degrees.

### Relative movement command

```text
MOVE 10
MOVE -10
```

`MOVE` is relative to the current internal position.

The firmware does not send `DONE` until `stepper->wait_while_active()` confirms that the L6474 motion command has finished. This prevents a new Python command from being intentionally issued while the previous movement is still active.

## Python requirements

Install pyserial:

```bash
pip install pyserial
```

Find the board serial port, for example:

```text
Windows: COM6
Linux:   /dev/ttyACM0
```

## Usage examples

### Windows PowerShell — run from the repository root

For example, to test relative 90-degree moves on `COM10` for 10 repetitions:

```powershell
python .\Diagnostics\L6474SerialTest\motor_serial_test.py --port COM10 --angle 90 --repeats 10 --mode move
```

For a 10-degree relative-move test:

```powershell
python .\Diagnostics\L6474SerialTest\motor_serial_test.py --port COM10 --angle 10 --repeats 20 --mode move
```

For a 10-degree absolute-position test:

```powershell
python .\Diagnostics\L6474SerialTest\motor_serial_test.py --port COM10 --angle 10 --repeats 20 --mode goto
```

Alternatively, change into the test directory first:

```powershell
cd .\Diagnostics\L6474SerialTest
python .\motor_serial_test.py --port COM10 --angle 90 --repeats 10 --mode move
```

## Absolute-position test

Run from inside `Diagnostics/L6474SerialTest`:

```bash
python motor_serial_test.py --port COM6 --angle 45 --repeats 10 --mode goto
```

The commanded target sequence is:

```text
0 -> +45 -> 0 -> -45 -> 0
```

Therefore every physical move in the test should be 45 degrees.

For the 10-degree problem case:

```bash
python motor_serial_test.py --port COM6 --angle 10 --repeats 20 --mode goto
```

The commanded target sequence becomes:

```text
0 -> +10 -> 0 -> -10 -> 0
```

## Relative-movement test

To test the equivalent of `MOVE_BY` directly:

```bash
python motor_serial_test.py --port COM6 --angle 10 --repeats 20 --mode move
```

The relative command sequence is chosen so the internal position follows:

```text
0 -> +10 -> 0 -> -10 -> 0
```

This test uses `stepper->move(direction, steps)` rather than `go_to()`.

## Expected console output

A normal 10-degree absolute move should look approximately like:

```text
PC : GOTO 10.000
MCU: ACK GOTO target_deg=10.000 target_steps=89
MCU: DONE target_deg=10.000 pos_steps=89 pos_deg=10.013 l6474=0x....
```

Because 10 degrees cannot be represented exactly with 3200 pulses/revolution, the expected internal angle is about 10.013 degrees.

## How to interpret the result

### Both `goto` and `move` tests work reliably

If repeated 10-degree moves are smooth and repeatable in this diagnostic firmware, but the main `PendulumController` still fails, the problem is more likely in the main controller logic, command timing, state handling, safety logic, or interaction with other firmware components.

### Internal position changes correctly, but the physical shaft does not

Example:

```text
DONE target_deg=10.000 pos_steps=89 pos_deg=10.013
```

while the motor only twitches or moves much less than 10 degrees.

This strongly suggests that the step pulses were generated but the physical motor did not follow them. Possible causes include:

- lost steps,
- incorrect motor phase pairing,
- loose motor connection,
- insufficient phase current,
- motor supply problem,
- driver/power-stage problem,
- excessive mechanical load.

The L6474 internal position is based on commanded/generated steps; it is not an independent shaft encoder, so it cannot directly detect missed physical steps.

### `goto` works but `move` fails

This points toward a difference in how relative `move()` commands are being handled compared with absolute `go_to()` commands.

### Both tests fail intermittently

That makes the higher-level pendulum control logic less likely to be the primary cause. Investigate the motor wiring, L6474 current/speed profile, power supply, microstep mode, and mechanical load.

## Recommended test order

1. Remove or unload the pendulum arm if practical.
2. Flash `L6474SerialTest.ino`.
3. Start with `--angle 10 --mode goto`.
4. Run at least 20 repetitions.
5. Repeat with `--mode move`.
6. Compare the actual shaft motion with the reported `pos_steps` / `pos_deg` values.
7. If failures remain, repeat the experiment with full-step mode to separate microstep behavior from general motor/driver behavior.

## Safety

The motor can move unexpectedly during diagnostics. Keep hands, wires, and loose objects clear of the rotor and linkage. Be ready to remove motor power if the mechanism behaves unexpectedly.
