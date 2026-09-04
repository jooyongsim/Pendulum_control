# Pendulum_control

Improved controller for the STMicroelectronics STEVAL-EDUKIT01 rotary inverted pendulum.

This repository is based on the serial-control and simple PID examples in `jooyongsim/inverted_pendulum`, with changes aimed at more reliable L6474 motor control and safer PID experiments.

## What changed

- L6474 torque regulation current is set to **800 mA** for the STEVAL-EDUKIT01 motor configuration. Verify your motor rating before use.
- L6474 over-current threshold is set to `L6474_OCD_TH_2250mA`, following the EDUKIT-oriented configuration.
- Acceleration/deceleration: **6000 pps^2**.
- Maximum commanded speed: **2000 pps**.
- Minimum run speed: **30 pps** so the PID can request small velocities.
- Added signed velocity command (`CMD_SET_VELOCITY`) using `run()` and `set_max_speed()` rather than repeatedly issuing overlapping position moves.
- Direction reversal performs a hard stop before restarting in the opposite direction.
- Added rotor travel safety limit and explicit hard-stop/query/reset commands.
- L6474 FLAG ISR no longer prints text into the serial JSON stream.
- The notebook converts MCU timestamps from milliseconds to seconds before PID integration/differentiation.
- Removed the old test override that replaced the calculated PID output with a constant move command.
- Added angle wrapping, derivative filtering, output limiting, anti-windup, pendulum guard angle, rotor soft limit, and `finally` hard-stop behavior.

## Files

- `PendulumController/PendulumController.ino` - STM32/Arduino firmware.
- `PendulumController/control-comms.hpp` - newline-delimited JSON serial interface for the firmware.
- `control_comms.py` - Python serial interface.
- `pendulum_pid_velocity.ipynb` - improved PID experiment notebook.

## Arduino dependencies

Install libraries compatible with your STM32 Arduino core:

- `STM32duino X-NUCLEO-IHM01A1` (L6474)
- `RotaryEncoder`
- `ArduinoJson` v6

The pin assignment follows the original STEVAL-EDUKIT01/Nucleo setup used in the source project.

## Python dependencies

```bash
python -m pip install pyserial matplotlib
```

## Typical workflow

1. Flash `PendulumController/PendulumController.ino` to the Nucleo board.
2. Open `pendulum_pid_velocity.ipynb` in Jupyter.
3. Set `SERIAL_PORT` for your computer.
4. Run the connection and status cells.
5. Run the low-speed direction test first. If positive command drives the rotor in the wrong control direction, change `CONTROL_SIGN` in the notebook.
6. Manually place the pendulum close to upright (180 degrees) before starting the PID cell.
7. Start with conservative gains and increase them gradually.

## Command protocol

Commands use one floating-point action value:

| Command | ID | Action |
|---|---:|---|
| Set home | 0 | ignored |
| Move to angle | 1 | rotor degrees relative to home |
| Move by angle | 2 | delta degrees |
| Set microstep mode | 3 | 0=full, 1=1/2, 2=1/4, 3=1/8, 4=1/16 |
| Set signed velocity | 4 | signed pps |
| Hard stop | 5 | ignored |
| Query state | 6 | ignored |
| Reset safety/fault latch | 7 | ignored |

Observation array:

```text
[pendulum_angle_deg, rotor_angle_deg, signed_motor_speed_pps, last_l6474_status]
```

`pendulum_angle_deg` is wrapped to `[0, 360)`. `rotor_angle_deg` is signed relative to the position where `CMD_SET_HOME` was issued.

## Safety

An inverted pendulum can move quickly. Keep hands and loose objects away from the mechanism and be ready to remove motor power. The supplied gains are starting values, not a guarantee of stable control. Confirm the motor current rating before using the 800 mA TVAL setting.

## Source references

The communication structure and initial PID example were adapted from `jooyongsim/inverted_pendulum`. The motor-control API is provided by STMicroelectronics' STM32duino X-NUCLEO-IHM01A1 L6474 library.
