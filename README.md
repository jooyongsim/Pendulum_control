# Pendulum_control

Improved controller and experiment notebooks for the STMicroelectronics STEVAL-EDUKIT01 rotary inverted pendulum.

This repository is based on the serial-control and simple PID examples in `jooyongsim/inverted_pendulum`, the original EDUKIT/UCLA teaching project, STMicroelectronics reference firmware and motor-driver libraries, and Shawn Hymel's DigiKey inverted-pendulum examples.

## What changed

- L6474 torque regulation current is set to **800 mA** for the STEVAL-EDUKIT01-oriented configuration. Verify your motor rating before use.
- L6474 over-current threshold is set to `L6474_OCD_TH_2250mA`.
- Acceleration/deceleration: **6000 pps^2**.
- Maximum commanded speed: **2000 pps**.
- Minimum run speed: **30 pps** so feedback controllers can request small velocities.
- Added signed velocity command (`CMD_SET_VELOCITY`) using `run()` and `set_max_speed()` rather than repeatedly issuing overlapping position moves.
- Direction reversal performs a hard stop before restarting in the opposite direction.
- Added rotor travel safety limit and explicit hard-stop/query/reset commands.
- L6474 FLAG ISR no longer prints text into the serial JSON stream.
- PID notebook uses seconds for integration/differentiation, angle wrapping, derivative filtering, output limiting, anti-windup and safety guards.
- Added a manual keyboard-control notebook that logs observations and control inputs for later system identification and controller design.

## Files

- `PendulumController/PendulumController.ino` - STM32/Arduino firmware.
- `PendulumController/control-comms.hpp` - newline-delimited JSON serial interface for the firmware.
- `control_comms.py` - Python serial interface.
- `pendulum_pid_velocity.ipynb` - improved PID experiment notebook.
- `manual_keyboard_balance.ipynb` - manual keyboard balancing, complete data logger and plotting notebook.
- `requirements.txt` - Python dependencies.

## Arduino dependencies

Install libraries compatible with your STM32 Arduino core:

- `STM32duino X-NUCLEO-IHM01A1` (L6474)
- `RotaryEncoder`
- `ArduinoJson` v6

The pin assignment follows the STEVAL-EDUKIT01 / NUCLEO-F401RE setup used by the source projects.

## Python dependencies

```bash
python -m pip install -r requirements.txt
```

The manual keyboard notebook uses `pynput`, and experiment data processing uses `pandas` and `matplotlib`.

## Manual keyboard balancing and data collection

Open `manual_keyboard_balance.ipynb` after flashing the firmware.

Keyboard controls:

| Key | Action |
|---|---|
| Left arrow / `A` | negative motor velocity |
| Right arrow / `D` | positive motor velocity |
| Up arrow / `W` | increase command magnitude |
| Down arrow / `S` | decrease command magnitude |
| Space | immediate hard stop |
| Esc | finish the experiment, stop motor, save data |

The initial manual command is intentionally conservative (`250 pps`). Change `CONTROL_SIGN` in the notebook if the physical left/right direction is opposite to what you expect.

During the experiment the notebook continuously records the complete observation returned by the firmware plus the control input and host-side timing/key state. Logged columns include:

- host Unix time and host elapsed time
- MCU timestamp and MCU elapsed time
- firmware response status and termination flag
- pendulum angle
- pendulum error relative to upright (`180 deg`)
- estimated pendulum angular velocity
- rotor angle relative to the software home position
- estimated rotor angular velocity
- observed signed stepper speed in pps
- raw L6474 status register value
- command ID sent
- commanded signed velocity in pps
- current manual speed setting
- left/right key states
- emergency-stop state

At the end of an experiment the notebook writes:

```text
data/manual_balance_YYYYMMDD_HHMMSS.csv
data/manual_balance_YYYYMMDD_HHMMSS.png
```

The generated figure plots pendulum angle/error/velocity, rotor angle/velocity, commanded and observed motor speed, and driver/firmware status over time. A final table also shows only the time points where the manual control input changed.

This dataset is intended to make it easier to compare human balancing behavior with PID/LQR/RL control, inspect latency and motor response, and estimate useful system dynamics from real hardware measurements.

## PID workflow

1. Flash `PendulumController/PendulumController.ino` to the Nucleo board.
2. Open `pendulum_pid_velocity.ipynb` in Jupyter.
3. Set `SERIAL_PORT` for your computer.
4. Run the connection and status cells.
5. Run the low-speed direction test first. If positive command drives the rotor in the wrong control direction, change `CONTROL_SIGN`.
6. Manually place the pendulum close to upright (`180 deg`) before starting the PID cell.
7. Start with conservative gains and tune on the real mechanism.

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

## References and upstream projects

### This project and source-code lineage

- This repository: https://github.com/jooyongsim/Pendulum_control
- Previous working repository used as the immediate source for the serial/PID experiments: https://github.com/jooyongsim/inverted_pendulum
- Shawn Hymel's original Python/Arduino PID demo for the STEVAL-EDUKIT01: https://github.com/ShawnHymel/pendulum-pid
- Shawn Hymel's hardware reinforcement-learning follow-up: https://github.com/ShawnHymel/pendulum-rl
- STM32duino X-NUCLEO-IHM01A1 Arduino library containing the L6474 driver API: https://github.com/stm32duino/X-NUCLEO-IHM01A1
- L6474 Arduino class header used to understand `run()`, `set_max_speed()`, `hard_stop()` and register access: https://github.com/stm32duino/X-NUCLEO-IHM01A1/blob/main/src/L6474.h

### UCLA / original EDUKIT educational project

The ST motor-control curriculum credits **Prof. William J. Kaiser, UCLA** for the control-systems educational material associated with this rotary inverted pendulum platform.

- UCLA/EDUKIT project source by William J. Kaiser: https://github.com/wjkaiser/Edukit_Rotary_Inverted_Pendulum_Project
- Example one-motor motion-control source used for comparison: https://github.com/wjkaiser/Edukit_Rotary_Inverted_Pendulum_Project/blob/main/Edukit_Rotary_Inverted_Pendulum_Project/Projects/Multi/Examples/MotionControl/IHM01A1_ExampleFor1Motor/Src/main.c
- ST Motor Control and Control Systems educational curriculum / instructor material: https://www.st.com/content/st_com/en/campaigns/educationalplatforms/motorcontrol-edu.html

### DigiKey tutorials

- Shawn Hymel / DigiKey video: **How to Tune a PID Controller for an Inverted Pendulum**: https://www.digikey.com/en/videos/d/digi-key-electronics/how-to-tune-a-pid-controller-for-an-inverted-pendulum-digikey
- DigiKey Maker article: **How to Tune a PID Controller**: https://www.digikey.com/en/maker/projects/how-to-tune-a-pid-controller/9ee9a111aef049af9f84f785779989ec

### STMicroelectronics hardware and firmware

- STEVAL-EDUKIT01 product page: https://www.st.com/en/evaluation-tools/steval-edukit01.html
- STEVAL-EDUKIT01 data brief: https://www.st.com/resource/en/data_brief/steval-edukit01.pdf
- STSW-EDUKIT01 official firmware: https://www.st.com/en/embedded-software/stsw-edukit01.html
- ST Motor Control and Control Systems educational portal: https://www.st.com/content/st_com/en/campaigns/educationalplatforms/motorcontrol-edu.html

The STEVAL-EDUKIT01 uses a NUCLEO-F401RE board, an X-NUCLEO-IHM01A1 expansion board with the L6474PD microstepping driver, a stepper motor, and a quadrature rotary encoder.

## Safety

An inverted pendulum can move quickly. Keep hands, cables, and loose objects away from the mechanism and be ready to remove motor power. The supplied PID gains are starting values rather than guaranteed stable settings. Confirm the actual motor current rating before using the 800 mA TVAL configuration, and begin manual tests at low speed.
