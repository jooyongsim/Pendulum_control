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
- Added a stepper/rotor dynamic-response notebook that records position, velocity and acceleration and fits the second-order rotor model used by the UCLA/ST instructor manual.
- Added a detailed analysis of the original EDUKIT `main.c` control architecture.

## Files

- `PendulumController/PendulumController.ino` - STM32/Arduino firmware.
- `PendulumController/control-comms.hpp` - newline-delimited JSON serial interface for the firmware.
- `control_comms.py` - Python serial interface.
- `pendulum_pid_velocity.ipynb` - improved PID experiment notebook.
- `manual_keyboard_balance.ipynb` - manual keyboard balancing, complete data logger and plotting notebook.
- `stepper_dynamic_response.ipynb` - position-step experiment, velocity/acceleration estimation and second-order rotor-model identification.
- `docs/simple_pendulum_free_response.md` - fixed-pivot free-response derivation: linear/nonlinear and undamped/damped cases, including time-domain and Laplace solutions.
- `docs/linear_poles_and_pid_response.md` - exponential trial solution, natural frequency/damping ratio, pole stability, and PID closed-loop modal response.
- `simple_pendulum_simulation.ipynb` - analytical-versus-numerical simulation of the fixed-pivot pendulum models.
- `encoder_release_experiment.ipynb` - non-live encoder acquisition, timestamped CSV logging, plotting, and basic period/damping estimation for a hand-release test.
- `encoder_live_monitor.py` - live encoder plot and CSV logger for the same hand-release test.
- `docs/edukit_main_c_analysis.md` - detailed explanation of PID/LQR layers, `ACCEL_CONTROL`, `GoTo()`, and the L6474 motion layer in the original EDUKIT firmware.
- `requirements.txt` - Python dependencies.

## Arduino dependencies

Install libraries compatible with your STM32 Arduino core:

- `STM32duino X-NUCLEO-IHM01A1` (L6474)
- `RotaryEncoder`
- `ArduinoJson` v6

The pin assignment follows the STEVAL-EDUKIT01 / NUCLEO-F401RE setup used by the source projects.

### STM32 board does not appear in Arduino IDE Boards Manager

If the STM32 board package does not appear in **Boards Manager**, the most common cause is that the STM32 Boards Manager URL has not been added, or an old URL is still being used.

For Arduino IDE 2.x:

1. Open **File -> Preferences**.
2. Add the following URL to **Additional Boards Manager URLs**:

```text
https://github.com/stm32duino/BoardManagerFiles/raw/main/package_stmicroelectronics_index.json
```

3. Click **OK** and restart Arduino IDE.
4. Open **Tools -> Board -> Boards Manager**.
5. Search for:

```text
STM32
```

6. Install:

```text
STM32 MCU based boards
by STMicroelectronics
```

If it still does not appear, check the following:

- Make sure there are no spaces before or after the Boards Manager URL.
- Remove or replace obsolete STM32 package-index URLs if you previously configured one.
- Check whether a company, school, proxy, VPN, firewall, or security product is blocking access to `github.com` or `raw.githubusercontent.com`.
- Clear the Boards Manager search box completely and search for `STM32` again.
- Restart Arduino IDE after changing **Additional Boards Manager URLs**.
- Prefer a current Arduino IDE 2.x release.

This project uses the **NUCLEO-F401RE** included with the STEVAL-EDUKIT01, so after installing the STM32 core select the appropriate Nucleo F401RE board configuration before compiling and uploading `PendulumController/PendulumController.ino`.

## Python dependencies

```bash
python -m pip install -r requirements.txt
```

The notebooks use `pyserial`, `pandas`, `matplotlib`, `scipy`, and `pynput` where keyboard input is required.

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

During the experiment the notebook continuously records the complete observation returned by the firmware plus the control input and host-side timing/key state. Logged columns include host and MCU timing, pendulum angle/error/velocity, rotor angle/velocity, observed stepper speed, raw L6474 status, firmware status, command velocity and keyboard state.

At the end of an experiment the notebook writes:

```text
data/manual_balance_YYYYMMDD_HHMMSS.csv
data/manual_balance_YYYYMMDD_HHMMSS.png
```

## Stepper / rotor dynamic-response identification

Open `stepper_dynamic_response.ipynb` with the pendulum removed or safely hanging down and the rotor centered.

The notebook applies repeated small rotor position steps and records:

- commanded rotor target angle `phi_RC`
- reported rotor angle / internal step position `phi`
- L6474 reported step rate in pps
- pendulum encoder angle
- estimated rotor angular velocity
- estimated rotor angular acceleration
- firmware and L6474 status values

After acquisition it fits the UCLA/ST manual's small-signal rotor model:

```text
                     a
G_rotor(s) = ------------------
              s^2 + b s + c
```

and also reports the equivalent natural frequency, damping ratio, DC gain, RMSE and correlation. A Bode plot of the identified model is generated.

Important measurement limitation: the current Arduino firmware obtains rotor position from the L6474/step-count state. This is not an independent physical rotor-shaft encoder, so lost steps cannot be detected directly from the reported rotor angle. Use conservative motion first and treat audible stalls or `드드드득` as invalid identification data.

The instructor manual gives example profiles with 3000 step/s^2 acceleration/deceleration and compares High (`Max=1000, Min=300`), Medium (`Max=1000, Min=200`) and Low (`Max=200, Min=200`) speed configurations. This repository's current firmware profile differs, so the notebook is intended to identify coefficients for the actual hardware/configuration rather than reproduce the manual coefficients exactly.

## PID workflow

1. Flash `PendulumController/PendulumController.ino` to the Nucleo board.
2. Open `pendulum_pid_velocity.ipynb` in Jupyter.
3. Set `SERIAL_PORT` for your computer.
4. Run the connection and status cells.
5. Run the low-speed direction test first. If positive command drives the rotor in the wrong control direction, change `CONTROL_SIGN`.
6. Manually place the pendulum close to upright (`180 deg`) before starting the PID cell.
7. Start with conservative gains and tune on the real mechanism.

## Fixed-pivot hand-release experiment

For an experiment with no rotary motion, start with [`docs/simple_pendulum_free_response.md`](docs/simple_pendulum_free_response.md), then run `simple_pendulum_simulation.ipynb`.

To record the real encoder after lifting and releasing the pendulum by hand, use `encoder_release_experiment.ipynb`. It sends an initial hard-stop and then query commands only; it does not command motor motion. If a live display is required, run:

```bash
python encoder_live_monitor.py --duration 20 --countdown 3
```

Both acquisition programs calibrate the downward angle, retain raw and zero-referenced angles, save host and MCU timestamps, and write a timestamped CSV under `data/`.

## Original EDUKIT `main.c`: controller hierarchy

See [`docs/edukit_main_c_analysis.md`](docs/edukit_main_c_analysis.md) for the full code-level analysis.

The short version is:

```text
Pendulum PID / LQR ----+
                       +--> control output --> motor actuation
Rotor-position PID ----+
```

There is **not** another hidden mechanical position PID inside the L6474 motor layer.

The original source has two important actuation modes:

- `ACCEL_CONTROL == 1` (defined as the default in the inspected `main.c`): the controller output is passed to `apply_acceleration()`, integrated into target velocity, and converted to STEP PWM period. Direction and PWM timing are explicitly managed by firmware.
- `ACCEL_CONTROL == 0`: the firmware calls `BSP_MotorControl_GoTo()`. In this mode the L6474/BSP generates a deterministic target-position trajectory constrained by minimum speed, maximum speed, acceleration and deceleration. This trajectory generator is not a PID.

The second-order `G_rotor(s)` in the instructor manual is therefore an **experimentally identified equivalent plant model** of the motor-controller/rotor response. It should not be interpreted as proof of an internal second-order PID loop.

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

### Instructor manual / dynamic motor model

- **The Integrated Rotary Inverted Pendulum — An Open and Configurable System With Digital Motor Controller Technology**, Prof. William J. Kaiser (UCLA), 129-page instructor manual: https://www.st.com/content/dam/AME/2019/Educational%20Curriculums/motor-control/Introduction_to_Integrated_Rotary_Inverted_Pendulum_v2.pdf
- ST Motor Control and Control Systems curriculum page hosting the manual: https://www.st.com/content/st_com/en/campaigns/educationalplatforms/motorcontrol-edu.html

Section 4 of the manual describes the rotor/motor-controller dynamic response, the effect of maximum/minimum speed and acceleration/deceleration, and the identified second-order `G_rotor(s)` model used by `stepper_dynamic_response.ipynb`.

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
- Example one-motor motion-control source analyzed in this repository: https://github.com/wjkaiser/Edukit_Rotary_Inverted_Pendulum_Project/blob/main/Edukit_Rotary_Inverted_Pendulum_Project/Projects/Multi/Examples/MotionControl/IHM01A1_ExampleFor1Motor/Src/main.c
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
