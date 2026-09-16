# Real-Time PID Pendulum Simulator

This folder contains a small real-time simulation for a rod pendulum controlled by PID:

- [`pid_pendulum_realtime.py`](./pid_pendulum_realtime.py)

The simulator was written to reproduce the pendulum dynamics and PID behavior used in the project without requiring the physical NUCLEO/L6474 hardware.

## What the simulator does

The model is a nonlinear rotational pendulum:

```text
J * theta_ddot + c * theta_dot + k * sin(theta) = tau
```

where:

- `theta` is the pendulum angle,
- `theta_dot` is angular velocity,
- `J` is the rotational inertia,
- `c` is viscous damping,
- `k` is the gravity-related restoring coefficient,
- `tau` is the controller input.

The implementation normalizes the model by `J`, so the simulation uses:

```text
J = 1
k / J = wn^2
c / J = 2 * zeta * wn
u = tau / J
```

The plant is integrated with a fixed-step fourth-order Runge-Kutta method (RK4).

## Identified plant parameters

The current script uses:

| Parameter | Value |
|---|---:|
| Natural frequency `wn` | `7.499 rad/s` |
| Damping ratio `zeta` | `0.01172` |
| Gravity `g` | `9.81 m/s^2` |
| Simulation step `DT` | `0.002 s` |
| Reference step | `5 deg` |
| Initial reference | `20 deg` |

For drawing the rod, its effective length is calculated as:

```text
L = 3g / (2 wn^2)
```

This gives approximately `0.262 m`.

## PID controller

The control law is:

```text
tau = Kp * e + Ki * integral(e) - Kd * theta_dot
```

with:

```text
e = reference - theta
```

The derivative term is applied to the measured angular velocity rather than differentiating the error directly.

The PID gains are generated from the desired closed-loop design parameters:

```text
wn_des   = 12.0
zeta_des = 0.7
alpha    = 8.0
```

The controller output is saturated at:

```text
|u| <= 3 * wn^2
```

The simulator also implements simple anti-windup: the integral state is not accumulated while the unsaturated PID output exceeds the actuator limit.

## Requirements

Python 3 is required.

Install the required packages:

```bash
pip install numpy matplotlib
```

## Usage

From the repository root:

### Windows PowerShell

```powershell
python .\simulator\pid_pendulum_realtime.py
```

### Linux / macOS

```bash
python3 simulator/pid_pendulum_realtime.py
```

This opens the real-time simulator window.

## Quick numerical check

The script also provides a non-interactive check mode. It runs a 5-second response and prints the final angle, overshoot, controller effort, and model information.

### Windows PowerShell

```powershell
python .\simulator\pid_pendulum_realtime.py --check
```

### Linux / macOS

```bash
python3 simulator/pid_pendulum_realtime.py --check
```

This is useful when testing the model on a machine without a GUI or when checking that Python dependencies are installed correctly.

## Keyboard controls

Click the Matplotlib window first so it receives keyboard input.

| Key | Action |
|---|---|
| `Up` | Increase reference angle by `5 deg` |
| `Down` | Decrease reference angle by `5 deg` |
| `Left` | Apply a negative angular-velocity disturbance |
| `Right` | Apply a positive angular-velocity disturbance |
| `c` | Toggle PID controller ON/OFF |
| `r` | Reset simulation state, time, and plots |
| `Space` | Pause/resume simulation |

The left/right keys are useful for disturbance-rejection tests. The disturbance is applied by directly changing the pendulum angular velocity.

## Display

The left panel shows the pendulum animation and reference angle.

The right panel shows approximately the most recent 10 seconds of:

```text
pendulum angle
reference angle
```

The status text reports values such as:

```text
time
current angle
reference angle
control input
controller ON/OFF state
pause state
```

## Real-time behavior

The physics solver uses a fixed simulation step:

```text
DT = 0.002 s
```

while the Matplotlib animation normally updates at about 50 FPS.

At each graphics update, the script measures how much wall-clock time has elapsed and advances the RK4 simulation by as many `DT` steps as necessary.

The elapsed time used in one graphics update is capped at `0.1 s` so a temporary GUI stall does not cause a very large simulation jump.

## Coordinate convention

The simulator uses:

```text
theta = 0 deg
```

for the rod pointing downward.

Positive and negative angles rotate the pendulum away from this downward equilibrium.

This is important when comparing the simulator angle with the physical pendulum encoder convention used elsewhere in the repository.

## Controller ON/OFF behavior

When the controller is OFF:

```text
tau = 0
```

and the rod evolves only according to the identified nonlinear pendulum dynamics and damping.

Whenever the controller is toggled ON or OFF, the integral state is cleared to avoid carrying an old integral value into the new control mode.

## Reset behavior

Pressing `r` resets:

```text
theta
angular velocity
integral error
simulation time
plot history
```

The current reference angle is intentionally preserved.

## Wayland / Linux GUI note

If the Matplotlib window does not open correctly under Wayland, try forcing the Tk backend:

```bash
MPLBACKEND=TkAgg python3 simulator/pid_pendulum_realtime.py
```

You may need to install the Tk Python package provided by your Linux distribution.

## Why this simulator is useful

This program can be used to:

- inspect the identified pendulum dynamics before running hardware,
- verify PID gains and saturation behavior,
- test reference changes,
- test disturbance rejection,
- compare controller ON/OFF motion,
- check anti-windup behavior,
- build intuition before applying related control laws to the physical pendulum system.

The simulator is intentionally lightweight and is not a full model of the stepper motor, L6474 driver, communication delay, encoder quantization, backlash, or missed motor steps. Those effects need to be considered separately when comparing simulation results with the real system.
