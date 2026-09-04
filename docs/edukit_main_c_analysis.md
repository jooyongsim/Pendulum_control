# EDUKIT `main.c` motor/control architecture analysis

## Scope

This note analyzes the UCLA/STMicroelectronics EDUKIT firmware example:

- Source repository: https://github.com/wjkaiser/Edukit_Rotary_Inverted_Pendulum_Project
- Main source: https://github.com/wjkaiser/Edukit_Rotary_Inverted_Pendulum_Project/blob/main/Edukit_Rotary_Inverted_Pendulum_Project/Projects/Multi/Examples/MotionControl/IHM01A1_ExampleFor1Motor/Src/main.c
- Configuration header: https://github.com/wjkaiser/Edukit_Rotary_Inverted_Pendulum_Project/blob/main/Edukit_Rotary_Inverted_Pendulum_Project/Projects/Multi/Examples/MotionControl/IHM01A1_ExampleFor1Motor/Inc/edukit_system.h
- Instructor manual: https://www.st.com/content/dam/AME/2019/Educational%20Curriculums/motor-control/Introduction_to_Integrated_Rotary_Inverted_Pendulum_v2.pdf

The important distinction is between the **pendulum/rotor feedback controllers** and the **stepper-motion generator** below them.

## Short answer: is there a hidden inner position PID?

No, not in the sense of a conventional cascaded servo drive with a position PID -> velocity PID -> current PID.

The firmware contains explicit feedback controllers for the pendulum angle and, when enabled, rotor angle:

```text
Pendulum error -> PID_Pend ----+
                              +--> rotor_control_target_steps
Rotor error ----> PID_Rotor ---+
```

But the lower-level motor actuation is not another position PID. There are two selectable actuation paths.

### `ACCEL_CONTROL == 1` (the supplied `main.c` defines this as 1)

The controller output is interpreted as an acceleration-like command. `apply_acceleration()` integrates it to obtain a target step velocity and converts that velocity to a PWM step-clock period.

```text
outer control output
        |
        v
 acceleration command
        |
        |  v[k+1] = v[k] + a[k] Ts
        v
 target step velocity
        |
        |  PWM period = clock / |velocity|
        v
 DIR + STEP PWM
        |
        v
      L6474
        |
        v
  stepper motor rotor
```

`apply_acceleration()` also limits acceleration/deceleration, limits maximum velocity, handles sign/direction changes, and updates the remaining PWM period to reduce discontinuity during speed or direction changes.

Therefore, with `ACCEL_CONTROL == 1`, `MAX_ACCEL/MAX_DECEL/MAX_SPEED/MIN_SPEED` from the ordinary L6474 `GoTo()` profile are not the main inner closed-loop dynamics. The custom PWM-frequency update code determines the step schedule.

### `ACCEL_CONTROL == 0`

The code instead executes:

```c
BSP_MotorControl_GoTo(0, rotor_control_target_steps/2);
```

This invokes the L6474/BSP position-target motion logic. In this path, a new target position is supplied and the library generates a deterministic motion profile constrained by configured minimum speed, maximum speed, acceleration, and deceleration.

This is still **not a PID position servo**. It is a command/profile generator for an open-loop stepper motor. The library tracks commanded step position internally and shapes the pulse frequency so that the target is approached according to the configured motion limits.

## Why the ST/UCLA PDF models the rotor response as a second-order system

The instructor manual explicitly treats the mapping from rotor angle command, `phi_RC`, to rotor angle, `phi`, as a dynamic plant. It says that target angles are updated every control cycle, often much faster than the motor can reach an individual target. As a result, the sequence of target updates plus speed/acceleration limits produces a deterministic but nonlinear actuator response.

For small-signal controller design this behavior is approximated by

```text
                    a
G_rotor(s) = -----------------
             s^2 + b s + c
```

with unity DC gain requiring `a = c` (apart from the sign convention used in the original physical system). The manual identifies different coefficients for different motor speed configurations.

This second-order model is therefore an **identified equivalent plant model**, not evidence of an internal second-order PID controller.

## `main.c` control layers

### 1. Measurement layer

The pendulum optical encoder is read by `encoder_position_read()`. Rotor position is obtained from the motor-control position state by `rotor_position_read()`.

The code calibrates the downward pendulum angle, applies the 180-degree reference for inverted operation, applies optional inclination/encoder-offset correction, and enforces rotor and pendulum safety limits.

### 2. Primary pendulum controller

The code configures a CMSIS-style PID state structure:

```c
PID_Pend.Kp = proportional * CONTROLLER_GAIN_SCALE;
PID_Pend.Ki = integral * CONTROLLER_GAIN_SCALE;
PID_Pend.Kd = derivative * CONTROLLER_GAIN_SCALE;
```

The pendulum error is converted to the rotor-step scale and passed to:

```c
pid_filter_control_execute(&PID_Pend, current_error_steps,
                           sample_period, Deriv_Filt_Pend);
```

Its output becomes the principal motor/rotor command:

```c
rotor_control_target_steps = PID_Pend.control_output;
```

Although the variable name contains `steps`, its physical meaning depends on the actuation mode. Under `ACCEL_CONTROL == 1`, the value is passed to `apply_acceleration()` and behaves as an acceleration command, not a direct target position.

### 3. Secondary rotor-position PID

When the dual-controller architecture is enabled, the firmware computes rotor-angle error and executes `PID_Rotor`:

```c
pid_filter_control_execute(&PID_Rotor, current_error_rotor_steps,
                           sample_period_rotor, Deriv_Filt_Rotor);
```

Then it sums the two controller outputs:

```c
rotor_control_target_steps =
    PID_Pend.control_output + PID_Rotor.control_output;
```

This secondary PID is the rotor-position feedback controller referred to in the comments. It is an **outer control-system controller**, not an internal L6474 servo loop.

### 4. Optional state-feedback / plant-shaping paths

The source also includes LQR/state-feedback operation, integral compensation, feed-forward terms, disturbance-rejection tests, chirp/comb excitation, and optional rotor-plant filters. These can modify `rotor_control_target_steps` before motor actuation.

An optional digital filter implements a selected second-order rotor plant shape. This should not be confused with the physical motor model from the PDF: it is an optional signal-shaping/design feature inserted by the firmware.

### 5. Motor actuation layer: acceleration-control path

The default compile-time selection is:

```c
#define ACCEL_CONTROL 1
```

The motor command is then applied as:

```c
apply_acceleration(&rotor_control_target_steps,
                   &target_velocity_prescaled,
                   Tsample);
```

Conceptually:

```text
rotor_control_target_steps ~= requested acceleration

target_velocity[k+1]
  = target_velocity[k] + acceleration[k] * Ts

step_frequency ~= abs(target_velocity)
PWM_period = timer_clock / step_frequency
```

The function also changes the DIR GPIO when the velocity changes sign.

This implementation is especially suitable for an inverted pendulum because the desired rotor acceleration can change every 2 ms without repeatedly starting independent `GoTo()` trajectories.

### 6. Motor actuation layer: target-position path

If acceleration control is disabled:

```c
BSP_MotorControl_GoTo(0, rotor_control_target_steps/2);
```

The L6474 motion library then computes its pulse schedule using speed, acceleration and deceleration settings. The instructor manual's discussion of repeated target-position updates corresponds closely to this style of actuation.

## Relationship to a conventional servo architecture

A conventional servo might be:

```text
position PID -> velocity PID -> current/torque PID -> motor
      ^             ^                ^
  encoder pos   encoder vel      current sensor
```

The EDUKIT stepper architecture is closer to:

```text
pendulum PID/LQR + optional rotor PID
                 |
                 v
     acceleration or position command
                 |
                 v
     deterministic STEP/DIR scheduler
                 |
                 v
       L6474 current regulation
                 |
                 v
             stepper motor
```

The L6474 has current/torque regulation at the electrical drive level (`TVAL`), but it does not close a mechanical shaft-position PID loop using an independent rotor encoder in this EDUKIT configuration.

## Why speed/acceleration appear to have their own dynamics after a position command

With the `GoTo()` path, a position target is not converted to an instantaneous jump. The motion generator starts at minimum speed (subject to implementation/state), accelerates up to maximum speed, decides when braking must begin, then decelerates to arrive at the target. Therefore position, velocity and acceleration naturally show time-domain profiles even without a PID.

With the custom `ACCEL_CONTROL` path, this relation is even more direct: acceleration is the command, velocity is its integral, and STEP frequency represents velocity.

That is why an outer PID or LQR can command rotor motion while the actuator itself exhibits an identifiable dynamic response. The actuator response is generated by pulse scheduling, mechanical inertia/load, stepper torque limits, and nonlinear effects such as missed steps/resonance—not by a hidden position PID.

## Dynamic-response identification recommended for this repository

Use `stepper_dynamic_response.ipynb` to record:

- commanded rotor target angle
- reported rotor angle/step position
- L6474 step rate
- estimated rotor angular velocity
- estimated angular acceleration
- pendulum angle (useful for identifying load coupling)
- driver status and safety state

The notebook applies repeated small position steps and fits the same second-order structure used in the instructor manual. Tests should begin with the pendulum removed or safely hanging down and with small rotor travel.

Important limitation: the current Arduino firmware's rotor angle is based on commanded/internal step position. It is not an independent physical shaft encoder. Missed steps therefore cannot be inferred from rotor position alone. The pendulum encoder can reveal coupled motion but does not directly measure rotor shaft angle.

## Key conclusion

- **Yes:** `main.c` has explicit pendulum PID and optional rotor-position PID/LQR control.
- **No:** there is no additional hidden mechanical position PID inside the L6474 motion layer.
- **Position-target mode:** L6474/BSP generates a speed/acceleration-limited step trajectory.
- **Default `ACCEL_CONTROL=1` mode:** custom firmware integrates acceleration to velocity and directly updates PWM step frequency.
- The PDF's `G_rotor(s)` is an experimentally identified equivalent motor-controller plant model, not the transfer function of a hidden internal PID.
