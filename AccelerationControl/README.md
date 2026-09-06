# AccelerationControl

Host-driven acceleration-control implementation inspired by the `ACCEL_CONTROL` / `apply_acceleration()` path in the UCLA EDUKIT `main.c`.

## Architecture

```text
Jupyter controller
    |
    | signed acceleration command [pps^2]
    v
AccelerationController.ino
    |
    | v[k+1] = v[k] + a[k] Ts
    v
signed target step velocity [pps]
    |
    | set_max_speed(abs(v)) + run(direction)
    v
L6474 STEP/DIR generation
    v
stepper motor
```

The feedback/control-law calculation stays on the host. The MCU only receives acceleration, maintains the velocity integrator, enforces motor/rotor safety limits, and applies the resulting signed step rate.

Keeping the acceleration-to-velocity integrator on the MCU avoids host USB/serial jitter directly changing the integration interval. The Jupyter host still owns the controller calculation and updates acceleration commands at its chosen control rate.

## Files

- `AccelerationController/AccelerationController.ino` - Arduino/STM32 firmware.
- `AccelerationController/control-comms.hpp` - JSON serial protocol helper.
- `acceleration_control.ipynb` - open-loop acceleration test and host-side feedback-controller template.

## Commands

| Command | ID | Action |
|---|---:|---|
| Set home | 0 | ignored |
| Set acceleration | 1 | signed acceleration in pps^2 |
| Hard stop | 2 | ignored |
| Query | 3 | ignored |
| Reset safety | 4 | ignored |
| Zero velocity | 5 | ignored |

Observation:

```text
[
  pendulum_angle_deg,
  rotor_angle_deg,
  measured_signed_pps,
  integrated_target_pps,
  acceleration_command_pps2,
  l6474_status,
  host_command_age_ms
]
```

## Relation to the original EDUKIT code

The original `main.c` defines `ACCEL_CONTROL 1` and implements `apply_acceleration()`. Its central operation is an acceleration integral followed by conversion of target velocity into STEP timing. The original implementation writes PWM periods directly and carefully rescales the remaining timer period during speed/direction changes.

This Arduino implementation preserves the same control-domain idea but uses the STM32duino L6474 API (`set_max_speed()` + `run()`) instead of directly calling `L6474_Board_Pwm1SetPeriod()`. Therefore it is **conceptually equivalent, not cycle-for-cycle identical** to the original low-level timer implementation.

Original source:

https://github.com/wjkaiser/Edukit_Rotary_Inverted_Pendulum_Project/blob/main/Edukit_Rotary_Inverted_Pendulum_Project/Projects/Multi/Examples/MotionControl/IHM01A1_ExampleFor1Motor/Src/main.c

## Safety

- Default host acceleration is clamped to +/-12000 pps^2 in firmware.
- Target velocity is clamped to +/-2000 pps.
- Rotor software travel limit is +/-90 degrees.
- If host commands stop arriving for 150 ms, firmware zeros velocity and stops the motor.
- Start with the pendulum hanging down or removed and use the open-loop test first.
- The rotor angle reported here is step-count based, not an independent shaft encoder; missed steps are not directly observable.
