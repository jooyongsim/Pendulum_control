# Rotary bar pendulum model (input: arm angular acceleration)

This folder uses the following convention.

- `phi` (\(\phi\)): rotary-arm angle
- `theta` (\(\theta\)): pendulum angle measured from the equilibrium being analyzed
- \(u=\ddot\phi\): commanded rotary-arm angular acceleration
- uniform pendulum bar: mass \(m\), total length \(l\)
- rotary arm radius: \(r\)
- gravity: \(g=9.81\,\mathrm{m/s^2}\)
- numerical values used here: \(l=0.235\,\mathrm{m}\), \(r=0.14\,\mathrm{m}\)
- low-speed approximation: \(\dot\phi^2\), \(\dot\theta^2\), and \(\dot\phi\dot\theta\) terms are neglected.

For a uniform bar pivoted at one end,

\[
l_c=\frac{l}{2},\qquad J_p=\frac{1}{3}ml^2.
\]

## 1. Suspended equilibrium: \(\theta=0\) downward

The linearized equation with input \(u=\ddot\phi\) is

\[
\ddot\theta+\frac{3g}{2l}\theta=-\frac{3r}{2l}u.
\]

Therefore,

\[
\boxed{\frac{\Theta(s)}{U(s)}=-\frac{\frac{3r}{2l}}{s^2+\frac{3g}{2l}}}
\]

With \(l=0.235\,\mathrm m\), \(r=0.14\,\mathrm m\),

\[
a=\frac{3g}{2l}=62.6170\,\mathrm{s^{-2}},\qquad
b=\frac{3r}{2l}=0.893617,
\]

so

\[
\boxed{\frac{\Theta(s)}{U(s)}=-\frac{0.893617}{s^2+62.6170}}.
\]

`01_suspended_acceleration_response.ipynb` applies a rectangular acceleration command for 1 s and then returns the acceleration command to zero. The notebook also integrates \(u\) to show the resulting arm velocity and angle.

> Important: setting angular acceleration back to zero after 1 s does **not** make arm velocity return to zero. Under the ideal model the arm continues at the velocity accumulated during the first second. For a physical stepper system, use acceleration/velocity limits and, when needed, a deceleration segment.

## 2. Inverted equilibrium: \(\theta=0\) upright

The linearized equation is

\[
\ddot\theta-\frac{3g}{2l}\theta=\frac{3r}{2l}u,
\]

and the plant from arm angular acceleration to pendulum angle is

\[
\boxed{\frac{\Theta(s)}{U(s)}=\frac{\frac{3r}{2l}}{s^2-\frac{3g}{2l}}}
=\boxed{\frac{0.893617}{s^2-62.6170}}.
\]

The positive real pole \(+\sqrt{62.6170}=+7.9131\,\mathrm{s^{-1}}\) is the open-loop inverted instability.

`02_inverted_pid_control.ipynb` designs a PID around this acceleration-input plant and simulates recovery from a 5 degree initial tilt. The commanded arm acceleration is limited to a configurable value to represent an actuator constraint.

## Files

- [`DERIVATION.md`](DERIVATION.md): derivation from torque/inertia through suspended and inverted linear models.
- [`01_suspended_acceleration_response.ipynb`](01_suspended_acceleration_response.ipynb): open-loop suspended response to a 1 s acceleration pulse.
- [`02_inverted_pid_control.ipynb`](02_inverted_pid_control.ipynb): PID stabilization of the inverted linearized model.
- [`figures/suspended_acceleration_pulse_response.png`](figures/suspended_acceleration_pulse_response.png)
- [`figures/inverted_pid_response.png`](figures/inverted_pid_response.png)

## Stepper-motor interpretation

The control input in these notebooks is \(u=\ddot\phi\), not motor torque. For a stepper implementation, integrate the acceleration command to obtain commanded angular velocity, then convert angular velocity to step-pulse frequency. The admissible \(|u|\) should be chosen from the usable torque-speed envelope and the equivalent inertia so that missed steps are avoided.
