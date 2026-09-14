"""Simulator for the rotary inverted pendulum, using the identified parameters.

The plant comes from `pendulum_model_id.py` / `docs/pendulum_model_identification.md`.
The point of this module is to try controllers against the *measured* dynamics
before putting them on the hardware, including the effects that usually decide
whether a balancing controller works: sample rate, encoder quantisation,
derivative noise, actuator saturation and dry friction.

Angle convention
----------------
theta is measured from UPRIGHT, so theta = 0 is balanced and theta = pi is
hanging. The firmware reports 0..360 deg with 180 = upright, so

    theta_rad = deg2rad(firmware_angle_deg - 180.0)

Model
-----
With the pivot accelerating at `a` (m/s^2) in the plane of the swing,

    J theta'' = m g l sin(theta) - m l a cos(theta) - c theta' - Coulomb

Dividing by J and using the identified combinations

    wn^2 = m g l / J,     2 zeta wn = c / J,     L_eff = J / (m l) = g / wn^2

gives the form this module integrates:

    theta'' = wn^2 sin(theta) - 2 zeta wn theta' - (a / L_eff) cos(theta) - F sgn(theta')

Note that the input gain -1/L_eff is fixed by the SAME L_eff the free-swing
identification produced: no extra experiment is needed for it. What is NOT
identified is the arm radius `r` that converts rotor angular acceleration into
pivot acceleration (a = r * alpha''); measure it, or identify it from a driven
experiment. It is a plain scale factor on the control input.
"""

from dataclasses import dataclass, field

import numpy as np

G = 9.80665


# ---------------------------------------------------------------------------
# plant
# ---------------------------------------------------------------------------
@dataclass
class PendulumParams:
    """Identified pendulum model. Defaults are the values from the 60 s logs."""

    wn: float = 7.508            # rad/s   natural frequency
    zeta: float = 0.010629       # -       damping ratio
    coulomb_dps: float = 3.149   # deg/s   amplitude lost per second to dry friction
    arm_radius_m: float = 0.085  # m       NOT identified -- measure it

    @property
    def sigma(self):
        """Decay rate zeta*wn [1/s]."""
        return self.zeta * self.wn

    @property
    def wd(self):
        """Damped natural frequency [rad/s]."""
        return self.wn * np.sqrt(max(0.0, 1.0 - self.zeta ** 2))

    @property
    def L_eff(self):
        """Effective length g/wn^2 = J/(m l) [m]."""
        return G / self.wn ** 2

    @property
    def b(self):
        """Input gain: theta'' per unit pivot acceleration [1/m]."""
        return -1.0 / self.L_eff

    @property
    def coulomb_accel(self):
        """Dry-friction angular deceleration [rad/s^2].

        A Coulomb torque F removes 4F/wn^2 of amplitude per period, so a
        measured amplitude loss rate c_A [rad/s] corresponds to F = c_A wn^2 T/4.
        """
        return np.deg2rad(self.coulomb_dps) * self.wn ** 2 * (2 * np.pi / self.wn) / 4.0

    def physical(self, J):
        """Resolve c, m*l and m g l from a measured inertia J [kg m^2].

        Free-swing data fixes only the ratios mgl/J and c/J, so one measured
        quantity is needed to put numbers on the individual parameters.
        """
        ml = J / self.L_eff
        return dict(J=J, c=2 * self.sigma * J, ml=ml, mgl=ml * G)

    def summary(self):
        return (f"wn={self.wn:.4f} rad/s  zeta={self.zeta:.6f}  wd={self.wd:.4f} rad/s\n"
                f"sigma=zeta*wn={self.sigma:.4f} 1/s   L_eff=g/wn^2={self.L_eff:.5f} m\n"
                f"mgl/J=wn^2={self.wn ** 2:.3f} 1/s^2   c/J=2*zeta*wn={2 * self.sigma:.4f} 1/s\n"
                f"input gain b=-1/L_eff={self.b:.3f} (rad/s^2)/(m/s^2)")


def derivative(theta, omega, accel_pivot, p: PendulumParams, linear=False):
    """Right-hand side of the pendulum ODE. `accel_pivot` in m/s^2."""
    if linear:
        # linearised about upright; used to check against the analytic solution
        restoring = p.wn ** 2 * theta
        coupling = accel_pivot / p.L_eff
        friction = 0.0
    else:
        restoring = p.wn ** 2 * np.sin(theta)
        coupling = accel_pivot / p.L_eff * np.cos(theta)
        # tanh instead of sign: sign() makes a stiff discontinuity at omega = 0
        friction = p.coulomb_accel * np.tanh(omega / 1e-3)
    return omega, restoring - 2 * p.sigma * omega - coupling - friction


# ---------------------------------------------------------------------------
# analytic solution of the initial value problem (linear, viscous only)
# ---------------------------------------------------------------------------
def ivp_hanging(t, theta0, omega0, p: PendulumParams):
    """theta'' + 2 zeta wn theta' + wn^2 theta = 0, theta measured from hanging.

    Underdamped (zeta < 1):
        theta(t) = e^(-sigma t) [ theta0 cos(wd t) + (omega0 + sigma theta0)/wd sin(wd t) ]
    """
    s, wd = p.sigma, p.wd
    if p.zeta < 1.0:
        return np.exp(-s * t) * (theta0 * np.cos(wd * t)
                                 + (omega0 + s * theta0) / wd * np.sin(wd * t))
    if np.isclose(p.zeta, 1.0):                      # critically damped
        return np.exp(-s * t) * (theta0 + (omega0 + s * theta0) * t)
    r = p.wn * np.sqrt(p.zeta ** 2 - 1)              # overdamped
    s1, s2 = -s + r, -s - r
    c1 = (omega0 - s2 * theta0) / (s1 - s2)
    return c1 * np.exp(s1 * t) + (theta0 - c1) * np.exp(s2 * t)


def ivp_upright(t, theta0, omega0, p: PendulumParams):
    """theta'' + 2 zeta wn theta' - wn^2 theta = 0, theta measured from upright.

    Gravity now drives the error instead of restoring it, so the roots are real
    and one of them is positive: the solution diverges for almost every IC.
    """
    s = p.sigma
    r = np.sqrt(s ** 2 + p.wn ** 2)
    s1, s2 = -s + r, -s - r                          # s1 > 0 > s2
    c1 = (omega0 - s2 * theta0) / (s1 - s2)
    return c1 * np.exp(s1 * t) + (theta0 - c1) * np.exp(s2 * t)


# ---------------------------------------------------------------------------
# controllers
# ---------------------------------------------------------------------------
@dataclass
class State:
    """What a controller is allowed to see each control step."""

    theta: float = 0.0        # rad, from upright (quantised as the encoder would)
    omega: float = 0.0        # rad/s
    alpha: float = 0.0        # rad, rotor angle
    alpha_dot: float = 0.0    # rad/s


class Controller:
    """Base class: returns a pivot acceleration command in m/s^2."""

    name = "none"

    def reset(self):
        pass

    def __call__(self, s: State, dt: float) -> float:
        return 0.0


class PID(Controller):
    """Textbook PID on the upright angle, with derivative filtering and anti-windup."""

    name = "PID"

    def __init__(self, kp, ki, kd, setpoint=0.0, u_max=np.inf, tau_d=0.02):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.setpoint, self.u_max, self.tau_d = setpoint, u_max, tau_d
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.d_state = 0.0
        self.prev_error = None

    def __call__(self, s: State, dt: float) -> float:
        error = s.theta - self.setpoint
        if self.prev_error is None:
            self.prev_error = error
        raw_d = (error - self.prev_error) / dt
        # first-order filter: raw differences of a quantised angle are mostly noise
        alpha = dt / (self.tau_d + dt)
        self.d_state += alpha * (raw_d - self.d_state)
        self.prev_error = error

        u = self.kp * error + self.ki * self.integral + self.kd * self.d_state
        if abs(u) <= self.u_max:                     # conditional integration
            self.integral += error * dt
        return float(np.clip(u, -self.u_max, self.u_max))


class StateFeedback(Controller):
    """u = -(k1 theta + k2 theta_dot), theta measured from upright.

    Angle-only feedback balances the pendulum but says nothing about where the
    rotor ends up, so the arm drifts away and eventually hits its travel limit.
    Add k3/k4 to regulate the rotor as well -- that is what makes the difference
    between "balances in simulation" and "balances on the bench".
    """

    name = "state feedback"

    def __init__(self, k1, k2, k3=0.0, k4=0.0, u_max=np.inf):
        self.k1, self.k2, self.k3, self.k4, self.u_max = k1, k2, k3, k4, u_max

    def __call__(self, s: State, dt: float) -> float:
        u = -(self.k1 * s.theta + self.k2 * s.omega
              + self.k3 * s.alpha + self.k4 * s.alpha_dot)
        return float(np.clip(u, -self.u_max, self.u_max))


def place_poles(p: PendulumParams, wn_des, zeta_des):
    """Gains that put the closed-loop poles at wn_des, zeta_des.

    Closed loop with u = -(k1 theta + k2 theta'):
        theta'' = (wn^2 + k1/L) theta + (-2 sigma + k2/L) theta'
    so matching s^2 + 2 zeta_d wn_d s + wn_d^2 gives the expressions below.
    """
    L, a1, a0 = p.L_eff, 2 * zeta_des * wn_des, wn_des ** 2
    k1 = -L * (a0 + p.wn ** 2)
    k2 = L * (2 * p.sigma - a1)
    return k1, k2


# ---------------------------------------------------------------------------
# simulation
# ---------------------------------------------------------------------------
@dataclass
class SimConfig:
    duration: float = 5.0
    dt_plant: float = 1e-4        # integrator step
    control_hz: float = 100.0     # discrete controller rate
    encoder_deg: float = 0.3      # quantisation, 1200 CPR
    accel_max: float = np.inf     # |pivot acceleration| limit [m/s^2]
    rotor_limit_deg: float = np.inf
    delay_steps: int = 0          # controller-output delay, in control steps
    linear: bool = False          # integrate the linearised plant instead
    theta0_deg: float = 5.0
    omega0_dps: float = 0.0


def simulate(p: PendulumParams, controller: Controller, cfg: SimConfig):
    """Integrate the plant with a discrete controller in the loop."""
    controller.reset()
    n = int(cfg.duration / cfg.dt_plant)
    every = max(1, int(round(1.0 / (cfg.control_hz * cfg.dt_plant))))
    quant = np.deg2rad(cfg.encoder_deg)

    theta = np.deg2rad(cfg.theta0_deg)
    omega = np.deg2rad(cfg.omega0_dps)
    alpha = alpha_dot = 0.0                    # rotor angle / rate
    u = 0.0
    pending = [0.0] * (cfg.delay_steps + 1)

    log = {k: np.empty(n) for k in
           ("t", "theta", "omega", "u", "alpha", "theta_meas", "saturated")}

    for i in range(n):
        t = i * cfg.dt_plant
        if i % every == 0:
            # the controller only ever sees a quantised angle
            theta_meas = np.round(theta / quant) * quant if np.isfinite(quant) and quant > 0 else theta
            dt_c = every * cfg.dt_plant
            cmd = controller(State(theta_meas, omega, alpha, alpha_dot), dt_c)
            pending.append(cmd)
            u_raw = pending.pop(0)
            u = float(np.clip(u_raw, -cfg.accel_max, cfg.accel_max))
            # a rotor at its travel limit cannot accelerate further outwards
            if np.isfinite(cfg.rotor_limit_deg):
                at_limit = abs(np.rad2deg(alpha)) >= cfg.rotor_limit_deg
                if at_limit and np.sign(u) == np.sign(alpha):
                    u = 0.0
        else:
            theta_meas = log["theta_meas"][i - 1]

        # rotor kinematics: pivot acceleration a = r * alpha''
        alpha_ddot = u / p.arm_radius_m
        alpha_dot += alpha_ddot * cfg.dt_plant
        alpha += alpha_dot * cfg.dt_plant

        d1, d2 = derivative(theta, omega, u, p, linear=cfg.linear)
        omega += d2 * cfg.dt_plant
        theta += d1 * cfg.dt_plant

        log["t"][i] = t
        log["theta"][i] = theta
        log["omega"][i] = omega
        log["u"][i] = u
        log["alpha"][i] = alpha
        log["theta_meas"][i] = theta_meas
        log["saturated"][i] = abs(u) >= cfg.accel_max - 1e-12

    log["theta_deg"] = np.rad2deg(log["theta"])
    log["alpha_deg"] = np.rad2deg(log["alpha"])
    log["settled"] = bool(np.all(np.abs(log["theta_deg"][-int(0.5 / cfg.dt_plant):]) < 2.0))
    return log


def free_swing(p: PendulumParams, theta0_deg, duration=15.0, from_hanging=True, **kw):
    """Convenience wrapper: no controller, released from rest."""
    cfg = SimConfig(duration=duration, theta0_deg=180.0 + theta0_deg if from_hanging else theta0_deg,
                    control_hz=1000.0, encoder_deg=0.0, **kw)
    return simulate(p, Controller(), cfg)


def state_space(p: PendulumParams):
    """Continuous 4-state model  x = [theta, theta_dot, alpha, alpha_dot].

    theta'' = wn^2 theta - 2 sigma theta' - u / L_eff      (u = pivot accel, m/s^2)
    alpha'' = u / r                                        (rotor kinematics)
    """
    A = np.array([[0.0, 1.0, 0.0, 0.0],
                  [p.wn ** 2, -2 * p.sigma, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 1.0],
                  [0.0, 0.0, 0.0, 0.0]])
    B = np.array([[0.0], [-1.0 / p.L_eff], [0.0], [1.0 / p.arm_radius_m]])
    return A, B


def place_poles4(p: PendulumParams, poles):
    """Ackermann pole placement for the full 4-state model.

    Returns (k1, k2, k3, k4) for u = -(k1 theta + k2 theta' + k3 alpha + k4 alpha').

    Balancing while holding the rotor in place is genuinely 4th order: picking
    k3/k4 by hand on top of a 2-state design destabilises the loop, because the
    rotor can only be moved one way by first tipping the pendulum the other way.
    """
    A, B = state_space(p)
    n = A.shape[0]
    ctrb = np.hstack([np.linalg.matrix_power(A, i) @ B for i in range(n)])
    if abs(np.linalg.det(ctrb)) < 1e-12:
        raise ValueError("model is not controllable with this arm radius")

    phi = np.polynomial.polynomial.polyfromroots(np.asarray(poles, dtype=complex))
    phi = np.real(phi[::-1])                      # highest power first
    phi_A = np.zeros_like(A)
    for coeff in phi:                             # Horner on the matrix
        phi_A = phi_A @ A + coeff * np.eye(n)

    last_row = np.linalg.solve(ctrb.T, np.eye(n)[:, -1])
    K = last_row @ phi_A
    return tuple(float(v) for v in K)
