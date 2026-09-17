"""Rotary inverted pendulum (STEVAL-EDUKIT01) under acceleration control: plant, motor, controllers.

Shared by rotary_balance_realtime.py and rotary_swingup_realtime.py. The numbers and control
laws are the ones running on the real rig (swing-up/balance_control.py, swing-up/swingup_control.py,
swing-up/SwingupController/SwingupController.ino).

theta = pendulum angle from UPRIGHT (0 = balanced, +-pi = hanging), phi = arm angle,
u = phi'' [rad/s^2] (the host commands the arm acceleration).

    theta'' = (3g/2l) sin(theta) + b cos(theta) phi'' + sin(theta) cos(theta) phi'^2
              - 2 zeta wn theta' - Fc sign(theta')
    b = 3r/2l

Identified on our unit: wn = 7.499 rad/s (3g/2l = wn^2, l = 262 mm), zeta = 0.01172,
dry friction 2.705 deg/s of amplitude, b = 0.730 (r = 127 mm) from the open-loop test.

Motor = the firmware, not an ideal acceleration source:
    v += a dt           |a| <= 24000 pps^2, |v| <= 4000 pps   (target velocity)
    |v| < 30 pps        motor stopped (L6474 minimum run speed)
    |phi| >= 150 deg    latched stop
Measurement: 100 Hz loop, 4 ms from reading to the new command, 0.3 deg encoder,
arm angle in whole microsteps, theta' by filtered differencing.
"""
import math
from collections import deque

import numpy as np

# ---------------------------------------------------------------- plant (identified)
G = 9.81
WN, ZETA = 7.499, 0.01172
L = 3 * G / (2 * WN ** 2)                       # 0.262 m, uniform rod
A_G = WN ** 2                                   # 3g/2l [1/s^2]
C_D = 2 * ZETA * WN                             # viscous [1/s]
B_U = 0.730                                     # 3r/2l from the open-loop test
R_ARM = 2 * L * B_U / 3                         # 0.127 m
COULOMB_DPS = 2.705                             # amplitude lost per second to dry friction
F_C = math.radians(COULOMB_DPS) * WN ** 2 * (2 * math.pi / WN) / 4.0   # [rad/s^2]

# ---------------------------------------------------------------- motor (SwingupController.ino)
STEP_RAD = 2 * math.pi / 3200                   # 1/16 microstep
ACCEL_MAX = 24000 * STEP_RAD                    # 47.1 rad/s^2
SPEED_MAX = 4000 * STEP_RAD                     # 7.85 rad/s
SPEED_MIN = 30 * STEP_RAD                       # below this the motor does not run
ARM_LATCH = math.radians(150.0)

# ---------------------------------------------------------------- loop / sensor
CONTROL_HZ = 100.0
DELAY = 0.004                                   # s, reading -> new command
ENC_RAD = math.radians(0.3)                     # 1200 CPR
TAU_V = 0.01                                    # s, theta' filter
DT = 0.001                                      # plant step

# ---------------------------------------------------------------- controllers (hardware defaults)
POLES = [-6 + 6j, -6 - 6j, -1.5 + 1j, -1.5 - 1j]
SWING = dict(pump=30.0, ke=0.3, etarget=1.0, kpa=3.0, kda=5.0,
             catch=12.0, catch_w=4.0, catch_arm=60.0, fall=30.0, bias_tau=6.0)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def state_matrices(b=B_U):
    A = np.array([[0, 1, 0, 0], [A_G, -C_D, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]], float)
    B = np.array([[0], [b], [0], [1]], float)
    return A, B


def place4(poles=POLES, b=B_U):
    """Ackermann: K = e_n^T C^-1 phi(A), u = -K [theta, theta', phi, phi']."""
    A, B = state_matrices(b)
    ctrb = np.hstack([np.linalg.matrix_power(A, i) @ B for i in range(4)])
    phi = np.zeros_like(A)
    for c in np.real(np.poly(poles)):
        phi = phi @ A + c * np.eye(4)
    return np.linalg.solve(ctrb.T, np.eye(4)[:, -1]) @ phi


def place2(wn=12.0, zeta=0.7, b=B_U):
    """Rod only: s^2 + (c + b k2) s + (b k1 - a) = s^2 + 2 zeta wn s + wn^2. The arm is ignored."""
    return np.array([(wn ** 2 + A_G) / b, (2 * zeta * wn - C_D) / b, 0.0, 0.0])


def energy(th, w):
    """0 when upright at rest, -2 A_G (= -112) when hanging at rest."""
    return 0.5 * w * w + A_G * (math.cos(th) - 1)


class Plant:
    """x = [theta, theta', phi, v]; v is the firmware's target velocity, integrated from the command."""

    def __init__(self, theta0=0.0, b=B_U, coulomb=True, motor_limits=True):
        self.b, self.coulomb, self.motor_limits = b, coulomb, motor_limits
        self.reset(theta0)

    def reset(self, theta0=0.0):
        self.x = np.array([theta0, 0.0, 0.0, 0.0])
        self.a = 0.0                                # applied arm acceleration command
        self.t = 0.0
        self.latched = False

    def running(self, v):
        return (not self.motor_limits) or abs(v) >= SPEED_MIN

    def deriv(self, x, a):
        th, w, phi, v = x
        on = self.running(v)
        acc = a if on else 0.0
        phid = v if on else 0.0
        dw = (A_G * math.sin(th) + self.b * math.cos(th) * acc
              + math.sin(th) * math.cos(th) * phid * phid - C_D * w)
        if self.coulomb:
            dw -= F_C * math.tanh(w / 0.01)
        return np.array([w, dw, phid, a])

    def step(self):
        a = 0.0 if self.latched else self.a
        if self.motor_limits:
            a = max(-ACCEL_MAX, min(ACCEL_MAX, a))
        x = self.x
        k1 = self.deriv(x, a)
        k2 = self.deriv(x + DT / 2 * k1, a)
        k3 = self.deriv(x + DT / 2 * k2, a)
        k4 = self.deriv(x + DT * k3, a)
        self.x = x + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        self.t += DT
        if self.motor_limits:
            self.x[3] = max(-SPEED_MAX, min(SPEED_MAX, self.x[3]))
            if abs(self.x[2]) >= ARM_LATCH and not self.latched:
                self.latched = True
                self.x[3] = 0.0

    @property
    def arm_rate(self):
        v = self.x[3]
        return v if self.running(v) else 0.0

    def kick(self, dw):
        self.x[1] += dw


class Sensor:
    """What the host sees: quantised encoder, whole microsteps, filtered theta'."""

    def __init__(self, realistic=True):
        self.realistic = realistic
        self.prev = None
        self.w = 0.0

    def reset(self):
        self.prev, self.w = None, 0.0

    def read(self, plant, offset=0.0):
        """offset = encoder zero error: the true vertical reads as `offset`."""
        th, w, phi, _ = plant.x
        if not self.realistic:
            return np.array([wrap(th + offset), w, phi, plant.arm_rate])
        m = round((th + offset) / ENC_RAD) * ENC_RAD
        ts = 1.0 / CONTROL_HZ
        if self.prev is not None:
            self.w += (ts / (TAU_V + ts)) * (wrap(m - self.prev) / ts - self.w)
        self.prev = m
        return np.array([wrap(m), self.w, round(phi / STEP_RAD) * STEP_RAD, plant.arm_rate])


class Balancer:
    """u = -K x. Gives up beyond `guard`."""

    def __init__(self, K, guard_deg=25.0):
        self.K, self.guard = np.asarray(K, float), math.radians(guard_deg)
        self.status = "OK"

    def reset(self):
        self.status = "OK"

    def __call__(self, z, t):
        if abs(z[0]) > self.guard:
            self.status = "FALLEN"
        return 0.0 if self.status != "OK" else float(-self.K @ z)


class SwingUp:
    """Same state machine as swing-up/swingup_control.py.

    swing   : +pump 0.15 s, -pump 0.15 s, then
              u = pump * clip(kE (E_target - E), 0, 1) * sign(theta' cos theta) - kda phi' - kpa phi
    catch   : |theta| < catch, |theta'| < catch_w, |phi| < catch_arm -> 4-state balancer,
              arm reference = arm angle at the catch, decaying with 3 s
    balance : slow upright-offset adaptation; beyond `fall` back to swing
    """

    def __init__(self, K, **kw):
        self.K = np.asarray(K, float)
        self.p = dict(SWING, **kw)
        self.gamma = abs(self.K[2]) / (self.K[0] * self.p["bias_tau"]) if self.p["bias_tau"] > 0 else 0.0
        self.reset()

    def reset(self, offset_guess=0.0):
        self.mode, self.phi_ref, self.b_hat = "swing", 0.0, offset_guess
        self.t_start = None
        self.t_catch, self.catches, self.falls = None, 0, 0
        self.E = float("nan")

    def __call__(self, z, t):
        p, ts = self.p, 1.0 / CONTROL_HZ
        if self.t_start is None:
            self.t_start = t
        tr = t - self.t_start
        th_raw, w, phi, phid = z
        th = wrap(th_raw - self.b_hat)
        self.E = energy(th, w)
        if (self.mode == "swing" and tr > 0.3 and abs(math.degrees(th)) < p["catch"]
                and abs(w) < p["catch_w"] and abs(math.degrees(phi)) < p["catch_arm"]):
            self.mode, self.phi_ref = "balance", phi
            self.catches += 1
            if self.t_catch is None:
                self.t_catch = tr
        elif self.mode == "balance" and abs(math.degrees(th)) > p["fall"]:
            self.mode = "swing"
            self.falls += 1
        if self.mode == "swing":
            if tr < 0.15:
                return p["pump"]
            if tr < 0.30:
                return -p["pump"]
            scale = min(1.0, max(0.0, p["ke"] * (p["etarget"] - self.E)))
            sgn = 1.0 if w * math.cos(th) > 0 else -1.0
            return p["pump"] * scale * sgn - p["kda"] * phid - p["kpa"] * phi
        self.phi_ref *= math.exp(-ts / 3.0)
        self.b_hat += self.gamma * (phi - self.phi_ref) * ts
        self.b_hat = max(-math.radians(5), min(math.radians(5), self.b_hat))
        return float(-self.K @ np.array([th, w, phi - self.phi_ref, phid]))


class Loop:
    """Plant at 1 kHz, controller at 100 Hz, each command applied DELAY after its reading."""

    def __init__(self, plant, controller, sensor=None, arm_limit_deg=140.0, offset=0.0):
        self.plant, self.ctrl = plant, controller
        self.sensor = sensor or Sensor()
        self.arm_limit = math.radians(arm_limit_deg)
        self.offset = offset                        # encoder zero error: true vertical reads as this
        self.on = True
        self.reset()

    def reset(self):
        self.next_ctrl = 0.0
        self.queue = deque()                        # (apply time, command)
        self.u = 0.0
        self.stop = ""
        self.sensor.reset()

    def _check_stop(self):
        pl = self.plant
        if pl.latched:
            self.stop = "FIRMWARE LATCH (150 deg)"
        elif abs(pl.x[2]) > self.arm_limit:
            self.stop = f"ARM LIMIT ({math.degrees(self.arm_limit):.0f} deg)"
        if self.stop:                               # host sends ZERO_VELOCITY + HARD_STOP
            self.queue.clear()
            pl.a, pl.x[3] = 0.0, 0.0

    def advance(self, n):
        pl = self.plant
        for _ in range(n):
            if pl.t >= self.next_ctrl - 1e-9:
                self.next_ctrl += 1.0 / CONTROL_HZ
                if not self.stop:
                    self._check_stop()
                z = self.sensor.read(pl, self.offset)
                u = self.ctrl(z, pl.t) if (self.on and not self.stop) else 0.0
                self.u = max(-ACCEL_MAX, min(ACCEL_MAX, u))
                if not self.stop:
                    lag = DELAY if self.sensor.realistic else 0.0
                    self.queue.append((pl.t + lag, self.u))
            while self.queue and pl.t >= self.queue[0][0] - 1e-9:
                pl.a = self.queue.popleft()[1]
            pl.step()
