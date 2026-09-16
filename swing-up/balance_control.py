"""Acceleration control on the real STEVAL-EDUKIT01 (SwingupController.ino firmware).

Modes (each logs a CSV to data/):
  damp     Pendulum HANGING. The arm adds swing damping: phi'' = (k/b) theta_h' (hanging model
           theta_h'' = -wn^2 theta_h - 2 zeta wn theta_h' - b phi''), plus a weak arm-centering loop.
           With --excite the arm first swings the pendulum with the same doublet as the open-loop
           test, so the decay can be compared with the free decay. Safe first closed-loop check.
  balance  Pendulum UPRIGHT. 4-state pole placement on the slide model
               theta'' = (3g/2l) sin(theta) + b cos(theta) phi'' - 2 zeta wn theta'
           Lift the rod upright by hand; control engages once |theta| < --engage deg for 0.2 s,
           then let go.

Identified: wn = 7.499 rad/s, zeta = 0.01172 (free swing); b = 3r/2l = 0.730 -> r = 0.127 m and
the sign (same as Lec01) from the open-loop test on our unit.
Units: u [rad/s^2] -> pps^2 = u * 3200 / (2 pi), clamped to the firmware's 24000 pps^2.
Guards: |theta| > --guard (balance), |phi| > --arm-limit, firmware latch -> zero velocity + stop.
Ctrl+C always stops the motor.

    python3 swing-up/balance_control.py --mode damp --excite
    python3 swing-up/balance_control.py --mode balance
"""
import argparse
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from control_comms import ControlComms, StatusCode, DebugLevel, find_board_port  # noqa: E402

CMD_SET_HOME, CMD_SET_ACCEL, CMD_HARD_STOP, CMD_QUERY, CMD_RESET_SAFETY, CMD_ZERO_VEL = 0, 1, 2, 3, 4, 5
OBS_PEND, OBS_ROTOR, OBS_MEAS_PPS = 0, 1, 2
STEP_RAD = 2 * math.pi / 3200
PPS2_PER_RAD = 1.0 / STEP_RAD
ACCEL_MAX_PPS2 = 24000.0     # must match MAX_ACCEL_PPS2 in SwingupController.ino

G, WN, ZETA = 9.81, 7.499, 0.01172
L = 3 * G / (2 * WN ** 2)
A_G, C_D = WN ** 2, 2 * ZETA * WN
ENC_RAD = math.radians(0.3)


def wrap_rad(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def place4(b, poles):
    A = np.array([[0, 1, 0, 0], [A_G, -C_D, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]], float)
    B = np.array([[0], [b], [0], [1]], float)
    ctrb = np.hstack([np.linalg.matrix_power(A, i) @ B for i in range(4)])
    if abs(np.linalg.det(ctrb)) < 1e-9:
        raise ValueError("not controllable")
    phi = np.zeros_like(A)
    for c in np.real(np.poly(poles)):
        phi = phi @ A + c * np.eye(4)
    return np.linalg.solve(ctrb.T, np.eye(4)[:, -1]) @ phi


class Link:
    def __init__(self, port):
        self.port = find_board_port(port)
        self.c = ControlComms(timeout=0.2, debug_level=DebugLevel.DEBUG_ERROR)
        if self.c.connect(self.port, 500000) is not StatusCode.OK:
            raise SystemExit(f"could not open {self.port}")
        time.sleep(2.0)

    def step(self, cmd, val=0.0):
        return self.c.step(cmd, [float(val)])

    def stop(self):
        for cmd in (CMD_SET_ACCEL, CMD_ZERO_VEL, CMD_HARD_STOP):
            try:
                self.c.step(cmd, [0.0])
            except Exception:
                pass

    def close(self):
        self.stop()
        self.c.close()


class Estimator:
    """theta from the encoder (wrapped), theta' by filtered difference; phi, phi' from the firmware."""

    def __init__(self, zero_deg, tau=0.01):
        self.zero = math.radians(zero_deg)
        self.tau, self.prev, self.w, self.prev_t = tau, None, 0.0, None

    def update(self, obs, t):
        th = wrap_rad(math.radians(obs[OBS_PEND]) - self.zero)
        if self.prev is not None and t > self.prev_t:
            dt = t - self.prev_t
            raw = wrap_rad(th - self.prev) / dt
            self.w += (dt / (self.tau + dt)) * (raw - self.w)
        self.prev, self.prev_t = th, t
        return np.array([th, self.w, math.radians(obs[OBS_ROTOR]), obs[OBS_MEAS_PPS] * STEP_RAD])


def doublet(t, a, T):
    return a if t < T else (-a if t < 3 * T else (a if t < 4 * T else 0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("damp", "balance"), required=True)
    ap.add_argument("--duration", type=float, default=8.0, help="s of control")
    ap.add_argument("--rate", type=float, default=100.0)
    ap.add_argument("--b", type=float, default=0.730, help="3r/2l from the open-loop test")
    ap.add_argument("--poles", default="-8+8j,-8-8j,-2+1j,-2-1j", help="balance: closed-loop poles")
    ap.add_argument("--kdamp", type=float, default=1.5, help="damp: added damping [1/s]")
    ap.add_argument("--excite", action="store_true", help="damp: swing the pendulum first")
    ap.add_argument("--engage", type=float, default=3.0, help="balance: engage below this tilt [deg]")
    ap.add_argument("--guard", type=float, default=25.0, help="balance: give up beyond this tilt [deg]")
    ap.add_argument("--arm-limit", type=float, default=80.0, help="deg")
    ap.add_argument("--port", default=None)
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    poles = [complex(p.replace(" ", "")) for p in args.poles.split(",")]
    K = place4(args.b, poles)
    print(f"mode={args.mode}  b={args.b}  (r={2*L*args.b/3*1000:.0f} mm)  rate={args.rate:.0f} Hz")
    if args.mode == "balance":
        print(f"poles {poles}\nK = [{', '.join(f'{k:.3f}' for k in K)}]   u = -K [theta, theta', phi, phi']")
    else:
        print(f"damping +{args.kdamp} 1/s on top of the natural {C_D:.3f} 1/s")

    link = Link(args.port)
    rows, dropped, reason = [], 0, "duration"
    period = 1.0 / args.rate
    try:
        link.step(CMD_HARD_STOP)
        link.step(CMD_SET_HOME)
        link.step(CMD_RESET_SAFETY)
        time.sleep(0.1)
        link.step(CMD_RESET_SAFETY)     # second read clears the UVLO flag latched at motor power-on
        q = link.step(CMD_QUERY)
        if q is None or len(q[3]) != 7:
            raise SystemExit("no/odd reply -- is SwingupController.ino on the board?")
        # RESET_SAFETY refreshed the L6474 status. 0xFFFF/0x0000 means the driver is not answering
        # on SPI (motor supply off or browned out); the arm would not move at all.
        drv = int(q[3][5])
        healthy = all(drv & m for m in (0x0200, 0x0800, 0x1000))   # UVLO, TH_SD, OCD are active-low
        if drv in (0xFFFF, 0x0000) or not healthy:
            raise SystemExit(f"L6474 driver not healthy (status 0x{drv:04X}): check the motor power supply, "
                             "then reset the board with the pendulum hanging still")
        print(f"driver OK (L6474 0x{drv:04X})")
        zero = 0.0 if args.mode == "damp" else 180.0
        est = Estimator(zero)

        if args.mode == "damp":
            if abs(math.degrees(wrap_rad(math.radians(q[3][OBS_PEND])))) > 5:
                raise SystemExit("encoder zero is not at hanging: reset the board with the pendulum hanging still")
            if not args.yes:
                input("Pendulum hanging, arm clear. Enter to start ... ")
        else:
            print(f"\nLift the rod upright. Control engages when |theta| < {args.engage} deg for 0.2 s "
                  "(Ctrl+C to quit).")
            held, t_wait = 0, time.perf_counter()
            while True:
                r = link.step(CMD_QUERY)
                if r:
                    th = math.degrees(wrap_rad(math.radians(r[3][OBS_PEND]) - math.pi))
                    held = held + 1 if abs(th) < args.engage else 0
                    if time.perf_counter() - t_wait > 0.5:
                        print(f"\r   theta from upright {th:+7.2f} deg   ", end="", flush=True); t_wait = time.perf_counter()
                    if held >= int(0.2 * args.rate):
                        print("\n   ENGAGED -- let go"); break
                time.sleep(period)

        excite_T = 0.25 if (args.mode == "damp" and args.excite) else 0.0
        excite_end = 4 * excite_T
        t0 = time.perf_counter()
        k, zeroed, active = 0, excite_T == 0.0, excite_T == 0.0
        stuck_since, stuck_phi = None, 0.0
        while True:
            t = time.perf_counter() - t0
            if t >= excite_end + args.duration:
                break
            r_obs = link.step(CMD_QUERY) if k == 0 else last
            # control uses the observation returned by the previous command (one sample old at most)
            x = est.update(r_obs[3], t) if r_obs else None
            if x is None:
                dropped += 1; u = 0.0
            elif t < excite_end:
                u = doublet(t, 2000.0, excite_T) * STEP_RAD
            elif args.mode == "damp":
                u = (args.kdamp / args.b) * x[1] - 4.0 * x[2] - 4.0 * x[3]
            else:
                u = float(-K @ x)
            pps2 = float(np.clip(u * PPS2_PER_RAD, -ACCEL_MAX_PPS2, ACCEL_MAX_PPS2))
            tq = time.perf_counter()
            if t >= excite_end and not zeroed:
                resp = link.step(CMD_ZERO_VEL); zeroed = True
            else:
                resp = link.step(CMD_SET_ACCEL, pps2)
            rtt = (time.perf_counter() - tq) * 1000
            if resp is None:
                dropped += 1; last = r_obs
            else:
                last = resp
                st, _, latched, obs = resp
                if x is not None:
                    rows.append((t, int(t >= excite_end), math.degrees(x[0]), math.degrees(x[1]),
                                 obs[OBS_ROTOR], math.degrees(x[3]), u, pps2, rtt, st, int(latched)))
                if latched:
                    reason = f"firmware latched (status {st})"; break
                # Saturated commands must move the arm; if they do not for 0.3 s the driver is dead.
                if t >= excite_end and abs(pps2) >= ACCEL_MAX_PPS2:
                    if stuck_since is None:
                        stuck_since, stuck_phi = t, obs[OBS_ROTOR]
                    elif t - stuck_since > 0.3 and abs(obs[OBS_ROTOR] - stuck_phi) < 0.5:
                        reason = "arm not following saturated commands -- motor driver/power?"; break
                else:
                    stuck_since = None
                if abs(obs[OBS_ROTOR]) > args.arm_limit:
                    reason = f"arm {obs[OBS_ROTOR]:.1f} deg"; break
                limit = args.guard if args.mode == "balance" else 40.0   # damp: catch a wrong sign
                if x is not None and abs(math.degrees(x[0])) > limit:
                    reason = f"pendulum {math.degrees(x[0]):.1f} deg"; break
            k += 1
            deadline = t0 + k * period
            while time.perf_counter() < deadline:
                time.sleep(max(0.0, min(0.002, deadline - time.perf_counter())))
    except KeyboardInterrupt:
        reason = "Ctrl+C"
    finally:
        link.close()

    print(f"\nstopped: {reason} | samples {len(rows)} | dropped {dropped}")
    if not rows:
        return
    d = np.array(rows)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv = ROOT / "data" / f"control_{args.mode}_{stamp}.csv"
    np.savetxt(csv, d, delimiter=",", comments="",
               header="t_s,active,theta_deg,theta_dot_dps,phi_deg,phi_dot_dps,u_rad_s2,cmd_pps2,rtt_ms,status,latched")
    print(f"CSV -> {csv}")
    dt = np.diff(d[:, 0]) * 1000
    print(f"loop: median {np.median(dt):.2f} ms, max {dt.max():.2f} ms | rtt median {np.median(d[:, 8]):.2f} ms")
    act = d[d[:, 1] == 1]
    if len(act):
        sat = np.mean(np.abs(act[:, 7]) >= ACCEL_MAX_PPS2) * 100
        print(f"control: |theta| max {np.abs(act[:, 2]).max():.2f} deg, last 2 s rms "
              f"{np.sqrt(np.mean(act[act[:, 0] > act[-1, 0] - 2, 2] ** 2)):.2f} deg | "
              f"|phi| max {np.abs(act[:, 4]).max():.1f} deg | saturated {sat:.1f}%")


if __name__ == "__main__":
    main()
