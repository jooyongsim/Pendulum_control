"""Swing-up + catch on the real STEVAL-EDUKIT01 (SwingupController.ino).

Separate from balance_control.py, which is only imported for its serial link, state estimator
and pole-placement helper -- that file is not modified.

Start with the pendulum HANGING still (board reset while it hung: encoder 0 = hanging).

  swing    zero-mean kick (+pump, -pump), then near-bang-bang energy pumping
               E = 1/2 theta'^2 + (3g/2l)(cos theta - 1)      (upright at rest = 0, hanging = -112)
               u = pump * clip(kE (E_target - E), 0, 1) * sign(theta' cos theta) - kda phi' - kpa phi
           dE/dt = b theta' cos(theta) u, so the sign term always adds energy; the arm-velocity
           damping keeps the zero-mean pumping from walking the arm into its limit.
  balance  when |theta| < catch deg and |theta'| < catch_w rad/s: the 4-state balancer
           (same poles as balance_control), arm reference = arm angle at the catch, decaying to 0.
           Beyond 30 deg it falls back to swing.

Designed in a firmware-aware simulation (30 pps floor, step-edge rate updates, 4 ms delay, 0.3 deg
encoder, Coulomb friction, centrifugal term). Defaults pump 30, kpa 3, kda 5 succeeded in 23/24 randomised
cases (b 0.8-1.2x, Coulomb 0.5-2x, delay 4-12 ms, start tilt +-2.5 deg). Earlier: pump 35, kpa 1, kda 5 caught in ~1.8 s with the arm
<= 69 deg, and still caught (arm <= 98 deg) with b -20%/+20% and doubled Coulomb friction. The gentler
pump 25 / kpa 3 kept the arm under 37 deg nominally but ran into the arm limit when b was 20% low.

Upright reference: the encoder zero is wherever the rod rested at board reset, anywhere inside the
+-2 deg stiction band, and a 1.6 deg error parks the arm ~90 deg away (k1/|k3| = 57 deg per deg).
So before swinging (--calibrate, default) the arm gives the hanging rod a small doublet and the
centre of the free swing is taken as the true vertical; while balancing, a slow adaptation
(--bias-tau) removes the rest: an arm pushed off its reference means the upright estimate is off.

Guards: |phi| > --arm-limit, large command with a motionless arm, firmware latch -> stop. Ctrl+C stops.

    python3 swing-up/swingup_control.py
"""
import argparse
import math
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import balance_control as bc  # noqa: E402  (reused, not modified)

ROOT = bc.ROOT


def home_arm(link, period, timeout=8.0):
    """Bring the arm back to the current firmware zero before re-homing. Every script calls
    SET_HOME at start, so starting with the arm left off-centre by a previous run would move the
    zero -- and the arm limit -- away from the cable's neutral position."""
    t0, k = time.perf_counter(), 0
    while time.perf_counter() - t0 < timeout:
        r = link.step(bc.CMD_QUERY)
        if r:
            phi, phid = math.radians(r[3][bc.OBS_ROTOR]), r[3][bc.OBS_MEAS_PPS] * bc.STEP_RAD
            if abs(math.degrees(phi)) < 0.5 and abs(phid) < 0.05:
                break
            u = max(-6.0, min(6.0, -4.0 * phi - 4.0 * phid))
            link.step(bc.CMD_SET_ACCEL, u * bc.PPS2_PER_RAD)
        k += 1
        while time.perf_counter() < t0 + k * period:
            time.sleep(0.001)
    link.step(bc.CMD_ZERO_VEL)
    r = link.step(bc.CMD_QUERY)
    return r[3][bc.OBS_ROTOR] if r else float('nan')


def calibrate_centre(link, period, accel_pps2=3000.0, T=0.2, watch=3.5):
    """Small arm doublet (ends at rest where it started), then the centre of the free swing."""
    T0 = 2 * math.pi / bc.WN
    t0, k, zeroed, rows = time.perf_counter(), 0, False, []
    while True:
        t = time.perf_counter() - t0
        if t >= 4 * T + watch:
            break
        if t < 4 * T:
            a = accel_pps2 if t < T else (-accel_pps2 if t < 3 * T else accel_pps2)
            r = link.step(bc.CMD_SET_ACCEL, a)
        elif not zeroed:
            r = link.step(bc.CMD_ZERO_VEL); zeroed = True
        else:
            r = link.step(bc.CMD_QUERY)
        if r:
            rows.append((t, math.degrees(bc.wrap_rad(math.radians(r[3][bc.OBS_PEND])))))
        k += 1
        while time.perf_counter() < t0 + k * period:
            time.sleep(0.001)
    link.step(bc.CMD_ZERO_VEL)
    d = np.array(rows)
    t, x = d[:, 0], d[:, 1]
    mids, a = [], 4 * T + 0.2
    while a + T0 <= t[-1]:
        w = (t >= a) & (t < a + T0)
        if w.sum() > 20 and x[w].max() - x[w].min() > 3.0:
            mids.append((x[w].max() + x[w].min()) / 2)
        a += T0 / 2
    if len(mids) < 3:
        return None, len(mids)
    return float(np.median(mids)), len(mids)


def wait_still(link, timeout=60.0):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        vals, t1 = [], time.perf_counter()
        while time.perf_counter() - t1 < 1.0:
            r = link.step(bc.CMD_QUERY)
            if r:
                vals.append(math.degrees(bc.wrap_rad(math.radians(r[3][bc.OBS_PEND]))))
            time.sleep(0.02)
        if vals and max(vals) - min(vals) <= 0.6:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=25.0, help="s")
    ap.add_argument("--rate", type=float, default=100.0)
    ap.add_argument("--b", type=float, default=0.730, help="3r/2l")
    ap.add_argument("--poles", default="-6+6j,-6-6j,-1.5+1j,-1.5-1j", help="balancer poles (use --poles=...)")
    ap.add_argument("--pump", type=float, default=30.0, help="swing acceleration [rad/s^2]")
    ap.add_argument("--ke", type=float, default=0.3, help="energy error to pump scale")
    ap.add_argument("--etarget", type=float, default=1.0, help="target energy (upright = 0)")
    ap.add_argument("--kpa", type=float, default=3.0, help="arm position pull during swing")
    ap.add_argument("--kda", type=float, default=5.0, help="arm velocity damping during swing")
    ap.add_argument("--catch", type=float, default=12.0,
                    help="catch below this tilt [deg]; at 25 deg the balancer saturated for ~0.3 s")
    ap.add_argument("--catch-w", type=float, default=4.0, help="catch below this rate [rad/s]")
    ap.add_argument("--catch-arm", type=float, default=60.0,
                    help="only catch with the arm within this angle [deg]; a catch at -88 deg ran into the limit")
    ap.add_argument("--fall", type=float, default=30.0, help="balance -> swing beyond this tilt [deg]")
    ap.add_argument("--arm-limit", type=float, default=140.0, help="deg (cable checked to 145)")
    ap.add_argument("--calibrate", action=argparse.BooleanOptionalAction, default=True,
                    help="measure the true vertical from a free swing before starting")
    ap.add_argument("--upright-offset", type=float, default=0.0,
                    help="encoder offset of the true vertical [deg], used with --no-calibrate")
    ap.add_argument("--bias-tau", type=float, default=6.0, help="upright adaptation time constant [s], 0 = off")
    ap.add_argument("--port", default=None)
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    poles = [complex(p.replace(" ", "")) for p in args.poles.split(",")]
    K = bc.place4(args.b, poles)
    A_G = bc.A_G
    print(f"swing: pump {args.pump} rad/s^2, kE {args.ke}, kpa {args.kpa}, kda {args.kda}, "
          f"catch |theta|<{args.catch} deg & |theta'|<{args.catch_w} rad/s")
    print(f"balance: poles {poles}  K=[{', '.join(f'{k:.2f}' for k in K)}]")

    link = bc.Link(args.port)
    rows, dropped, reason = [], 0, "duration"
    period = 1.0 / args.rate
    try:
        link.step(bc.CMD_HARD_STOP)
        link.step(bc.CMD_RESET_SAFETY)
        time.sleep(0.1)
        link.step(bc.CMD_RESET_SAFETY)
        q = link.step(bc.CMD_QUERY)
        if q is None or len(q[3]) != 7:
            raise SystemExit("no/odd reply -- is SwingupController.ino on the board?")
        drv = int(q[3][5])
        if drv in (0xFFFF, 0x0000) or not all(drv & m for m in (0x0200, 0x0800, 0x1000)):
            raise SystemExit(f"L6474 driver not healthy (0x{drv:04X}): check motor power, reset the board")
        arm0 = q[3][bc.OBS_ROTOR]
        if abs(arm0) > 2.0:
            print(f"arm is at {arm0:+.1f} deg from its zero -- returning it before homing ...")
            print(f"   arm now {home_arm(link, period):+.2f} deg")
        if not wait_still(link):
            raise SystemExit("pendulum did not settle within 60 s")
        link.step(bc.CMD_SET_HOME)
        q = link.step(bc.CMD_QUERY)
        hang = math.degrees(bc.wrap_rad(math.radians(q[3][bc.OBS_PEND])))
        if abs(hang) > 5:
            raise SystemExit(f"pendulum reads {hang:+.1f} deg from hanging: let it hang still and reset the board")
        print(f"driver OK (0x{drv:04X}), pendulum {hang:+.1f} deg from hanging")
        offset = args.upright_offset
        if args.calibrate:
            c, n = calibrate_centre(link, period)
            if c is None:
                raise SystemExit(f"calibration swing too small ({n} windows) -- rerun")
            offset = c
            print(f"calibrated vertical: {offset:+.2f} deg from the encoder zero ({n} half-period windows)")
            if abs(offset) > 5.0:
                raise SystemExit("offset larger than the stiction band -- check the encoder zero")
            if not wait_still(link):
                raise SystemExit("pendulum did not settle after calibration")
        b_hat = math.radians(offset)
        gamma = (abs(K[2]) / (K[0] * args.bias_tau)) if args.bias_tau > 0 else 0.0
        if not args.yes:
            input("The rod will swing over the top and the arm will move up to ~100 deg (limit "
                  f"{args.arm_limit:.0f}). Clear the area, press Enter ... ")

        est = bc.Estimator(180.0)                       # theta from upright; hanging = +-180 deg
        mode, phi_ref, t_catch, reswings = "swing", 0.0, None, 0
        stuck = deque()                                 # (t, arm deg) while the command is large
        t0 = time.perf_counter()
        k, last = 0, link.step(bc.CMD_QUERY)
        while True:
            t = time.perf_counter() - t0
            if t >= args.duration:
                break
            x = est.update(last[3], t) if last else None
            if x is None:
                dropped += 1; u = 0.0; E = float("nan")
            else:
                th_raw, w, phi, phid = x
                th = bc.wrap_rad(th_raw - b_hat)            # tilt from the estimated true vertical
                E = 0.5 * w * w + A_G * (math.cos(th) - 1)
                if (mode == "swing" and t > 0.3 and abs(math.degrees(th)) < args.catch and abs(w) < args.catch_w
                        and abs(math.degrees(phi)) < args.catch_arm):
                    mode, phi_ref = "balance", phi
                    t_catch = t if t_catch is None else t_catch
                    print(f"   CATCH at {t:.2f} s (theta {math.degrees(th):+.1f} deg, arm {math.degrees(phi):+.1f} deg)")
                elif mode == "balance" and abs(math.degrees(th)) > args.fall:
                    mode = "swing"; reswings += 1
                    print(f"   fell at {t:.2f} s -> swinging again")
                if mode == "swing":
                    if t < 0.15:
                        u = args.pump
                    elif t < 0.30:
                        u = -args.pump
                    else:
                        s = w * math.cos(th)
                        scale = min(1.0, max(0.0, args.ke * (args.etarget - E)))
                        u = args.pump * scale * (1.0 if s > 0 else -1.0) - args.kda * phid - args.kpa * phi
                else:
                    phi_ref *= math.exp(-period / 3.0)
                    b_hat += gamma * (phi - phi_ref) * period
                    b_hat = max(-math.radians(5), min(math.radians(5), b_hat))
                    u = float(-K @ np.array([th, w, phi - phi_ref, phid]))
            pps2 = float(np.clip(u * bc.PPS2_PER_RAD, -bc.ACCEL_MAX_PPS2, bc.ACCEL_MAX_PPS2))
            tq = time.perf_counter()
            resp = link.step(bc.CMD_SET_ACCEL, pps2)
            rtt = (time.perf_counter() - tq) * 1000
            if resp is None:
                dropped += 1
            else:
                last = resp
                st, _, latched, obs = resp
                if x is not None:
                    rows.append((t, int(mode == "balance"), math.degrees(x[0] - b_hat), math.degrees(x[1]),
                                 obs[bc.OBS_ROTOR], math.degrees(x[3]), E, u, pps2, rtt, st, int(latched),
                                 math.degrees(b_hat)))
                if latched:
                    reason = f"firmware latched (status {st})"; break
                if abs(obs[bc.OBS_ROTOR]) > args.arm_limit:
                    reason = f"arm {obs[bc.OBS_ROTOR]:.1f} deg"; break
                # A large command must move the arm. Judge the arm's travel over the last 0.3 s,
                # not against where it was when the command first became large: during pumping
                # the arm swings out and back, and returning to the same angle is not a stall.
                if abs(u) > 10.0:
                    stuck.append((t, obs[bc.OBS_ROTOR]))
                    while stuck and t - stuck[0][0] > 0.3:
                        stuck.popleft()
                    arms = [a for _, a in stuck]
                    if t - stuck[0][0] >= 0.28 and max(arms) - min(arms) < 0.5:
                        reason = "arm not following -- motor driver/power?"; break
                else:
                    stuck.clear()
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
    csv = ROOT / "data" / f"swingup_{stamp}.csv"
    np.savetxt(csv, d, delimiter=",", comments="",
               header="t_s,balancing,theta_deg,theta_dot_dps,phi_deg,phi_dot_dps,energy,u_rad_s2,cmd_pps2,rtt_ms,status,latched,upright_offset_deg")
    print(f"CSV -> {csv}")
    bal = d[d[:, 1] == 1]
    print(f"catch: {'%.2f s' % t_catch if t_catch is not None else 'never'} | re-swings {reswings} | "
          f"arm max {np.abs(d[:, 4]).max():.1f} deg | max energy {np.nanmax(d[:, 6]):+.1f}")
    if len(bal):
        tail = bal[bal[:, 0] > bal[-1, 0] - 2]
        print(f"balancing: {len(bal)/args.rate:.1f} s total, last 2 s |theta| rms {np.sqrt(np.mean(tail[:, 2]**2)):.2f} deg, "
              f"arm now {d[-1, 4]:+.1f} deg")
    print(f"upright offset: start {d[0, 12]:+.2f} deg -> end {d[-1, 12]:+.2f} deg "
          f"(next run on the same board power-up: --no-calibrate --upright-offset {d[-1, 12]:.2f})")


if __name__ == "__main__":
    main()
