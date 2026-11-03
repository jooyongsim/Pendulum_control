"""PILCO on the real STEVAL-EDUKIT01 (SwingupController.ino): learn swing-up + balance from a few real trials.

Algorithm 1 of Deisenroth & Rasmussen (ICML 2011), with the rig as the system:
    trial 0      random actions (gentle arm centring added) for --horizon s     -> data
    repeat       fit the GP dynamics model on ALL real data           (simulator/pilco_core.py)
                 optimise the RBF policy on the GP by moment matching  (no simulator involved)
                 run the policy on the rig for --horizon s             -> data
The policy runs at 1/--dt Hz (default 20 Hz, held over the 100 Hz serial loop), state x = [theta', phi, phi', theta],
theta from the calibrated vertical (0 = upright), u = arm acceleration, |u| <= 47.1 rad/s^2 (firmware limit).

Same start-up, serial link, estimator and guards as swingup_control.py / swingup_rl_control.py (imported, not
modified). Between trials the arm is braked and brought home, the GP and policy are trained on this PC (needs torch,
a few minutes), then the script waits for the rod to hang still. Everything is saved in data/pilco_<time>/ after
every trial, so a run can be continued with --resume.

    python swing-up/swingup_pilco_control.py                              # learn on the rig (asks before each trial)
    python swing-up/swingup_pilco_control.py --resume data/pilco_2026...  # continue a learning run
    python swing-up/swingup_pilco_control.py --run simulator/pilco_results/pilco_policy.npz   # run a learned policy
    python swing-up/swingup_pilco_control.py --sim --yes --trials 3      # no board: the same script on the rig model
"""
import argparse
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "simulator"))
import balance_control as bc            # noqa: E402  serial link, estimator, constants (not modified)
import swingup_control as sc            # noqa: E402  home_arm, calibrate_centre, wait_still (not modified)

ROOT = bc.ROOT
U_MAX = bc.ACCEL_MAX_PPS2 * bc.STEP_RAD                # 47.12 rad/s^2


def policy_from_npz(path):
    """torch-free RBF policy (same as pilco_core.policy_numpy)"""
    P = np.load(path)
    c, w, ell, umax = P["c"], P["w"], P["ell"], float(P["u_max"])

    def pi(x):
        p = np.array([x[0], x[1], x[2], math.sin(x[3]), math.cos(x[3])])
        return umax * math.sin(float(w @ np.exp(-0.5 * (((p - c) / ell) ** 2).sum(1))))
    return pi, (float(P["dt"]) if "dt" in P else 0.05)


def random_policy(rng):
    """trial 0: random arm accelerations (PILCO Alg. 1, step 1) with a gentle pull back to the centre"""
    return lambda x: rng.uniform(-0.6, 0.6) * U_MAX - 6.0 * x[1] - 3.0 * x[2]


# ------------------------------------------------------------------ the rig
def startup(link, args, period):
    """driver check, arm home, wait still, SET_HOME, upright calibration (as swingup_rl_control.py) -> offset [rad]"""
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
        arm1 = sc.home_arm(link, period)
        print(f"   arm now {arm1:+.2f} deg")
        if abs(arm1) > 5.0:
            raise SystemExit("the arm did not come home (past the 150 deg latch?): turn it back to the centre by hand, "
                             "reset the board with the rod hanging still, and start again")
    if not sc.wait_still(link):
        raise SystemExit("pendulum did not settle within 60 s")
    link.step(bc.CMD_SET_HOME)
    q = link.step(bc.CMD_QUERY)
    hang = math.degrees(bc.wrap_rad(math.radians(q[3][bc.OBS_PEND])))
    if abs(hang) > 5:
        raise SystemExit(f"pendulum reads {hang:+.1f} deg from hanging: let it hang still and reset the board")
    print(f"driver OK (0x{drv:04X}), pendulum {hang:+.1f} deg from hanging")
    offset = args.upright_offset
    if args.calibrate:
        c, n = sc.calibrate_centre(link, period)
        if c is None:
            raise SystemExit(f"calibration swing too small ({n} windows) -- rerun")
        offset = c
        print(f"calibrated vertical: {offset:+.2f} deg from the encoder zero ({n} half-period windows)")
        if abs(offset) > 5.0:
            raise SystemExit("offset larger than the stiction band -- check the encoder zero")
    return math.radians(offset)


def brake_and_home(link, args, period):
    """stop the arm with the full deceleration, then bring it back to its zero slowly"""
    if args.sim:
        link.plant.reset(math.pi + np.random.uniform(-0.03, 0.03))   # rig model: next trial from hanging still
        link.home = 0.0
        return
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 1.0:
        r = link.step(bc.CMD_QUERY)
        if not r:
            continue
        v = r[3][bc.OBS_MEAS_PPS]
        if abs(v) < 60:
            break
        link.step(bc.CMD_SET_ACCEL, -math.copysign(bc.ACCEL_MAX_PPS2, v))
        time.sleep(period)
    link.step(bc.CMD_ZERO_VEL)
    print(f"   arm home: {sc.home_arm(link, period):+.1f} deg")


def wait_ready(link, args):
    if args.sim:
        return
    print("   waiting for the rod to hang still ...")
    if not sc.wait_still(link, timeout=120.0):
        print("   !! still swinging after 120 s -- starting anyway")


def run_trial(link, pi, seconds, dt, args, offset, period):
    """one trial at 100 Hz, the policy every dt. Returns X (T+1, 4), U (T), the 100 Hz log and the stop reason."""
    hold = max(1, int(round(dt / period)))
    est = bc.Estimator(180.0)
    Xs, Us, rows, reason, u = [], [], [], "duration", 0.0
    stuck_t, stuck_arm = None, None
    t0 = time.perf_counter()
    k, last = 0, link.step(bc.CMD_QUERY)
    while True:
        t = (k * period) if args.sim else (time.perf_counter() - t0)
        if t >= seconds:
            break
        x = est.update(last[3], t) if last else None
        if x is not None and k % hold == 0:
            th = bc.wrap_rad(x[0] - offset)
            s = np.array([x[1], x[2], x[3], th])            # PILCO state [theta', phi, phi', theta]
            u = float(np.clip(pi(s), -U_MAX, U_MAX))
            Xs.append(s); Us.append(u)
        pps2 = float(np.clip(u * bc.PPS2_PER_RAD, -bc.ACCEL_MAX_PPS2, bc.ACCEL_MAX_PPS2))
        tq = time.perf_counter()
        resp = link.step(bc.CMD_SET_ACCEL, pps2)
        rtt = (time.perf_counter() - tq) * 1000
        if resp is not None:
            last = resp
            st, _, latched, obs = resp
            if x is not None:
                rows.append((t, math.degrees(bc.wrap_rad(x[0] - offset)), math.degrees(x[1]), obs[bc.OBS_ROTOR],
                             math.degrees(x[3]), u, pps2, rtt, st, int(latched)))
            if latched:
                reason = f"firmware latched (status {st})"; break
            if abs(obs[bc.OBS_ROTOR]) > args.arm_limit:
                reason = f"arm {obs[bc.OBS_ROTOR]:.1f} deg"; break
            if abs(u) > 10.0:                               # a large command must move the arm within 0.3 s
                if stuck_t is None or abs(obs[bc.OBS_ROTOR] - stuck_arm) > 0.5:
                    stuck_t, stuck_arm = t, obs[bc.OBS_ROTOR]
                elif t - stuck_t > 0.3:
                    reason = "arm not following -- motor driver/power?"; break
            else:
                stuck_t = None
        k += 1
        if not args.sim:
            deadline = t0 + k * period
            while time.perf_counter() < deadline:
                time.sleep(max(0.0, min(0.002, deadline - time.perf_counter())))
    link.step(bc.CMD_SET_ACCEL, 0.0)
    X = np.array(Xs) if Xs else np.zeros((0, 4))
    if len(X):
        X[:, 3] = np.unwrap(X[:, 3])
    return X, np.array(Us[:-1]), np.array(rows), reason


def summary(X, dt):
    th = np.degrees(bc.wrap_rad(X[:, 3]))
    up = np.abs(th) < 10
    t_up = f"{np.argmax(up) * dt:.2f} s" if up.any() else "never"
    n1 = max(1, int(round(1.0 / dt)))
    return (f"first upright {t_up}, upright in the last 1 s {100 * up[-n1:].mean():.0f}%, "
            f"arm max {np.degrees(np.abs(X[:, 1])).max():.0f} deg")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="run this learned policy (.npz) instead of learning")
    ap.add_argument("--duration", type=float, default=20.0, help="--run: s")
    ap.add_argument("--trials", type=int, default=15, help="PILCO trials after the random one")
    ap.add_argument("--horizon", type=float, default=3.0, help="trial length = planning horizon [s]")
    ap.add_argument("--dt", type=float, default=0.05, help="policy period [s]")
    ap.add_argument("--n-basis", type=int, default=50)
    ap.add_argument("--gp-max-points", type=int, default=250)
    ap.add_argument("--policy-iters", type=int, default=100)
    ap.add_argument("--sigma-c", type=float, default=0.5)
    ap.add_argument("--resume", default=None, help="continue a learning run from this data/pilco_* folder")
    ap.add_argument("--rate", type=float, default=100.0, help="serial loop rate [Hz]")
    ap.add_argument("--arm-limit", type=float, default=120.0, help="deg (stop the trial beyond this)")
    ap.add_argument("--calibrate", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--upright-offset", type=float, default=0.0, help="deg, with --no-calibrate")
    ap.add_argument("--port", default=None)
    ap.add_argument("--yes", action="store_true", help="do not ask before each trial")
    ap.add_argument("--sim", action="store_true", help="no board: the rig model answers (swingup_rl_control.SimLink)")
    ap.add_argument("--sim-b", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    period = 1.0 / args.rate

    if args.sim:
        from swingup_rl_control import SimLink
        link = SimLink(tilt_deg=0.0, b=args.sim_b)
        print("SIMULATION: the board is replaced by rotary_model (4 ms delay, 0.3 deg encoder, firmware limits)")
    else:
        link = bc.Link(args.port)
    try:
        offset = 0.0 if args.sim else startup(link, args, period)

        if args.run:                                       # ---- run a learned policy
            pi, dt = policy_from_npz(args.run)
            print(f"policy {args.run} (RBF, {1 / dt:.0f} Hz)")
            if not args.yes and not args.sim:
                input(f"The policy will swing the rod up; the arm may move up to ~{args.arm_limit:.0f} deg. Enter ... ")
            X, U, rows, reason = run_trial(link, pi, args.duration, dt, args, offset, period)
            brake_and_home(link, args, period)
            print(f"stopped: {reason} | {summary(X, dt)}")
            out = ROOT / "data" / f"pilco_run_{datetime.now():%Y%m%d_%H%M%S}{'_sim' if args.sim else ''}.npz"
            out.parent.mkdir(exist_ok=True)
            np.savez(out, X=X, U=U, rows=rows)
            print("data ->", out)
            return

        # ---- learn (PILCO, Alg. 1)
        import pilco_core as pc                            # needs torch
        import torch
        rng = np.random.default_rng(args.seed)
        torch.manual_seed(args.seed)
        out = Path(args.resume) if args.resume else \
            ROOT / "data" / f"pilco_{datetime.now():%Y%m%d_%H%M%S}{'_sim' if args.sim else ''}"
        out.mkdir(parents=True, exist_ok=True)
        files = sorted(out.glob("trial_*.npz"))
        X_data, Y_data = [], []
        for f in files:
            d = np.load(f)
            inp, tgt = pc.transitions(d["X"], d["U"])
            X_data.append(inp); Y_data.append(tgt)
        print(f"output -> {out}  ({len(files)} earlier trials loaded)")
        H = int(round(args.horizon / args.dt))
        cost = pc.Cost(sigma_c=args.sigma_c, arm_lim=math.radians(args.arm_limit))
        m0 = np.array([0.0, 0.0, 0.0, math.pi])
        S0 = np.diag([0.05, 0.05, 0.01, 0.03]) ** 2
        gp = pc.GPModel(max_points=args.gp_max_points, seed=args.seed)
        policy = pc.RBFPolicy(n_basis=args.n_basis, u_max=U_MAX, seed=args.seed)
        if files and (out / "policy_last.pt").exists():
            policy.load_state_dict(torch.load(out / "policy_last.pt"))

        for k in range(len(files), args.trials + 1):
            if k == 0:
                pi, kind = random_policy(rng), "random"
            else:
                t0 = time.time()
                print(f"trial {k}: fitting the GP on {sum(len(x) for x in X_data)} transitions ...", flush=True)
                gp.fit(np.concatenate(X_data), np.concatenate(Y_data), iters=100)
                print("   optimising the policy on the GP ...", flush=True)
                J = pc.optimize_policy(policy, gp, cost, m0, S0, H, iters=args.policy_iters, verbose=True)
                torch.save(policy.state_dict(), out / "policy_last.pt")
                np.savez(out / f"policy_{k:02d}.npz", **policy.numpy_params(), dt=args.dt)
                print(f"   predicted J = {J:.2f} ({time.time() - t0:.0f} s of training)")
                pi, kind = pc.policy_numpy(policy.numpy_params()), "pilco"
            wait_ready(link, args)
            if not args.yes:
                input(f"trial {k} ({kind}, {args.horizon:.1f} s): clear the area and press Enter ... ")
            X, U, rows, reason = run_trial(link, pi, args.horizon, args.dt, args, offset, period)
            brake_and_home(link, args, period)
            np.savez(out / f"trial_{k:02d}.npz", X=X, U=U, rows=rows, kind=kind, reason=reason)
            if len(U):
                inp, tgt = pc.transitions(X, U)
                X_data.append(inp); Y_data.append(tgt)
            print(f"trial {k} ({kind}): stopped by {reason} | {summary(X, args.dt)}", flush=True)
        np.savez(out / "pilco_policy.npz", **policy.numpy_params(), dt=args.dt)
        print(f"done -> {out / 'pilco_policy.npz'}  (run it: --run {out / 'pilco_policy.npz'})")
    except KeyboardInterrupt:
        print("\nCtrl+C")
    finally:
        link.close()


if __name__ == "__main__":
    main()
