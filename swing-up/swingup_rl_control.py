"""Swing-up + balance on the real STEVAL-EDUKIT01 with a LEARNED policy (reinforcement / imitation learning).

RL version of swingup_control.py: the same start-up (driver check, arm homing, wait still, SET_HOME,
upright calibration from a free swing), the same serial link / estimator (balance_control.py), the same
guards and CSV log -- only the control law is replaced by a neural-network policy trained in
simulator/rl_tutorial_cartpole_rotary_v3 (rotary_model.py: identified plant + firmware limits).
swingup_control.py and balance_control.py are imported, not modified.

Policy (as in training):
    obs = [cos th, sin th, th', phi, phi']     th = tilt from the calibrated vertical (0 = upright)
    a   = policy(obs) in [-1, 1]  ->  u = a * 47.1 rad/s^2  (= a * 24000 pps^2, the firmware limit)
    recomputed every 20 ms (50 Hz, the training control period) and sent on every 100 Hz loop tick.
One policy does everything: pump energy, catch near the top, balance, and swing up again after a fall.
It is evaluated with numpy only (no torch), so it runs wherever swingup_control.py runs.

    --policy sac        SAC (default; the most robust on the rig model: 4-12 ms delay, b -20%)
                        weights: simulator/rl_models/rotary_swingup_sac.pt (already in the repository)
    --policy ppo / dqn / dagger / td3 / ...   see POLICIES below
    --model FILE --kind sac|ppo|dqn|bc|ars|gauss   any other saved model
Weights are looked up in simulator/rl_models/ (existing, in the repository) first, then in
simulator/rl_results/models/ (written by the v2/v3 tutorials: PPO, DAgger, TD3, ...).

Start with the pendulum HANGING still (board reset while it hung: encoder 0 = hanging).
Guards: |phi| > --arm-limit, large command with a motionless arm, firmware latch -> stop. Ctrl+C stops.

    python swing-up/swingup_rl_control.py                     # SAC, real board
    python swing-up/swingup_rl_control.py --policy ppo
    python swing-up/swingup_rl_control.py --sim --yes         # no board: same script on the rig model
"""
import argparse
import math
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "simulator"))
import balance_control as bc            # noqa: E402  serial link, estimator, constants (not modified)
import swingup_control as sc            # noqa: E402  home_arm, calibrate_centre, wait_still (not modified)
from rl_rotary_realtime import load_state_dict   # noqa: E402  .pt reader without torch

ROOT = bc.ROOT
# existing weights first: simulator/rl_models/ is in the repository (SAC, DQN, REINFORCE);
# rl_results/models/ is written by the v2/v3 tutorials (adds PPO, DAgger, TD3, ...)
MODEL_DIRS = [HERE.parent / "simulator" / "rl_models", HERE.parent / "simulator" / "rl_results" / "models"]
POLICY_HZ = 50.0                                     # training control period 20 ms
ACCEL_MAX = bc.ACCEL_MAX_PPS2 * bc.STEP_RAD          # 47.12 rad/s^2

# name -> (model file, kind, rig-model check from the tutorial (section 15): realistic sensor / 4 ms delay)
POLICIES = {
    "sac":         ("rotary_swingup_sac.pt", "sac", "ok: caught in 1.0 s; also ok with 12 ms delay and b -20%"),
    "ppo":         ("rotary_swingup_ppo.pt", "ppo", "ok: caught in 1.1 s; ok with 12 ms delay, FAILS with b -20%"),
    "dqn":         ("rotary_swingup_dqn.pt", "dqn", "ok: caught in 1.1 s, but uses the arm up to 135 deg; FAILS with 12 ms delay"),
    "dagger":      ("rotary_dagger.pt", "bc", "ok: caught in 2.6 s (imitates the energy controller); FAILS with b -20%"),
    "ppo_state":   ("rotary_swingup_ppo_state.pt", "ppo_state", "WARNING: did not balance on the rig model (30%)"),
    "td3":         ("rotary_swingup_td3.pt", "td3", "WARNING: did not balance on the rig model"),
    "bc":          ("rotary_bc.pt", "bc", "WARNING: hit the arm limit on the rig model"),
    "reinforce":   ("rotary_swingup_reinforce.pt", "gauss", "WARNING: does not catch (spins the rod)"),
    "ars":         ("rotary_swingup_ars.pt", "ars", "WARNING: linear policy, does not catch"),
}
DQN_LEVELS = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])


# ------------------------------------------------------------------ policy (numpy forward pass)
class MLP:
    """nn.Sequential of Linear/ReLU stored under `prefix` ('net.', 'pi.', 'mu.' or '')."""

    def __init__(self, sd, prefix):
        idx = sorted({int(k[len(prefix):].split(".")[0]) for k in sd
                      if k.startswith(prefix) and k[len(prefix):].split(".")[0].isdigit()})
        if not idx:
            raise ValueError(f"no layers with prefix '{prefix}' in the model")
        self.layers = [(np.asarray(sd[f"{prefix}{i}.weight"], np.float64), np.asarray(sd[f"{prefix}{i}.bias"], np.float64))
                       for i in idx]

    def __call__(self, x):
        h = np.asarray(x, np.float64)
        for i, (W, b) in enumerate(self.layers):
            h = W @ h + b
            if i < len(self.layers) - 1:
                h = np.maximum(h, 0.0)
        return h


def make_policy(path, kind):
    """obs (5,) -> a in [-1, 1]: the deterministic action the tutorial evaluates."""
    sd = load_state_dict(path)
    if kind in ("sac", "td3", "bc"):                  # tanh(first output); SAC actor = [mu, log_std]
        net = MLP(sd, "net.")
        return lambda o: float(np.tanh(net(o)[0]))
    if kind in ("ppo", "ppo_state"):                  # Gaussian mean (state sigma: [mu, log_std])
        net = MLP(sd, "pi.")
        return lambda o: float(np.clip(net(o)[0], -1, 1))
    if kind == "gauss":                               # REINFORCE Gaussian policy mean
        net = MLP(sd, "mu.")
        return lambda o: float(np.clip(net(o)[0], -1, 1))
    if kind == "dqn":                                 # argmax over 5 acceleration levels
        net = MLP(sd, "")
        return lambda o: float(DQN_LEVELS[int(np.argmax(net(o)))])
    if kind == "ars":                                 # linear: clip(M (o - mean) / std)
        M, m, v = (np.asarray(sd[k], np.float64) for k in ("M", "mean", "var"))
        return lambda o: float(np.clip(M @ ((np.asarray(o, float) - m) / np.sqrt(v + 1e-8)), -1, 1)[0])
    raise ValueError(f"unknown kind {kind}")


# ------------------------------------------------------------------ --sim: the board replaced by the rig model
class SimLink:
    """Answers the same serial commands as SwingupController.ino, from rotary_model.Plant
    (identified plant, firmware accel/speed limits, 30 pps floor, 150 deg latch, Coulomb friction).
    Each step() = one 10 ms loop tick; the new command is applied 4 ms after the reading (rig delay).
    Encoder: 0.3 deg resolution, 0 = hanging (as after a board reset with the rod hanging)."""

    def __init__(self, tilt_deg=1.0, b=None, delay=0.004):
        import rotary_model as rm
        self.rm, self.delay = rm, delay
        self.plant = rm.Plant(math.pi + math.radians(tilt_deg), **({} if b is None else {"b": b}))
        self.home = 0.0
        self.port = "SIM"

    def _obs(self):
        rm, pl = self.rm, self.plant
        pend = math.degrees(rm.wrap(pl.x[0] + math.pi))                 # 0 = hanging
        pend = round(pend / 0.3) * 0.3
        rotor = round((pl.x[2] - self.home) / bc.STEP_RAD) * bc.STEP_RAD
        return (pend, math.degrees(rotor), pl.arm_rate / bc.STEP_RAD, pl.x[3] / bc.STEP_RAD,
                pl.a / bc.STEP_RAD, float(0x1A00), 0.0)

    def _advance(self, seconds):
        for _ in range(int(round(seconds / self.rm.DT))):
            self.plant.step()

    def step(self, cmd, val=0.0):
        obs = self._obs()                                                # reading at the start of the tick
        new_a = None
        if cmd == bc.CMD_SET_ACCEL:
            new_a = float(val) * bc.STEP_RAD
        elif cmd in (bc.CMD_HARD_STOP, bc.CMD_ZERO_VEL):
            new_a = 0.0
            self.plant.x[3] = 0.0
        elif cmd == bc.CMD_SET_HOME:
            self.home = self.plant.x[2]
        elif cmd == bc.CMD_RESET_SAFETY:
            self.plant.latched = False
        self._advance(self.delay)
        if new_a is not None:
            self.plant.a = new_a
        self._advance(0.010 - self.delay)
        return (0, 0, bool(self.plant.latched), obs)

    def stop(self):
        self.plant.a, self.plant.x[3] = 0.0, 0.0

    def close(self):
        self.stop()


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="sac", choices=sorted(POLICIES), help="trained policy to run")
    ap.add_argument("--model", default=None, help="use this .pt file instead of --policy (needs --kind)")
    ap.add_argument("--kind", default=None, choices=("sac", "td3", "bc", "ppo", "ppo_state", "gauss", "dqn", "ars"))
    ap.add_argument("--model-dir", default=None,
                    help="look for --policy weights only here (default: simulator/rl_models, then simulator/rl_results/models)")
    ap.add_argument("--duration", type=float, default=25.0, help="s")
    ap.add_argument("--rate", type=float, default=100.0, help="serial loop rate [Hz]")
    ap.add_argument("--arm-limit", type=float, default=140.0, help="deg (cable checked to 145)")
    ap.add_argument("--calibrate", action=argparse.BooleanOptionalAction, default=True,
                    help="measure the true vertical from a free swing before starting")
    ap.add_argument("--upright-offset", type=float, default=0.0,
                    help="encoder offset of the true vertical [deg], used with --no-calibrate")
    ap.add_argument("--port", default=None)
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--sim", action="store_true", help="no board: run the same script on the rig model")
    ap.add_argument("--sim-b", type=float, default=None, help="--sim: model b (default 0.730)")
    ap.add_argument("--sim-delay", type=float, default=0.004, help="--sim: command delay [s]")
    args = ap.parse_args()

    # ---- policy
    if args.model:
        if not args.kind:
            raise SystemExit("--model needs --kind")
        path, kind, note = Path(args.model), args.kind, "custom model (not checked on the rig model)"
    else:
        fname, kind, note = POLICIES[args.policy]
        dirs = [Path(args.model_dir)] if args.model_dir else MODEL_DIRS
        path = next((d / fname for d in dirs if (d / fname).exists()), dirs[0] / fname)
    if not path.exists():
        raise SystemExit(f"model not found: {path}\n-> searched {', '.join(str(d) for d in MODEL_DIRS)}; "
                         "run simulator/rl_tutorial_cartpole_rotary_v3 to train it")
    print(f"weights: {path}")
    policy = make_policy(path, kind)
    hold = max(1, int(round(args.rate / POLICY_HZ)))
    a_hang = policy(np.array([-1.0, 0.0, 0.0, 0.0, 0.0]))
    a_up = policy(np.array([1.0, 0.0, 0.0, 0.0, 0.0]))
    print(f"policy {path.name} ({kind}) | rig-model check: {note}")
    print(f"policy at {args.rate / hold:.0f} Hz (held {hold} ticks), |u| <= {ACCEL_MAX:.1f} rad/s^2 | "
          f"u(hanging, still) = {a_hang * ACCEL_MAX:+.1f}, u(upright, still) = {a_up * ACCEL_MAX:+.1f} rad/s^2")
    if note.startswith("WARNING") and not args.sim:
        print("   !! this policy failed the rig-model check in simulation -- expect it not to balance")

    link = sc.bc.Link(args.port) if not args.sim else SimLink(b=args.sim_b, delay=args.sim_delay)
    if args.sim:
        print("SIMULATION: the board is replaced by rotary_model (4 ms delay, 0.3 deg encoder, firmware limits)")
    rows, dropped, reason = [], 0, "duration"
    period = 1.0 / args.rate
    try:
        # ---- start-up: identical to swingup_control.py
        link.step(bc.CMD_HARD_STOP)
        link.step(bc.CMD_RESET_SAFETY)
        if not args.sim:
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
            print(f"   arm now {sc.home_arm(link, period):+.2f} deg")
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
            if not sc.wait_still(link):
                raise SystemExit("pendulum did not settle after calibration")
        b_hat = math.radians(offset)
        if not args.yes:
            input(f"The {args.policy if not args.model else path.name} policy will swing the rod over the top; "
                  f"the arm may move up to ~{args.arm_limit:.0f} deg. Clear the area, press Enter ... ")

        # ---- control loop
        est = bc.Estimator(180.0)                       # theta from upright; hanging = +-180 deg
        a, t_up, up_since, falls, was_up = 0.0, None, None, 0, False
        stuck = deque()                                 # (t, arm deg) while the command is large
        t0 = time.perf_counter()
        k, last = 0, link.step(bc.CMD_QUERY)
        while True:
            t = (k * period) if args.sim else (time.perf_counter() - t0)
            if t >= args.duration:
                break
            x = est.update(last[3], t) if last else None
            if x is None:
                dropped += 1; u = 0.0; th = float("nan")
            else:
                th_raw, w, phi, phid = x
                th = bc.wrap_rad(th_raw - b_hat)            # tilt from the calibrated vertical
                if k % hold == 0:                           # new action at 50 Hz, held in between
                    obs = np.array([math.cos(th), math.sin(th), w, phi, phid])
                    a = policy(obs)
                    if not np.isfinite(a):
                        reason = "policy output not finite"; break
                u = a * ACCEL_MAX
                up = abs(math.degrees(th)) < 10.0
                if up and t_up is None:
                    t_up = t
                    print(f"   UPRIGHT at {t:.2f} s (arm {math.degrees(phi):+.1f} deg)")
                if was_up and abs(math.degrees(th)) > 30.0:
                    falls += 1; was_up = False
                    print(f"   fell at {t:.2f} s -> the policy swings up again")
                if up:
                    was_up = True
            pps2 = float(np.clip(u * bc.PPS2_PER_RAD, -bc.ACCEL_MAX_PPS2, bc.ACCEL_MAX_PPS2))
            tq = time.perf_counter()
            resp = link.step(bc.CMD_SET_ACCEL, pps2)
            rtt = (time.perf_counter() - tq) * 1000
            if resp is None:
                dropped += 1
            else:
                last = resp
                st, _, latched, obs_raw = resp
                if x is not None:
                    E = 0.5 * x[1] ** 2 + bc.A_G * (math.cos(th) - 1)
                    rows.append((t, int(abs(math.degrees(th)) < 10.0), math.degrees(th), math.degrees(x[1]),
                                 obs_raw[bc.OBS_ROTOR], math.degrees(x[3]), E, a, u, pps2, rtt, st, int(latched),
                                 math.degrees(b_hat)))
                if latched:
                    reason = f"firmware latched (status {st})"; break
                if abs(obs_raw[bc.OBS_ROTOR]) > args.arm_limit:
                    reason = f"arm {obs_raw[bc.OBS_ROTOR]:.1f} deg"; break
                # same stall guard as swingup_control.py: a large command must move the arm within 0.3 s
                if abs(u) > 10.0:
                    stuck.append((t, obs_raw[bc.OBS_ROTOR]))
                    while stuck and t - stuck[0][0] > 0.3:
                        stuck.popleft()
                    arms = [v for _, v in stuck]
                    if t - stuck[0][0] >= 0.28 and max(arms) - min(arms) < 0.5:
                        reason = "arm not following -- motor driver/power?"; break
                else:
                    stuck.clear()
            k += 1
            if not args.sim:
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
    tag = (path.stem if args.model else args.policy) + ("_sim" if args.sim else "")
    csv = ROOT / "data" / f"swingup_rl_{tag}_{stamp}.csv"
    csv.parent.mkdir(exist_ok=True)
    np.savetxt(csv, d, delimiter=",", comments="",
               header="t_s,upright,theta_deg,theta_dot_dps,phi_deg,phi_dot_dps,energy,action,u_rad_s2,cmd_pps2,"
                      "rtt_ms,status,latched,upright_offset_deg")
    print(f"CSV -> {csv}")
    tail = d[d[:, 0] > d[-1, 0] - 2]
    print(f"first upright: {'%.2f s' % t_up if t_up is not None else 'never'} | falls {falls} | "
          f"arm max {np.abs(d[:, 4]).max():.1f} deg | upright in the last 2 s {100 * tail[:, 1].mean():.0f}%")
    if tail[:, 1].mean() > 0.95:
        print(f"balancing: last 2 s |theta| rms {np.sqrt(np.mean(tail[:, 2] ** 2)):.2f} deg, arm now {d[-1, 4]:+.1f} deg, "
              f"|u| rms {np.sqrt(np.mean(tail[:, 8] ** 2)):.1f} rad/s^2")


if __name__ == "__main__":
    main()
