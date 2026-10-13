"""Behavior cloning of the swing-up + balance controller in the rotary pendulum simulator.

Teacher  = rotary_model.SwingUp (the controller that runs on the real rig).
Student  = small MLP: state -> arm acceleration command u [rad/s^2].
Loss     = MSE between the student's u and the teacher's u (both scaled by ACCEL_MAX).

    1. generate  teacher rollouts in simulator/rotary_model.py (randomised start, b, friction,
                 plus random pushes so the data contains falls and re-swings)
    2. train     MLP with MSE loss (default 1 epoch)
    3. evaluate  validation MSE, then the student alone drives the simulator (closed loop)

The state the student sees is what the teacher sees (Sensor.read):
    z = [theta (0 = upright), theta', phi (arm), phi']
theta enters as (sin, cos) so that +179 deg and -179 deg look alike to the network.

The teacher is not a pure function of z: it kicks open-loop for the first 0.3 s and switches
swing/balance with hysteresis. The kick samples are left out (no state can explain them);
the hysteresis band is what the student has to approximate.

    python behavior_cloning/bc_train.py                 # 1 epoch, prints losses + rollout
    python behavior_cloning/bc_train.py --epochs 30
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "simulator"))
import rotary_model as rm  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"


# ---------------------------------------------------------------- 1. teacher data
class Recorder:
    """Wraps the teacher: logs (z, u, mode) at every 100 Hz control tick."""

    def __init__(self, teacher):
        self.teacher, self.rows = teacher, []

    def __call__(self, z, t):
        u = self.teacher(z, t)
        tr = t - self.teacher.t_start
        u_sent = max(-rm.ACCEL_MAX, min(rm.ACCEL_MAX, u))   # Loop clips the same way
        self.rows.append((*z, u_sent, self.teacher.mode == "balance", tr >= 0.30))
        return u


def rollout_teacher(rng, duration):
    tilt = math.radians(rng.uniform(0.5, 3.0)) * rng.choice([-1, 1])
    plant = rm.Plant(math.pi + tilt, b=rm.B_U * rng.uniform(0.9, 1.1))
    rec = Recorder(rm.SwingUp(rm.place4()))
    lp = rm.Loop(plant, rec, rm.Sensor(True))
    f_c = rm.F_C
    rm.F_C = f_c * rng.uniform(0.5, 2.0)
    next_push = rng.uniform(4.0, 7.0)
    try:
        while plant.t < duration and not lp.stop:
            if plant.t >= next_push:                 # a hand slapping the rod
                plant.kick(rng.uniform(1.0, 4.0) * rng.choice([-1, 1]))
                next_push += rng.uniform(3.0, 6.0)
            lp.advance(10)                           # 10 x 1 ms = one control tick
    finally:
        rm.F_C = f_c
    d = np.array(rec.rows, dtype=np.float32)
    return d[d[:, 6] > 0]                            # drop the open-loop kick


def make_dataset(episodes, duration, seed):
    rng = np.random.default_rng(seed)
    eps = [rollout_teacher(rng, duration) for _ in range(episodes)]
    return eps


def features(z):
    """[theta, theta', phi, phi'] -> network input [sin, cos, theta', phi, phi'] (unscaled)."""
    z = np.asarray(z, dtype=np.float32)
    return np.stack([np.sin(z[..., 0]), np.cos(z[..., 0]), z[..., 1], z[..., 2], z[..., 3]], -1)


# ---------------------------------------------------------------- 2. student
class Policy(nn.Module):
    def __init__(self, mean, std, hidden=64):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean))
        self.register_buffer("std", torch.as_tensor(std))
        self.net = nn.Sequential(
            nn.Linear(5, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):                            # x: raw features -> u / ACCEL_MAX
        return self.net((x - self.mean) / self.std).squeeze(-1)

    @torch.no_grad()
    def act(self, z):
        x = torch.as_tensor(features(z)).unsqueeze(0)
        return float(self(x)) * rm.ACCEL_MAX


def run_epoch(model, X, Y, opt=None, batch=256):
    train = opt is not None
    model.train(train)
    idx = torch.randperm(len(X)) if train else torch.arange(len(X))
    total = 0.0
    with torch.set_grad_enabled(train):
        for i in range(0, len(X), batch):
            b = idx[i:i + batch]
            loss = nn.functional.mse_loss(model(X[b]), Y[b])
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total += loss.item() * len(b)
    return total / len(X)


# ---------------------------------------------------------------- 3. student drives the simulator
def rollout_student(model, duration, seed):
    rng = np.random.default_rng(seed)
    tilt = math.radians(rng.uniform(0.5, 3.0)) * rng.choice([-1, 1])
    plant = rm.Plant(math.pi + tilt)
    teacher = rm.SwingUp(rm.place4())                # used only for the 0.3 s open-loop kick
    start = {}

    def ctrl(z, t):
        start.setdefault("t", t)
        if t - start["t"] < 0.30:
            return teacher(z, t)
        return model.act(z)

    lp = rm.Loop(plant, ctrl, rm.Sensor(True))
    up_time, t_up = 0.0, None
    while plant.t < duration and not lp.stop:
        lp.advance(10)
        if abs(rm.wrap(plant.x[0])) < math.radians(12):
            up_time += 1.0 / rm.CONTROL_HZ
            t_up = plant.t if t_up is None else t_up
    return dict(upright_s=up_time, first_up_s=t_up, stop=lp.stop or "duration")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--duration", type=float, default=15.0, help="s per teacher episode")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-rollout", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    OUT.mkdir(exist_ok=True)

    t0 = time.time()
    eps = make_dataset(args.episodes, args.duration, args.seed)
    n_val = max(1, len(eps) // 5)                    # split by episode, not by sample
    train, val = np.concatenate(eps[:-n_val]), np.concatenate(eps[-n_val:])
    print(f"teacher data: {len(train)} train / {len(val)} val samples from {len(eps)} episodes "
          f"({100 * train[:, 5].mean():.0f}% balancing) in {time.time() - t0:.1f} s")

    Xtr, Xva = (torch.as_tensor(features(d[:, :4])) for d in (train, val))
    Ytr, Yva = (torch.as_tensor(d[:, 4] / rm.ACCEL_MAX) for d in (train, val))
    model = Policy(Xtr.mean(0), Xtr.std(0) + 1e-6)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"before training: val MSE {run_epoch(model, Xva, Yva):.4f}  "
          f"(predict-zero baseline {float((Yva ** 2).mean()):.4f})")
    for ep in range(1, args.epochs + 1):
        tr = run_epoch(model, Xtr, Ytr, opt)
        va = run_epoch(model, Xva, Yva)
        print(f"epoch {ep:3d}: train MSE {tr:.4f}  val MSE {va:.4f}  "
              f"(val RMS error {math.sqrt(va) * rm.ACCEL_MAX:.1f} rad/s^2)")

    ckpt = OUT / "bc_policy.pt"
    torch.save(model.state_dict(), ckpt)
    print(f"saved {ckpt.relative_to(ROOT)}")

    if not args.no_rollout:
        r = rollout_student(model, 15.0, seed=1000 + args.seed)
        first = f"{r['first_up_s']:.2f} s" if r["first_up_s"] is not None else "never"
        print(f"student alone in the simulator (15 s): first reached upright {first}, "
              f"{r['upright_s']:.1f} s within 12 deg, ended by {r['stop']}")


if __name__ == "__main__":
    main()
