"""Soft Actor-Critic on the rotary pendulum, one reward design per run.

    python rl_yeen/train_sac.py                                   full reward, balance, 30k steps
    python rl_yeen/train_sac.py --reward no_arm                   drop the arm-centering term
    python rl_yeen/train_sac.py --task swingup --steps 200000     the hard task
    python rl_yeen/train_sac.py --sweep                           every reward in rl_common.REWARDS

SAC rather than REINFORCE or DQN because the arm acceleration is continuous and the plant is
cheap to simulate but expensive in wall-clock per environment step: SAC is off-policy, so every
transition is reused many times from the replay buffer instead of being thrown away after one
gradient step.

Checkpoints are .npz of the actor's mean head plus the run's report, so evaluation and the rig
need numpy only -- the same arrangement as behavior_cloning_yeen. check_numpy_matches_torch()
guards the export the way train_bc.py does.
"""
import rl_common as rl       # first: sets the OpenMP env vars torch needs (see rl_common)

import argparse              # noqa: E402
import json                  # noqa: E402
import math                  # noqa: E402
import time                  # noqa: E402

import numpy as np           # noqa: E402
import torch                 # noqa: E402
import torch.nn as nn        # noqa: E402
import torch.nn.functional as F  # noqa: E402

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0


def mlp(n_in, n_out, hidden):
    layers, d = [], n_in
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    return nn.Sequential(*layers, nn.Linear(d, n_out))


class Actor(nn.Module):
    """Gaussian policy squashed through tanh so the action stays inside [-1, 1]."""

    def __init__(self, hidden):
        super().__init__()
        self.net = mlp(rl.OBS_DIM, 2 * rl.ACT_DIM, hidden)

    def forward(self, obs):
        mu, log_std = self.net(obs).chunk(2, dim=-1)
        return mu, log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)

    def sample(self, obs):
        mu, log_std = self(obs)
        std = log_std.exp()
        pre = mu + std * torch.randn_like(mu)                  # reparameterised, so grads flow
        a = torch.tanh(pre)
        # change of variables for the tanh squash: log p(a) = log p(pre) - log|da/dpre|
        logp = (-0.5 * (((pre - mu) / std) ** 2) - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)
        logp = logp - torch.log(1 - a ** 2 + 1e-6).sum(-1)
        return a, logp

    def act(self, obs, deterministic=True):
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            mu, log_std = self(o)
            pre = mu if deterministic else mu + log_std.exp() * torch.randn_like(mu)
            return float(torch.tanh(pre)[0, 0])


class Critic(nn.Module):
    """Twin Q networks. The minimum of the two is used as the target, which damps the
    overestimation bias that a single Q network accumulates through its own max."""

    def __init__(self, hidden):
        super().__init__()
        self.q1 = mlp(rl.OBS_DIM + rl.ACT_DIM, 1, hidden)
        self.q2 = mlp(rl.OBS_DIM + rl.ACT_DIM, 1, hidden)

    def forward(self, obs, act):
        x = torch.cat([obs, act], -1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


class Replay:
    def __init__(self, size):
        self.o = np.zeros((size, rl.OBS_DIM), np.float32)
        self.a = np.zeros((size, rl.ACT_DIM), np.float32)
        self.r = np.zeros(size, np.float32)
        self.o2 = np.zeros((size, rl.OBS_DIM), np.float32)
        self.d = np.zeros(size, np.float32)
        self.size, self.i, self.n = size, 0, 0

    def add(self, o, a, r, o2, d):
        i = self.i
        self.o[i], self.a[i], self.r[i], self.o2[i], self.d[i] = o, a, r, o2, d
        self.i = (i + 1) % self.size
        self.n = min(self.n + 1, self.size)

    def batch(self, m, rng):
        j = rng.integers(0, self.n, m)
        t = lambda x: torch.as_tensor(x[j])        # noqa: E731
        return t(self.o), t(self.a), t(self.r), t(self.o2), t(self.d)


def train(task="balance", reward="full", steps=30_000, hidden=(256, 256), lr=3e-4, gamma=0.98,
          tau=0.005, batch=256, buffer=300_000, start_steps=5_000, learn_start=1_000,
          eval_every=2_500, seed=0, quiet=False):
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    rng = np.random.default_rng(seed)

    env = rl.RotaryEnv(task, reward, seed=seed)
    actor, critic = Actor(hidden), Critic(hidden)
    target = Critic(hidden)
    target.load_state_dict(critic.state_dict())
    for p in target.parameters():
        p.requires_grad_(False)

    opt_a = torch.optim.Adam(actor.parameters(), lr=lr)
    opt_c = torch.optim.Adam(critic.parameters(), lr=lr)
    # alpha trades reward against policy entropy. Tuning it to hit a target entropy beats
    # hand-picking a constant, which has to be re-picked whenever the reward scale changes --
    # and this study changes the reward scale on purpose.
    log_alpha = torch.zeros(1, requires_grad=True)
    opt_al = torch.optim.Adam([log_alpha], lr=lr)
    target_entropy = -float(rl.ACT_DIM)

    buf = Replay(buffer)
    obs, _ = env.reset()
    ep_ret, ep_len, ep_rets = 0.0, 0, []
    curve, best = [], None
    t0 = time.perf_counter()

    for t in range(1, steps + 1):
        a = rng.uniform(-1, 1, rl.ACT_DIM) if t <= start_steps else [actor.act(obs, False)]
        obs2, r, term, trunc, _ = env.step(a)
        ep_ret += r
        ep_len += 1
        # `trunc` is the time limit, not a real end of the world: bootstrapping must continue
        # through it or the agent learns that the clock running out is a failure.
        buf.add(obs, a, r, obs2, float(term))
        obs = obs2
        if term or trunc:
            ep_rets.append(ep_ret)
            obs, _ = env.reset()
            ep_ret, ep_len = 0.0, 0

        if t > learn_start:
            o, ac, rw, o2, d = buf.batch(batch, rng)
            alpha = log_alpha.exp().detach()
            with torch.no_grad():
                a2, logp2 = actor.sample(o2)
                q1t, q2t = target(o2, a2)
                y = rw + gamma * (1 - d) * (torch.min(q1t, q2t) - alpha * logp2)
            q1, q2 = critic(o, ac)
            loss_c = F.mse_loss(q1, y) + F.mse_loss(q2, y)
            opt_c.zero_grad(); loss_c.backward(); opt_c.step()

            for p in critic.parameters():
                p.requires_grad_(False)
            anew, logp = actor.sample(o)
            qa1, qa2 = critic(o, anew)
            loss_a = (alpha * logp - torch.min(qa1, qa2)).mean()
            opt_a.zero_grad(); loss_a.backward(); opt_a.step()
            for p in critic.parameters():
                p.requires_grad_(True)

            loss_al = -(log_alpha.exp() * (logp.detach() + target_entropy)).mean()
            opt_al.zero_grad(); loss_al.backward(); opt_al.step()

            with torch.no_grad():
                for p, pt in zip(critic.parameters(), target.parameters()):
                    pt.mul_(1 - tau).add_(tau * p)      # polyak averaging of the target critic

        if t % eval_every == 0 or t == steps:
            ev = rl.evaluate(lambda o_: actor.act(o_, True), task, reward)
            curve.append(dict(step=t, train_ret=float(np.mean(ep_rets[-10:])) if ep_rets else float("nan"),
                              alpha=float(log_alpha.exp()), **{k: ev[k] for k in
                              ("ret", "hold10", "hold5", "bias_deg", "arm_max_deg", "failures")}))
            if best is None or ev["ret"] > best[0]:
                best = (ev["ret"], {k: v.detach().clone() for k, v in actor.state_dict().items()}, ev)
            if not quiet:
                c = curve[-1]
                print(f"    {t:7d}/{steps}  train {c['train_ret']:8.1f}  eval {c['ret']:8.1f}"
                      f"  hold10 {c['hold10']:5.1f}%  bias {c['bias_deg']:+6.2f}deg"
                      f"  arm {c['arm_max_deg']:6.1f}deg  fail {c['failures']}/{ev['n_starts']}"
                      f"  alpha {c['alpha']:.3f}")

    actor.load_state_dict(best[1])                      # keep the best evaluation, not the last
    rep = dict(task=task, reward=reward, reward_cfg=rl.REWARDS[reward], steps=steps,
               hidden=list(hidden), gamma=gamma, lr=lr, batch=batch, seed=seed,
               params=sum(p.numel() for p in actor.parameters()),
               train_s=time.perf_counter() - t0, curve=curve, final=best[2])
    return actor, rep


def export(actor, name, rep):
    """Save the actor's mean head as .npz and check numpy reproduces torch."""
    sd = actor.state_dict()
    weights = [v.detach().numpy().astype(np.float64) for k, v in sd.items() if k.endswith("weight")]
    biases = [v.detach().numpy().astype(np.float64) for k, v in sd.items() if k.endswith("bias")]
    meta = dict(name=name, n_layers=len(weights), activation="relu",
                obs_names=list(rl.OBS_NAMES), u_max=rl.U_MAX, dt_ctrl=rl.DT_CTRL, report=rep)
    path = rl.save_actor(rl.CKPT_DIR / f"{name}.npz", weights, biases, meta)
    rep["export_err"] = check_numpy_matches_torch(actor, path)
    # rewrite with the check result inside the report
    rl.save_actor(path, weights, biases, dict(meta, report=rep))
    return path


def check_numpy_matches_torch(actor, path, n=512, tol=1e-4):
    """The deployed policy runs on NumpyActor, not torch. A weight-ordering mistake in the
    export would otherwise show up only as a policy that quietly behaves differently."""
    np_actor = rl.NumpyActor(path)
    rng = np.random.default_rng(0)
    th = rng.uniform(-math.pi, math.pi, n)
    obs = np.stack([np.cos(th), np.sin(th), rng.uniform(-10, 10, n),
                    rng.uniform(-2.4, 2.4, n), rng.uniform(-7.8, 7.8, n)], 1)
    a_t = np.array([actor.act(o, True) for o in obs.astype(np.float32)])
    a_n = np.array([np_actor(o) for o in obs])
    err = float(np.max(np.abs(a_t - a_n)))              # torch is float32, NumpyMLP float64
    if not err < tol:
        raise SystemExit(f"numpy actor disagrees with torch by {err:.2e} -- export is broken")
    return err


def line(name, rep):
    f = rep["final"]
    return (f"{name:<22s} ret {f['ret']:8.1f}  hold10 {f['hold10']:5.1f}%  hold5 {f['hold5']:5.1f}%"
            f"  bias {f['bias_deg']:+6.2f}deg  arm {f['arm_max_deg']:6.1f}deg"
            f"  fail {f['failures']}/{int(f['n_starts'])} {f['why']:<5s} [{rep['train_s']:.0f}s]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="balance", choices=list(rl.EPISODE_S))
    ap.add_argument("--reward", default="full", choices=list(rl.REWARDS))
    ap.add_argument("--sweep", action="store_true", help="train every reward in REWARDS")
    ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.98)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--eval-every", type=int, default=2_500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="", help="suffix for the checkpoint name")
    a = ap.parse_args()

    rewards = list(rl.REWARDS) if a.sweep else [a.reward]
    kw = dict(task=a.task, steps=a.steps, hidden=tuple(a.hidden), lr=a.lr, gamma=a.gamma,
              batch=a.batch, eval_every=a.eval_every, seed=a.seed)

    done = []
    for rw in rewards:
        name = f"sac_{a.task}_{rw}{a.tag}"
        print(f"\n--- {name}  ({rl.REWARDS[rw]['doc']}) " + "-" * 20)
        actor, rep = train(reward=rw, **kw)
        p = export(actor, name, rep)
        print(f"  {line(name, rep)}")
        print(f"  numpy/torch export agreement: {rep['export_err']:.1e}   -> {p}")
        done.append((name, rep))

    if len(done) > 1:
        print("\nreward ablation (" + a.task + f", {a.steps} steps, seed {a.seed}):")
        for name, rep in done:
            print(f"  {line(name, rep)}")
        print("\n  Return is NOT comparable across rewards -- each row optimises a different"
              "\n  number. Compare hold10/hold5, bias and arm instead: those are the same"
              "\n  physical question asked of every agent. Run eval_rl.py for the full table.")


if __name__ == "__main__":
    main()
