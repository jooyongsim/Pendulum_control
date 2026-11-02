"""Imitation first, reinforcement after: BC -> DAgger -> SAC fine-tune, one policy throughout.

Why fuse them at all. The swing-up task punishes pure RL: to collect a single transition that
pays the upright term, random exploration has to pump a hanging rod up through +-180 deg and
arrive slow, inside a 140 deg arm budget. That almost never happens by chance, so the critic
never sees the behaviour it is supposed to value. Cloning the expert produces that behaviour on
the first episode instead -- and cloning is all it can do, because a cloned policy has no way to
be *better* than what it copied. So:

    expert demos --BC--> actor init --DAgger--> better init --SAC--> past the expert

The whole point is that it is ONE network the whole way. That only works if every stage speaks
the same language, which the two earlier folders did not:

    behavior_cloning_yeen   [sin, cos, th', phi, phi'] x K frames, u in rad/s^2, 100 Hz
    rl_yeen                 [cos, sin, th', phi, phi'] x 1,        a in [-1, 1],  50 Hz

So the imitation stages are redone here in the RL convention -- rl_common.RotaryEnv, 50 Hz,
obs as above, action squashed into [-1, 1]. The BC target is the action itself:

    loss = mean( (tanh(mu(o)) - a*)^2 )

fitted through the squash rather than onto atanh(a*), which would explode as a* approaches the
saturated command the expert spends much of the swing-up issuing.

Three things have to be right or the warm start is destroyed in the first few hundred updates:

  1. The critic starts as noise, so its gradient is noise. `critic_warm` updates the critics
     alone, on the demonstrations, before the policy is allowed to move at all.
  2. SAC pushes the policy towards its entropy target. Left at the default the entropy bonus
     randomises a freshly cloned actor immediately, so log_std starts small (ALPHA0, LOG_STD0)
     and alpha is allowed to climb only if the reward justifies it.
  3. There is no uniform-random `start_steps` phase. Filling the buffer with noise is what a
     cold start needs and precisely what a warm start must not do.

A BC regularisation term keeps the policy near the demonstrations early on and is decayed to
zero, so the final policy is free to leave the expert behind:

    loss_actor = alpha * logp - min(Q1, Q2) + bc_coef(t) * MSE(tanh(mu(o_demo)), a*_demo)

    python pipeline_yeen/fusion.py --task swingup --steps 60000 --baseline
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import argparse              # noqa: E402
import math                  # noqa: E402
import sys                   # noqa: E402
import time                  # noqa: E402
from pathlib import Path     # noqa: E402

import numpy as np           # noqa: E402

import arena                 # noqa: E402  -- also puts simulator/, rl_yeen/ on sys.path
import rl_common as rl       # noqa: E402
from eval_rl import teacher  # noqa: E402  -- the rig controller as a function of the RL obs

ROOT = Path(__file__).resolve().parent
CKPT_DIR = ROOT / "checkpoints"

LOG_STD0 = -2.3              # exp(-2.3) ~ 0.1: the cloned policy explores a little, not wildly
ALPHA0 = 0.02                # start the entropy weight low; it is tuned upward if it pays


# ---------------------------------------------------------------- demonstrations
def collect(n_ep, task="swingup", student=None, beta=1.0, noise=0.05, seed=0, max_steps=None):
    """Roll out beta*teacher + (1-beta)*student and label every state with the teacher.

    Two different things come out of one pass, and conflating them is the usual DAgger bug:

        a_exec  what was actually applied -> belongs in the replay buffer, with the reward and
                next state it actually produced
        a_star  what the teacher would do here -> belongs in the BC dataset, as the label

    beta = 1 is plain expert demonstration. beta = 0 is the student driving with the teacher
    only annotating, which is the DAgger round and the only way the dataset ever covers the
    states the student's own mistakes lead to.
    """
    rng = np.random.default_rng(seed)
    O, A_star, A_exec, R, O2, D = [], [], [], [], [], []
    rets, ups = [], []
    for ep in range(n_ep):
        env = rl.RotaryEnv(task, "full", seed=seed * 1000 + ep)
        if max_steps:
            env.max_steps = max_steps
        obs, _ = env.reset()
        ep_ret, n_up, n = 0.0, 0, 0
        term = trunc = False
        while not (term or trunc):
            a_star = teacher(obs)
            if student is not None and rng.random() > beta:
                a_exec = student(obs)
            else:
                a_exec = a_star
            a_exec = float(np.clip(a_exec + noise * rng.standard_normal(), -1.0, 1.0))
            obs2, r, term, trunc, _ = env.step(a_exec)
            O.append(obs); A_star.append(a_star); A_exec.append(a_exec)
            R.append(r); O2.append(obs2); D.append(float(term))
            ep_ret += r
            n_up += abs(rl.wrap(env.x[0])) < math.radians(rl.UPRIGHT_DEG)
            n += 1
            obs = obs2
        rets.append(ep_ret)
        ups.append(n_up / max(n, 1))
    out = dict(obs=np.asarray(O, np.float32), a_star=np.asarray(A_star, np.float32).reshape(-1, 1),
               a_exec=np.asarray(A_exec, np.float32).reshape(-1, 1),
               rew=np.asarray(R, np.float32), obs2=np.asarray(O2, np.float32),
               done=np.asarray(D, np.float32))
    out["stats"] = dict(episodes=n_ep, transitions=len(O), ret=float(np.mean(rets)),
                        up_frac=float(np.mean(ups)), beta=beta)
    return out


def merge(*sets):
    keys = ("obs", "a_star", "a_exec", "rew", "obs2", "done")
    out = {k: np.concatenate([s[k] for s in sets]) for k in keys}
    out["stats"] = dict(episodes=sum(s["stats"]["episodes"] for s in sets),
                        transitions=len(out["obs"]))
    return out


# ---------------------------------------------------------------- stages (torch)
def _torch():
    import torch
    import torch.nn.functional as F
    import train_sac as ts
    return torch, F, ts


def bc_fit(actor, data, epochs=60, lr=1e-3, batch=512, seed=0, val_frac=0.1, quiet=False):
    """Fit the actor's mean head to the teacher's action, through the tanh squash."""
    torch, F, _ = _torch()
    torch.manual_seed(seed)
    o = torch.as_tensor(data["obs"])
    a = torch.as_tensor(data["a_star"])
    n_val = max(1, int(val_frac * len(o)))
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(o), generator=g)
    vi, ti = perm[:n_val], perm[n_val:]
    opt = torch.optim.Adam(actor.parameters(), lr=lr)
    hist = []
    for ep in range(1, epochs + 1):
        actor.train()
        idx = ti[torch.randperm(len(ti), generator=g)]
        tot = 0.0
        for s in range(0, len(idx), batch):
            j = idx[s:s + batch]
            mu, _ = actor(o[j])
            loss = F.mse_loss(torch.tanh(mu), a[j])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(j)
        with torch.no_grad():
            mu, _ = actor(o[vi])
            val = float(F.mse_loss(torch.tanh(mu), a[vi]))
        hist.append(dict(epoch=ep, train=tot / len(idx), val=val))
        if not quiet and (ep % max(1, epochs // 6) == 0 or ep == epochs):
            print(f"      bc epoch {ep:3d}/{epochs}  train {hist[-1]['train']:.5f}"
                  f"  val {val:.5f}  (rmse {math.sqrt(val):.4f} of a in [-1,1])")
    # exploration has to start small or the entropy term undoes the clone in a few hundred steps
    with torch.no_grad():
        last = [m for m in actor.net if hasattr(m, "bias")][-1]
        last.weight[rl.ACT_DIM:] *= 0.1
        last.bias[rl.ACT_DIM:] = LOG_STD0
    return hist


def dagger(actor, data, task="swingup", iters=3, episodes=20, beta0=0.0, beta_decay=0.5,
           epochs=40, seed=7, quiet=False):
    """Student drives, teacher labels, dataset grows, policy refits. Returns (data, log)."""
    _, _, ts = _torch()
    log = []
    for it in range(1, iters + 1):
        beta = beta0 * (beta_decay ** (it - 1))
        student = lambda o_: actor.act(o_, False)        # noqa: E731
        fresh = collect(episodes, task, student=student, beta=beta, noise=0.02,
                        seed=seed * 100 + it)
        data = merge(data, fresh)
        if not quiet:
            print(f"    dagger iter {it}: +{fresh['stats']['transitions']:,} from the student's"
                  f" own states (beta {beta:.2f}, its return {fresh['stats']['ret']:.1f})"
                  f"  -> {data['stats']['transitions']:,} total")
        h = bc_fit(actor, data, epochs=epochs, seed=seed + it, quiet=quiet)
        ev = rl.evaluate(lambda o_: actor.act(o_, True), task, "full")
        log.append(dict(iter=it, beta=beta, transitions=data["stats"]["transitions"],
                        val=h[-1]["val"], **{k: ev[k] for k in
                        ("ret", "hold10", "hold2", "bias_deg", "arm_max_deg", "failures")}))
        if not quiet:
            c = log[-1]
            print(f"      -> eval ret {c['ret']:8.1f}  hold2 {c['hold2']:5.1f}%"
                  f"  bias {c['bias_deg']:+6.2f}deg  arm {c['arm_max_deg']:6.1f}deg"
                  f"  fail {c['failures']}")
    return data, log


def finetune(actor, data, task="swingup", reward="full", steps=60_000, hidden=(256, 256),
             lr=3e-4, gamma=0.98, tau=0.005, batch=256, buffer=400_000, critic_warm=2_000,
             learn_start=0, bc_coef=1.0, bc_decay=20_000, eval_every=2_500, seed=0,
             warm=True, quiet=False):
    """SAC from a given actor, demonstrations already in the buffer.

    warm=False is the cold baseline: the same budget, same everything, random actor and a
    uniform-random exploration phase instead of the demonstrations. That contrast is the whole
    experiment -- "SAC solved it" means nothing without the run that had no head start.
    """
    torch, F, ts = _torch()
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    rng = np.random.default_rng(seed)

    env = rl.RotaryEnv(task, reward, seed=seed)
    critic, target = ts.Critic(hidden), ts.Critic(hidden)
    target.load_state_dict(critic.state_dict())
    for p in target.parameters():
        p.requires_grad_(False)

    opt_a = torch.optim.Adam(actor.parameters(), lr=lr)
    opt_c = torch.optim.Adam(critic.parameters(), lr=lr)
    log_alpha = torch.tensor([math.log(ALPHA0 if warm else 1.0)], requires_grad=True)
    opt_al = torch.optim.Adam([log_alpha], lr=lr)
    target_entropy = -float(rl.ACT_DIM)

    buf = ts.Replay(buffer)
    demo_o = demo_a = None
    if warm and data is not None:
        for o, a, r, o2, d in zip(data["obs"], data["a_exec"], data["rew"],
                                  data["obs2"], data["done"]):
            buf.add(o, a, r, o2, d)
        demo_o = torch.as_tensor(data["obs"])
        demo_a = torch.as_tensor(data["a_star"])
        if not quiet:
            print(f"    buffer seeded with {buf.n:,} demonstration transitions")

    def batch_of(n):
        j = rng.integers(0, buf.n, n)
        t = lambda x: torch.as_tensor(x[j])              # noqa: E731
        return t(buf.o), t(buf.a), t(buf.r), t(buf.o2), t(buf.d)

    def critic_step(o, ac, rw, o2, d):
        alpha = log_alpha.exp().detach()
        with torch.no_grad():
            a2, logp2 = actor.sample(o2)
            q1t, q2t = target(o2, a2)
            y = rw + gamma * (1 - d) * (torch.min(q1t, q2t) - alpha * logp2)
        q1, q2 = critic(o, ac)
        loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        opt_c.zero_grad(); loss.backward(); opt_c.step()
        with torch.no_grad():
            for p, pt in zip(critic.parameters(), target.parameters()):
                pt.mul_(1 - tau).add_(tau * p)
        return float(loss)

    # 1) the critic alone, on the demonstrations: a noisy Q would wreck the clone immediately
    if warm and buf.n > batch and critic_warm:
        t0 = time.perf_counter()
        for i in range(critic_warm):
            critic_step(*batch_of(batch))
        if not quiet:
            print(f"    critic warm-up {critic_warm} steps on demos only"
                  f" [{time.perf_counter()-t0:.0f}s]")

    # 2) ordinary SAC, plus a decaying pull towards the demonstrations
    obs, _ = env.reset()
    ep_ret, ep_rets, curve, best = 0.0, [], [], None
    start_steps = 0 if warm else 5_000
    t0 = time.perf_counter()
    for t in range(1, steps + 1):
        if t <= start_steps:
            a = rng.uniform(-1, 1, rl.ACT_DIM)
        else:
            a = [actor.act(obs, False)]
        obs2, r, term, trunc, _ = env.step(a)
        buf.add(obs, a, r, obs2, float(term))            # bootstrap through trunc: see train_sac
        ep_ret += r
        obs = obs2
        if term or trunc:
            ep_rets.append(ep_ret)
            obs, _ = env.reset()
            ep_ret = 0.0

        if t > learn_start and buf.n > batch:
            o, ac, rw, o2, d = batch_of(batch)
            critic_step(o, ac, rw, o2, d)

            alpha = log_alpha.exp().detach()
            for p in critic.parameters():
                p.requires_grad_(False)
            anew, logp = actor.sample(o)
            qa1, qa2 = critic(o, anew)
            loss_a = (alpha * logp - torch.min(qa1, qa2)).mean()
            coef = bc_coef * max(0.0, 1.0 - t / bc_decay) if demo_o is not None else 0.0
            if coef > 0.0:
                j = rng.integers(0, len(demo_o), batch)
                mu, _ = actor(demo_o[j])
                loss_a = loss_a + coef * F.mse_loss(torch.tanh(mu), demo_a[j])
            opt_a.zero_grad(); loss_a.backward(); opt_a.step()
            for p in critic.parameters():
                p.requires_grad_(True)

            loss_al = -(log_alpha.exp() * (logp.detach() + target_entropy)).mean()
            opt_al.zero_grad(); loss_al.backward(); opt_al.step()

        if t % eval_every == 0 or t == steps:
            ev = rl.evaluate(lambda o_: actor.act(o_, True), task, reward)
            curve.append(dict(step=t, bc_coef=bc_coef * max(0.0, 1.0 - t / bc_decay),
                              train_ret=float(np.mean(ep_rets[-10:])) if ep_rets else float("nan"),
                              alpha=float(log_alpha.exp()),
                              **{k: ev[k] for k in ("ret", "hold10", "hold2", "bias_deg",
                                                    "arm_max_deg", "failures")}))
            if best is None or ev["ret"] > best[0]:
                best = (ev["ret"], {k: v.detach().clone() for k, v in actor.state_dict().items()}, ev)
            if not quiet:
                c = curve[-1]
                print(f"      {t:7d}/{steps}  train {c['train_ret']:8.1f}  eval {c['ret']:8.1f}"
                      f"  hold2 {c['hold2']:5.1f}%  bias {c['bias_deg']:+6.2f}deg"
                      f"  arm {c['arm_max_deg']:6.1f}deg  fail {c['failures']}"
                      f"  alpha {c['alpha']:.3f}  bc {c['bc_coef']:.2f}")

    actor.load_state_dict(best[1])
    rep = dict(stage="sac_warm" if warm else "sac_cold", task=task, reward=reward, steps=steps,
               hidden=list(hidden), gamma=gamma, lr=lr, batch=batch, seed=seed,
               bc_coef=bc_coef, bc_decay=bc_decay, critic_warm=critic_warm if warm else 0,
               demos=int(len(data["obs"])) if (warm and data is not None) else 0,
               train_s=time.perf_counter() - t0, curve=curve, final=best[2])
    return actor, rep


# ---------------------------------------------------------------- the pipeline
def run(task="swingup", demo_episodes=40, bc_epochs=60, dagger_iters=3, dagger_episodes=20,
        steps=60_000, hidden=(256, 256), bc_coef=1.0, bc_decay=20_000, critic_warm=2_000,
        eval_every=2_500, seed=0, baseline=False, quiet=False):
    torch, _, ts = _torch()
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    stages, out = {}, []

    def save(actor, name, rep):
        sd = actor.state_dict()
        W = [v.detach().numpy().astype(np.float64) for k, v in sd.items() if k.endswith("weight")]
        b = [v.detach().numpy().astype(np.float64) for k, v in sd.items() if k.endswith("bias")]
        meta = dict(name=name, n_layers=len(W), activation="relu", obs_names=list(rl.OBS_NAMES),
                    u_max=rl.U_MAX, dt_ctrl=rl.DT_CTRL, report=rep)
        p = rl.save_actor(CKPT_DIR / f"{name}.npz", W, b, meta)
        err = ts.check_numpy_matches_torch(actor, p)
        rep["export_err"] = err
        rl.save_actor(p, W, b, dict(meta, report=rep))
        return p

    def score(name, actor, rep):
        ev = rl.evaluate(lambda o_: actor.act(o_, True), task, "full")
        ar = arena.evaluate(rl.NumpyActor(CKPT_DIR / f"{name}.npz").as_rig_controller(), task)
        row = dict(stage=name, ret=ev["ret"], hold2=ev["hold2"], bias=ev["bias_deg"],
                   arm=ev["arm_max_deg"], fail=ev["failures"],
                   arena_ret=ar["ret"], arena_hold2=ar["hold2"], arena_stops=ar["stops"],
                   train_s=rep.get("train_s", 0.0))
        out.append(row)
        return row

    print(f"\n=== 0) teacher reference ({task}) " + "-" * 24)
    t_ev = rl.evaluate(teacher, task, "full")
    t_ar = arena.evaluate(arena.expert(), task)
    print(f"  teacher        ret {t_ev['ret']:8.1f}  hold2 {t_ev['hold2']:5.1f}%"
          f"  bias {t_ev['bias_deg']:+6.2f}deg  arm {t_ev['arm_max_deg']:6.1f}deg")
    print(f"  rig expert in arena: ret {t_ar['ret']:8.1f}  hold2 {t_ar['hold2']:5.1f}%")
    out.append(dict(stage="teacher", ret=t_ev["ret"], hold2=t_ev["hold2"],
                    bias=t_ev["bias_deg"], arm=t_ev["arm_max_deg"], fail=t_ev["failures"],
                    arena_ret=t_ar["ret"], arena_hold2=t_ar["hold2"], arena_stops=t_ar["stops"],
                    train_s=0.0))

    print(f"\n=== 1) demonstrations ({demo_episodes} episodes) " + "-" * 18)
    data = collect(demo_episodes, task, beta=1.0, noise=0.05, seed=seed + 1)
    print(f"  {data['stats']['transitions']:,} transitions | teacher return"
          f" {data['stats']['ret']:.1f} | upright {100*data['stats']['up_frac']:.1f}% of ticks")

    print(f"\n=== 2) behaviour cloning ({bc_epochs} epochs) " + "-" * 20)
    actor = ts.Actor(hidden)
    hist = bc_fit(actor, data, epochs=bc_epochs, seed=seed, quiet=quiet)
    rep = dict(stage="bc", task=task, bc_epochs=bc_epochs, val=hist[-1]["val"],
               demos=int(len(data["obs"])), hidden=list(hidden), curve=hist, train_s=0.0)
    name = f"bc_{task}"
    save(actor, name, rep)
    stages[name] = {k: v.detach().clone() for k, v in actor.state_dict().items()}
    r = score(name, actor, rep)
    print(f"  {name:<16s} ret {r['ret']:8.1f}  hold2 {r['hold2']:5.1f}%  bias {r['bias']:+6.2f}deg"
          f"  arm {r['arm']:6.1f}deg  fail {r['fail']}")

    if dagger_iters:
        print(f"\n=== 3) DAgger ({dagger_iters} x {dagger_episodes} episodes) " + "-" * 14)
        data, dlog = dagger(actor, data, task, iters=dagger_iters, episodes=dagger_episodes,
                            epochs=max(20, bc_epochs // 2), seed=seed + 7, quiet=quiet)
        rep = dict(stage="dagger", task=task, iters=dagger_iters, demos=int(len(data["obs"])),
                   hidden=list(hidden), curve=dlog, train_s=0.0)
        name = f"dagger_{task}"
        save(actor, name, rep)
        stages[name] = {k: v.detach().clone() for k, v in actor.state_dict().items()}
        r = score(name, actor, rep)
        print(f"  {name:<16s} ret {r['ret']:8.1f}  hold2 {r['hold2']:5.1f}%"
              f"  bias {r['bias']:+6.2f}deg  arm {r['arm']:6.1f}deg  fail {r['fail']}")

    print(f"\n=== 4) SAC fine-tune from the imitated policy ({steps:,} steps) " + "-" * 6)
    actor, rep = finetune(actor, data, task=task, steps=steps, hidden=hidden, bc_coef=bc_coef,
                          bc_decay=bc_decay, critic_warm=critic_warm, eval_every=eval_every,
                          seed=seed, warm=True, quiet=quiet)
    name = f"sac_warm_{task}"
    save(actor, name, rep)
    r = score(name, actor, rep)
    print(f"  {name:<16s} ret {r['ret']:8.1f}  hold2 {r['hold2']:5.1f}%  bias {r['bias']:+6.2f}deg"
          f"  arm {r['arm']:6.1f}deg  fail {r['fail']}  [{rep['train_s']:.0f}s]")

    if baseline:
        print(f"\n=== 5) cold SAC, same budget, no demonstrations " + "-" * 16)
        cold = ts.Actor(hidden)
        cold, crep = finetune(cold, None, task=task, steps=steps, hidden=hidden, bc_coef=0.0,
                              critic_warm=0, learn_start=1_000, eval_every=eval_every,
                              seed=seed, warm=False, quiet=quiet)
        name = f"sac_cold_{task}"
        save(cold, name, crep)
        r = score(name, cold, crep)
        print(f"  {name:<16s} ret {r['ret']:8.1f}  hold2 {r['hold2']:5.1f}%"
              f"  bias {r['bias']:+6.2f}deg  arm {r['arm']:6.1f}deg  fail {r['fail']}"
              f"  [{crep['train_s']:.0f}s]")

    print("\n" + "=" * 86)
    print(f"{'stage':<16s}{'ret':>9s}{'hold2':>8s}{'bias':>8s}{'arm':>8s}{'fail':>6s}"
          f"{'arena_ret':>11s}{'arena_h2':>10s}{'stops':>7s}")
    for r in out:
        print(f"{r['stage']:<16s}{r['ret']:9.1f}{r['hold2']:8.1f}{r['bias']:+8.2f}"
              f"{r['arm']:8.1f}{r['fail']:6.0f}{r['arena_ret']:11.1f}{r['arena_hold2']:10.1f}"
              f"{r['arena_stops']:7.0f}")
    print("\n  ret/hold2/bias/arm  = 50 Hz training env, the numbers SAC optimised"
          "\n  arena_*             = 100 Hz rm.Loop, quantised encoder, 4 ms delay (the rig)"
          "\n  a stage that only wins in the left columns did not survive the transfer")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="swingup", choices=list(rl.EPISODE_S))
    ap.add_argument("--demo-episodes", type=int, default=40)
    ap.add_argument("--bc-epochs", type=int, default=60)
    ap.add_argument("--dagger-iters", type=int, default=3)
    ap.add_argument("--dagger-episodes", type=int, default=20)
    ap.add_argument("--steps", type=int, default=60_000)
    ap.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    ap.add_argument("--bc-coef", type=float, default=1.0,
                    help="weight of the pull towards the demonstrations at step 0")
    ap.add_argument("--bc-decay", type=int, default=20_000,
                    help="steps over which that weight falls to zero")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--baseline", action="store_true",
                    help="also train cold SAC on the same budget -- the contrast is the result")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run (~1 min) that exercises every stage: check the plumbing "
                         "before committing to the real one")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    if a.smoke:
        a.demo_episodes, a.bc_epochs = 4, 5
        a.dagger_iters, a.dagger_episodes = 1, 2
        a.steps, a.bc_decay = 600, 300
        smoke = dict(critic_warm=50, eval_every=300)
        print("SMOKE RUN -- the numbers are meaningless, only the plumbing is being tested")
    else:
        smoke = {}

    rows = run(task=a.task, demo_episodes=a.demo_episodes, bc_epochs=a.bc_epochs,
               dagger_iters=a.dagger_iters, dagger_episodes=a.dagger_episodes, steps=a.steps,
               hidden=tuple(a.hidden), bc_coef=a.bc_coef, bc_decay=a.bc_decay, seed=a.seed,
               baseline=a.baseline, **smoke)
    if a.csv:
        import csv
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\n  -> {a.csv}")


if __name__ == "__main__":
    main()
