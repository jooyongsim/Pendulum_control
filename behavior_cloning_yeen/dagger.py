"""DAgger (Ross, Gordon & Bagnell 2011) on top of the behaviour-cloned swing-up policy.

Behaviour cloning only ever sees states the expert visits. The cloned policy's own small errors
push the rod into states the expert never reached -- here, an arm that sits biased around -15 deg
and oscillates -- and the policy has no training signal there. That is covariate shift, and it is
what caps bc_h5 at 97.5%.

DAgger closes the loop:

    for each iteration
        the POLICY drives the plant                 -> visits the policy's own state distribution
        the EXPERT shadows the same observations    -> labels every one of those states
        aggregate the labels into the dataset, retrain from scratch

The expert here is a state machine, not a pure function of the observation, so it cannot simply
be queried at an arbitrary state. It is instead run alongside the policy on the same observation
stream, so its mode / phi_ref / b_hat evolve along the trajectory the policy actually produced.
That shadow call is what makes the label meaningful.

Iteration 0 is the existing expert dataset, which is the beta = 1 round of the algorithm, so the
DAgger rounds here default to beta = 0 (the policy drives throughout).

    python behavior_cloning_yeen/dagger.py           3 iterations x 40 episodes
    python behavior_cloning_yeen/dagger.py --iters 5 --episodes 60 --beta0 0.3
"""
import il_common as il       # first: sets the OpenMP env vars torch needs (see il_common)

import argparse              # noqa: E402
import time                  # noqa: E402

import numpy as np           # noqa: E402

import train_bc              # noqa: E402
from eval_bc import eval_case, fmt, sweep  # noqa: E402


class ShadowExpert:
    """The policy acts; the expert watches the same observations and says what it would do.

    Both are called exactly once per control tick, so both internal states (the policy's frame
    history, the expert's mode / phi_ref / b_hat) stay in step with the trajectory on screen.
    """

    def __init__(self, policy, expert, beta=0.0, rng=None):
        self.policy, self.expert, self.beta = policy, expert, beta
        self.rng = rng or np.random.default_rng()
        self.feat, self.act, self.mode, self.ts = [], [], [], []
        self.pol = []                                # what the policy wanted, for the gap metric

    def reset(self, *a, **kw):
        self.policy.reset()
        self.expert.reset()

    def __call__(self, z, t):
        u_pol = self.policy(z, t)
        u_exp = self.expert(z, t)                    # advances the expert's state machine
        self.feat.append(il.features(z))
        self.act.append(float(np.clip(u_exp, -il.U_MAX, il.U_MAX)))     # label = expert
        self.pol.append(u_pol)
        self.mode.append(1 if self.expert.mode == "balance" else 0)
        self.ts.append(t)
        # beta > 0 hands control back to the expert now and then, which keeps early iterations
        # from spending every episode in states so bad that the labels are useless
        return u_exp if self.rng.random() < self.beta else u_pol

    def arrays(self):
        return (np.asarray(self.feat, np.float32), np.asarray(self.act, np.float32),
                np.asarray(self.mode, np.uint8), np.asarray(self.ts, np.float32),
                np.asarray(self.pol, np.float32))


def gather(ckpt, cases, duration, beta, rng, ep_offset):
    """Run the policy on every case, collecting expert labels along its own trajectories."""
    F, A, M, T, EP, P, n_stop = [], [], [], [], [], [], 0
    for i, case in enumerate(cases):
        lab = ShadowExpert(il.BCPolicy(ckpt), il.expert(), beta, rng)
        stats, _ = il.rollout(lab, duration=duration, **case)
        f, u, m, ts, pol = lab.arrays()
        F.append(f); A.append(u); M.append(m); T.append(ts); P.append(pol)
        EP.append(np.full(len(f), ep_offset + i, np.int32))
        n_stop += bool(stats["stop"])
    f, u, m, ts, ep, pol = (np.concatenate(x) for x in (F, A, M, T, EP, P))
    # the quantity BC cannot see: how far the policy is from the expert on the policy's OWN
    # states. A validation error measured on expert trajectories says nothing about this.
    gap = dict(all=float(np.sqrt(np.mean((pol - u) ** 2))),
               swing=float(np.sqrt(np.mean((pol - u)[m == 0] ** 2))) if (m == 0).any() else 0.0,
               balance=float(np.sqrt(np.mean((pol - u)[m == 1] ** 2))) if (m == 1).any() else 0.0)
    return f, u, m, ts, ep, n_stop, gap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="iteration-0 dataset (default data/expert.npz)")
    ap.add_argument("--init", default=None, help="starting policy (default checkpoints/bc_h5.npz)")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=40, help="rollouts per iteration")
    ap.add_argument("--duration", type=float, default=15.0)
    ap.add_argument("--kick-prob", type=float, default=0.5)
    ap.add_argument("--beta0", type=float, default=0.0, help="expert takeover rate, iteration 1")
    ap.add_argument("--beta-decay", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--eval-episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    data = a.data or (il.DATA_DIR / "expert.npz")
    ckpt = a.init or (il.CKPT_DIR / "bc_h5.npz")
    hist = il.BCPolicy(ckpt).k
    rng = np.random.default_rng(a.seed)

    d0, meta0 = il.load_dataset(data)
    agg = {k: [d0[k]] for k in il.DATASET_KEYS}
    n_ep = int(d0["ep"].max()) + 1
    print(f"iteration 0: {data.name if hasattr(data, 'name') else data}  "
          f"{len(d0['act'])} samples / {n_ep} episodes, policy {ckpt}  (history {hist})\n")

    # one fixed test set for every iteration, disjoint from both collection and DAgger seeds
    erng = np.random.default_rng(1234)
    ecases = [eval_case(erng, a.duration, False) for _ in range(a.eval_episodes)]
    print(f"held-out swing-up test: {a.eval_episodes} plants, no disturbance, {a.duration:.0f} s")
    print("  " + fmt("expert", sweep(il.expert, ecases, a.duration)))
    print("  " + fmt("iter 0", sweep(lambda: il.BCPolicy(ckpt), ecases, a.duration)) + "\n")

    for it in range(1, a.iters + 1):
        beta = a.beta0 * (a.beta_decay ** (it - 1))
        t0 = time.perf_counter()
        cases = [eval_case(rng, a.duration, rng.random() < a.kick_prob) for _ in range(a.episodes)]
        f, u, m, ts, ep, n_stop, gap = gather(ckpt, cases, a.duration, beta, rng, n_ep)
        n_ep += a.episodes
        for k, v in zip(il.DATASET_KEYS, (f, u, m, ts, ep)):
            agg[k].append(v)
        cat = {k: np.concatenate(v) for k, v in agg.items()}
        meta = dict(source="dagger", iteration=it, beta=beta, base=str(data),
                    n_episodes=n_ep, control_hz=il.rm.CONTROL_HZ,
                    feat_names=list(il.FEAT_NAMES), u_max=il.U_MAX, seed=a.seed,
                    duration=meta0.get("duration"))
        dpath = il.save_dataset(il.DATA_DIR / f"dagger_iter{it}.npz", cat["feat"], cat["act"],
                                cat["mode"], cat["t"], cat["ep"], meta)

        ck, rep, _ = train_bc.train(dpath, hist, hidden=(256, 256), epochs=a.epochs, batch=1024,
                                    lr=1e-3, swing_weight=2.0, seed=0, quiet=True,
                                    name=f"dag_i{it}_h{hist}")
        ckpt = train_bc.save(ck, il.CKPT_DIR / f"dagger_h{hist}_iter{it}.npz")

        print(f"iteration {it}  beta {beta:.2f}  +{len(u)} samples "
              f"(total {len(cat['act'])}, {100 * np.mean(m == 0):.1f}% swing)  "
              f"{n_stop}/{a.episodes} rollouts stopped  [{time.perf_counter() - t0:.0f}s]")
        print(f"  gap on the policy's own states (before retraining): {gap['all']:.3f}"
              f"  swing {gap['swing']:.3f}  balance {gap['balance']:.3f}")
        print(f"  after retraining, on expert states:  RMSE {rep['rmse']:.3f}"
              f"  swing {rep['rmse_swing']:.3f}  balance {rep['rmse_balance']:.3f}"
              f"  R2 {rep['r2']:.3f}")
        print("  " + fmt(f"iter {it}", sweep(lambda c=ckpt: il.BCPolicy(c), ecases, a.duration)))
        print()

    print(f"final policy -> {ckpt}")


if __name__ == "__main__":
    main()
