"""Collect expert demonstrations from the SwingUp controller in the simulator.

Each episode runs the real rig's controller on a randomised plant (the same error bands
rotary_swingup_realtime.py --check uses for its robustness sweep) and logs every control tick.
Random kicks are injected so the dataset also contains falls and re-swings; a behaviour-cloned
policy that has only ever seen clean swing-ups has nothing to do when it drifts off that path.

    python behavior_cloning_yeen/collect_expert.py         150 episodes x 25 s -> data/expert.npz
    python behavior_cloning_yeen/collect_expert.py --episodes 40 --out data/expert_small.npz
"""
import argparse
import json
import time

import numpy as np

import il_common as il

B_NOM = il.rm.B_U


def random_case(rng, duration, kick_p=0.7):
    """Plant + sensor errors the real rig could plausibly have, plus disturbances.

    Kicks are deliberately hard enough (up to 9 rad/s) to knock the rod past the expert's 30 deg
    fall threshold, because a fall is what produces more swing-mode data.
    """
    kicks = []
    if rng.random() < kick_p:
        for t in np.sort(rng.uniform(4.0, max(5.0, duration - 2.0), rng.integers(1, 4))):
            kicks.append((float(t), float(rng.choice([-1, 1]) * rng.uniform(3.0, 9.0))))
    return dict(
        b=float(B_NOM * rng.uniform(0.8, 1.2)),
        coulomb=float(rng.uniform(0.5, 2.0)),
        delay=float(rng.uniform(0.004, 0.012)),
        tilt_deg=float(rng.uniform(-2.5, 2.5)),
        offset_deg=float(rng.uniform(-0.5, 0.5)),
        realistic=True,
        kicks=kicks,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=250)
    ap.add_argument("--duration", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kick-prob", type=float, default=0.7)
    ap.add_argument("--keep-failures", action="store_true",
                    help="keep episodes where the expert never caught the rod (default: drop)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    il.DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = a.out or (il.DATA_DIR / "expert.npz")
    rng = np.random.default_rng(a.seed)

    F, A, M, T, EP, cases, dropped = [], [], [], [], [], [], 0
    t0 = time.perf_counter()
    for i in range(a.episodes):
        case = random_case(rng, a.duration, a.kick_prob)
        stats, rec = il.rollout(il.expert(), duration=a.duration, log=True, **case)
        # Keep anything the expert actually swung up. `success` scores the last 5 s, so a late
        # kick would fail an episode whose recovery is exactly the data we want.
        usable = not stats["stop"] and stats["t_catch"] is not None
        if not usable and not a.keep_failures:
            dropped += 1
            continue
        f, u, m, ts = rec.arrays()
        F.append(f); A.append(u); M.append(m); T.append(ts)
        EP.append(np.full(len(f), len(cases), np.int32))
        cases.append(dict(case, **{k: v for k, v in stats.items() if k != "stop"},
                          stop=stats["stop"]))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1:4d}/{a.episodes} episodes, {sum(len(x) for x in F):7d} samples, "
                  f"{dropped} dropped, {time.perf_counter() - t0:5.1f}s")

    if not F:
        raise SystemExit("no usable episodes -- the expert failed on every randomised plant")

    feat, act = np.concatenate(F), np.concatenate(A)
    mode, ts, ep = np.concatenate(M), np.concatenate(T), np.concatenate(EP)
    meta = dict(source="expert", n_episodes=len(cases), duration=a.duration,
                control_hz=il.rm.CONTROL_HZ, feat_names=list(il.FEAT_NAMES), u_max=il.U_MAX,
                seed=a.seed, dropped_failures=dropped, cases=cases)
    il.save_dataset(out, feat, act, mode, ts, ep, meta)

    sw = mode == 0
    print(f"\n{len(cases)} episodes kept ({dropped} expert failures dropped), "
          f"{len(feat)} samples -> {out}")
    print(f"  swing  {sw.sum():7d} ({100 * sw.mean():4.1f}%)  "
          f"u in [{act[sw].min():+6.1f}, {act[sw].max():+6.1f}]  std {act[sw].std():5.2f}")
    print(f"  balance{(~sw).sum():7d} ({100 * (~sw).mean():4.1f}%)  "
          f"u in [{act[~sw].min():+6.1f}, {act[~sw].max():+6.1f}]  std {act[~sw].std():5.2f}")
    print(f"  saturated |u| = u_max: {100 * np.mean(np.abs(act) >= il.U_MAX - 1e-9):4.1f}%")
    print(f"  first catch  median {np.median([c['t_catch'] for c in cases]):.2f} s")
    print(f"  episodes ending upright: {sum(c['success'] for c in cases)}/{len(cases)}")
    print(f"  collection took {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
