"""Closed-loop evaluation: put the cloned policy in the control loop and see if it swings up.

Open-loop RMSE against the expert's actions is a weak proxy. Behaviour cloning fails through
covariate shift -- small action errors move the rod into states the expert never visited, where
the policy was never trained -- and that only shows up when the policy drives the plant itself.

Every policy is run on the *same* randomised plants as the expert (paired comparison), with a
test seed disjoint from the one used for collection.

    python behavior_cloning_yeen/eval_bc.py              expert vs every checkpoint, 40 episodes
    python behavior_cloning_yeen/eval_bc.py --episodes 100
    python behavior_cloning_yeen/eval_bc.py --plot       trajectory overlay for one plant
"""
import argparse
import math

import numpy as np

import il_common as il

B_NOM = il.rm.B_U


def eval_case(rng, duration, kicks=True):
    """A test plant. Separate from collect_expert.random_case on purpose.

    Collection wants hard, late kicks because falls generate swing-mode data. Evaluation wants
    kicks early enough that there is time to resettle before the scoring window, otherwise the
    metric measures how brutal the kick was rather than how well the policy recovers.
    """
    ks = []
    if kicks:
        hi = max(5.0, duration - il.SETTLE_S - 3.0)
        for t in np.sort(rng.uniform(4.0, hi, rng.integers(1, 3))):
            ks.append((float(t), float(rng.choice([-1, 1]) * rng.uniform(3.0, 7.0))))
    return dict(b=float(B_NOM * rng.uniform(0.8, 1.2)), coulomb=float(rng.uniform(0.5, 2.0)),
                delay=float(rng.uniform(0.004, 0.012)), tilt_deg=float(rng.uniform(-2.5, 2.5)),
                offset_deg=float(rng.uniform(-0.5, 0.5)), realistic=True, kicks=ks)


def sweep(make_ctrl, cases, duration):
    rows = []
    for case in cases:
        stats, _ = il.rollout(make_ctrl(), duration=duration, **case)
        rows.append(stats)
    ok = [r for r in rows if r["success"]]
    catches = [r["t_catch"] for r in rows if r["t_catch"] is not None]
    stops = {}
    for r in rows:
        if r["stop"]:
            stops[r["stop"]] = stops.get(r["stop"], 0) + 1
    return dict(n=len(rows), success=len(ok), rate=100.0 * len(ok) / len(rows),
                caught=100.0 * len(catches) / len(rows),
                t_catch=float(np.median(catches)) if catches else float("nan"),
                theta_rms=float(np.median([r["theta_rms_deg"] for r in ok])) if ok else float("nan"),
                arm_max=float(np.median([r["arm_max_deg"] for r in rows])),
                sat=float(np.median([r["sat_pct"] for r in rows])), stops=stops)


def fmt(name, s):
    catch = f"{s['t_catch']:5.2f}s" if not math.isnan(s["t_catch"]) else "  --  "
    rms = f"{s['theta_rms']:4.2f}d" if not math.isnan(s["theta_rms"]) else "  -- "
    stop = ", ".join(f"{k} x{v}" for k, v in sorted(s["stops"].items())) or "-"
    return (f"{name:<12} up {s['success']:3d}/{s['n']:<3d} ({s['rate']:5.1f}%)  caught {s['caught']:5.1f}%"
            f"  catch {catch}  theta rms {rms}  arm {s['arm_max']:5.1f}d  sat {s['sat']:4.1f}%  {stop}")


def plot_one(ckpts, case, duration, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [("expert", il.expert(), "0.25")] + [
        (il.BCPolicy(p).name, il.BCPolicy(p), f"C{i}") for i, p in enumerate(ckpts)]
    fig, ax = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)
    for name, ctrl, color in runs:
        rec = il.Recorder(ctrl)
        with il.model_errors(case.get("coulomb", 1.0), case.get("delay")):
            lp = il.make_loop(rec, realistic=case.get("realistic", True), b=case["b"],
                              tilt_deg=case["tilt_deg"], offset_deg=case["offset_deg"])
            pending = sorted(case.get("kicks", []))
            T, TH, PH = [], [], []
            while lp.plant.t < duration:
                while pending and lp.plant.t >= pending[0][0]:
                    lp.plant.kick(pending.pop(0)[1])
                lp.advance(10)
                T.append(lp.plant.t)
                TH.append(math.degrees(il.rm.wrap(lp.plant.x[0])))
                PH.append(math.degrees(lp.plant.x[2]))
        lw = 2.2 if name == "expert" else 1.3
        ax[0].plot(T, TH, color=color, lw=lw, label=name)
        ax[1].plot(T, PH, color=color, lw=lw)
        ax[2].plot(rec.ts, rec.act, color=color, lw=lw, alpha=.85)
    for t, dw in case.get("kicks", []):
        for a_ in ax:
            a_.axvline(t, color="r", ls=":", lw=0.8)
    ax[0].axhspan(-il.rm.SWING["catch"], il.rm.SWING["catch"], color="#2E8B57", alpha=.12)
    ax[0].set(ylabel="theta from up [deg]", ylim=(-190, 190), yticks=[-180, -90, 0, 90, 180])
    ax[1].set(ylabel="arm phi [deg]")
    ax[2].set(ylabel="u [rad/s^2]", xlabel="t [s]")
    ax[0].legend(ncol=4, fontsize=8, title="dotted red = kick")
    for a_ in ax:
        a_.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"  trajectory overlay -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="*", default=None, help="default: every checkpoints/bc_h*.npz")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--duration", type=float, default=15.0, help="clean suite")
    ap.add_argument("--duration-dist", type=float, default=25.0,
                    help="disturbed suite: needs room for the kick plus the settling window")
    ap.add_argument("--suite", choices=("clean", "disturbed", "both"), default="both")
    ap.add_argument("--seed", type=int, default=1234, help="disjoint from the collection seed")
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()

    ckpts = [str(p) for p in (a.ckpt or sorted(il.CKPT_DIR.glob("bc_h*.npz")))]
    if not ckpts:
        raise SystemExit("no checkpoints -- run train_bc.py first")

    suites = []
    if a.suite in ("clean", "both"):
        suites.append(("swing-up only (no disturbance)", False, a.duration))
    if a.suite in ("disturbed", "both"):
        suites.append(("disturbance recovery", True, a.duration_dist))

    plot_case, plot_dur = None, a.duration
    for title, kicks, dur in suites:
        rng = np.random.default_rng(a.seed)       # same plants for every policy, and per suite
        cases = [eval_case(rng, dur, kicks) for _ in range(a.episodes)]
        plot_case, plot_dur = cases[0], dur       # ends up the disturbed suite when both run
        print(f"=== {title}: {a.episodes} randomised plants (seed {a.seed}), {dur:.0f} s each")
        print(f"    success = within {il.UPRIGHT_DEG:.0f} deg of upright for >95% of the last "
              f"{il.SETTLE_S:.0f} s\n")
        print("    " + fmt("expert", sweep(il.expert, cases, dur)))
        for p in ckpts:
            pol = il.BCPolicy(p)
            r = pol.meta["report"]
            print("    " + fmt(pol.name, sweep(lambda p=p: il.BCPolicy(p), cases, dur))
                  + f"   | open-loop RMSE {r['rmse']:.3f}")
        print()

    if a.plot:
        il.FIG_DIR.mkdir(parents=True, exist_ok=True)
        plot_one(ckpts, plot_case, plot_dur, il.FIG_DIR / "bc_closed_loop.png")


if __name__ == "__main__":
    main()
