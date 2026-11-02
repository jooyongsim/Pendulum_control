"""Score the trained actors against each other, the teacher, and model error. numpy only.

    python rl_yeen/eval_rl.py                        every checkpoint, clean plant
    python rl_yeen/eval_rl.py --suite robust         also +-20% b and 2x Coulomb friction
    python rl_yeen/eval_rl.py --task swingup --csv out.csv

Two rules this table follows, both learned from the imitation-learning study:

1. Every agent is scored with the SAME reward (`--score`, default `full`), whatever it was
   trained on. A return computed under the agent's own reward is not a measurement, it is that
   agent's own opinion of itself -- `trap` would "win" by scoring -th^2 on a short episode.

2. hold is reported at +-10, +-5 and +-2 deg. The cloned policies passed +-10 while sitting
   8.9 deg off vertical, so a single band hides the difference between a policy that balances
   and one that merely does not fall over.
"""
import argparse
import math
from pathlib import Path

import numpy as np

import rl_common as rl

rm = rl.rm

# ---------------------------------------------------------------- reference controller
K4 = rm.place4()
CATCH_DEG, CATCH_W = 20.0, 4.0


def teacher(obs):
    """The rig's controller as a stateless function of the RL observation.

    rotary_model.SwingUp carries mode/phi_ref/b_hat between calls and runs at 100 Hz, so it
    cannot be dropped into a 50 Hz env as-is. This is the same simplification the course
    notebook makes for its teacher: energy pumping until the rod is close and slow, then the
    4-state pole-placement gain.
    """
    c, s, w, phi, phid = obs
    th = math.atan2(s, c)
    if abs(math.degrees(th)) < CATCH_DEG and abs(w) < CATCH_W:
        u = float(-K4 @ np.array([th, w, phi, phid]))
    else:
        E = rm.energy(th, w)
        scale = min(1.0, max(0.0, rm.SWING["ke"] * (rm.SWING["etarget"] - E)))
        sgn = 1.0 if w * math.cos(th) > 0 else -1.0
        u = rm.SWING["pump"] * scale * sgn - rm.SWING["kda"] * phid - rm.SWING["kpa"] * phi
    return float(np.clip(u / rl.U_MAX, -1.0, 1.0))


# ---------------------------------------------------------------- suites
SUITES = dict(
    clean=[dict(label="clean", b=rm.B_U, friction=1.0)],
    robust=[dict(label="clean", b=rm.B_U, friction=1.0),
            dict(label="b -20%", b=0.8 * rm.B_U, friction=1.0),
            dict(label="b +20%", b=1.2 * rm.B_U, friction=1.0),
            dict(label="2x fric", b=rm.B_U, friction=2.0)],
)

COLS = [("ret", "return", "{:8.1f}"), ("hold10", "hold10", "{:8.1f}"), ("hold5", "hold5", "{:8.1f}"),
        ("hold2", "hold2", "{:8.1f}"), ("bias_deg", "bias", "{:+8.2f}"),
        ("wobble_deg", "wobble", "{:8.2f}"), ("t_up", "t_up", "{:8.2f}"),
        ("arm_max_deg", "arm", "{:8.1f}"), ("u_rms", "u_rms", "{:8.2f}"),
        ("sat_pct", "sat%", "{:8.1f}"), ("failures", "fail", "{:8.0f}")]


def agents(task, ckpt_dir, include_teacher=True):
    out = []
    if include_teacher:
        out.append(("teacher", teacher, None))
    for p in sorted(Path(ckpt_dir).glob(f"sac_{task}_*.npz")):
        a = rl.NumpyActor(p)
        out.append((a.name.replace(f"sac_{task}_", ""), a, a.meta["report"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="balance", choices=list(rl.EPISODE_S))
    ap.add_argument("--score", default="full", choices=list(rl.REWARDS),
                    help="reward used to score every agent (not the one it trained on)")
    ap.add_argument("--suite", default="clean", choices=list(SUITES))
    ap.add_argument("--ckpt-dir", default=str(rl.CKPT_DIR))
    ap.add_argument("--no-teacher", action="store_true")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    ags = agents(a.task, a.ckpt_dir, not a.no_teacher)
    if not ags:
        raise SystemExit(f"no checkpoints in {a.ckpt_dir} -- run train_sac.py --sweep first")

    print(f"task {a.task} | scored with reward '{a.score}' | "
          f"{len(rl.BALANCE_STARTS if a.task == 'balance' else rl.SWINGUP_STARTS)} starts per cell")
    rows = []
    for cond in SUITES[a.suite]:
        label = cond["label"]
        plant = {k: v for k, v in cond.items() if k != "label"}
        print(f"\n--- {label} " + "-" * 60)
        print(f"{'agent':<12s}" + "".join(f"{h:>8s}" for _, h, _ in COLS))
        for name, act, rep in ags:
            ev = rl.evaluate(act, a.task, a.score, **plant)
            print(f"{name:<12s}" + "".join(fmt.format(ev[k]) for k, _, fmt in COLS))
            rows.append(dict(cond=label, agent=name, **{k: ev[k] for k, _, _ in COLS},
                             trained_on=(rep or {}).get("reward", "-")))

    print("\n  hold = % of the last 2 s inside the band | bias = mean angle there"
          "\n  (a steady policy can still sit several degrees off vertical: that is bias, not wobble)"
          "\n  t_up = first time inside +-10 deg | arm = largest |phi| seen (limit 140)")

    if a.csv:
        import csv
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\n  -> {a.csv}")


if __name__ == "__main__":
    main()
