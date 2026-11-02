"""One arena where the expert, the cloned policies and the RL policies are scored side by side.

The two earlier folders cannot be compared as they stand:

    behavior_cloning_yeen   rm.Loop at 100 Hz, quantised encoder, 4 ms delay, u in rad/s^2,
                            25 s swing-up, scored as "fraction of the tail inside +-10 deg"
    rl_yeen                 its own 50 Hz RK4 env, perfect sensing, a in [-1, 1],
                            5 s balance / 10 s swing-up, scored as a reward return

A number from one is not a number from the other. So every agent is brought here instead, and
here is the rig's own loop: rotary_model.Loop, 100 Hz, realistic sensor, 140 deg arm stop. Three
consequences worth stating up front, because they are the point of doing it this way:

  * The RL policies were trained at 50 Hz with perfect sensing and are evaluated at 100 Hz with
    a quantised encoder and a 4 ms delay. That mismatch is not a flaw in the comparison, it IS
    the sim-to-real question -- a policy that only works in its own training env is not a result.
    (NumpyActor.as_rig_controller() holds each command for the 10 ms tick, as the rig would.)
  * The return is summed every OTHER tick, so it is on the same 50 Hz scale as the numbers
    printed by rl_yeen/eval_rl.py and the two are directly readable against each other.
  * `offset` is in the robustness suite on purpose. An encoder zero error is exactly what the
    expert absorbs with its hidden b_hat integrator and what the cloned policies could not --
    the 8.9 deg bias. No policy here has an integrator either, so this column predicts how each
    one would sit on a rig whose vertical is not quite where the encoder thinks it is.

    python pipeline_yeen/arena.py                      swing-up, clean plant
    python pipeline_yeen/arena.py --task balance --suite robust --csv out.csv
"""
import os

# conda's MKL and pip's torch each bundle their own Intel OpenMP; same workaround as the other
# two folders, set before anything can pull torch in.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import argparse               # noqa: E402
import contextlib             # noqa: E402
import math                   # noqa: E402
import sys                    # noqa: E402
from pathlib import Path      # noqa: E402

import numpy as np            # noqa: E402

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figs"


def _sibling(name, must_contain):
    """Find a package folder without assuming how deep this file is nested.

    This folder has been moved once already (Yeeun/pipeline_yeen), and the ones it imports from
    may move too. So walk up from here and take the first ancestor that actually holds the
    folder, instead of hard-coding parents[1].
    """
    for base in _BASES:
        p = base / name
        if (p / must_contain).is_file():
            return p
    return None


# Path.parents is not sliceable before 3.10, so build the list by hand.
_BASES = tuple(list(ROOT.parents)[:5])


SIM_DIR = _sibling("simulator", "rotary_model.py")
RL_DIR = _sibling("rl_yeen", "rl_common.py")
BC_DIR = _sibling("behavior_cloning_yeen", "il_common.py")      # optional: only for its ckpts

if SIM_DIR is None or RL_DIR is None:
    missing = [n for n, d in (("simulator", SIM_DIR), ("rl_yeen", RL_DIR)) if d is None]
    raise ImportError(
        f"cannot find {' and '.join(missing)} above {ROOT}.\n"
        f"  Both must sit somewhere on the way up from this file -- in the repository root\n"
        f"  alongside it, or beside this folder. Looked in: "
        + ", ".join(str(b) for b in _BASES))

REPO = SIM_DIR.parent
for p in (SIM_DIR, RL_DIR, BC_DIR):
    if p is not None:
        sys.path.insert(0, str(p))

import rotary_model as rm     # noqa: E402
import rl_common as rl        # noqa: E402

if BC_DIR is not None:
    import il_common as il    # noqa: E402
else:                         # the imitation folder is not needed to run the fusion pipeline
    il = None

DT_TICK = 0.01                  # 100 Hz: one Loop control step is advance(10)
ARM_LIMIT_DEG = 140.0           # the host-side stop, same as il_common and the rig
U_MAX = rm.ACCEL_MAX
K4 = rm.place4()


@contextlib.contextmanager
def model_errors(coulomb=1.0, delay=None):
    """rotary_model keeps Coulomb friction and the loop delay in module globals.

    Same helper as il_common.model_errors, kept here so this arena works with or without the
    imitation-learning folder present.
    """
    f0, d0 = rm.F_C, rm.DELAY
    rm.F_C = f0 * coulomb
    rm.DELAY = d0 if delay is None else delay
    try:
        yield
    finally:
        rm.F_C, rm.DELAY = f0, d0

# ---------------------------------------------------------------- tasks
# A swing-up agent has to get there; a balance agent is handed the rod already up. Scoring the
# second one from hanging would just print "it fell over", which measures nothing.
TASKS = dict(
    swingup=dict(seconds=25.0, tail_s=5.0,
                 starts=[("+1 deg", 180.0 + 1.0), ("-1 deg", 180.0 - 1.0), ("+2 deg", 180.0 + 2.0)],
                 doc="hanging rod, the rig's actual task"),
    balance=dict(seconds=10.0, tail_s=5.0,
                 starts=[("+5 deg", 5.0), ("-8 deg", -8.0), ("+11 deg", 11.0), ("-3 deg", -3.0)],
                 doc="handed the rod near upright"),
)

SUITES = dict(
    clean=[dict(label="clean")],
    robust=[dict(label="clean"),
            dict(label="b -20%", b=0.8 * rm.B_U),
            dict(label="b +20%", b=1.2 * rm.B_U),
            dict(label="2x fric", coulomb=2.0),
            dict(label="offset 2", offset_deg=2.0)],
)


# ---------------------------------------------------------------- the loop
def run(ctrl, task="swingup", theta0_deg=None, seconds=None, b=rm.B_U, coulomb=1.0, delay=None,
        offset_deg=0.0, realistic=True, kicks=()):
    """One episode in rm.Loop. `ctrl(z, t) -> u [rad/s^2]`, the rig's controller contract.

    Returns arrays at 100 Hz plus the host's stop reason ("" if it never tripped).
    """
    spec = TASKS[task]
    seconds = spec["seconds"] if seconds is None else seconds
    th0 = spec["starts"][0][1] if theta0_deg is None else theta0_deg

    if hasattr(ctrl, "reset"):
        ctrl.reset()                      # Loop.reset() does not do this for us
    pending = sorted(kicks)
    keys = ("t", "th", "w", "phi", "phi_dot", "u")
    log = {k: [] for k in keys}
    with model_errors(coulomb, delay):
        plant = rm.Plant(math.radians(th0), b=b)
        lp = rm.Loop(plant, ctrl, rm.Sensor(realistic), arm_limit_deg=ARM_LIMIT_DEG,
                     offset=math.radians(offset_deg))
        while plant.t < seconds:
            while pending and plant.t >= pending[0][0]:
                plant.kick(pending.pop(0)[1])
            lp.advance(int(round(DT_TICK / rm.DT)))
            for k, v in zip(keys, (plant.t, rm.wrap(plant.x[0]), plant.x[1],
                                   plant.x[2], plant.x[3], lp.u)):
                log[k].append(v)
    out = {k: np.asarray(v, np.float64) for k, v in log.items()}
    out["stop"] = lp.stop
    out["task"] = task
    return out


# ---------------------------------------------------------------- scoring
def score(lg, reward="full", tail_s=None):
    """The imitation study's metrics and the RL study's return, from the same episode.

    bias and wobble stay separate: a policy can be perfectly steady and still hold the rod
    several degrees off vertical. That is what ±10 deg alone hid, so ±5 and ±2 are here too.
    """
    cfg = rl.REWARDS[reward] if isinstance(reward, str) else reward
    tail_s = TASKS[lg["task"]]["tail_s"] if tail_s is None else tail_s

    th_deg = np.degrees(lg["th"])
    # never score more than the second half of the episode: on a shortened run (`seconds=`)
    # a fixed 5 s window would reach back into the swing-in transient and call it a hold
    n = max(1, int(min(tail_s, 0.5 * len(th_deg) * DT_TICK) / DT_TICK))
    tail = th_deg[-n:]
    up = np.abs(th_deg) < rl.UPRIGHT_DEG

    # every other tick -> the 50 Hz scale rl_yeen/eval_rl.py prints, so the two tables compare
    r_up = r_arm = r_eff = 0.0
    for th, phi, u in zip(lg["th"][::2], lg["phi"][::2], lg["u"][::2]):
        a = float(np.clip(u / U_MAX, -1.0, 1.0))
        du, da, de = rl.reward_terms(cfg, th, phi, a)
        r_up, r_arm, r_eff = r_up + du, r_arm + da, r_eff + de
    ret = r_up + r_arm + r_eff + (cfg["term"] if lg["stop"] else 0.0)

    return dict(
        ret=ret, ret_up=r_up, ret_arm=r_arm, ret_eff=r_eff,
        stopped=bool(lg["stop"]), stop=lg["stop"],
        t_up=float(lg["t"][np.argmax(up)]) if up.any() else float("nan"),
        bias_deg=float(tail.mean()), wobble_deg=float(tail.std()),
        hold10=float(np.mean(np.abs(tail) < 10.0) * 100.0),
        hold5=float(np.mean(np.abs(tail) < 5.0) * 100.0),
        hold2=float(np.mean(np.abs(tail) < 2.0) * 100.0),
        arm_max_deg=float(np.abs(np.degrees(lg["phi"])).max()),
        u_rms=float(np.sqrt(np.mean(lg["u"] ** 2))),
        sat_pct=float(np.mean(np.abs(lg["u"]) >= U_MAX - 1e-9) * 100.0),
    )


def evaluate(ctrl, task="swingup", reward="full", **plant):
    """Mean score over the task's fixed starts -- the same starts for every agent."""
    ms = [score(run(ctrl, task, theta0_deg=d, **plant), reward)
          for _, d in TASKS[task]["starts"]]
    out = {k: float(np.nanmean([m[k] for m in ms])) for k, v in ms[0].items()
           if isinstance(v, (int, float)) and not isinstance(v, bool)}
    out["stops"] = sum(m["stopped"] for m in ms)
    out["why"] = ",".join(sorted({m["stop"] for m in ms if m["stop"]})) or "-"
    out["n_starts"] = len(ms)
    return out


# ---------------------------------------------------------------- the line-up
def expert():
    c = rm.SwingUp(K4)
    c.label = "expert"
    return c


def bc_controllers(ckpt_dir=None):
    """Every behaviour-cloning / DAgger checkpoint from the imitation folder, newest stage last.

    Empty if that folder is not there: those checkpoints use the 100 Hz stacked-frame convention
    and belong to the earlier study, so the fusion pipeline does not need them.
    """
    if ckpt_dir is None:
        if BC_DIR is None:
            return []
        ckpt_dir = BC_DIR / "checkpoints"
    out = []
    for p in sorted(Path(ckpt_dir).glob("*.npz")):
        c = il.BCPolicy(p)
        c.label = p.stem
        out.append(c)
    return out


def rl_controllers(task=None, ckpt_dir=None):
    """Every SAC actor, wrapped so it drives the 100 Hz rig loop.

    A balance-only actor is still listed for the swing-up task: that it cannot swing up is a
    fact about what it was trained on, not a gap in the table.
    """
    d = Path(ckpt_dir or rl.CKPT_DIR)
    out = []
    for p in sorted(d.glob("sac_*.npz")):
        if task and not p.stem.startswith(f"sac_{task}_"):
            continue
        a = rl.NumpyActor(p)
        c = a.as_rig_controller()
        c.label = p.stem.replace("sac_", "")
        out.append(c)
    return out


def line_up(task="swingup", bc=True, rl_=True, teacher=True):
    out = []
    if teacher:
        out.append(expert())
    if bc:
        out += bc_controllers()
    if rl_:
        out += rl_controllers(task)
    return out


# ---------------------------------------------------------------- table
COLS = [("ret", "return", "{:8.1f}"), ("hold10", "hold10", "{:8.1f}"), ("hold5", "hold5", "{:8.1f}"),
        ("hold2", "hold2", "{:8.1f}"), ("bias_deg", "bias", "{:+8.2f}"),
        ("wobble_deg", "wobble", "{:8.2f}"), ("t_up", "t_up", "{:8.2f}"),
        ("arm_max_deg", "arm", "{:8.1f}"), ("u_rms", "u_rms", "{:8.2f}"),
        ("sat_pct", "sat%", "{:8.1f}"), ("stops", "stop", "{:8.0f}")]


def print_table(rows, title=""):
    if title:
        print(f"\n--- {title} " + "-" * max(0, 58 - len(title)))
    print(f"{'agent':<18s}" + "".join(f"{h:>8s}" for _, h, _ in COLS))
    for r in rows:
        print(f"{r['agent']:<18s}" + "".join(fmt.format(r[k]) for k, _, fmt in COLS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="swingup", choices=list(TASKS))
    ap.add_argument("--suite", default="clean", choices=list(SUITES))
    ap.add_argument("--score", default="full", choices=list(rl.REWARDS),
                    help="every agent is scored with this reward, whatever it trained on")
    ap.add_argument("--no-bc", action="store_true")
    ap.add_argument("--no-rl", action="store_true")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    ags = line_up(a.task, bc=not a.no_bc, rl_=not a.no_rl)
    spec = TASKS[a.task]
    print(f"task {a.task} ({spec['doc']}) | {spec['seconds']:.0f} s | "
          f"{len(spec['starts'])} starts | 100 Hz rm.Loop, realistic sensor | "
          f"scored with reward '{a.score}'")

    rows = []
    for cond in SUITES[a.suite]:
        plant = {k: v for k, v in cond.items() if k != "label"}
        here = []
        for c in ags:
            ev = evaluate(c, a.task, a.score, **plant)
            here.append(dict(agent=c.label, **ev))
            rows.append(dict(cond=cond["label"], agent=c.label,
                             **{k: ev[k] for k, _, _ in COLS}, why=ev["why"]))
        print_table(here, cond["label"])

    print("\n  hold = % of the last "
          f"{spec['tail_s']:.0f} s inside the band | bias = mean angle there, wobble = its std"
          "\n  (steady and off-vertical at the same time is possible: that is bias, not wobble)"
          "\n  t_up = first time inside +-10 deg | arm = largest |phi| (host stops at 140)"
          "\n  return is on the 50 Hz scale of rl_yeen/eval_rl.py, so the two tables compare")

    if a.csv:
        import csv
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\n  -> {a.csv}")


if __name__ == "__main__":
    main()
