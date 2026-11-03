"""Swing-up videos and one evaluation table for all three approaches, scored in one arena.

    1 classic      energy swing-up + a fixed gain      (no learning)
    2 imitation    behaviour cloning / DAgger          (learned from the expert)
    3 fusion       BC -> DAgger -> SAC fine-tune       (learned from the reward, warm-started)

Everything is driven through rotary_model.Loop at 100 Hz with a quantised encoder and the 4 ms
delay -- the rig's own loop -- so a number from one group is a number from the others. That is
the whole reason this file exists: the three studies were built in three different conventions
and their printed scores were never comparable.

    python Yeeun/pipeline_yeen/gallery.py --list
    python Yeeun/pipeline_yeen/gallery.py --table
    python Yeeun/pipeline_yeen/gallery.py --video all            # needs ffmpeg

On the classic group: a PID/PD law cannot swing up. `-K x` from hanging is meaningless, so the
fixed-gain controllers are paired with the same energy-pumping state machine the rig uses and
differ only in the catch gain:

    pid_rod     place2() = [k1, k2, 0, 0]   the rod only -- the arm is not fed back at all
    pole_4state place4()                    the rod and the arm  (this is the expert)

The arm term is the whole difference, and it is visible in the video rather than argued about:
place2 has no phi feedback, so SwingUp's upright-offset estimator is dead too (its gain is
|K[2]| / ..., which is zero), and nothing holds the arm near the middle.
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import argparse              # noqa: E402
import math                  # noqa: E402
from pathlib import Path     # noqa: E402

import numpy as np           # noqa: E402

import arena                 # noqa: E402  -- puts simulator/, rl_yeen/, bc/ on sys.path
import rotary_model as rm    # noqa: E402
import rl_common as rl       # noqa: E402

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "gallery"

# Default palette. The data-size study's lightest tone (#C9D9E6) is left out: it is legible as
# the first of four shaded blues but not as a panel title on white.
COLORS = ["#3E7CB1", "#E07A3F", "#2E8B57", "#8E5A9E", "#6B7A85", "#1F4E79"]

# A controller may carry its own .color; the groups below set it where the colour means
# something -- light to dark across the data-size sweep, one hue per approach elsewhere.
BLUES = ["#A9C4D9", "#6E9CC4", "#3E7CB1", "#1F4E79"]
HUE = dict(pid="#8E5A9E", expert="#3E7CB1", bc="#6E9CC4", dagger="#E07A3F",
           sac_warm="#2E8B57", sac_cold="#6B7A85")
BAND = rl.UPRIGHT_DEG            # the +-10 deg upright wedge
VIDEO_S = 12.0                   # swing-up finishes in ~2.5 s; 12 s shows the hold without bloat
STRIDE = 4                       # 100 Hz / 4 = 25 fps, i.e. real time


# ---------------------------------------------------------------- the line-up
def pid_rod():
    """Energy swing-up with the rod-only gain. No arm feedback anywhere."""
    c = rm.SwingUp(rm.place2())
    c.label, c.color = "PID (rod only)", HUE["pid"]
    return c


def pole_4state():
    c = rm.SwingUp(rm.place4())
    # labels stay <= 17 chars: arena.print_table pads the agent column to 18
    c.label, c.color = "4-state (expert)", HUE["expert"]
    return c


def _bc(path, label, color=None):
    import il_common as il
    c = il.BCPolicy(path)
    c.label, c.color = label, color
    return c


def _actor(path, label, color=None):
    c = rl.NumpyActor(path).as_rig_controller()
    c.label, c.color = label, color
    return c


def _episodes(path):
    """Training episodes recorded in a BC checkpoint, for sorting and labelling."""
    import json
    rep = json.loads(str(np.load(path, allow_pickle=False)["meta_json"])).get("report", {})
    return int(rep.get("episodes", 0))


def classic():
    return [pid_rod(), pole_4state()]


def imitation(sweep_dir=None, with_dagger=True):
    """The data-size sweep if it is there, plus whatever is committed.

    The sweep checkpoints are NOT in the repository -- the data-size notebook wrote them to
    /content/sweep and they were never committed. Retrain them (gallery.ipynb does) or this
    group falls back to the committed bc_h1/h5/h30 + DAgger iterations.
    """
    out = []
    sweep = Path(sweep_dir) if sweep_dir else None
    if sweep and sweep.is_dir():
        ps = sorted(sweep.glob("*.npz"), key=_episodes)
        for i, p in enumerate(ps):                   # light -> dark as the data grows
            n = _episodes(p)
            shade = BLUES[min(i * len(BLUES) // max(len(ps), 1), len(BLUES) - 1)]
            out.append(_bc(p, f"BC {n} ep" if n else p.stem, shade))
    if arena.BC_DIR is not None:
        ck = arena.BC_DIR / "checkpoints"
        if not out:                                  # no sweep -> show the history variants
            def hist_n(p):                           # bc_h1 < bc_h5 < bc_h30, not lexicographic
                return int(p.stem.split("bc_h")[-1])
            ps = sorted(ck.glob("bc_h*.npz"), key=hist_n)
            for i, p in enumerate(ps):
                out.append(_bc(p, f"BC hist {hist_n(p)}", BLUES[i % len(BLUES)]))
        # the DAgger model as a reference point beside the sweep; left out when the question is
        # purely "how much data", since DAgger changes WHERE the data comes from as well
        last = sorted(ck.glob("dagger_*.npz"))
        if with_dagger and last:
            out.append(_bc(last[-1], "DAgger " + last[-1].stem.split("_")[-1], HUE["dagger"]))
    return out


def dagger_growth():
    """bc_h5 -> dagger iter1 -> iter2 -> iter3: the same policy as its dataset grows.

    A data-size axis that needs no retraining, because the committed DAgger checkpoints already
    are one: history 5 throughout, 206 -> 246 -> 286 -> 326 episodes, 40 added per round.

    It is NOT the same question as the data-size sweep (31/47/65/206 expert episodes). Here the
    added episodes come from the STUDENT's own states, labelled by the teacher, so this measures
    "more of the right data" while the sweep measures "more data". The sweep's checkpoints were
    written to /content/sweep and never committed, so they have to be retrained -- see
    gallery.ipynb section 2.
    """
    if arena.BC_DIR is None:
        return []
    ck = arena.BC_DIR / "checkpoints"
    out, ps = [], [ck / "bc_h5.npz"] + sorted(ck.glob("dagger_h5_iter*.npz"))
    for i, p in enumerate(ps):
        if not p.is_file():
            continue
        n = _episodes(p)
        shade = BLUES[min(i, len(BLUES) - 1)]
        out.append(_bc(p, f"{n} ep" + ("" if i == 0 else f" (D{i})"), shade))
    return out


def fusion(ckpt_dir=None):
    d = Path(ckpt_dir) if ckpt_dir else (ROOT / "checkpoints")
    nice = dict(bc="BC (RL obs)", dagger="DAgger (RL obs)",
                sac_warm="SAC warm", sac_cold="SAC cold (ctrl)")
    out = []
    if not d.is_dir():
        return out
    for key, label in nice.items():
        for p in sorted(d.glob(f"{key}_*.npz")):
            out.append(_actor(p, label, HUE.get(key, HUE["bc"])))
    return out


def best_of_each(sweep_dir=None, ckpt_dir=None):
    """One representative per approach -- the figure for the report."""
    out = [pole_4state()]
    im = imitation(sweep_dir)
    if im:
        out.append(im[-1])                           # the DAgger one if present, else the largest
    fu = fusion(ckpt_dir)
    warm = [c for c in fu if "warm" in c.label]
    out += warm or fu[-1:]
    return out


def four_approaches(ckpt_dir=None, with_expert=True, **kw):
    """PID / BC / BC+DAgger / BC+DAgger+RL -- the four the report compares.

    The last three are the SAME network at three points in its life, so this row is a history
    of one policy rather than four rivals. PID is in front of it as the thing that was there
    before any learning. The expert leads as the line the imitation stages are copying.
    """
    d = Path(ckpt_dir) if ckpt_dir else (ROOT / "checkpoints")
    want = [("bc", "BC"), ("dagger", "BC+DAgger"), ("sac_warm", "BC+DAgger+RL")]
    out = [pid_rod()]
    if with_expert:
        out.append(pole_4state())
    for key, label in want:
        ps = sorted(d.glob(f"{key}_*.npz"))
        if ps:
            out.append(_actor(ps[0], label, HUE[key]))
        else:
            print(f"  (missing {key}_*.npz in {d} -- '{label}' left out)")
    return out


GROUPS = {
    "classic": lambda **kw: classic(),
    "imitation": lambda sweep_dir=None, **kw: imitation(sweep_dir),
    "fusion": lambda ckpt_dir=None, **kw: fusion(ckpt_dir),
    "growth": lambda **kw: dagger_growth(),
    "four": lambda ckpt_dir=None, **kw: four_approaches(ckpt_dir),
    "summary": lambda **kw: best_of_each(**kw),
}

TITLES = {
    "classic": "1. classic control -- energy swing-up + fixed gain",
    # the sweep varies training episodes; without it the fallback varies frames of history
    "imitation": "2. imitation learning -- behaviour cloning variants",
    "fusion": "3. imitation + DAgger + reinforcement learning",
    "growth": "the dataset grows: 206 -> 246 -> 286 -> 326 episodes (history 5)",
    "four": "PID  /  BC  /  BC+DAgger  /  BC+DAgger+RL",
    "summary": "one from each approach",
}


# ---------------------------------------------------------------- video
def render(entries, out_path, task="swingup", seconds=VIDEO_S, stride=STRIDE, fps=25,
           title="", dpi=80, **plant):
    """Pendulum panels over a shared theta(t) / phi(t) plot, written as mp4 (ffmpeg) or gif.

    One panel per controller, same start, same plant: anything that differs between panels is
    the controller and nothing else.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Wedge

    runs = []
    for i, c in enumerate(entries):
        lg = arena.run(c, task, seconds=seconds, **plant)
        runs.append(dict(label=getattr(c, "label", f"#{i}"),
                         color=getattr(c, "color", None) or COLORS[i % len(COLORS)],
                         t=lg["t"], th=np.degrees(lg["th"]), phi=np.degrees(lg["phi"]),
                         stop=lg["stop"], score=arena.score(lg)))

    n = len(runs)
    n_tick = max(len(r["t"]) for r in runs)
    idx = np.arange(0, n_tick, stride)
    t_end = max(r["t"][-1] for r in runs)

    fig = plt.figure(figsize=(3.0 * n + 0.6, 6.4), dpi=dpi)
    L, lim = rm.L, 1.34 * rm.L
    w = 0.96 / n
    panels = []
    for i, r in enumerate(runs):
        ax = fig.add_axes([0.02 + i * w, 0.555, w * 0.92, 0.37])
        ax.set_aspect("equal")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.add_patch(Wedge((0, 0), L * 1.06, 90 - BAND, 90 + BAND,
                           facecolor="#2E8B57", alpha=0.16, edgecolor="none"))
        for s in (1, -1):
            a = math.radians(BAND)
            ax.plot([0, s * L * 1.06 * math.sin(a)], [0, L * 1.06 * math.cos(a)],
                    color="#2E8B57", lw=0.9, ls=":")
        ax.add_patch(plt.Circle((0, 0), L, fill=False, color="0.9", ls="--", lw=0.7))
        rod, = ax.plot([], [], lw=6, color=r["color"], solid_capstyle="round")
        bob, = ax.plot([], [], "o", ms=8, color=r["color"])
        ax.plot([0], [0], "ko", ms=4)
        ax.set_title(r["label"], fontsize=10, weight="bold", color=r["color"])
        txt = ax.text(0, -1.24 * L, "", ha="center", fontsize=9, family="monospace")
        panels.append((rod, bob, txt))

    ax_th = fig.add_axes([0.07, 0.30, 0.89, 0.21])
    ax_ph = fig.add_axes([0.07, 0.07, 0.89, 0.17], sharex=ax_th)
    lines_th, lines_ph, heads = [], [], []
    for r in runs:
        lt, = ax_th.plot([], [], lw=1.6, color=r["color"], label=r["label"])
        lp, = ax_ph.plot([], [], lw=1.6, color=r["color"])
        hd, = ax_th.plot([], [], "o", ms=4, color=r["color"])
        lines_th.append(lt)
        lines_ph.append(lp)
        heads.append(hd)
    ax_th.axhspan(-BAND, BAND, color="#2E8B57", alpha=0.16)
    ax_th.text(0.15, 0, "upright band", fontsize=8, color="#2E8B57", va="center")
    ax_th.set_xlim(0, t_end)
    ax_th.set_ylim(-190, 190)
    ax_th.set_yticks([-180, -90, 0, 90, 180])
    ax_th.set_ylabel("theta from up [deg]", fontsize=9)
    ax_th.tick_params(labelsize=8, labelbottom=False)
    ax_th.grid(alpha=0.3)
    ax_th.legend(ncol=min(n, 4), fontsize=8, loc="lower right")

    for s in (1, -1):
        ax_ph.axhline(s * arena.ARM_LIMIT_DEG, color="#C0392B", lw=1, ls="--")
    ax_ph.text(0.15, arena.ARM_LIMIT_DEG * 0.78, "arm limit -- the host stops here",
               fontsize=8, color="#C0392B", va="center")
    ax_ph.set_ylim(-1.15 * arena.ARM_LIMIT_DEG, 1.15 * arena.ARM_LIMIT_DEG)
    ax_ph.set_ylabel("arm phi [deg]", fontsize=9)
    ax_ph.set_xlabel("t [s]", fontsize=9)
    ax_ph.tick_params(labelsize=8)
    ax_ph.grid(alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=11, y=0.985)

    def frame(k):
        out = []
        for (rod, bob, txt), lt, lp, hd, r in zip(panels, lines_th, lines_ph, heads, runs):
            i = min(idx[k], len(r["t"]) - 1)
            th = math.radians(r["th"][i])
            x, y = L * math.sin(th), L * math.cos(th)
            inside = abs(r["th"][i]) < BAND
            rod.set_data([0, x], [0, y])
            bob.set_data([x], [y])
            bob.set_color("#2E8B57" if inside else r["color"])
            stopped = bool(r["stop"]) and i >= len(r["t"]) - 1
            txt.set_text(f"{r['th'][i]:+6.1f} deg" + ("\nSTOPPED" if stopped else ""))
            txt.set_color("#C0392B" if stopped else ("#2E8B57" if inside else "0.45"))
            lt.set_data(r["t"][:i + 1], r["th"][:i + 1])
            lp.set_data(r["t"][:i + 1], r["phi"][:i + 1])
            hd.set_data([r["t"][i]], [r["th"][i]])
            out += [rod, bob, txt, lt, lp, hd]
        return out

    ani = FuncAnimation(fig, frame, frames=len(idx), interval=1000 / fps, blit=True)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".mp4":
        try:
            ani.save(str(out_path), writer="ffmpeg", fps=fps, dpi=dpi,
                     savefig_kwargs=dict(facecolor="white"))
        except Exception as e:                       # no ffmpeg -> a gif still plays anywhere
            out_path = out_path.with_suffix(".gif")
            print(f"    ffmpeg unavailable ({type(e).__name__}) -- writing {out_path.name}")
            ani.save(str(out_path), writer="pillow", fps=fps, dpi=dpi)
    else:
        ani.save(str(out_path), writer="pillow", fps=fps, dpi=dpi)
    plt.close(fig)
    return out_path, {r["label"]: r["score"] for r in runs}


# ---------------------------------------------------------------- table
def rows_for(entries, task="swingup", suite="clean", reward="full", group=""):
    out = []
    for cond in arena.SUITES[suite]:
        plant = {k: v for k, v in cond.items() if k != "label"}
        for c in entries:
            ev = arena.evaluate(c, task, reward, **plant)
            out.append(dict(group=group, agent=getattr(c, "label", "?"), cond=cond["label"],
                            **{k: ev[k] for k, _, _ in arena.COLS}, why=ev["why"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="all",
                    choices=list(GROUPS) + ["all"], nargs="+")
    ap.add_argument("--task", default="swingup", choices=list(arena.TASKS))
    ap.add_argument("--suite", default="clean", choices=list(arena.SUITES))
    ap.add_argument("--sweep-dir", default=None,
                    help="folder of the data-size BC checkpoints (bc_*.npz)")
    ap.add_argument("--ckpt-dir", default=None, help="fusion checkpoints")
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--seconds", type=float, default=VIDEO_S)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    groups = list(GROUPS) if "all" in a.group else a.group
    kw = dict(sweep_dir=a.sweep_dir, ckpt_dir=a.ckpt_dir)
    built = {g: GROUPS[g](**kw) for g in groups}

    print(f"arena: 100 Hz rm.Loop, realistic sensor, arm stop {arena.ARM_LIMIT_DEG:.0f} deg | "
          f"task {a.task}")
    for g, es in built.items():
        names = [getattr(c, "label", "?") for c in es]
        print(f"  {g:<10s} {len(es)}: {', '.join(names) if names else '(none found)'}")
    if a.list or not (a.video or a.table):
        if not (a.video or a.table):
            print("\n  nothing to do -- pass --table and/or --video")
        return

    out = Path(a.out)
    all_rows = []
    for g, es in built.items():
        if not es:
            print(f"\n--- {g}: no controllers found, skipped")
            continue
        if a.table:
            print(f"\n--- {g} " + "-" * 60)
            rows = rows_for(es, a.task, a.suite, group=g)
            all_rows += rows
            arena.print_table([r for r in rows if r["cond"] == "clean"], TITLES[g])
        if a.video:
            p, _ = render(es, out / f"{g}_{a.task}.mp4", task=a.task, seconds=a.seconds,
                          fps=a.fps, title=TITLES[g])
            print(f"  video -> {p}  ({p.stat().st_size / 1e6:.1f} MB)")

    if a.csv and all_rows:
        import csv
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
            w.writeheader()
            w.writerows(all_rows)
        print(f"\n  table -> {a.csv}")


if __name__ == "__main__":
    main()
