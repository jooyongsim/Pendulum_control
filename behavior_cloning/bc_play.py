"""Run a trained behavior-cloning policy (bc_policy.pt) in the rotary pendulum simulator.

    python behavior_cloning/bc_play.py                    real-time window, student drives
    python behavior_cloning/bc_play.py --check            numbers only: student vs teacher, 20 starts
    python behavior_cloning/bc_play.py --model path.pt    another checkpoint

Like in training, the teacher gives the 0.3 s open-loop start kick; after that the network alone
sets the arm acceleration from [theta, theta', phi, phi'].

Window keys (click the window first):
    left / right : push the rod (theta' -/+ 1 rad/s)
    up           : knock it over (theta' + 5 rad/s)
    t            : switch driver student <-> teacher (restarts from hanging)
    r            : restart from hanging
    space        : pause
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bc_train  # noqa: E402  (also puts simulator/ on sys.path)
from bc_train import Policy, rm  # noqa: E402

ARM_LIMIT_DEG = 140.0
UP_DEG = 12.0                    # "upright" = within the teacher's catch window


def load_policy(path):
    sd = torch.load(path, map_location="cpu")
    model = Policy(sd["mean"], sd["std"], hidden=sd["net.0.weight"].shape[0])
    model.load_state_dict(sd)
    model.eval()
    return model


class Student:
    """Teacher's 0.3 s start kick, then the network."""

    def __init__(self, model):
        self.model, self.kick, self.t0 = model, rm.SwingUp(rm.place4()), None

    def __call__(self, z, t):
        self.t0 = t if self.t0 is None else self.t0
        return self.kick(z, t) if t - self.t0 < 0.30 else self.model.act(z)


def make(driver, model, tilt_deg=1.0, b=rm.B_U):
    plant = rm.Plant(math.pi + math.radians(tilt_deg), b=b)
    ctrl = Student(model) if driver == "student" else rm.SwingUp(rm.place4())
    return rm.Loop(plant, ctrl, rm.Sensor(True), arm_limit_deg=ARM_LIMIT_DEG)


def episode(driver, model, tilt_deg, b, f_c_scale, duration=15.0):
    f_c = rm.F_C
    rm.F_C = f_c * f_c_scale
    try:
        lp = make(driver, model, tilt_deg, b)
        up, t_up, arm_max = 0.0, None, 0.0
        while lp.plant.t < duration and not lp.stop:
            lp.advance(10)
            arm_max = max(arm_max, abs(math.degrees(lp.plant.x[2])))
            if abs(math.degrees(rm.wrap(lp.plant.x[0]))) < UP_DEG:
                up += 1.0 / rm.CONTROL_HZ
                t_up = lp.plant.t if t_up is None else t_up
    finally:
        rm.F_C = f_c
    return dict(t_up=t_up, up=up, arm=arm_max, stop=lp.stop)


def check(model, n=20, seed=123):
    """Same randomised starts as training (tilt, b, friction), no pushes."""
    rng = np.random.default_rng(seed)
    starts = [(rng.uniform(0.5, 3.0) * rng.choice([-1, 1]), rm.B_U * rng.uniform(0.9, 1.1),
               rng.uniform(0.5, 2.0)) for _ in range(n)]
    print(f"{n} starts, 15 s each; 'up' = |theta| < {UP_DEG:.0f} deg")
    print(f"{'':>3} {'tilt':>6} {'b':>5} {'Fc x':>5} | {'student: first up / time up / arm max':<40}| teacher")
    res = {"student": [], "teacher": []}
    for i, (tilt, b, fc) in enumerate(starts):
        row = []
        for who in ("student", "teacher"):
            r = episode(who, model, tilt, b, fc)
            res[who].append(r)
            first = f"{r['t_up']:5.2f} s" if r["t_up"] is not None else "  never"
            row.append(f"{first} / {r['up']:4.1f} s / {r['arm']:5.1f} deg" + (f" {r['stop']}" if r["stop"] else ""))
        print(f"{i:3d} {tilt:+6.2f} {b:5.3f} {fc:5.2f} | {row[0]:<40}| {row[1]}")
    for who, rs in res.items():
        ok = [r for r in rs if r["t_up"] is not None and not r["stop"]]
        mean_up = np.mean([r["up"] for r in rs])
        first = np.median([r["t_up"] for r in ok]) if ok else float("nan")
        print(f"{who:8s}: reached upright and kept running {len(ok)}/{len(rs)}, "
              f"median first-up {first:.2f} s, mean time upright {mean_up:.1f} of 15 s")


def run(model):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    S = {"driver": "student", "paused": False, "last": None}

    def reset():
        S.update(lp=make(S["driver"], model), last=None, T=[], TH=[], PH=[], U=[])

    reset()
    fig = plt.figure(figsize=(13, 6.5))
    ax_side = fig.add_axes([0.03, 0.10, 0.36, 0.78])
    ax_top = fig.add_axes([0.28, 0.66, 0.11, 0.22])
    ax_th = fig.add_axes([0.47, 0.66, 0.50, 0.25])
    ax_ph = fig.add_axes([0.47, 0.37, 0.50, 0.25], sharex=ax_th)
    ax_u = fig.add_axes([0.47, 0.08, 0.50, 0.25], sharex=ax_th)

    lim = 1.25 * rm.L
    ax_side.set_aspect("equal"); ax_side.set_xlim(-lim, lim); ax_side.set_ylim(-lim, lim)
    ax_side.set_title("side view (up = balanced)", loc="left")
    ax_side.add_patch(plt.Circle((0, 0), rm.L, fill=False, color="0.85", ls="--"))
    rod, = ax_side.plot([], [], lw=8, color="#E97820", solid_capstyle="round")
    ax_side.plot([0], [0], "ko", ms=7)
    info = ax_side.text(-0.98 * lim, -0.70 * lim, "", fontsize=8, family="monospace", va="top")
    who_txt = ax_side.text(0, 1.12 * rm.L, "", ha="center", fontsize=13, weight="bold")

    ax_top.set_aspect("equal"); ax_top.set_xlim(-1.3, 1.3); ax_top.set_ylim(-1.3, 1.3)
    ax_top.set_xticks([]); ax_top.set_yticks([]); ax_top.set_title("arm (top view)", fontsize=8)
    ax_top.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="0.7"))
    for s in (1, -1):
        a = math.radians(ARM_LIMIT_DEG)
        ax_top.plot([0, s * math.sin(a)], [0, math.cos(a)], "r:", lw=0.8)
    arm, = ax_top.plot([], [], lw=4, color="#2774B8")

    line_th, = ax_th.plot([], [], color="#E97820")
    line_ph, = ax_ph.plot([], [], color="#2774B8")
    line_u, = ax_u.plot([], [], color="#7B3FA0", lw=0.8)
    ax_th.set(ylabel="theta from up [deg]", ylim=(-190, 190), yticks=[-180, -90, 0, 90, 180])
    ax_th.axhspan(-UP_DEG, UP_DEG, color="#2E8B57", alpha=0.15)
    ax_ph.set(ylabel="arm phi [deg]", ylim=(-160, 160))
    for s in (1, -1):
        ax_ph.axhline(s * ARM_LIMIT_DEG, color="r", ls=":", lw=0.8)
    ax_u.set(ylabel="command u [rad/s^2]", xlabel="t [s]", ylim=(-50, 50))
    for a in (ax_th, ax_ph, ax_u):
        a.grid(alpha=.3)

    def on_key(ev):
        pl = S["lp"].plant
        if ev.key == "left": pl.kick(-1.0)
        elif ev.key == "right": pl.kick(1.0)
        elif ev.key == "up": pl.kick(5.0)
        elif ev.key == "t":
            S["driver"] = "teacher" if S["driver"] == "student" else "student"; reset()
        elif ev.key == "r": reset()
        elif ev.key == " ": S["paused"] = not S["paused"]; S["last"] = None
    fig.canvas.mpl_connect("key_press_event", on_key)

    def update(_):
        lp = S["lp"]
        now = time.perf_counter()
        S["last"] = now if S["last"] is None else S["last"]
        elapsed = min(now - S["last"], 0.1); S["last"] = now
        if not S["paused"]:
            lp.advance(int(elapsed / rm.DT))
        pl = lp.plant
        th, ph = rm.wrap(pl.x[0]), pl.x[2]
        up = abs(math.degrees(th)) < UP_DEG
        S["T"].append(pl.t); S["TH"].append(math.degrees(th)); S["PH"].append(math.degrees(ph)); S["U"].append(lp.u)
        rod.set_data([0, rm.L * math.sin(th)], [0, rm.L * math.cos(th)])
        rod.set_color("#2E8B57" if up else "#E97820")
        arm.set_data([0, math.sin(ph)], [0, math.cos(ph)])
        who_txt.set_text(lp.stop or ("STUDENT (network)" if S["driver"] == "student" else "TEACHER (SwingUp)"))
        who_txt.set_color("r" if lp.stop else "#7B3FA0" if S["driver"] == "student" else "#2774B8")
        info.set_text(f"t={pl.t:6.2f}s  theta={math.degrees(th):7.1f}  arm={math.degrees(ph):7.1f} deg\n"
                      f"u={lp.u:+6.1f} rad/s^2{'  PAUSED' if S['paused'] else ''}\n"
                      "keys: <- -> push   up knock over\n"
                      "      t student/teacher   r restart   space pause")
        t0 = max(0.0, pl.t - 12.0)
        ax_th.set_xlim(t0, t0 + 12.0)
        line_th.set_data(S["T"], S["TH"]); line_ph.set_data(S["T"], S["PH"]); line_u.set_data(S["T"], S["U"])
        return rod, arm, line_th, line_ph, line_u, info, who_txt

    S["ani"] = FuncAnimation(fig, update, interval=20, cache_frame_data=False)
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(bc_train.OUT / "bc_policy.pt"))
    ap.add_argument("--check", action="store_true", help="numbers only, no window")
    ap.add_argument("-n", type=int, default=20, help="starts for --check")
    args = ap.parse_args()
    model = load_policy(args.model)
    print(f"loaded {args.model}")
    if args.check:
        check(model, args.n)
    else:
        run(model)


if __name__ == "__main__":
    main()
