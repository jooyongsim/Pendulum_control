"""Rotary pendulum swing-up on the rig simulator with a policy trained THROUGH the differentiable world model.

Same window and same simulated rig as Pendulum_control/simulator/rotary_swingup_realtime.py (1 kHz nonlinear
plant, firmware motor limits, 100 Hz loop, 4 ms delay, 0.3 deg encoder) -- only the controller is different:

    diff : one smooth law learned by BPTT through the differentiable RK4 model (no RL, no real rollouts)
           u = U_max tanh(((1-g) u_swing + g u_LQR)/U_max + NN(x)),  g = ((1 + cos theta)/2)^10
    rig  : the hand-designed controller of the real rig (energy pumping, catch, pole-placement balancer)

    python diff_swingup_realtime.py              real-time window (starts with the differentiable policy)
    python diff_swingup_realtime.py --check      numbers only: both controllers, nominal + robustness
    python diff_swingup_realtime.py --plot       one 10 s run of each controller -> swingup_compare.png
    python diff_swingup_realtime.py --policy FILE   use another trained policy (.pt)

Controls (click the window first):
    p            : switch controller  differentiable policy <-> rig swing-up (resets)
    left / right : kick the pendulum (theta' -/+ 1 rad/s)
    up           : knock it over (theta' + 5 rad/s) -> watch it swing up again
    o            : encoder zero error +1 deg
    c            : controller on / off
    x            : realistic (100 Hz, 4 ms delay, 0.3 deg encoder) / ideal
    r            : reset: hanging still, arm at 0
    space        : pause
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np

from diff_controller import DiffPolicyController, load_policy, rig_controller, rm

HERE = Path(__file__).resolve().parent
ARM_LIMIT_DEG = 140.0
START_TILT_DEG = 1.0
ap = argparse.ArgumentParser()
ap.add_argument("--policy", default=None)
ap.add_argument("--check", action="store_true")
ap.add_argument("--plot", action="store_true")
ARGS = ap.parse_args()
POLICY, POLICY_PATH = load_policy(ARGS.policy)


def controller(kind):
    return DiffPolicyController(POLICY) if kind == "diff" else rig_controller()


def make(kind="diff", realistic=True, b=rm.B_U, tilt_deg=START_TILT_DEG, offset_deg=0.0):
    plant = rm.Plant(math.pi + math.radians(tilt_deg), b=b)
    return rm.Loop(plant, controller(kind), rm.Sensor(realistic), arm_limit_deg=ARM_LIMIT_DEG,
                   offset=math.radians(offset_deg))


# ---------------------------------------------------------------- numbers
def simulate(kind, duration=15.0, **kw):
    f0, d0 = rm.F_C, rm.DELAY                       # model-error cases change the shared constants
    rm.F_C, rm.DELAY = f0 * kw.pop("coulomb", 1.0), kw.pop("delay", d0)
    try:
        lp = make(kind, **kw)
        arm_max, sat, n = 0.0, 0, 0
        while lp.plant.t < duration:
            lp.advance(10)
            arm_max = max(arm_max, abs(lp.plant.x[2]))
            sat += abs(lp.u) >= rm.ACCEL_MAX - 1e-9
            n += 1
    finally:
        rm.F_C, rm.DELAY = f0, d0
    c = lp.ctrl
    th = abs(rm.wrap(lp.plant.x[0]))
    result = lp.stop or ("balanced" if th < math.radians(5) else "not caught")
    return dict(result=result, catch=c.t_catch, arm_max=math.degrees(arm_max), falls=c.falls, sat=100.0 * sat / max(n, 1))


def fmt(r):
    catch = f"{r['catch']:5.2f}s" if r["catch"] is not None else "  --  "
    return f"{r['result']:<24} catch {catch}  arm max {r['arm_max']:5.1f}d  re-swings {r['falls']}  sat {r['sat']:4.1f}%"


def check():
    print(f"policy: {POLICY_PATH}")
    cases = [("nominal (realistic)", {}), ("ideal sensor, no delay", dict(realistic=False)),
             ("encoder zero +1.5 deg", dict(offset_deg=1.5)),
             ("b -20%", dict(b=0.8 * rm.B_U)), ("b +20%", dict(b=1.2 * rm.B_U)),
             ("Coulomb x0.5", dict(coulomb=0.5)), ("Coulomb x2", dict(coulomb=2.0)),
             ("delay 12 ms", dict(delay=0.012)), ("start tilt -2.5 deg", dict(tilt_deg=-2.5)),
             ("start tilt +2.5 deg", dict(tilt_deg=2.5))]
    for kind in ("diff", "rig"):
        print(f"\n=== {controller(kind).name} ===")
        for name, kw in cases:
            print(f"  {name:<26}{fmt(simulate(kind, **dict(kw)))}")
        rng = np.random.default_rng(0)
        ok, n = 0, 24
        for _ in range(n):
            kw = dict(b=rm.B_U * rng.uniform(0.8, 1.2), coulomb=rng.uniform(0.5, 2.0),
                      delay=rng.uniform(0.004, 0.012), tilt_deg=rng.uniform(-2.5, 2.5))
            ok += simulate(kind, **kw)["result"] == "balanced"
        print(f"  randomised (b 0.8-1.2x, Coulomb 0.5-2x, delay 4-12 ms, tilt +-2.5 deg): {ok}/{n} balanced")


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for kind, col in (("diff", "#2774B8"), ("rig", "#E97820")):
        lp = make(kind)
        T, TH, PH, U = [], [], [], []
        while lp.plant.t < 10.0:
            lp.advance(10)
            T.append(lp.plant.t); TH.append(abs(math.degrees(rm.wrap(lp.plant.x[0]))))
            PH.append(math.degrees(lp.plant.x[2])); U.append(lp.u)
        c = lp.ctrl
        lab = f"{c.name} (catch {c.t_catch:.2f} s)" if c.t_catch is not None else c.name
        ax[0].plot(T, TH, color=col, label=lab); ax[1].plot(T, PH, color=col); ax[2].plot(T, U, color=col, lw=0.7)
    ax[0].set(ylabel="|theta| from up [deg]", ylim=(-5, 190), yticks=[0, 45, 90, 135, 180])
    ax[0].axhspan(0, 12, color="#2E8B57", alpha=0.12)
    ax[1].set(ylabel="arm phi [deg]"); ax[2].set(ylabel="u [rad/s$^2$]", xlabel="t [s]")
    for s in (1, -1):
        ax[1].axhline(s * ARM_LIMIT_DEG, color="r", ls=":", lw=0.8)
    for a in ax:
        a.grid(alpha=.3)
    ax[0].legend(fontsize=8, loc="lower right")
    ax[0].set_title("rig simulator (realistic: 100 Hz, 4 ms delay, 0.3 deg encoder, motor limits)", fontsize=9)
    plt.tight_layout()
    out = HERE / "swingup_compare.png"
    plt.savefig(out, dpi=130)
    print("saved", out)


# ---------------------------------------------------------------- real-time window
def run():
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    S = {"kind": "diff", "lp": make("diff"), "paused": False, "last": None, "T": [], "TH": [], "PH": [], "E": []}
    fig = plt.figure(figsize=(13.5, 7))
    ax_side = fig.add_axes([0.03, 0.10, 0.36, 0.78])
    ax_top = fig.add_axes([0.28, 0.66, 0.11, 0.22])
    ax_th = fig.add_axes([0.47, 0.66, 0.50, 0.25])
    ax_ph = fig.add_axes([0.47, 0.37, 0.50, 0.25], sharex=ax_th)
    ax_e = fig.add_axes([0.47, 0.08, 0.50, 0.25], sharex=ax_th)

    lim = 1.25 * rm.L
    ax_side.set_aspect("equal"); ax_side.set_xlim(-lim, lim); ax_side.set_ylim(-lim, lim)
    title = ax_side.set_title("", loc="left", fontsize=10)
    ax_side.add_patch(plt.Circle((0, 0), rm.L, fill=False, color="0.85", ls="--"))
    for s in (1, -1):
        c = math.radians(rm.SWING["catch"])
        ax_side.plot([0, s * rm.L * math.sin(c)], [0, rm.L * math.cos(c)], color="#2E8B57", lw=0.8, ls=":")
    rod, = ax_side.plot([], [], lw=8, color="#E97820", solid_capstyle="round")
    ax_side.plot([0], [0], "ko", ms=7)
    info = ax_side.text(-0.98 * lim, -0.62 * lim, "", fontsize=8, family="monospace", va="top")
    mode_txt = ax_side.text(0, 1.12 * rm.L, "", ha="center", fontsize=13, weight="bold")

    lim_rad = math.radians(ARM_LIMIT_DEG)
    ax_top.set_aspect("equal"); ax_top.set_xlim(-1.3, 1.3); ax_top.set_ylim(-1.3, 1.3)
    ax_top.set_xticks([]); ax_top.set_yticks([]); ax_top.set_title("arm (top view)", fontsize=8)
    ax_top.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="0.7"))
    for s in (1, -1):
        ax_top.plot([0, s * math.sin(lim_rad)], [0, math.cos(lim_rad)], "r:", lw=0.8)
    arm, = ax_top.plot([], [], lw=4, color="#2774B8")

    line_th, = ax_th.plot([], [], color="#E97820")
    line_ph, = ax_ph.plot([], [], color="#2774B8")
    line_e, = ax_e.plot([], [], color="#7B3FA0")
    ax_th.set(ylabel="theta from up [deg]", ylim=(-190, 190), yticks=[-180, -90, 0, 90, 180])
    ax_th.axhspan(-rm.SWING["catch"], rm.SWING["catch"], color="#2E8B57", alpha=0.15)
    ax_ph.set(ylabel="arm phi [deg]", ylim=(-160, 160))
    for s in (1, -1):
        ax_ph.axhline(s * ARM_LIMIT_DEG, color="r", ls=":", lw=0.8)
    ax_e.set(ylabel="energy E", xlabel="t [s]", ylim=(-125, 30))
    ax_e.axhline(0, color="#2E8B57", lw=0.8); ax_e.axhline(-2 * rm.A_G, color="0.6", lw=0.8, ls=":")
    for a in (ax_th, ax_ph, ax_e):
        a.grid(alpha=.3)
    fig.text(0.47, 0.935, f"policy: {POLICY_PATH.name}   |u|<={rm.ACCEL_MAX:.0f} rad/s^2  |phi'|<={rm.SPEED_MAX:.1f} rad/s  "
             f"(green: catch window)   p: switch controller", fontsize=7.5, family="monospace")

    def reset(realistic, kind=None):
        S["kind"] = kind or S["kind"]
        S["lp"] = make(S["kind"], realistic=realistic)
        S.update(T=[], TH=[], PH=[], E=[], last=None)

    def on_key(ev):
        lp = S["lp"]
        if ev.key == "left": lp.plant.kick(-1.0)
        elif ev.key == "right": lp.plant.kick(1.0)
        elif ev.key == "up": lp.plant.kick(5.0)
        elif ev.key == "o": lp.offset += math.radians(1.0)
        elif ev.key == "c": lp.on = not lp.on
        elif ev.key == "x": reset(not lp.sensor.realistic)
        elif ev.key == "p": reset(lp.sensor.realistic, "rig" if S["kind"] == "diff" else "diff")
        elif ev.key == " ": S["paused"] = not S["paused"]; S["last"] = None
        elif ev.key == "r": reset(lp.sensor.realistic)
    fig.canvas.mpl_connect("key_press_event", on_key)

    def update(_):
        lp = S["lp"]
        now = time.perf_counter()
        if S["last"] is None:
            S["last"] = now
        elapsed = min(now - S["last"], 0.1); S["last"] = now
        if not S["paused"]:
            lp.advance(int(elapsed / rm.DT))
        pl, c = lp.plant, lp.ctrl
        th, ph = rm.wrap(pl.x[0]), pl.x[2]
        E = rm.energy(th, pl.x[1])
        S["T"].append(pl.t); S["TH"].append(math.degrees(th)); S["PH"].append(math.degrees(ph)); S["E"].append(E)
        title.set_text(f"controller: {c.name}")
        rod.set_data([0, rm.L * math.sin(th)], [0, rm.L * math.cos(th)])
        rod.set_color("#2E8B57" if c.mode == "balance" else "#E97820")
        arm.set_data([0, math.sin(ph)], [0, math.cos(ph)])
        mode_txt.set_text(lp.stop or ("OFF" if not lp.on else c.mode.upper()))
        mode_txt.set_color("r" if lp.stop else ("#2E8B57" if c.mode == "balance" else "#E97820"))
        catch = f"{c.t_catch:.2f} s" if c.t_catch is not None else "--"
        info.set_text(f"t={pl.t:6.2f}s  theta={math.degrees(th):7.1f}  arm={math.degrees(ph):7.1f} deg\n"
                      f"u={lp.u:+6.1f} rad/s^2  E={E:+7.1f}\n"
                      f"first catch {catch}   re-swings {c.falls}\n"
                      f"encoder error {math.degrees(lp.offset):+.1f} deg\n"
                      f"{'realistic' if lp.sensor.realistic else 'ideal'}"
                      f"{'  PAUSED' if S['paused'] else ''}\n"
                      "keys: p switch controller  <- -> kick  up knock over\n"
                      "      o encoder +1 deg  c ctrl  x realistic/ideal  r reset")
        t0 = max(0.0, pl.t - 12.0)
        ax_th.set_xlim(t0, t0 + 12.0)
        line_th.set_data(S["T"], S["TH"]); line_ph.set_data(S["T"], S["PH"]); line_e.set_data(S["T"], S["E"])
        return rod, arm, line_th, line_ph, line_e, info, mode_txt, title

    S["ani"] = FuncAnimation(fig, update, interval=20, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    if ARGS.check:
        check()
    elif ARGS.plot:
        plot()
    else:
        run()
