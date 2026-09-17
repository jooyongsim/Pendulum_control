"""Rotary pendulum SWING-UP: pump energy with the arm, catch near the top, balance, re-swing after a fall.

Same controller as the real rig (swing-up/swingup_control.py) on the full nonlinear model with the
firmware's motor limits (see rotary_model.py). The rod starts hanging.

    energy  : E = 1/2 theta'^2 + (3g/2l)(cos theta - 1)      upright at rest = 0, hanging = -112
              dE/dt = b theta' cos(theta) phi''  -> push the arm with sign(theta' cos theta)
    swing   : +30 rad/s^2 for 0.15 s, -30 for 0.15 s, then
              u = 30 clip(0.3 (1 - E), 0, 1) sign(theta' cos theta) - 5 phi' - 3 phi
    catch   : |theta| < 12 deg, |theta'| < 4 rad/s, |arm| < 60 deg -> 4-state balancer
              (poles -6+-6j, -1.5+-1j), arm reference decays from the catch angle with 3 s
    fall    : |theta| > 30 deg -> swing again

    python3 simulator/rotary_swingup_realtime.py            real-time window
    python3 simulator/rotary_swingup_realtime.py --check    numbers only (nominal + robustness)

Controls (click the window first):
    left / right : kick the pendulum (theta' -/+ 1 rad/s)
    up           : knock it over (theta' + 5 rad/s) -> watch it swing up again
    o            : encoder zero error +1 deg (the rig has up to +-2 deg after a board reset);
                   the slow upright adaptation removes it while balancing
    c            : controller on / off
    x            : realistic (100 Hz, 4 ms delay, 0.3 deg encoder) / ideal
    r            : reset: hanging still, arm at 0
    space        : pause
If the window does not open under Wayland:  MPLBACKEND=TkAgg python3 ...
"""
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rotary_model as rm  # noqa: E402

ARM_LIMIT_DEG = 140.0
START_TILT_DEG = 1.0            # a perfectly still hanging rod gives the kick nothing to grow from
K4 = rm.place4()


def make(realistic=True, b=rm.B_U, tilt_deg=START_TILT_DEG, offset_deg=0.0):
    plant = rm.Plant(math.pi + math.radians(tilt_deg), b=b)
    return rm.Loop(plant, rm.SwingUp(K4), rm.Sensor(realistic), arm_limit_deg=ARM_LIMIT_DEG,
                   offset=math.radians(offset_deg))


def simulate(duration=15.0, **kw):
    f0, d0 = rm.F_C, rm.DELAY                  # model-error cases change the shared constants
    rm.F_C, rm.DELAY = f0 * kw.pop("coulomb", 1.0), kw.pop("delay", d0)
    try:
        lp = make(**kw)
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
    result = lp.stop or ("balanced" if c.mode == "balance" and th < math.radians(3) else "not caught")
    return dict(result=result, catch=c.t_catch, arm_max=math.degrees(arm_max), falls=c.falls,
                sat=100.0 * sat / max(n, 1), offset_end=math.degrees(c.b_hat))


def fmt(r):
    catch = f"{r['catch']:5.2f}s" if r["catch"] is not None else "  --  "
    return (f"{r['result']:<26} catch {catch}  arm max {r['arm_max']:5.1f}d  "
            f"re-swings {r['falls']}  saturated {r['sat']:4.1f}%")


def check():
    print(f"wn={rm.WN} zeta={rm.ZETA} b={rm.B_U} Coulomb {rm.F_C:.3f} rad/s^2 | "
          f"K=[{', '.join(f'{k:.2f}' for k in K4)}] | arm stop {ARM_LIMIT_DEG:.0f} deg")
    print(f"{'nominal (realistic)':<34}{fmt(simulate())}")
    print(f"{'ideal sensor, no delay':<34}{fmt(simulate(realistic=False))}")
    r = simulate(duration=25, offset_deg=1.5)
    print(f"{'encoder zero +1.5 deg':<34}{fmt(r)}  estimate after 25 s {r['offset_end']:+.2f}d")
    print("\nrobustness (model errors the real rig could have):")
    for name, kw in (("b -20%", dict(b=0.8 * rm.B_U)), ("b +20%", dict(b=1.2 * rm.B_U)),
                     ("Coulomb x0.5", dict(coulomb=0.5)), ("Coulomb x2", dict(coulomb=2.0)),
                     ("delay 12 ms", dict(delay=0.012)), ("start tilt -2.5 deg", dict(tilt_deg=-2.5)),
                     ("start tilt +2.5 deg", dict(tilt_deg=2.5))):
        print(f"  {name:<32}{fmt(simulate(**kw))}")
    rng = np.random.default_rng(0)
    ok, n = 0, 24
    for _ in range(n):
        kw = dict(b=rm.B_U * rng.uniform(0.8, 1.2), coulomb=rng.uniform(0.5, 2.0),
                  delay=rng.uniform(0.004, 0.012), tilt_deg=rng.uniform(-2.5, 2.5))
        ok += simulate(**kw)["result"] == "balanced"
    print(f"\nrandomised (b 0.8-1.2x, Coulomb 0.5-2x, delay 4-12 ms, tilt +-2.5 deg): {ok}/{n} balanced")


def run():
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    S = {"lp": make(), "paused": False, "last": None, "T": [], "TH": [], "PH": [], "E": []}
    fig = plt.figure(figsize=(13.5, 7))
    ax_side = fig.add_axes([0.03, 0.10, 0.36, 0.78])
    ax_top = fig.add_axes([0.28, 0.66, 0.11, 0.22])
    ax_th = fig.add_axes([0.47, 0.66, 0.50, 0.25])
    ax_ph = fig.add_axes([0.47, 0.37, 0.50, 0.25], sharex=ax_th)
    ax_e = fig.add_axes([0.47, 0.08, 0.50, 0.25], sharex=ax_th)

    lim = 1.25 * rm.L
    ax_side.set_aspect("equal"); ax_side.set_xlim(-lim, lim); ax_side.set_ylim(-lim, lim)
    ax_side.set_title("side view (up = balanced)", loc="left")
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
        ax_ph.axhline(s * rm.SWING["catch_arm"], color="#2E8B57", ls=":", lw=0.8)
    ax_e.set(ylabel="energy E", xlabel="t [s]", ylim=(-125, 30))
    ax_e.axhline(0, color="#2E8B57", lw=0.8); ax_e.axhline(-2 * rm.A_G, color="0.6", lw=0.8, ls=":")
    for a in (ax_th, ax_ph, ax_e):
        a.grid(alpha=.3)
    fig.text(0.47, 0.935, f"wn={rm.WN}  zeta={rm.ZETA}  b={rm.B_U}  |u|<={rm.ACCEL_MAX:.0f} rad/s^2  "
             f"|phi'|<={rm.SPEED_MAX:.1f} rad/s  pump {rm.SWING['pump']:.0f}  catch <{rm.SWING['catch']:.0f} deg  "
             f"(green: catch window)", fontsize=7.5, family="monospace")

    def reset(realistic):
        S["lp"] = make(realistic=realistic)
        S.update(T=[], TH=[], PH=[], E=[], last=None)

    def on_key(ev):
        lp = S["lp"]
        if ev.key == "left": lp.plant.kick(-1.0)
        elif ev.key == "right": lp.plant.kick(1.0)
        elif ev.key == "up": lp.plant.kick(5.0)
        elif ev.key == "o": lp.offset += math.radians(1.0)
        elif ev.key == "c": lp.on = not lp.on
        elif ev.key == "x": reset(not lp.sensor.realistic)
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
        rod.set_data([0, rm.L * math.sin(th)], [0, rm.L * math.cos(th)])
        rod.set_color("#2E8B57" if c.mode == "balance" else "#E97820")
        arm.set_data([0, math.sin(ph)], [0, math.cos(ph)])
        mode_txt.set_text(lp.stop or ("OFF" if not lp.on else c.mode.upper()))
        mode_txt.set_color("r" if lp.stop else ("#2E8B57" if c.mode == "balance" else "#E97820"))
        catch = f"{c.t_catch:.2f} s" if c.t_catch is not None else "--"
        info.set_text(f"t={pl.t:6.2f}s  theta={math.degrees(th):7.1f}  arm={math.degrees(ph):7.1f} deg\n"
                      f"u={lp.u:+6.1f} rad/s^2  E={E:+7.1f}\n"
                      f"first catch {catch}   re-swings {c.falls}\n"
                      f"encoder error {math.degrees(lp.offset):+.1f} deg, "
                      f"estimate {math.degrees(c.b_hat):+.2f} deg\n"
                      f"{'realistic' if lp.sensor.realistic else 'ideal'}"
                      f"{'  PAUSED' if S['paused'] else ''}\n"
                      "keys: <- -> kick   up knock over   o encoder +1 deg\n"
                      "      c ctrl  x realistic/ideal  r reset  space")
        t0 = max(0.0, pl.t - 12.0)
        ax_th.set_xlim(t0, t0 + 12.0)
        line_th.set_data(S["T"], S["TH"]); line_ph.set_data(S["T"], S["PH"]); line_e.set_data(S["T"], S["E"])
        return rod, arm, line_th, line_ph, line_e, info, mode_txt

    S["ani"] = FuncAnimation(fig, update, interval=20, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    else:
        run()
