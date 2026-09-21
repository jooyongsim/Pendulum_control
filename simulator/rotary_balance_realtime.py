"""Rotary inverted pendulum: balancing by ARM ACCELERATION (Lec01 "Linearize About Upright").

The rod starts 5 deg from upright and the controller moves the arm to keep it up.
Plant, motor limits and sensor are the real rig's (see rotary_model.py).

    linear model : theta'' = (3g/2l) theta + b phi'' - 2 zeta wn theta'      b = 3r/2l = 0.730
    2-state      : u = -(k1 theta + k2 theta')            rod balances, arm runs away
    4-state      : u = -K [theta, theta', phi, phi']      pole placement, arm stays near 0
                   poles -6+-6j, -1.5+-1j (used on the rig) -> K = [236.2, 26.5, -4.16, -4.55]

    python3 simulator/rotary_balance_realtime.py            real-time window
    python3 simulator/rotary_balance_realtime.py --check    numbers only
    python3 simulator/rotary_balance_realtime.py --plot     analytic vs simulated response figure

Controls (click the window first):
    left / right : kick the pendulum (theta' -/+ 1 rad/s)
    up           : big kick (theta' + 2.5 rad/s)
    t            : tilt the pendulum +3 deg
    2 / 4        : 2-state / 4-state controller
    c            : controller on / off
    x            : realistic (100 Hz, 4 ms delay, 0.3 deg encoder) / ideal
    r            : reset to theta0 = 5 deg
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

THETA0_DEG = 5.0
ARM_LIMIT_DEG = 140.0
K2, K4 = rm.place2(), rm.place4()


def make(K, realistic=True, limits=True, theta0_deg=THETA0_DEG):
    plant = rm.Plant(math.radians(theta0_deg), coulomb=limits, motor_limits=limits)
    return rm.Loop(plant, rm.Balancer(K), rm.Sensor(realistic),
                   arm_limit_deg=ARM_LIMIT_DEG if limits else 1e9)


def run_log(K, duration=6.0, **kw):
    lp = make(K, **kw)
    rows = []
    for _ in range(int(round(duration / rm.DT / 5))):
        rows.append((lp.plant.t, lp.plant.x[0], lp.plant.x[2], lp.u))
        lp.advance(5)
    d = np.array(rows)
    status = lp.stop or lp.ctrl.status
    theta = (d[:, 1] + np.pi) % (2 * np.pi) - np.pi
    return {"t": d[:, 0], "theta": np.rad2deg(theta), "phi": np.rad2deg(d[:, 2]), "u": d[:, 3],
            "status": status}


def analytic(K, t, theta0_deg=THETA0_DEG):
    from scipy.linalg import expm
    A, B = rm.state_matrices()
    Acl = A - B @ np.asarray(K, float).reshape(1, 4)
    x0 = np.array([math.radians(theta0_deg), 0, 0, 0])
    X = np.array([expm(Acl * tt) @ x0 for tt in t])
    return {"t": t, "theta": np.rad2deg(X[:, 0]), "phi": np.rad2deg(X[:, 2]), "u": -(X @ np.asarray(K, float))}


def summary():
    return (f"wn={rm.WN} zeta={rm.ZETA}  l={rm.L*1000:.0f} mm  b=3r/2l={rm.B_U} (r={rm.R_ARM*1000:.0f} mm)  "
            f"K4=[{', '.join(f'{k:.2f}' for k in K4)}]\n"
            f"|u|<={rm.ACCEL_MAX:.1f} rad/s^2  |phi'|<={rm.SPEED_MAX:.2f} rad/s  arm stop {ARM_LIMIT_DEG:.0f} deg  "
            f"{rm.CONTROL_HZ:.0f} Hz")


def check():
    A, B = rm.state_matrices()
    print(summary())
    print("open-loop poles :", np.round(np.linalg.eigvals(A), 3))
    print("2-state poles   :", np.round(np.linalg.eigvals(A - B @ K2.reshape(1, 4)), 3))
    print("4-state poles   :", np.round(np.linalg.eigvals(A - B @ K4.reshape(1, 4)), 3))
    print(f"\n{'case':<30}{'final theta':>12}{'max|theta|':>11}{'final phi':>11}{'max|phi|':>10}{'max|u|':>8}  status")
    for name, K, kw in (("4-state ideal (no limits)", K4, dict(realistic=False, limits=False)),
                        ("4-state realistic", K4, dict(realistic=True)),
                        ("2-state ideal (no limits)", K2, dict(realistic=False, limits=False)),
                        ("2-state realistic", K2, dict(realistic=True)),
                        ("no control", np.zeros(4), dict(realistic=False, limits=False))):
        g = run_log(K, **kw)
        print(f"{name:<30}{g['theta'][-1]:>11.2f}d{np.abs(g['theta']).max():>10.1f}d"
              f"{g['phi'][-1]:>10.1f}d{np.abs(g['phi']).max():>9.1f}d{np.abs(g['u']).max():>8.1f}  {g['status']}")
    print("\nlargest kick the 4-state controller survives (realistic, from upright):")
    lo, hi = 0.0, 10.0
    while hi - lo > 0.05:
        mid = (lo + hi) / 2
        lp = make(K4, theta0_deg=0.0)
        lp.plant.kick(mid)
        lp.advance(int(6 / rm.DT))
        ok = not lp.stop and lp.ctrl.status == "OK"
        lo, hi = (mid, hi) if ok else (lo, mid)
    print(f"   theta' = {lo:.2f} rad/s ({math.degrees(lo):.0f} deg/s)")


def plot(show=True):
    import matplotlib.pyplot as plt
    t = np.arange(0, 6.0, 0.005)
    curves = [("4-state analytic (linear, ideal)", analytic(K4, t), "k", "-", 2.4),
              ("4-state nonlinear, ideal", run_log(K4, realistic=False, limits=False), "#2E8B57", "--", 1.4),
              ("4-state nonlinear, realistic", run_log(K4), "#E97820", "-", 1.3),
              ("2-state nonlinear, realistic", run_log(K2), "#7B3FA0", "-", 1.2)]
    fig, ax = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for name, g, c, ls, lw in curves:
        lab = name + ("" if g.get("status", "OK") == "OK" else f"  [{g['status']}]")
        ax[0].plot(g["t"], g["theta"], color=c, ls=ls, lw=lw, label=lab)
        ax[1].plot(g["t"], g["phi"], color=c, ls=ls, lw=lw)
        ax[2].plot(g["t"], g["u"], color=c, ls=ls, lw=lw)
    ax[0].set(ylabel="theta from upright [deg]", ylim=(-8, 8),
              title=f"Rotary pendulum, acceleration control, theta0 = {THETA0_DEG} deg")
    for s in (1, -1):
        ax[1].axhline(s * ARM_LIMIT_DEG, color="r", ls=":", lw=0.8)
        ax[2].axhline(s * rm.ACCEL_MAX, color="r", ls=":", lw=0.8)
    ax[1].set(ylabel="arm phi [deg]")
    ax[2].set(ylabel="u = phi'' [rad/s^2]", xlabel="t [s]")
    for a in ax:
        a.grid(alpha=.3)
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    out = Path(__file__).with_name("rotary_balance_response.png")
    fig.savefig(out, dpi=120)
    print(f"figure -> {out}")
    if show:
        plt.show()


def run():
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    S = {"lp": make(K4), "design": 4, "paused": False, "last": None, "T": [], "TH": [], "PH": []}
    fig = plt.figure(figsize=(13, 6))
    ax_side = fig.add_axes([0.03, 0.08, 0.36, 0.80])
    ax_top = fig.add_axes([0.27, 0.62, 0.12, 0.26])
    ax_th = fig.add_axes([0.47, 0.55, 0.50, 0.35])
    ax_ph = fig.add_axes([0.47, 0.10, 0.50, 0.35], sharex=ax_th)

    lim = 1.3 * rm.L
    ax_side.set_aspect("equal"); ax_side.set_xlim(-lim, lim); ax_side.set_ylim(-0.4 * lim, lim)
    ax_side.set_title("side view (theta = 0 upright)", loc="left")
    ax_side.axhline(0, color="0.6", lw=3)
    rod, = ax_side.plot([], [], lw=8, color="#E97820", solid_capstyle="round")
    ax_side.plot([0], [0], "ko", ms=7)
    info = ax_side.text(-0.97 * lim, -0.12 * lim, "", fontsize=8, family="monospace", va="top")

    lim_rad = math.radians(ARM_LIMIT_DEG)
    ax_top.set_aspect("equal"); ax_top.set_xlim(-1.3, 1.3); ax_top.set_ylim(-1.3, 1.3)
    ax_top.set_xticks([]); ax_top.set_yticks([]); ax_top.set_title("arm (top view)", fontsize=8)
    ax_top.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="0.7"))
    for s in (1, -1):
        ax_top.plot([0, s * math.sin(lim_rad)], [0, math.cos(lim_rad)], "r:", lw=0.8)
    arm, = ax_top.plot([], [], lw=4, color="#2774B8")

    line_th, = ax_th.plot([], [], color="#E97820")
    line_ph, = ax_ph.plot([], [], color="#2774B8")
    ax_th.set(ylabel="theta [deg]", ylim=(-30, 30)); ax_th.axhline(0, color="0.5", lw=0.8); ax_th.grid(alpha=.3)
    ax_ph.set(ylabel="arm phi [deg]", xlabel="t [s]", ylim=(-160, 160)); ax_ph.grid(alpha=.3)
    for s in (1, -1):
        ax_ph.axhline(s * ARM_LIMIT_DEG, color="r", ls=":", lw=0.8)
    fig.text(0.47, 0.935, summary(), fontsize=7.5, family="monospace", va="bottom")

    def on_key(ev):
        lp = S["lp"]
        if ev.key == "left": lp.plant.kick(-1.0)
        elif ev.key == "right": lp.plant.kick(1.0)
        elif ev.key == "up": lp.plant.kick(2.5)
        elif ev.key == "t": lp.plant.x[0] += math.radians(3.0)
        elif ev.key in ("2", "4"):
            S["design"] = int(ev.key); lp.ctrl.K = K2 if ev.key == "2" else K4
        elif ev.key == "c": lp.on = not lp.on
        elif ev.key == "x": lp.sensor.realistic = not lp.sensor.realistic; lp.sensor.reset()
        elif ev.key == " ": S["paused"] = not S["paused"]; S["last"] = None
        elif ev.key == "r":
            K = lp.ctrl.K
            S["lp"] = make(K, realistic=lp.sensor.realistic)
            S.update(T=[], TH=[], PH=[], last=None)
    fig.canvas.mpl_connect("key_press_event", on_key)

    def update(_):
        lp = S["lp"]
        now = time.perf_counter()
        if S["last"] is None:
            S["last"] = now
        elapsed = min(now - S["last"], 0.1); S["last"] = now
        if not S["paused"]:
            lp.advance(int(elapsed / rm.DT))
        th, ph = lp.plant.x[0], lp.plant.x[2]
        S["T"].append(lp.plant.t); S["TH"].append(math.degrees(rm.wrap(th))); S["PH"].append(math.degrees(ph))
        rod.set_data([0, rm.L * math.sin(th)], [0, rm.L * math.cos(th)])
        arm.set_data([0, math.sin(ph)], [0, math.cos(ph)])
        status = lp.stop or lp.ctrl.status
        info.set_text(f"t={lp.plant.t:6.2f}s  theta={math.degrees(rm.wrap(th)):7.2f}  phi={math.degrees(ph):7.1f} deg\n"
                      f"u={lp.u:+6.2f} rad/s^2   {S['design']}-state  ctrl={'ON' if lp.on else 'OFF'}\n"
                      f"{'realistic' if lp.sensor.realistic else 'ideal'}  status={status}"
                      f"{'  PAUSED' if S['paused'] else ''}\n"
                      "keys: <- -> kick  up big kick  t tilt\n"
                      "      2/4  c  x  r  space")
        t0 = max(0.0, lp.plant.t - 10.0)
        ax_th.set_xlim(t0, t0 + 10.0)
        line_th.set_data(S["T"], S["TH"]); line_ph.set_data(S["T"], S["PH"])
        return rod, arm, line_th, line_ph, info

    S["ani"] = FuncAnimation(fig, update, interval=20, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    elif "--plot" in sys.argv:
        plot(show="--no-show" not in sys.argv)
    else:
        run()
