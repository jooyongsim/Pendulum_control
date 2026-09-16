"""Real-time PID rod-pendulum simulator (ADL_2026 Lec02 slides 24-25) with MY values.

Physics: J theta'' + c theta' + k sin(theta) = tau, RK4 at a fixed DT.
Control: tau = sat(Kp e + Ki int(e) - Kd theta'), anti-windup while saturated.
Each frame advances as many DT steps as wall-clock time has elapsed.

Keys   up / down : reference +/- 5 deg      left / right : kick the rod
       c         : controller on / off      r            : reset
       space     : pause

    python3 simulator/pid_pendulum_realtime.py
    python3 simulator/pid_pendulum_realtime.py --check     # no window: 5 s run, print numbers
If the window does not open under Wayland:  MPLBACKEND=TkAgg python3 ...
"""
import sys
import time

import numpy as np

# ---- plant: my identification (uniform rod) ----
G = 9.81
WN, ZETA = 7.499, 0.01172
L = 3 * G / (2 * WN ** 2)                     # 0.262 m, only used to draw the rod
# Normalised by J, so no mass is needed: J -> 1, k/J = wn^2, c/J = 2 zeta wn, u = tau/J
J, K = 1.0, WN ** 2
C = 2 * ZETA * WN

# ---- PID: pole placement scaled by J (same design as rod_pid_analysis.py) ----
WN_DES, ZETA_DES, ALPHA = 12.0, 0.7, 8.0
KD = J * (2 * ZETA_DES * WN_DES + ALPHA) - C
KP = J * (WN_DES ** 2 + 2 * ALPHA * ZETA_DES * WN_DES) - K
KI = J * ALPHA * WN_DES ** 2

TAU_MAX = 3 * K                               # rad/s^2 saturation on u = tau/J
DT = 0.002                                    # s physics step
REF_STEP = np.deg2rad(5.0)
KICK = np.deg2rad(200.0)                      # rad/s added to omega by a kick

S = {"x": np.zeros(3), "t": 0.0, "ref": np.deg2rad(20.0), "on": True, "paused": False,
     "T": [], "TH": [], "REF": [], "TAU": [], "last": None}


def pid_torque(x, ref):
    th, w, ie = x
    e = ref - th
    tau_raw = KP * e + KI * ie - KD * w       # derivative on the measurement
    return float(np.clip(tau_raw, -TAU_MAX, TAU_MAX)), e, tau_raw


def f(x, ref, on):
    th, w, ie = x
    if on:
        tau, e, tau_raw = pid_torque(x, ref)
        die = e if abs(tau_raw) < TAU_MAX else 0.0
    else:
        tau, die = 0.0, 0.0
    dw = (tau - C * w - K * np.sin(th)) / J   # exact nonlinear plant
    return np.array([w, dw, die])


def rk4_step(x, ref, dt, on=True):
    k1 = f(x, ref, on); k2 = f(x + dt / 2 * k1, ref, on)
    k3 = f(x + dt / 2 * k2, ref, on); k4 = f(x + dt * k3, ref, on)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def header():
    return (f"wn={WN} zeta={ZETA}  l={L*1000:.0f}mm  (per unit J, no mass)\n"
            f"Kp={KP:.4f} Ki={KI:.4f} Kd={KD:.5f}  |u|<={TAU_MAX:.0f} rad/s^2")


def check():
    x, ref = np.zeros(3), S["ref"]
    n = int(5.0 / DT)
    peak = 0.0
    for _ in range(n):
        x = rk4_step(x, ref, DT)
        peak = max(peak, x[0])
    tau = pid_torque(x, ref)[0]
    print(header())
    print(f"5 s step to {np.rad2deg(ref):.0f} deg: theta={np.rad2deg(x[0]):.3f} deg  "
          f"overshoot {100*(peak-ref)/ref:.1f}%  u={tau:.3f} vs wn^2 sin(ref)={K*np.sin(ref):.3f} rad/s^2")


def run():
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, (ax_anim, ax_plot) = plt.subplots(1, 2, figsize=(12, 5.5))
    lim = 1.3 * L
    ax_anim.set_aspect("equal"); ax_anim.set_xlim(-lim, lim); ax_anim.set_ylim(-lim, 0.5 * lim)
    ax_anim.set_title("rod pendulum (theta = 0 down)")
    rod, = ax_anim.plot([], [], lw=8, color="#E97820", solid_capstyle="round")
    refline, = ax_anim.plot([], [], "--", color="#2774B8")
    ax_anim.plot([0], [0], "ko", ms=6)
    info = ax_anim.text(-lim * 0.95, 0.45 * lim, "", fontsize=8, va="top", family="monospace")
    ax_anim.text(-lim * 0.95, -lim * 0.95, header() +
                 "\nup/down ref  left/right kick  c ctrl  r reset  space pause",
                 fontsize=7, family="monospace")

    line_th, = ax_plot.plot([], [], color="#E97820", label="theta")
    line_rf, = ax_plot.plot([], [], "--", color="#2774B8", label="reference")
    ax_plot.set_ylim(-100, 100); ax_plot.set_xlabel("t [s]"); ax_plot.set_ylabel("deg")
    ax_plot.grid(alpha=.3); ax_plot.legend(loc="upper right")

    # ---------------------------------------------------------------- controls
    #   up    : reference angle +5 deg   (REF_STEP)
    #   down  : reference angle -5 deg
    #   left  : kick the rod, omega -= KICK (200 deg/s)   -> disturbance test
    #   right : kick the rod, omega += KICK
    #   c     : controller ON/OFF (OFF = free swing with the identified damping);
    #           the integral state is cleared on every toggle
    #   space : pause / resume
    #   r     : reset time, state and plot (reference angle is kept)
    # Click the figure window first so it receives the key presses.
    def on_key(ev):
        if ev.key == "up": S["ref"] += REF_STEP
        elif ev.key == "down": S["ref"] -= REF_STEP
        elif ev.key == "left": S["x"][1] -= KICK
        elif ev.key == "right": S["x"][1] += KICK
        elif ev.key == "c": S["on"] = not S["on"]; S["x"][2] = 0.0
        elif ev.key == " ": S["paused"] = not S["paused"]; S["last"] = None
        elif ev.key == "r":
            S.update(x=np.zeros(3), t=0.0, T=[], TH=[], REF=[], TAU=[], last=None)
    fig.canvas.mpl_connect("key_press_event", on_key)

    def update(_):
        now = time.perf_counter()
        if S["last"] is None:
            S["last"] = now
        elapsed = min(now - S["last"], 0.1); S["last"] = now
        if not S["paused"]:
            for _ in range(int(elapsed / DT)):
                S["x"] = rk4_step(S["x"], S["ref"], DT, S["on"]); S["t"] += DT
        th, ref = S["x"][0], S["ref"]
        tau = pid_torque(S["x"], ref)[0] if S["on"] else 0.0
        S["T"].append(S["t"]); S["TH"].append(np.rad2deg(th)); S["REF"].append(np.rad2deg(ref))
        rod.set_data([0, L * np.sin(th)], [0, -L * np.cos(th)])
        refline.set_data([0, 1.1 * L * np.sin(ref)], [0, -1.1 * L * np.cos(ref)])
        info.set_text(f"t={S['t']:6.2f}s  theta={np.rad2deg(th):7.2f}  ref={np.rad2deg(ref):6.1f} deg\n"
                      f"u={tau:+.2f} rad/s^2  ctrl={'ON' if S['on'] else 'OFF'}"
                      f"{'  PAUSED' if S['paused'] else ''}")
        t0 = max(0.0, S["t"] - 10.0)
        ax_plot.set_xlim(t0, t0 + 10.0)
        line_th.set_data(S["T"], S["TH"]); line_rf.set_data(S["T"], S["REF"])
        return rod, refline, line_th, line_rf, info

    S["ani"] = FuncAnimation(fig, update, interval=20, cache_frame_data=False)   # ~50 fps
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    check() if "--check" in sys.argv else run()
