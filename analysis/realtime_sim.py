"""Real-time interactive simulator for the identified pendulum.

Runs the plant in step with the wall clock and draws it live, so a controller can
be poked at while it runs: disturb the pendulum, switch designs, turn the
controller off and watch it fall over.

The physics is `pendulum_sim.derivative()` -- the same function the batch
simulator and the identification use. Nothing is re-implemented here, so what
you see is the identified model, not a lookalike.

Controls
    space        pause / resume
    c            controller on / off  (off = watch it fall)
    1 2 3        switch design: 2-state, 4-state slow, 4-state fast
    left / right impulse disturbance
    r            reset
    d            damping: identified / viscous only / Coulomb only / none
    v            toggle: exact velocity  vs  velocity differenced from the
                 quantised angle (what the hardware actually has)
    q            quit

Why matplotlib: it is already a dependency of this project, needs no install,
and FuncAnimation with blitting is fast enough for a 50 Hz view. tkinter,
PySide6 and ipywidgets are also available here if a richer UI is ever wanted.

    python realtime_sim.py
    python realtime_sim.py --control-hz 50 --encoder-deg 0.3 --accel-max 6
"""
import argparse
import dataclasses
import time

import numpy as np

from _common import PROJECT  # noqa: F401  (puts the project on sys.path)
import pendulum_sim as ps


class Loop:
    """Plant + discrete controller advanced against the wall clock."""

    DAMPING_MODES = ("identified", "viscous only", "Coulomb only", "none")

    def __init__(self, p, cfg, design="4-state slow", estimate_velocity=False):
        self.p, self.cfg = p, cfg
        # One physics path only: each mode is the identified parameter set with
        # the terms that mode drops zeroed out, so ps.derivative() stays the
        # single source of the equations.
        self.plants = {
            "identified":   p,
            "viscous only": dataclasses.replace(p, coulomb_dps=0.0),
            "Coulomb only": dataclasses.replace(p, zeta=0.0),
            "none":         dataclasses.replace(p, zeta=0.0, coulomb_dps=0.0),
        }
        self.damping = "identified"
        self.estimate_velocity = estimate_velocity
        self.designs = {
            "2-state": ps.place_poles(p, 12.0, 0.8) + (0.0, 0.0),
            "4-state slow": ps.place_poles4(p, [-8 + 8j, -8 - 8j, -2 + 1j, -2 - 1j]),
            "4-state fast": ps.place_poles4(p, [-12 + 12j, -12 - 12j, -3 + 1j, -3 - 1j]),
        }
        self.design = design
        self.enabled = True
        self.reset()

    # ---------------- state ----------------
    def reset(self):
        self.theta = np.deg2rad(self.cfg.theta0_deg)
        self.omega = 0.0
        self.alpha = self.alpha_dot = 0.0
        self.u = 0.0
        self.t = 0.0
        self.since_control = np.inf
        self.prev_theta_meas = None
        self.prev_alpha_meas = None
        self.w_est = self.a_est = 0.0
        self.hist = {k: [] for k in ("t", "theta", "u", "alpha")}

    def kick(self, dps):
        self.omega += np.deg2rad(dps)

    # ---------------- one control decision ----------------
    def _control(self, dt_c):
        quant = np.deg2rad(self.cfg.encoder_deg)
        th_m = np.round(self.theta / quant) * quant if quant > 0 else self.theta
        al_m = self.alpha

        if self.estimate_velocity:
            # what the hardware really has: difference the quantised angle
            if self.prev_theta_meas is None:
                self.prev_theta_meas, self.prev_alpha_meas = th_m, al_m
            raw_w = (th_m - self.prev_theta_meas) / dt_c
            raw_a = (al_m - self.prev_alpha_meas) / dt_c
            self.prev_theta_meas, self.prev_alpha_meas = th_m, al_m
            beta = dt_c / (0.02 + dt_c)          # 20 ms derivative filter
            self.w_est += beta * (raw_w - self.w_est)
            self.a_est += beta * (raw_a - self.a_est)
            w, a = self.w_est, self.a_est
        else:
            w, a = self.omega, self.alpha_dot

        if not self.enabled:
            return 0.0
        k1, k2, k3, k4 = self.designs[self.design]
        u = -(k1 * th_m + k2 * w + k3 * al_m + k4 * a)
        return float(np.clip(u, -self.cfg.accel_max, self.cfg.accel_max))

    # ---------------- advance by real elapsed time ----------------
    def advance(self, wall_dt):
        cfg = self.cfg
        dt = cfg.dt_plant
        steps = int(min(wall_dt, 0.1) / dt)      # clamp: never chase a long stall
        period = 1.0 / cfg.control_hz
        for _ in range(steps):
            self.since_control += dt
            if self.since_control >= period:
                self.u = self._control(period)
                self.since_control = 0.0
                if np.isfinite(cfg.rotor_limit_deg):
                    at_limit = abs(np.rad2deg(self.alpha)) >= cfg.rotor_limit_deg
                    if at_limit and np.sign(self.u) == np.sign(self.alpha):
                        self.u = 0.0

            self.alpha_dot += (self.u / self.p.arm_radius_m) * dt
            self.alpha += self.alpha_dot * dt
            d1, d2 = ps.derivative(self.theta, self.omega, self.u,
                                   self.plants[self.damping])
            self.omega += d2 * dt
            self.theta += d1 * dt
            self.t += dt

        for k, v in (("t", self.t), ("theta", np.rad2deg(self.theta)),
                     ("u", self.u), ("alpha", np.rad2deg(self.alpha))):
            self.hist[k].append(v)
        keep = 1500
        if len(self.hist["t"]) > keep:
            for k in self.hist:
                self.hist[k] = self.hist[k][-keep:]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--control-hz", type=float, default=100.0)
    ap.add_argument("--encoder-deg", type=float, default=0.3)
    ap.add_argument("--accel-max", type=float, default=6.0)
    ap.add_argument("--rotor-limit-deg", type=float, default=90.0)
    ap.add_argument("--theta0-deg", type=float, default=5.0)
    ap.add_argument("--arm-radius", type=float, default=0.085)
    ap.add_argument("--window", type=float, default=6.0, help="trace window [s]")
    args = ap.parse_args()

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    p = ps.PendulumParams()
    p.arm_radius_m = args.arm_radius
    cfg = ps.SimConfig(control_hz=args.control_hz, encoder_deg=args.encoder_deg,
                       accel_max=args.accel_max, rotor_limit_deg=args.rotor_limit_deg,
                       theta0_deg=args.theta0_deg)
    loop = Loop(p, cfg)

    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 1.25], hspace=0.45, wspace=0.22)
    ax_view = fig.add_subplot(gs[:, 0])
    ax_th = fig.add_subplot(gs[0, 1])
    ax_u = fig.add_subplot(gs[1, 1])
    ax_al = fig.add_subplot(gs[2, 1])

    # --- the physical picture: pendulum on a pivot that slides horizontally ---
    L = 0.2610                                   # identified rod length [m]
    ax_view.set_xlim(-0.42, 0.42)
    ax_view.set_ylim(-0.33, 0.33)
    ax_view.set_aspect("equal")
    ax_view.grid(alpha=0.3)
    ax_view.set_title("pendulum (theta from upright)")
    ax_view.axhline(0, color="#999", lw=0.8)
    rail, = ax_view.plot([-0.4, 0.4], [0, 0], color="#bbb", lw=3, zorder=1)
    rod, = ax_view.plot([], [], lw=5, color="#2774B8", solid_capstyle="round", zorder=3)
    bob, = ax_view.plot([], [], "o", ms=10, color="#c0392b", zorder=4)
    pivot, = ax_view.plot([], [], "o", ms=9, color="#203864", zorder=4)
    arrow = ax_view.annotate("", xy=(0, 0), xytext=(0, 0),
                             arrowprops=dict(arrowstyle="->", color="#E97820", lw=2.5))
    banner = ax_view.text(0, 0.29, "", ha="center", fontsize=10, fontweight="bold")
    ax_view.text(-0.40, -0.245,
                 "identified plant
"
                 f"$\omega_n$ = {p.wn:.3f} rad/s    $\zeta$ = {p.zeta:.5f}
"
                 f"viscous  $\sigma=\zeta\omega_n$ = {p.sigma:.4f} 1/s
"
                 f"Coulomb  {p.coulomb_dps:.3f} deg/s "
                 f"({p.coulomb_accel:.3f} rad/s$^2$)
"
                 f"$L_{{eff}}$ = {p.L_eff*1000:.1f} mm    b = {p.b:.2f}",
                 fontsize=8, color="#404040", va="bottom", family="monospace")

    lines = {}
    for ax, key, label, color in ((ax_th, "theta", "theta [deg]", "#203864"),
                                  (ax_u, "u", "u [m/s^2]", "#E97820"),
                                  (ax_al, "alpha", "rotor [deg]", "#3D8C54")):
        lines[key], = ax.plot([], [], lw=1.4, color=color)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.axhline(0, color="#999", lw=0.8)
    ax_th.set_ylim(-30, 30)
    ax_u.set_ylim(-args.accel_max * 1.15, args.accel_max * 1.15)
    ax_al.set_ylim(-120, 120)
    ax_al.set_xlabel("time [s]")
    if np.isfinite(args.rotor_limit_deg):
        for s in (-1, 1):
            ax_al.axhline(s * args.rotor_limit_deg, color="#c0392b", ls="--", lw=1)

    paused = [False]
    last = [time.perf_counter()]

    def status():
        return (f"{loop.design}   controller {'ON' if loop.enabled else 'OFF'}   "
                f"{'EXACT' if not loop.estimate_velocity else 'DIFFERENCED'} velocity   "
                f"damping: {loop.damping}"
                f"{'   [PAUSED]' if paused[0] else ''}")

    def on_key(ev):
        if ev.key == " ":
            paused[0] = not paused[0]
        elif ev.key == "c":
            loop.enabled = not loop.enabled
        elif ev.key == "r":
            loop.reset()
        elif ev.key == "d":
            modes = Loop.DAMPING_MODES
            loop.damping = modes[(modes.index(loop.damping) + 1) % len(modes)]
        elif ev.key == "v":
            loop.estimate_velocity = not loop.estimate_velocity
            loop.prev_theta_meas = None
        elif ev.key == "left":
            loop.kick(-120)
        elif ev.key == "right":
            loop.kick(+120)
        elif ev.key in ("1", "2", "3"):
            loop.design = ["2-state", "4-state slow", "4-state fast"][int(ev.key) - 1]
        elif ev.key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)

    def update(_):
        now = time.perf_counter()
        wall_dt, last[0] = now - last[0], now
        if not paused[0]:
            loop.advance(wall_dt)

        # pivot position is only for the picture: integrate the commanded accel
        x = np.clip(loop.alpha * p.arm_radius_m, -0.34, 0.34)
        tip_x = x + L * np.sin(loop.theta)
        tip_y = L * np.cos(loop.theta)
        rod.set_data([x, tip_x], [0, tip_y])
        bob.set_data([tip_x], [tip_y])
        pivot.set_data([x], [0])
        arrow.set_position((x, -0.06))
        arrow.xy = (x + np.clip(loop.u, -6, 6) * 0.03, -0.06)
        fallen = abs(np.rad2deg(loop.theta)) > 90
        banner.set_text(status() + ("   FALLEN" if fallen else ""))
        banner.set_color("#c0392b" if (fallen or not loop.enabled) else "#203864")

        t = np.asarray(loop.hist["t"])
        if len(t):
            lo = max(0.0, t[-1] - args.window)
            for ax, key in ((ax_th, "theta"), (ax_u, "u"), (ax_al, "alpha")):
                lines[key].set_data(t, loop.hist[key])
                ax.set_xlim(lo, lo + args.window)
            ax_th.set_ylim(*(lambda m: (-m, m))(
                max(15.0, 1.2 * np.abs(loop.hist["theta"][-400:]).max())))
        return ()

    FuncAnimation(fig, update, interval=20, blit=False, cache_frame_data=False)
    fig.suptitle("Real-time pendulum simulator — "
                 "space pause | c controller | 1/2/3 design | d damping | arrows disturb | v velocity | r reset",
                 fontsize=10)
    plt.show()


if __name__ == "__main__":
    main()
