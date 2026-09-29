"""3) Swing-up: energy control from hanging, then catch with the balance design.

theta is measured from upright, so hanging is pi. Writing the (normalised)
energy of the pendulum as

    E = 0.5 * omega^2 + wn^2 * cos(theta)          E(upright, at rest) = +wn^2
                                                   E(hanging, at rest) = -wn^2

and differentiating along the plant (pivot acceleration a, viscous/Coulomb
losses ignored for the design):

    dE/dt = -omega * cos(theta) * a / L_eff

so pumping energy in means choosing

    a = sat( gain * (E - E_ref) * sign(omega * cos(theta)) )

which makes dE/dt = -(gain/L_eff) * (E - E_ref) * |omega cos(theta)| <= 0 in the
sense that drives E toward E_ref. The arm therefore pushes with the swing while
the pendulum is below and against it once it is above.

When the pendulum arrives near upright slowly enough it is handed over to the
4-state balance controller (pendulum + rotor), which is the only version that
also keeps the arm from drifting away.

    python swingup.py --save
    python swingup.py --gain 0.6 --accel-max 3.0 --save     # what a stronger axis would do
"""

import argparse

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pendulum_sim as ps                      # noqa: E402  (repo root)
from plant_local import add_plant_args, identify, rotor_limits   # noqa: E402


class SwingUp(ps.Controller):
    """Energy pumping below, balance controller near upright."""

    name = "swing-up + catch"

    def __init__(self, p, balance_k, gain=0.35, u_max=np.inf,
                 catch_deg=12.0, catch_dps=200.0, e_margin=0.0,
                 k_rotor=0.0, k_rotor_rate=0.0):
        self.p = p
        self.balance = ps.StateFeedback(*balance_k, u_max=u_max)
        self.gain, self.u_max = gain, u_max
        # Energy pumping alone says nothing about where the arm ends up, so the
        # rotor drifts away turn after turn. These two terms pull it back to
        # centre between pushes; they are small enough not to stall the pumping.
        self.k_rotor, self.k_rotor_rate = k_rotor, k_rotor_rate
        self.catch = np.deg2rad(catch_deg)
        self.catch_rate = np.deg2rad(catch_dps)
        self.e_ref = p.wn ** 2 * (1.0 + e_margin)
        self.reset()

    def reset(self):
        self.caught = False
        self.t_catch = None
        self.n_catches = 0
        self.n = 0

    def energy(self, theta, omega):
        return 0.5 * omega ** 2 + self.p.wn ** 2 * np.cos(theta)

    def __call__(self, s, dt):
        self.n += 1
        err = abs(np.arctan2(np.sin(s.theta), np.cos(s.theta)))
        if not self.caught and err < self.catch and abs(s.omega) < self.catch_rate:
            self.caught = True
            self.t_catch = self.n * dt
            self.n_catches += 1
        elif self.caught and err > 3 * self.catch:
            # the catch failed: go back to pumping instead of saturating for ever
            self.caught = False
        if self.caught:
            return self.balance(s, dt)

        e_err = self.energy(s.theta, s.omega) - self.e_ref
        direction = np.sign(s.omega * np.cos(s.theta)) or 1.0
        u = self.gain * e_err * direction
        u -= self.k_rotor * s.alpha + self.k_rotor_rate * s.alpha_dot
        return float(np.clip(u, -self.u_max, self.u_max))


def balance_gains(p, wn_des=12.0, zeta_des=0.8, rotor_poles=(-2.0, -2.6)):
    pole = -wn_des * zeta_des + 1j * wn_des * np.sqrt(1 - zeta_des ** 2)
    return ps.place_poles4(p, [pole, pole.conjugate(), *rotor_poles])


def run(p, gain, accel_max, duration=15.0, control_hz=200.0, theta0_deg=179.0,
        catch_deg=12.0, catch_dps=200.0, rotor_limit_deg=np.inf, encoder_deg=0.3,
        wn_des=12.0, zeta_des=0.8, k_rotor=0.6, k_rotor_rate=0.25):
    ctl = SwingUp(p, balance_gains(p, wn_des, zeta_des), gain=gain, u_max=accel_max,
                  catch_deg=catch_deg, catch_dps=catch_dps,
                  k_rotor=k_rotor, k_rotor_rate=k_rotor_rate)
    cfg = ps.SimConfig(duration=duration, control_hz=control_hz, encoder_deg=encoder_deg,
                       accel_max=accel_max, rotor_limit_deg=rotor_limit_deg,
                       theta0_deg=theta0_deg)
    log = ps.simulate(p, ctl, cfg)
    log["wrapped_deg"] = np.rad2deg(np.arctan2(np.sin(log["theta"]), np.cos(log["theta"])))
    log["energy"] = ctl.energy(log["theta"], log["omega"])
    log["e_ref"] = ctl.e_ref
    tail = log["t"] >= log["t"][-1] - 1.0
    log["caught"] = ctl.caught
    log["n_catches"] = ctl.n_catches
    log["t_catch"] = ctl.t_catch
    log["final_abs_deg"] = float(np.max(np.abs(log["wrapped_deg"][tail])))
    log["balanced"] = bool(ctl.caught and log["final_abs_deg"] < 5.0)
    log["caught_once"] = ctl.n_catches > 0
    log["peak_rotor_deg"] = float(np.max(np.abs(log["alpha_deg"])))
    log["peak_rate_dps"] = float(np.max(np.abs(np.rad2deg(log["omega"]))))
    return log, ctl


def summarise(log):
    if log["caught"]:
        msg = (f"caught at t={log['t_catch']:.2f} s, "
               f"final |theta| {log['final_abs_deg']:.2f} deg -> "
               + ("BALANCED" if log["balanced"] else "lost after the catch"))
    else:
        msg = f"never reached upright (peak |theta| swing {np.max(np.abs(log['wrapped_deg'])):.0f} deg)"
    return (f"{msg}\n  peak rotor {log['peak_rotor_deg']:.0f} deg, "
            f"peak pendulum rate {log['peak_rate_dps']:.0f} deg/s, "
            f"saturated {100 * np.mean(log['saturated']):.0f} % of the time")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_plant_args(ap)
    ap.add_argument("--gain", type=float, default=0.35, help="energy pumping gain")
    ap.add_argument("--accel-max", type=float, default=None,
                    help="pivot acceleration limit [m/s^2] (default: firmware profile)")
    ap.add_argument("--duration", type=float, default=15.0)
    ap.add_argument("--control-hz", type=float, default=200.0)
    ap.add_argument("--catch-deg", type=float, default=12.0)
    ap.add_argument("--catch-dps", type=float, default=200.0)
    ap.add_argument("--rotor-limit", type=float, default=np.inf, help="deg, inf = unlimited")
    ap.add_argument("--k-rotor", type=float, default=0.6, help="rotor centring gain")
    ap.add_argument("--k-rotor-rate", type=float, default=0.25, help="rotor rate damping")
    ap.add_argument("--sweep", action="store_true", help="scan the pumping gain")
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()

    p = identify(a.glob, a.arm)
    lim = rotor_limits(p.arm_radius_m)
    accel_max = a.accel_max if a.accel_max else lim["accel_max"]
    print(f"\npivot acceleration limit {accel_max:.2f} m/s^2"
          f"{'' if a.accel_max else ' (from the firmware profile)'}")

    if a.sweep:
        print(f"\n{'gain':>6s} {'caught':>8s} {'t_catch':>9s} {'peak rotor':>11s} {'balanced':>9s}")
        for g in [0.05, 0.1, 0.2, 0.35, 0.5, 0.8, 1.2, 2.0]:
            log, _ = run(p, g, accel_max, duration=a.duration, control_hz=a.control_hz,
                         catch_deg=a.catch_deg, catch_dps=a.catch_dps,
                         rotor_limit_deg=a.rotor_limit,
                         k_rotor=a.k_rotor, k_rotor_rate=a.k_rotor_rate)
            t_catch = f"{log['t_catch']:.2f} s" if log["caught"] else "-"
            print(f"{g:6.2f} {str(log['caught']):>8s} {t_catch:>9s} "
                  f"{log['peak_rotor_deg']:10.0f} {str(log['balanced']):>9s}")
        return

    log, ctl = run(p, a.gain, accel_max, duration=a.duration, control_hz=a.control_hz,
                   catch_deg=a.catch_deg, catch_dps=a.catch_dps, rotor_limit_deg=a.rotor_limit,
                   k_rotor=a.k_rotor, k_rotor_rate=a.k_rotor_rate)
    print("\n" + summarise(log))

    import matplotlib
    if a.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    ax[0].plot(log["t"], log["wrapped_deg"], color="#7B3FA0")
    ax[0].set_ylabel("θ from upright [deg]")
    ax[1].plot(log["t"], log["energy"], color="#E97820", label="E")
    ax[1].axhline(log["e_ref"], color="gray", ls="--", lw=1, label="E upright")
    ax[1].set_ylabel("energy [1/s²]")
    ax[2].plot(log["t"], log["alpha_deg"], color="#2774B8")
    ax[2].set_ylabel("rotor φ [deg]")
    ax[3].plot(log["t"], log["u"], color="#2CA02C")
    ax[3].axhline(log["u"].max() * 0 + ctl.u_max, color="r", ls=":", lw=0.8)
    ax[3].axhline(-ctl.u_max, color="r", ls=":", lw=0.8)
    ax[3].set_ylabel("pivot accel [m/s²]")
    ax[3].set_xlabel("t [s]")
    if log["t_catch"]:
        for x in ax:
            x.axvline(log["t_catch"], color="k", ls="--", lw=0.8)
    for x in ax:
        x.grid(alpha=0.3)
    ax[1].legend(fontsize=8)
    fig.suptitle(f"Swing-up (gain {a.gain}, limit {accel_max:.2f} m/s²) — "
                 + ("balanced" if log["balanced"] else "not balanced"))
    fig.tight_layout()
    if a.save:
        fig.savefig("swingup.png", dpi=130)
        print("-> swingup.png")
    else:
        plt.show()


if __name__ == "__main__":
    main()
