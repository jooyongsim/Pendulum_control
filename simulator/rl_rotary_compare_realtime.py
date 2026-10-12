"""Real-time rotary pendulum: switch between the SAC, DQN and REINFORCE policies of
rl_tutorial_cartpole_rotary_compare.py (section 10) while it runs.

Loads rl_models/rotary_{swingup,balance}_{sac,dqn,reinforce}.pt and runs them on the rig model of
rotary_model.py (100 Hz loop, 4 ms delay, 0.3 deg encoder, firmware motor limits). No training, no torch.

All policies see obs = [cos th, sin th, th', phi, phi'] and give a in [-1, 1] -> u = a * ACCEL_MAX,
recomputed every 20 ms (50 Hz, as in training):
    SAC        a = tanh(mu(obs))                 (actor output [mu, log_std])
    DQN        a = levels[argmax Q(obs, .)]      (5 levels -1, -0.5, 0, 0.5, 1)
    REINFORCE  a = clip(mu(obs), -1, 1)          (Gaussian policy mean)

Keys   m            : algorithm SAC -> DQN -> REINFORCE (the pendulum keeps its state)
       b            : model trained for swing-up <-> balance
       left / right : kick the pendulum (theta' -/+ 1 rad/s)
       up           : knock it over (theta' + 5 rad/s)
       x            : realistic sensor <-> ideal
       c            : controller on / off
       r            : reset (hanging for swing-up, 5 deg tilt for balance)
       space        : pause

    python simulator/rl_rotary_compare_realtime.py
    python simulator/rl_rotary_compare_realtime.py --check     # no window: numbers for all 6 policies
"""
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rotary_model as rm  # noqa: E402
from rl_rotary_realtime import ARM_LIMIT_DEG, RLController, load_state_dict  # noqa: E402

ALGOS = ("SAC", "DQN", "REINFORCE")
TASKS = ("swing-up", "balance")
DQN_LEVELS = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
COL = {"SAC": "#E97820", "DQN": "#2E8B57", "REINFORCE": "#7B3FA0"}


class MLP:
    """numpy forward of an nn.Sequential of Linear/ReLU stored under `prefix` (e.g. 'net.', 'mu.', '')."""

    def __init__(self, sd, prefix):
        idx = sorted({int(k[len(prefix):].split(".")[0]) for k in sd
                      if k.startswith(prefix) and k[len(prefix):].split(".")[0].isdigit()})
        self.layers = [(sd[f"{prefix}{i}.weight"], sd[f"{prefix}{i}.bias"]) for i in idx]

    def __call__(self, x):
        h = np.asarray(x, np.float32)
        for i, (W, b) in enumerate(self.layers):
            h = W @ h + b
            if i < len(self.layers) - 1:
                h = np.maximum(h, 0.0)
        return h


def make_policy(algo, path):
    """Returns obs -> a in [-1, 1], the same deterministic action the tutorial evaluates."""
    sd = load_state_dict(path)
    if algo == "SAC":
        net = MLP(sd, "net.")
        return lambda o: float(np.tanh(net(o)[0]))
    if algo == "DQN":
        net = MLP(sd, "")
        return lambda o: float(DQN_LEVELS[int(np.argmax(net(o)))])
    net = MLP(sd, "mu.")
    return lambda o: float(np.clip(net(o)[0], -1.0, 1.0))


def model_path(algo, task):
    return HERE / "rl_models" / f"rotary_{task.replace('-', '')}_{algo.lower()}.pt"


def load_policies():
    pols, missing = {}, []
    for algo in ALGOS:
        for task in TASKS:
            p = model_path(algo, task)
            if p.exists():
                pols[(algo, task)] = make_policy(algo, p)
            else:
                missing.append(str(p))
    if missing:
        print("not found (skipped):\n  " + "\n  ".join(missing) +
              "\n-> run rl_tutorial_cartpole_rotary_compare (.py or .ipynb) to train them")
    if not pols:
        sys.exit("no models")
    return pols


def make(policy, task, realistic=True):
    theta0 = math.pi + 0.02 if task == "swing-up" else math.radians(5.0)
    return rm.Loop(rm.Plant(theta0), RLController(policy), rm.Sensor(realistic), arm_limit_deg=ARM_LIMIT_DEG)


def simulate(policy, task, realistic=True, seconds=10.0):
    lp = make(policy, task, realistic)
    ths, phis, us, t_up = [], [], [], None
    while lp.plant.t < seconds:
        lp.advance(10)
        th = rm.wrap(lp.plant.x[0])
        ths.append(th); phis.append(lp.plant.x[2]); us.append(lp.u)
        if t_up is None and abs(th) < math.radians(10):
            t_up = lp.plant.t
    last = np.mean(np.abs(ths[-200:]) < math.radians(10))
    du = np.mean(np.abs(np.diff(us)))                       # how jumpy the command is
    return dict(result=lp.stop or ("balanced" if last > 0.95 else "not balanced"), t_up=t_up, last=last,
                arm=math.degrees(np.max(np.abs(phis))), du=du)


def check(pols):
    print(f"{'policy':<22}{'sensor':<10}{'result':<22}{'first up':>9}{'up(last 2s)':>12}{'max|arm|':>9}{'mean|du|':>9}")
    for (algo, task), pol in pols.items():
        for realistic in (False, True):
            r = simulate(pol, task, realistic)
            t_up = "--" if r["t_up"] is None else f"{r['t_up']:.2f}s"
            print(f"{algo + ' ' + task:<22}{'real' if realistic else 'ideal':<10}{r['result']:<22}{t_up:>9}"
                  f"{100 * r['last']:>11.0f}%{r['arm']:>8.0f}d{r['du']:>9.1f}")
    print("mean|du| = average change of u between 10 ms ticks [rad/s^2] (chattering indicator)")


def run(pols):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    S = {"algo": "SAC", "task": "swing-up", "paused": False, "last": None,
         "T": [], "TH": [], "PH": [], "U": [], "C": []}

    def current():
        key = (S["algo"], S["task"])
        if key not in pols:                                   # model missing: fall back to any available
            key = next(iter(pols)); S["algo"], S["task"] = key
        return pols[key]

    S["lp"] = make(current(), S["task"])

    fig = plt.figure(figsize=(13.5, 7))
    ax_side = fig.add_axes([0.03, 0.10, 0.36, 0.78])
    ax_top = fig.add_axes([0.28, 0.66, 0.11, 0.22])
    ax_th = fig.add_axes([0.47, 0.66, 0.50, 0.25])
    ax_ph = fig.add_axes([0.47, 0.37, 0.50, 0.25], sharex=ax_th)
    ax_u = fig.add_axes([0.47, 0.08, 0.50, 0.25], sharex=ax_th)

    L = rm.L
    lim = 1.25 * L
    ax_side.set_aspect("equal"); ax_side.set_xlim(-lim, lim); ax_side.set_ylim(-lim, lim)
    ax_side.set_title("side view (up = balanced)", loc="left")
    ax_side.add_patch(plt.Circle((0, 0), L, fill=False, color="0.85", ls="--"))
    rod, = ax_side.plot([], [], lw=8, solid_capstyle="round")
    ax_side.plot([0], [0], "ko", ms=7)
    info = ax_side.text(-0.98 * lim, -0.62 * lim, "", fontsize=8, family="monospace", va="top")
    mode_txt = ax_side.text(0, 1.12 * L, "", ha="center", fontsize=13, weight="bold")

    lim_rad = math.radians(ARM_LIMIT_DEG)
    ax_top.set_aspect("equal"); ax_top.set_xlim(-1.3, 1.3); ax_top.set_ylim(-1.3, 1.3)
    ax_top.set_xticks([]); ax_top.set_yticks([]); ax_top.set_title("arm (top view)", fontsize=8)
    ax_top.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="0.7"))
    for s in (1, -1):
        ax_top.plot([0, s * math.sin(lim_rad)], [0, math.cos(lim_rad)], "r:", lw=0.8)
    arm, = ax_top.plot([], [], lw=4, color="#2774B8")

    # one scatter-like line per algorithm so the colour shows which policy was driving
    lines = {a: (ax_th.plot([], [], color=COL[a], label=a)[0], ax_ph.plot([], [], color=COL[a])[0],
                 ax_u.plot([], [], color=COL[a], lw=0.8)[0]) for a in ALGOS}
    ax_th.set(ylabel="theta from up [deg]", ylim=(-190, 190), yticks=[-180, -90, 0, 90, 180])
    ax_th.axhspan(-10, 10, color="0.5", alpha=0.12)
    ax_th.legend(loc="upper right", fontsize=8, ncol=3)
    ax_ph.set(ylabel="arm phi [deg]", ylim=(-160, 160))
    for s in (1, -1):
        ax_ph.axhline(s * ARM_LIMIT_DEG, color="r", ls=":", lw=0.8)
    ax_u.set(ylabel="u [rad/s^2]", xlabel="t [s]", ylim=(-1.1 * rm.ACCEL_MAX, 1.1 * rm.ACCEL_MAX))
    for a in (ax_th, ax_ph, ax_u):
        a.grid(alpha=.3)
    fig.text(0.47, 0.935, "SAC: tanh(mu)   DQN: 5 levels of +-47 rad/s^2   REINFORCE: Gaussian mean   "
             "| policies at 50 Hz on a 100 Hz loop", fontsize=7.5, family="monospace")

    def reset(realistic):
        S["lp"] = make(current(), S["task"], realistic)
        S.update(T=[], TH=[], PH=[], U=[], C=[], last=None)

    def switch_policy():
        S["lp"].ctrl = RLController(current())                # keep the pendulum where it is

    def on_key(ev):
        lp = S["lp"]
        if ev.key == "left": lp.plant.kick(-1.0)
        elif ev.key == "right": lp.plant.kick(1.0)
        elif ev.key == "up": lp.plant.kick(5.0)
        elif ev.key == "m":
            S["algo"] = ALGOS[(ALGOS.index(S["algo"]) + 1) % len(ALGOS)]; switch_policy()
        elif ev.key == "b":
            S["task"] = "balance" if S["task"] == "swing-up" else "swing-up"; switch_policy()
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
        pl = lp.plant
        th, ph = rm.wrap(pl.x[0]), pl.x[2]
        S["T"].append(pl.t); S["TH"].append(math.degrees(th)); S["PH"].append(math.degrees(ph))
        S["U"].append(lp.u); S["C"].append(S["algo"])
        col = COL[S["algo"]]
        rod.set_data([0, L * math.sin(th)], [0, L * math.cos(th)]); rod.set_color(col)
        rod.set_alpha(1.0 if abs(th) < math.radians(10) else 0.6)
        arm.set_data([0, math.sin(ph)], [0, math.cos(ph)])
        mode_txt.set_text(lp.stop or (f"{S['algo']}  ({S['task']} model)" if lp.on else "OFF"))
        mode_txt.set_color("r" if lp.stop else col)
        info.set_text(f"t={pl.t:6.2f}s  theta={math.degrees(th):7.1f}  arm={math.degrees(ph):7.1f} deg\n"
                      f"u={lp.u:+6.1f} rad/s^2\n"
                      f"{'realistic sensor' if lp.sensor.realistic else 'ideal sensor'}"
                      f"{'  PAUSED' if S['paused'] else ''}\n"
                      "keys: m SAC/DQN/REINFORCE   b swing-up/balance model\n"
                      "      <- -> kick   up knock over   c ctrl   x sensor\n"
                      "      r reset   space pause")
        t0 = max(0.0, pl.t - 12.0)
        ax_th.set_xlim(t0, t0 + 12.0)
        C = np.array(S["C"])
        for a, (l_th, l_ph, l_u) in lines.items():          # NaN where another algorithm was driving
            m = C == a
            T = np.array(S["T"])
            l_th.set_data(T, np.where(m, S["TH"], np.nan))
            l_ph.set_data(T, np.where(m, S["PH"], np.nan))
            l_u.set_data(T, np.where(m, S["U"], np.nan))
        return rod, arm, info, mode_txt, *[l for ls in lines.values() for l in ls]

    S["ani"] = FuncAnimation(fig, update, interval=20, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    policies = load_policies()
    check(policies) if "--check" in sys.argv else run(policies)
