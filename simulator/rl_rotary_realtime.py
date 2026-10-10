"""Real-time rotary pendulum driven by the SAC policies trained in rl_tutorial_cartpole_rotary.py.

Loads rl_models/rotary_swingup_sac.pt (and rotary_balance_sac.pt) and runs them on the rig model of
rotary_model.py: 100 Hz loop, 4 ms delay, 0.3 deg encoder, firmware motor limits. No training here,
and torch is not needed: the weights are read and evaluated with numpy.

Policy: obs = [cos th, sin th, th', phi, phi'] -> a in [-1, 1] -> u = a * ACCEL_MAX,
        recomputed every 20 ms (50 Hz, as in training) and held for two 100 Hz loop ticks.

Keys   left / right : kick the pendulum (theta' -/+ 1 rad/s)
       up           : knock it over (theta' + 5 rad/s) -> watch it swing up again
       m            : policy swing-up <-> balance (balance policy cannot swing up)
       x            : realistic sensor <-> ideal
       c            : controller on / off
       r            : reset (hanging for swing-up, 5 deg tilt for balance)
       space        : pause

    python simulator/rl_rotary_realtime.py
    python simulator/rl_rotary_realtime.py --check     # no window: numbers only
"""
import collections
import io
import math
import pickle
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rotary_model as rm  # noqa: E402

MODELS = {"swing-up": HERE / "rl_models" / "rotary_swingup_sac.pt",
          "balance": HERE / "rl_models" / "rotary_balance_sac.pt"}
ARM_LIMIT_DEG = 140.0
HOLD = 2                                       # policy at 50 Hz on the 100 Hz loop


# ---- .pt reader without torch (torch.save = zip of a pickled state_dict + raw tensor bytes) ----
class _Storage:
    DTYPES = {"FloatStorage": np.float32, "DoubleStorage": np.float64}

    def __init__(self, name):
        self.dtype = self.DTYPES[name]


def _rebuild_tensor(storage, offset, size, stride, *_):
    item = storage.itemsize
    return np.lib.stride_tricks.as_strided(storage[offset:], shape=size,
                                           strides=[s * item for s in stride]).copy()


def load_state_dict(path):
    with zipfile.ZipFile(path) as z:
        root = z.namelist()[0].split("/")[0]

        class U(pickle.Unpickler):
            def find_class(self, mod, name):
                if mod == "collections" and name == "OrderedDict": return collections.OrderedDict
                if mod == "torch._utils" and name == "_rebuild_tensor_v2": return _rebuild_tensor
                if mod == "torch" and name.endswith("Storage"): return _Storage(name)
                raise pickle.UnpicklingError(f"unexpected global {mod}.{name}")

            def persistent_load(self, pid):            # ('storage', type, key, device, numel)
                _, st, key, _, _ = pid
                return np.frombuffer(z.read(f"{root}/data/{key}"), dtype=st.dtype).copy()

        return U(io.BytesIO(z.read(f"{root}/data.pkl"))).load()


class Policy:
    """Same MLP as the tutorial's Actor (ReLU hidden layers, output [mu, log_std]); deterministic tanh(mu)."""

    def __init__(self, path):
        sd = load_state_dict(path)
        n = sorted({int(k.split(".")[1]) for k in sd if k.startswith("net.")})
        self.layers = [(sd[f"net.{i}.weight"], sd[f"net.{i}.bias"]) for i in n]

    def __call__(self, obs):
        h = np.asarray(obs, np.float32)
        for i, (W, b) in enumerate(self.layers):
            h = W @ h + b
            if i < len(self.layers) - 1:
                h = np.maximum(h, 0.0)
        return float(np.tanh(h[0]))


class RLController:
    """rotary_model.Loop calls this at 100 Hz with z = [theta, theta', phi, phi'] (as the sensor sees them)."""

    def __init__(self, policy):
        self.policy, self.n, self.u = policy, 0, 0.0

    def __call__(self, z, t):
        if self.n % HOLD == 0:
            th, w, phi, phid = z
            self.u = self.policy([math.cos(th), math.sin(th), w, phi, phid]) * rm.ACCEL_MAX
        self.n += 1
        return self.u


def load_policies():
    missing = [str(p) for p in MODELS.values() if not p.exists()]
    if missing:
        sys.exit("model not found:\n  " + "\n  ".join(missing) +
                 "\n-> run rl_tutorial_cartpole_rotary (.py or .ipynb) once to train and save them")
    return {k: Policy(p) for k, p in MODELS.items()}


def make(policy, mode, realistic=True):
    theta0 = math.pi + 0.02 if mode == "swing-up" else math.radians(5.0)
    return rm.Loop(rm.Plant(theta0), RLController(policy), rm.Sensor(realistic), arm_limit_deg=ARM_LIMIT_DEG)


def check(policies):
    for mode, theta0_txt in (("swing-up", "hanging"), ("balance", "5 deg")):
        for realistic in (False, True):
            lp = make(policies[mode], mode, realistic)
            ths, t_up = [], None
            while lp.plant.t < 10.0:
                lp.advance(10)
                th = rm.wrap(lp.plant.x[0])
                ths.append(th)
                if t_up is None and abs(th) < math.radians(10):
                    t_up = lp.plant.t
            last = np.mean(np.abs(ths[-200:]) < math.radians(10))
            print(f"{mode:<9} from {theta0_txt:<8} {'realistic' if realistic else 'ideal':<9}: "
                  f"{lp.stop or ('balanced' if last > 0.95 else 'not balanced'):<12} "
                  f"first upright {t_up if t_up is None else round(t_up, 2)} s, upright(last 2 s) {100 * last:.0f}%")


def run(policies):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    S = {"mode": "swing-up", "paused": False, "last": None, "T": [], "TH": [], "PH": [], "U": []}
    S["lp"] = make(policies[S["mode"]], S["mode"])

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
    rod, = ax_side.plot([], [], lw=8, color="#E97820", solid_capstyle="round")
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

    line_th, = ax_th.plot([], [], color="#E97820")
    line_ph, = ax_ph.plot([], [], color="#2774B8")
    line_u, = ax_u.plot([], [], color="0.3", lw=1)
    ax_th.set(ylabel="theta from up [deg]", ylim=(-190, 190), yticks=[-180, -90, 0, 90, 180])
    ax_th.axhspan(-10, 10, color="#2E8B57", alpha=0.15)
    ax_ph.set(ylabel="arm phi [deg]", ylim=(-160, 160))
    for s in (1, -1):
        ax_ph.axhline(s * ARM_LIMIT_DEG, color="r", ls=":", lw=0.8)
    ax_u.set(ylabel="u [rad/s^2]", xlabel="t [s]", ylim=(-1.1 * rm.ACCEL_MAX, 1.1 * rm.ACCEL_MAX))
    for a in (ax_th, ax_ph, ax_u):
        a.grid(alpha=.3)
    fig.text(0.47, 0.935, f"SAC policy (rl_models/)  |u|<={rm.ACCEL_MAX:.0f} rad/s^2  |phi'|<={rm.SPEED_MAX:.1f} rad/s  "
             f"policy 50 Hz on a 100 Hz loop  (green band: upright +-10 deg)", fontsize=7.5, family="monospace")

    def reset(realistic):
        S["lp"] = make(policies[S["mode"]], S["mode"], realistic)
        S.update(T=[], TH=[], PH=[], U=[], last=None)

    def on_key(ev):
        lp = S["lp"]
        if ev.key == "left": lp.plant.kick(-1.0)
        elif ev.key == "right": lp.plant.kick(1.0)
        elif ev.key == "up": lp.plant.kick(5.0)
        elif ev.key == "m":                          # switch the policy, keep the pendulum where it is
            S["mode"] = "balance" if S["mode"] == "swing-up" else "swing-up"
            lp.ctrl = RLController(policies[S["mode"]])
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
        up = abs(th) < math.radians(10)
        S["T"].append(pl.t); S["TH"].append(math.degrees(th)); S["PH"].append(math.degrees(ph)); S["U"].append(lp.u)
        rod.set_data([0, L * math.sin(th)], [0, L * math.cos(th)])
        rod.set_color("#2E8B57" if up else "#E97820")
        arm.set_data([0, math.sin(ph)], [0, math.cos(ph)])
        mode_txt.set_text(lp.stop or (f"RL {S['mode'].upper()}" if lp.on else "OFF"))
        mode_txt.set_color("r" if lp.stop else ("#2E8B57" if up else "#E97820"))
        info.set_text(f"t={pl.t:6.2f}s  theta={math.degrees(th):7.1f}  arm={math.degrees(ph):7.1f} deg\n"
                      f"u={lp.u:+6.1f} rad/s^2\n"
                      f"{'realistic sensor' if lp.sensor.realistic else 'ideal sensor'}"
                      f"{'  PAUSED' if S['paused'] else ''}\n"
                      "keys: <- -> kick   up knock over   m swing-up/balance\n"
                      "      c ctrl  x realistic/ideal  r reset  space pause")
        t0 = max(0.0, pl.t - 12.0)
        ax_th.set_xlim(t0, t0 + 12.0)
        line_th.set_data(S["T"], S["TH"]); line_ph.set_data(S["T"], S["PH"]); line_u.set_data(S["T"], S["U"])
        return rod, arm, line_th, line_ph, line_u, info, mode_txt

    S["ani"] = FuncAnimation(fig, update, interval=20, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    pols = load_policies()
    check(pols) if "--check" in sys.argv else run(pols)
