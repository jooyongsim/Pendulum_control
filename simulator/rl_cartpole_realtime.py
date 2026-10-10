"""Real-time CartPole driven by the DQN / REINFORCE policies trained in rl_tutorial_cartpole_rotary.py.

Loads rl_models/cartpole_dqn.pt and rl_models/cartpole_reinforce.pt and runs them on the same CartPole
simulator as the tutorial (Euler, 50 Hz, +-10 N). No training here, and torch is not needed: the weights
are read and evaluated with numpy.

Rules as in training: the episode fails when |x| > 2.4 m or |theta| > 12 deg. Here there is no 500-step
limit; after a failure the cart is reset automatically after 1 s.

Keys   left / right : push the cart (-/+ 30 N for one step) -> disturbance test
       m            : policy DQN -> REINFORCE -> HUMAN (you drive with left/right, +-10 N)
       r            : reset
       space        : pause

    python simulator/rl_cartpole_realtime.py
    python simulator/rl_cartpole_realtime.py --check     # no window: steps survived per policy
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
MODELS = {"DQN": HERE / "rl_models" / "cartpole_dqn.pt",
          "REINFORCE": HERE / "rl_models" / "cartpole_reinforce.pt"}
PUSH = 30.0                                     # N, disturbance from the arrow keys


# ---- CartPole: same physics and limits as CartPoleEnv in the tutorial ----
G, M_CART, M_POLE, L_HALF, FORCE, DT = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02
X_LIM, TH_LIM = 2.4, math.radians(12)


def cartpole_step(s, force):
    x, x_dot, th, th_dot = s
    m_tot, pml = M_CART + M_POLE, M_POLE * L_HALF
    c, sn = math.cos(th), math.sin(th)
    tmp = (force + pml * th_dot ** 2 * sn) / m_tot
    th_acc = (G * sn - c * tmp) / (L_HALF * (4 / 3 - M_POLE * c ** 2 / m_tot))
    x_acc = tmp - pml * th_acc * c / m_tot
    return np.array([x + DT * x_dot, x_dot + DT * x_acc, th + DT * th_dot, th_dot + DT * th_acc])


def failed(s):
    return abs(s[0]) > X_LIM or abs(s[2]) > TH_LIM


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


class GreedyPolicy:
    """QNet (DQN: Q values) or PolicyNet (REINFORCE: logits); both act by argmax over the 2 outputs."""

    def __init__(self, path):
        sd = load_state_dict(path)
        n = sorted({int(k.split(".")[1]) for k in sd if k.startswith("net.")})
        self.layers = [(sd[f"net.{i}.weight"], sd[f"net.{i}.bias"]) for i in n]

    def outputs(self, obs):
        h = np.asarray(obs, np.float32)
        for i, (W, b) in enumerate(self.layers):
            h = W @ h + b
            if i < len(self.layers) - 1:
                h = np.maximum(h, 0.0)
        return h

    def __call__(self, obs):
        return int(np.argmax(self.outputs(obs)))      # 0 = push left, 1 = push right


def load_policies():
    missing = [str(p) for p in MODELS.values() if not p.exists()]
    if missing:
        sys.exit("model not found:\n  " + "\n  ".join(missing) +
                 "\n-> run rl_tutorial_cartpole_rotary (.py or .ipynb) once to train and save them")
    return {k: GreedyPolicy(p) for k, p in MODELS.items()}


def check(policies, seconds=60.0):
    rng = np.random.default_rng(0)
    for name, pol in policies.items():
        res = []
        for _ in range(5):
            s, k = rng.uniform(-0.05, 0.05, 4), 0
            while k < int(seconds / DT) and not failed(s):
                force = FORCE if pol(s) == 1 else -FORCE
                if k == int(5.0 / DT):
                    force += PUSH                        # one push at t = 5 s
                s, k = cartpole_step(s, force), k + 1
            res.append(k * DT)
        print(f"{name:<9}: survived {', '.join(f'{t:.1f}' for t in res)} s of {seconds:.0f} s "
              f"(one {PUSH:.0f} N push at t = 5 s)")


def run(policies):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    modes = list(policies) + ["HUMAN"]
    S = {"mode": modes[0], "s": np.zeros(4), "t": 0.0, "acc": 0.0, "last": None, "paused": False,
         "push": 0.0, "human": 0.0, "fail_t": None, "best": 0.0, "ep_start": 0.0,
         "T": [], "TH": [], "X": [], "F": []}
    COL = {"DQN": "#2E8B57", "REINFORCE": "#7B3FA0", "HUMAN": "#2774B8"}

    fig = plt.figure(figsize=(13, 6.5))
    ax_c = fig.add_axes([0.04, 0.42, 0.92, 0.50])
    ax_th = fig.add_axes([0.06, 0.08, 0.27, 0.25])
    ax_x = fig.add_axes([0.38, 0.08, 0.27, 0.25], sharex=ax_th)
    ax_f = fig.add_axes([0.70, 0.08, 0.27, 0.25], sharex=ax_th)

    ax_c.set_xlim(-2.8, 2.8); ax_c.set_ylim(-0.35, 1.35); ax_c.set_aspect("equal")
    ax_c.axhline(0, color="0.6", lw=1)
    for xl in (-X_LIM, X_LIM):
        ax_c.axvline(xl, color="r", ls=":", lw=1)
    cart = plt.Rectangle((0, 0), 0.4, 0.2, color=COL[S["mode"]])
    ax_c.add_patch(cart)
    pole, = ax_c.plot([], [], lw=6, color="#E97820", solid_capstyle="round")
    arrow, = ax_c.plot([], [], lw=3, color="r")
    title = ax_c.text(0, 1.25, "", ha="center", fontsize=13, weight="bold")
    info = ax_c.text(-2.75, 1.25, "", fontsize=8, family="monospace", va="top")
    ax_c.text(2.75, 1.25, "keys: <- -> push (30 N)   m DQN/REINFORCE/HUMAN\n"
                          "      r reset   space pause\n"
                          "HUMAN mode: hold <- / -> to drive (10 N)",
              fontsize=8, family="monospace", va="top", ha="right")
    ax_c.set_yticks([])

    line_th, = ax_th.plot([], [], color="#E97820")
    line_x, = ax_x.plot([], [], color="#2774B8")
    line_f, = ax_f.plot([], [], color="0.3", lw=0.8)
    ax_th.set(ylabel="theta [deg]", xlabel="t [s]", ylim=(-14, 14))
    ax_x.set(ylabel="x [m]", xlabel="t [s]", ylim=(-2.6, 2.6))
    ax_f.set(ylabel="force [N]", xlabel="t [s]", ylim=(-45, 45))
    for v in (12, -12):
        ax_th.axhline(v, color="r", ls=":", lw=.8)
    for v in (X_LIM, -X_LIM):
        ax_x.axhline(v, color="r", ls=":", lw=.8)
    for a in (ax_th, ax_x, ax_f):
        a.grid(alpha=.3)

    def reset():
        S.update(s=np.random.uniform(-0.05, 0.05, 4), fail_t=None, ep_start=S["t"], push=0.0)

    def on_key(ev):
        if ev.key in ("left", "right"):
            sign = -1.0 if ev.key == "left" else 1.0
            if S["mode"] == "HUMAN":
                S["human"] = sign * FORCE
            else:
                S["push"] = sign * PUSH
        elif ev.key == "m":
            S["mode"] = modes[(modes.index(S["mode"]) + 1) % len(modes)]
            cart.set_color(COL[S["mode"]])
            reset()
        elif ev.key == "r": reset()
        elif ev.key == " ": S["paused"] = not S["paused"]; S["last"] = None

    def on_release(ev):
        if ev.key in ("left", "right"):
            S["human"] = 0.0
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("key_release_event", on_release)

    def control_step():
        if S["fail_t"] is not None:                      # failed: wait 1 s, then reset
            if S["t"] - S["fail_t"] > 1.0:
                reset()
            S["t"] += DT
            return 0.0
        if S["mode"] == "HUMAN":
            force = S["human"]
        else:
            force = FORCE if policies[S["mode"]](S["s"]) == 1 else -FORCE
        force += S["push"]; S["push"] = 0.0
        S["s"] = cartpole_step(S["s"], force)
        S["t"] += DT
        if failed(S["s"]):
            S["fail_t"] = S["t"]
            S["best"] = max(S["best"], S["t"] - S["ep_start"])
        return force

    def update(_):
        now = time.perf_counter()
        if S["last"] is None:
            S["last"] = now
        elapsed = min(now - S["last"], 0.1); S["last"] = now
        force = S["F"][-1] if S["F"] else 0.0
        if not S["paused"]:
            S["acc"] += elapsed
            while S["acc"] >= DT:                         # fixed 50 Hz steps, as in training
                S["acc"] -= DT
                force = control_step()
        x, _, th, _ = S["s"]
        S["T"].append(S["t"]); S["TH"].append(math.degrees(th)); S["X"].append(x); S["F"].append(force)
        cart.set_xy((x - 0.2, -0.1))
        pole.set_data([x, x + 2 * L_HALF * math.sin(th)], [0.1, 0.1 + 2 * L_HALF * math.cos(th)])
        arrow.set_data([x, x + 0.02 * force], [-0.2, -0.2])
        alive = S["t"] - S["ep_start"] if S["fail_t"] is None else S["fail_t"] - S["ep_start"]
        title.set_text(("FAILED" if S["fail_t"] is not None else S["mode"]) + ("  (paused)" if S["paused"] else ""))
        title.set_color("r" if S["fail_t"] is not None else COL[S["mode"]])
        info.set_text(f"t={S['t']:7.2f} s   alive {alive:6.2f} s   best {max(S['best'], alive):6.2f} s\n"
                      f"x={x:+.2f} m   theta={math.degrees(th):+6.2f} deg   F={force:+5.1f} N")
        t0 = max(0.0, S["t"] - 10.0)
        ax_th.set_xlim(t0, t0 + 10.0)
        line_th.set_data(S["T"], S["TH"]); line_x.set_data(S["T"], S["X"]); line_f.set_data(S["T"], S["F"])
        return cart, pole, arrow, title, info, line_th, line_x, line_f

    reset()
    S["ani"] = FuncAnimation(fig, update, interval=20, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    pols = load_policies()
    check(pols) if "--check" in sys.argv else run(pols)
