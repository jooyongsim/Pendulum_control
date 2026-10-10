"""Real-time rod-pendulum simulator driven by the trained SAC policy (no training here).

Loads the actor weights saved by rl_pendulum_tutorial.py (sac_pendulum.pt) and runs it on the
same plant / window as pid_pendulum_realtime.py. Uses numpy only (torch is not imported). Press m to switch to the original PID and compare.

Physics: J theta'' + c theta' + k sin(theta) = u, RK4 at DT = 2 ms.
RL     : every 20 ms  u = sat(wn^2 sin(ref) + a * u_max),  a = actor(obs)   (zero-order hold)
PID    : the continuous-time PID of pid_pendulum_realtime.py

Keys   up / down : reference +/- 5 deg      left / right : kick the rod
       m         : controller RL <-> PID    c            : controller on / off
       r         : reset                    space        : pause

    python simulator/rl_pendulum_realtime.py
    python simulator/rl_pendulum_realtime.py --model path/to/actor.pt
    python simulator/rl_pendulum_realtime.py --check     # no window: 5 s step, RL vs PID numbers
Trained range: |ref| <= 60 deg. Outside that the policy is not guaranteed to work.
"""
import collections
import io
import pickle
import sys
import time
import zipfile
from pathlib import Path

import numpy as np                            # no torch needed: weights are read and evaluated with numpy

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pid_pendulum_realtime as sim           # plant parameters + PID (importing does not open a window)

L, J, K, C = sim.L, sim.J, sim.K, sim.C
U_MAX, DT = sim.TAU_MAX, sim.DT
DT_CTRL = 0.02                                # RL control period (same as training)
N_SUB = int(round(DT_CTRL / DT))              # physics steps per RL action (=10)
REF_STEP, KICK = sim.REF_STEP, sim.KICK
MODEL_PATH = HERE / "sac_pendulum.pt"


# ---- .pt reader without torch ----
# torch.save writes a zip: archive/data.pkl (pickled state_dict) + archive/data/<key> (raw tensor bytes).
# A small Unpickler rebuilds each tensor as a numpy array, so torch is only needed for training.
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
                return np.frombuffer(z.read(f"{root}/data/{key}"), dtype=st.dtype).astype(st.dtype)

        return U(io.BytesIO(z.read(f"{root}/data.pkl"))).load()


# ---- actor: same MLP as rl_pendulum_tutorial.Actor, numpy forward (deterministic output only) ----
class Actor:
    def __init__(self, sd):
        self.layers = [(sd[f"net.{i}.weight"], sd[f"net.{i}.bias"]) for i in (0, 2, 4)]

    def act(self, obs):
        h = np.asarray(obs, np.float32)
        for i, (W, b) in enumerate(self.layers):
            h = W @ h + b
            if i < len(self.layers) - 1:
                h = np.maximum(h, 0.0)                 # ReLU
        return float(np.tanh(h[0]))                    # output = [mu, log_std] -> tanh(mu) in [-1, 1]


def load_actor(path):
    if not Path(path).exists():
        sys.exit(f"model not found: {path}\n-> run rl_pendulum_tutorial.py once to train and save it")
    return Actor(load_state_dict(path))


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def obs(th, w, ref, a_prev):
    """same 6-D observation as PendulumTrackEnv._obs"""
    return np.array([np.sin(th), np.cos(th), w / 10.0, 5 * wrap(ref - th), ref, a_prev], np.float32)


def plant_rk4(th, w, u, dt=DT):
    def d(th, w): return w, (u - C * w - K * np.sin(th)) / J
    k1 = d(th, w); k2 = d(th + dt / 2 * k1[0], w + dt / 2 * k1[1])
    k3 = d(th + dt / 2 * k2[0], w + dt / 2 * k2[1]); k4 = d(th + dt * k3[0], w + dt * k3[1])
    return (th + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
            w + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]))


class Sim:
    """x = [theta, omega, int(e)] (int(e) only used by PID). RL holds u for N_SUB physics steps."""

    def __init__(self, actor, ref=np.deg2rad(20.0)):
        self.actor, self.ref = actor, ref
        self.mode, self.on = "RL", True
        self.reset()

    def reset(self):
        self.x, self.t, self.k = np.zeros(3), 0.0, 0
        self.a_prev, self.u = 0.0, 0.0

    def switch(self):
        self.mode = "PID" if self.mode == "RL" else "RL"
        self.x[2], self.a_prev = 0.0, 0.0

    def step(self):
        """advance one physics step DT"""
        if not self.on:
            self.u = 0.0
            self.x[0], self.x[1] = plant_rk4(self.x[0], self.x[1], 0.0)
        elif self.mode == "PID":
            self.x = sim.rk4_step(self.x, self.ref, DT, True)
            self.u = sim.pid_torque(self.x, self.ref)[0]
        else:
            if self.k % N_SUB == 0:                                   # new RL action every 20 ms
                a = self.actor.act(obs(self.x[0], self.x[1], self.ref, self.a_prev))
                self.u = float(np.clip(K * np.sin(self.ref) + a * U_MAX, -U_MAX, U_MAX))
                self.a_prev = a
            self.x[0], self.x[1] = plant_rk4(self.x[0], self.x[1], self.u)
        self.t += DT
        self.k += 1


def check(actor):
    ref = np.deg2rad(20.0)
    print(sim.header())
    for mode in ("RL", "PID"):
        s = Sim(actor, ref)
        if mode == "PID": s.switch()
        peak, TH = 0.0, []
        for _ in range(int(5.0 / DT)):
            s.step(); peak = max(peak, s.x[0]); TH.append(s.x[0])
        TH = np.array(TH)
        idx = np.where(np.abs(TH - ref) > 0.02 * ref)[0]
        ts = (idx[-1] + 1) * DT if len(idx) else 0.0
        print(f"{mode:3s} 5 s step to 20 deg: theta={np.rad2deg(s.x[0]):7.3f} deg  "
              f"overshoot {100*(peak-ref)/ref:5.1f}%  settle(2%) {ts:.2f} s  u={s.u:+.2f} rad/s^2")


def run(actor):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    s = Sim(actor)
    S = {"paused": False, "last": None, "T": [], "TH": [], "REF": [], "U": []}
    COL = {"RL": "#C8326E", "PID": "#E97820"}

    fig = plt.figure(figsize=(13, 6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.3], height_ratios=[2, 1])
    ax_anim = fig.add_subplot(gs[:, 0]); ax_th = fig.add_subplot(gs[0, 1])
    ax_u = fig.add_subplot(gs[1, 1], sharex=ax_th)

    lim = 1.3 * L
    ax_anim.set_aspect("equal"); ax_anim.set_xlim(-lim, lim); ax_anim.set_ylim(-lim, 0.5 * lim)
    ax_anim.set_title("rod pendulum (theta = 0 down)")
    rod, = ax_anim.plot([], [], lw=8, color=COL["RL"], solid_capstyle="round")
    refline, = ax_anim.plot([], [], "--", color="#2774B8")
    ax_anim.plot([0], [0], "ko", ms=6)
    info = ax_anim.text(-lim * 0.95, 0.45 * lim, "", fontsize=9, va="top", family="monospace")
    ax_anim.text(-lim * 0.95, -lim * 0.95, sim.header() +
                 "\nup/down ref  left/right kick  m RL<->PID\nc ctrl on/off  r reset  space pause",
                 fontsize=7, family="monospace")

    line_th, = ax_th.plot([], [], color=COL["RL"], label="theta")
    line_rf, = ax_th.plot([], [], "--", color="#2774B8", label="reference")
    ax_th.axhspan(-100, -60, color="0.9"); ax_th.axhspan(60, 100, color="0.9")   # outside trained ref range
    ax_th.set_ylim(-100, 100); ax_th.set_ylabel("deg"); ax_th.grid(alpha=.3)
    ax_th.legend(loc="upper right")
    line_u, = ax_u.plot([], [], color="0.3", lw=1)
    ax_u.set_ylim(-1.1 * U_MAX, 1.1 * U_MAX); ax_u.set_ylabel("u [rad/s^2]"); ax_u.set_xlabel("t [s]")
    ax_u.grid(alpha=.3)

    # Click the figure window first so it receives the key presses.
    def on_key(ev):
        if ev.key == "up": s.ref += REF_STEP
        elif ev.key == "down": s.ref -= REF_STEP
        elif ev.key == "left": s.x[1] -= KICK
        elif ev.key == "right": s.x[1] += KICK
        elif ev.key == "m": s.switch()
        elif ev.key == "c": s.on = not s.on; s.x[2] = 0.0
        elif ev.key == " ": S["paused"] = not S["paused"]; S["last"] = None
        elif ev.key == "r":
            s.reset(); S.update(T=[], TH=[], REF=[], U=[], last=None)
    fig.canvas.mpl_connect("key_press_event", on_key)

    def update(_):
        now = time.perf_counter()
        if S["last"] is None:
            S["last"] = now
        elapsed = min(now - S["last"], 0.1); S["last"] = now
        if not S["paused"]:
            for _ in range(int(elapsed / DT)):
                s.step()
        th, ref = s.x[0], s.ref
        S["T"].append(s.t); S["TH"].append(np.rad2deg(th)); S["REF"].append(np.rad2deg(ref))
        S["U"].append(s.u)
        rod.set_data([0, L * np.sin(th)], [0, -L * np.cos(th)]); rod.set_color(COL[s.mode])
        line_th.set_color(COL[s.mode])
        refline.set_data([0, 1.1 * L * np.sin(ref)], [0, -1.1 * L * np.cos(ref)])
        warn = "  (|ref|>60: untrained)" if abs(ref) > np.deg2rad(60) + 1e-9 and s.mode == "RL" else ""
        info.set_text(f"[{s.mode}] ctrl={'ON' if s.on else 'OFF'}{'  PAUSED' if S['paused'] else ''}\n"
                      f"t={s.t:6.2f}s  theta={np.rad2deg(th):7.2f}  ref={np.rad2deg(ref):6.1f} deg\n"
                      f"u={s.u:+.2f} rad/s^2{warn}")
        t0 = max(0.0, s.t - 10.0)
        ax_th.set_xlim(t0, t0 + 10.0)
        line_th.set_data(S["T"], S["TH"]); line_rf.set_data(S["T"], S["REF"])
        line_u.set_data(S["T"], S["U"])
        return rod, refline, line_th, line_rf, line_u, info

    S["ani"] = FuncAnimation(fig, update, interval=20, cache_frame_data=False)   # ~50 fps
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    path = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else MODEL_PATH
    actor = load_actor(path)
    check(actor) if "--check" in sys.argv else run(actor)
