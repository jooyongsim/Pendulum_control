"""Imitation learning on the rotary pendulum swing-up: pieces shared by collect / train / eval.

Expert = rotary_model.SwingUp, the energy-pump + catch + 4-state-balance state machine that runs
on the real rig. We log every (observation, action) pair it produces inside the 100 Hz control
loop and fit a feed-forward policy to it (behaviour cloning).

    obs  z = [theta, theta', phi, phi']  exactly what Loop hands the controller: quantised
                                         encoder, filtered theta', applied 4 ms later
    feat   = [sin theta, cos theta, theta', phi, phi']   sin/cos so the +-pi wrap is not a cliff
    act  u = arm acceleration command [rad/s^2], clipped to +-ACCEL_MAX (what Loop applies)

Two policy classes get compared:

    history = 1    memoryless      u = pi(feat_t)
    history = K    frame stacked   u = pi(feat_{t-K+1} ... feat_t)

The expert is a hybrid system: the same (theta, theta', phi, phi') can belong to the swing mode
or to the balance mode, and its first 0.30 s is an open-loop bang-bang kick. A memoryless policy
therefore cannot be an exact copy of it -- that gap is what the comparison is for.
"""
import os

# conda's MKL and pip's torch each bundle their own Intel OpenMP (Library/bin/libiomp5md.dll
# 20240820 vs torch/lib/libiomp5md.dll 20260213), and whichever loads second aborts the process.
# Replacing either means rebuilding numpy or torch in control2026, so we allow the duplicate and
# keep the thread count low. Set here because every script imports this module before torch;
# train_bc.check_numpy_matches_torch() guards the "may silently produce incorrect results" part.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import contextlib             # noqa: E402
import json                   # noqa: E402
import math                   # noqa: E402
import sys                    # noqa: E402
from collections import deque  # noqa: E402
from pathlib import Path      # noqa: E402

import numpy as np            # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simulator"))
import rotary_model as rm  # noqa: E402

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"
FIG_DIR = ROOT / "figs"

ARM_LIMIT_DEG = 140.0           # same host-side stop as rotary_swingup_realtime
START_TILT_DEG = 1.0            # a perfectly still hanging rod gives the kick nothing to grow from
K4 = rm.place4()

FEAT_NAMES = ("sin_th", "cos_th", "th_dot", "phi", "phi_dot")
N_FEAT = len(FEAT_NAMES)
U_MAX = rm.ACCEL_MAX


# ---------------------------------------------------------------- features
def features(z):
    """z = [theta, theta', phi, phi'] -> the 5 numbers the policy sees."""
    th = rm.wrap(z[0])
    return np.array([math.sin(th), math.cos(th), z[1], z[2], z[3]], dtype=np.float64)


def stack_episode(feat, k):
    """(T, N_FEAT) -> (T, k*N_FEAT), oldest frame first, start padded with the first frame.

    The padding has to match what BCPolicy does at runtime, or training and deployment disagree
    on the first k-1 steps of every episode.
    """
    if k == 1:
        return feat.copy()
    pad = np.repeat(feat[:1], k - 1, axis=0)
    g = np.concatenate([pad, feat], axis=0)
    return np.concatenate([g[i:i + len(feat)] for i in range(k)], axis=1)


# ---------------------------------------------------------------- simulator plumbing
@contextlib.contextmanager
def model_errors(coulomb=1.0, delay=None):
    """rotary_model keeps Coulomb friction and the loop delay in module globals."""
    f0, d0 = rm.F_C, rm.DELAY
    rm.F_C = f0 * coulomb
    rm.DELAY = d0 if delay is None else delay
    try:
        yield
    finally:
        rm.F_C, rm.DELAY = f0, d0


def make_loop(ctrl, realistic=True, b=rm.B_U, tilt_deg=START_TILT_DEG, offset_deg=0.0):
    """Hanging rod, arm at 0 -- the initial condition the rig starts from."""
    plant = rm.Plant(math.pi + math.radians(tilt_deg), b=b)
    return rm.Loop(plant, ctrl, rm.Sensor(realistic), arm_limit_deg=ARM_LIMIT_DEG,
                   offset=math.radians(offset_deg))


class Recorder:
    """Wraps a controller and logs (features, applied action) at every control tick."""

    def __init__(self, inner):
        self.inner = inner
        self.feat, self.act, self.mode, self.ts = [], [], [], []

    def reset(self, *a, **kw):
        self.inner.reset(*a, **kw)

    def __call__(self, z, t):
        u = self.inner(z, t)
        self.feat.append(features(z))
        self.act.append(float(np.clip(u, -U_MAX, U_MAX)))     # Loop clips before applying
        self.mode.append(1 if getattr(self.inner, "mode", "balance") == "balance" else 0)
        self.ts.append(t)
        return u

    def arrays(self):
        return (np.asarray(self.feat, np.float32), np.asarray(self.act, np.float32),
                np.asarray(self.mode, np.uint8), np.asarray(self.ts, np.float32))


# ---------------------------------------------------------------- rollout + scoring
UPRIGHT_DEG = 10.0              # "is it up?" band
SETTLE_S = 5.0                  # the window at the end of the episode we score on


def rollout(ctrl, duration=25.0, realistic=True, b=rm.B_U, coulomb=1.0, delay=None,
            tilt_deg=START_TILT_DEG, offset_deg=0.0, kicks=(), log=False):
    """One episode. Returns (stats, recorder or None).

    kicks = [(t, dtheta')] disturbances, so the data also covers falling and re-swinging.
    """
    rec = Recorder(ctrl) if log else None
    pending = sorted(kicks)
    with model_errors(coulomb, delay):
        lp = make_loop(rec or ctrl, realistic=realistic, b=b, tilt_deg=tilt_deg,
                       offset_deg=offset_deg)
        pl = lp.plant
        T, TH = [], []
        t_catch, arm_max, sat, n = None, 0.0, 0, 0
        while pl.t < duration:
            while pending and pl.t >= pending[0][0]:
                pl.kick(pending.pop(0)[1])
            lp.advance(10)                                     # 10 ms = one control step
            th = rm.wrap(pl.x[0])
            T.append(pl.t)
            TH.append(th)
            arm_max = max(arm_max, abs(pl.x[2]))
            sat += abs(lp.u) >= U_MAX - 1e-9
            n += 1
            if (t_catch is None and abs(th) < math.radians(rm.SWING["catch"])
                    and abs(pl.x[1]) < rm.SWING["catch_w"]):
                t_catch = pl.t
    TH = np.asarray(TH)
    tail = TH[np.asarray(T) >= max(0.0, duration - SETTLE_S)]
    up = float(np.mean(np.abs(tail) < math.radians(UPRIGHT_DEG))) if len(tail) else 0.0
    stats = dict(stop=lp.stop, t_catch=t_catch, up_frac=up,
                 success=bool(not lp.stop and up > 0.95),
                 theta_end_deg=math.degrees(abs(TH[-1])), arm_max_deg=math.degrees(arm_max),
                 sat_pct=100.0 * sat / max(n, 1),
                 theta_rms_deg=(math.degrees(float(np.sqrt(np.mean(tail ** 2))))
                                if len(tail) else float("nan")))
    return stats, rec


def expert():
    return rm.SwingUp(K4)


# ---------------------------------------------------------------- dataset i/o
DATASET_KEYS = ("feat", "act", "mode", "t", "ep")


def save_dataset(path, feat, act, mode, t, ep, meta):
    """One on-disk format for both the expert demonstrations and the DAgger aggregate."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, feat=np.asarray(feat, np.float32), act=np.asarray(act, np.float32),
                        mode=np.asarray(mode, np.uint8), t=np.asarray(t, np.float32),
                        ep=np.asarray(ep, np.int32), meta_json=json.dumps(meta))
    return Path(path)


def load_dataset(path):
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in DATASET_KEYS}, json.loads(str(d["meta_json"]))


# ---------------------------------------------------------------- numpy inference
class NumpyMLP:
    """Forward pass of the trained net in plain numpy.

    Inference lives here rather than in torch for two reasons: the closed-loop evaluation calls
    it 100 times per simulated second, and the real rig's host loop should not need torch.
    """

    def __init__(self, weights, biases, act="relu"):
        self.W = [np.asarray(w, np.float64) for w in weights]
        self.b = [np.asarray(c, np.float64) for c in biases]
        self.act = act

    def __call__(self, x):
        h = np.asarray(x, np.float64)
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W.T + b
            if i < len(self.W) - 1:
                h = np.maximum(h, 0.0) if self.act == "relu" else np.tanh(h)
        return h


class BCPolicy:
    """Drop-in replacement for rm.SwingUp: __call__(z, t) -> u [rad/s^2]."""

    def __init__(self, ckpt):
        ck = dict(np.load(ckpt, allow_pickle=False))
        self.meta = json.loads(str(ck["meta_json"]))
        n = int(self.meta["n_layers"])
        self.net = NumpyMLP([ck[f"W{i}"] for i in range(n)], [ck[f"b{i}"] for i in range(n)],
                            self.meta["activation"])
        self.k = int(self.meta["history"])
        self.x_mu, self.x_sd = ck["x_mu"], ck["x_sd"]
        self.a_mu, self.a_sd = float(ck["a_mu"]), float(ck["a_sd"])
        self.name = self.meta["name"]
        self.reset()

    def reset(self, *a, **kw):
        self.hist = deque(maxlen=self.k)
        self.mode = "bc"                       # Recorder/plots ask for this; the policy has none

    def __call__(self, z, t):
        f = features(z)
        if not self.hist:
            for _ in range(self.k):            # same start padding as stack_episode
                self.hist.append(f)
        else:
            self.hist.append(f)
        x = np.concatenate(self.hist) if self.k > 1 else f
        a = self.net((x - self.x_mu) / self.x_sd)
        return float(np.clip(a[0] * self.a_sd + self.a_mu, -U_MAX, U_MAX))
