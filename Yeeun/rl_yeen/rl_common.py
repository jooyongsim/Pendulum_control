"""Reward-design study on the rotary pendulum: the pieces shared by train / eval.

The imitation-learning study in ../behavior_cloning_yeen ended on two failures that the expert
itself does not have:

    * the cloned policy holds the rod a few degrees off vertical, because the expert's upright
      offset estimate (SwingUp.b_hat) is hidden state the student never observes;
    * DAgger started from a weak policy (31 episodes) drove the arm into its 140 deg limit.

Both are about *where the arm and the rod end up*, which behaviour cloning has no way to express:
it can only ask for the expert's action. Reinforcement learning can, because the reward says what
a good outcome is instead of what a good action is. So this folder asks the obvious follow-up:

    which term of the reward is actually doing the work?

Five rewards (REWARDS below) train the same SAC agent on the same plant and differ in one term
each. `trap` is the mistake the course notebook warns about in words -- a reward that is always
negative makes *ending the episode* profitable, so the agent is expected to drive the arm into
the limit on purpose. That one is meant to fail, and the failure is the result.

Environment (same identified plant as rotary_model.py, 1 kHz RK4 inside a 50 Hz agent step):

    obs    = [cos th, sin th, th', phi, phi']        th: upright = 0, hanging = +-pi
    action = a in [-1, 1]  ->  arm acceleration u = a * ACCEL_MAX
    task   = "balance" (start near upright, 5 s) | "swingup" (start hanging, 10 s)

Inference is plain numpy (NumpyActor), as in il_common: the host loop that drives the real rig
should not need torch, and the closed-loop evaluation calls the policy 50 times per simulated
second.
"""
import os

# conda's MKL and pip's torch each bundle their own Intel OpenMP and whichever loads second
# aborts the process. Same workaround as il_common -- set before anything imports torch.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import json                   # noqa: E402
import math                   # noqa: E402
import sys                    # noqa: E402
from pathlib import Path      # noqa: E402

import numpy as np            # noqa: E402

ROOT = Path(__file__).resolve().parent

# Find simulator/ without assuming how deep this folder sits in the repository -- it has been
# moved once already. Walk up and take the first ancestor that actually holds it.
# (Path.parents is not sliceable before 3.10, hence the list().)
for _base in list(ROOT.parents)[:5]:
    if (_base / "simulator" / "rotary_model.py").is_file():
        sys.path.insert(0, str(_base / "simulator"))
        break
else:
    raise ImportError(
        f"cannot find simulator/rotary_model.py above {ROOT} -- looked in "
        + ", ".join(str(b) for b in list(ROOT.parents)[:5]))

import rotary_model as rm  # noqa: E402

CKPT_DIR = ROOT / "checkpoints"
FIG_DIR = ROOT / "figs"

DT_CTRL = 0.02                  # 50 Hz agent step; the rig loop is 100 Hz, so hold each command twice
ARM_LIM = math.radians(140.0)   # same host-side stop as the rig
FALL = math.radians(30.0)       # "fallen" for the balance task
UPRIGHT_DEG = 10.0              # the band both studies score on
U_MAX = rm.ACCEL_MAX

OBS_NAMES = ("cos_th", "sin_th", "th_dot", "phi", "phi_dot")
OBS_DIM, ACT_DIM = len(OBS_NAMES), 1
EPISODE_S = dict(balance=5.0, swingup=10.0)

wrap = rm.wrap


# ---------------------------------------------------------------- reward designs
# up      how "being upright" is paid:
#           cos     0.5 (1 + cos th)   -- 1 upright, 0.5 horizontal, 0 hanging  (dense)
#           sparse  1 inside +-10 deg, 0 outside                               (no gradient)
#           square  -th^2                                                      (always <= 0)
# k_arm   weight on (phi / ARM_LIM)^2    -- keep the arm near the middle
# k_eff   weight on a^2                  -- do not use more acceleration than needed
# term    one-off penalty when the episode is terminated by a failure
REWARDS = {
    "full":      dict(up="cos",    k_arm=0.1, k_eff=0.01, term=-50.0,
                      doc="baseline: the reward the course notebook uses"),
    "no_arm":    dict(up="cos",    k_arm=0.0, k_eff=0.01, term=-50.0,
                      doc="arm-centering term removed -- does the arm drift to its limit?"),
    "no_effort": dict(up="cos",    k_arm=0.1, k_eff=0.0,  term=-50.0,
                      doc="effort term removed -- does the command get louder for no gain?"),
    "sparse":    dict(up="sparse", k_arm=0.1, k_eff=0.01, term=-50.0,
                      doc="upright paid only inside +-10 deg -- can random exploration find it?"),
    "trap":      dict(up="square", k_arm=0.0, k_eff=0.0,  term=0.0,
                      doc="r = -th^2 with no failure penalty -- ending the episode pays better"),
}


def reward_terms(cfg, th, phi, a):
    """The three parts kept separate so eval_rl.py can report where the return came from."""
    if cfg["up"] == "cos":
        up = 0.5 * (1.0 + math.cos(th))
    elif cfg["up"] == "sparse":
        up = 1.0 if abs(th) < math.radians(UPRIGHT_DEG) else 0.0
    elif cfg["up"] == "square":
        up = -th * th
    else:
        raise ValueError(cfg["up"])
    return up, -cfg["k_arm"] * (phi / ARM_LIM) ** 2, -cfg["k_eff"] * a * a


# ---------------------------------------------------------------- environment
class RotaryEnv:
    """Gymnasium-style env on the identified plant. Physics copied from rotary_model.Plant.deriv.

    The motor limits are part of the environment on purpose: the firmware will not run the
    stepper below SPEED_MIN and clamps its speed at SPEED_MAX, so a policy trained without them
    would learn commands the rig cannot execute.
    """

    obs_dim, act_dim = OBS_DIM, ACT_DIM

    def __init__(self, task="balance", reward="full", seed=None, b=rm.B_U, friction=1.0):
        if task not in EPISODE_S:
            raise ValueError(task)
        self.task = task
        self.cfg = REWARDS[reward] if isinstance(reward, str) else reward
        self.reward_name = reward if isinstance(reward, str) else "custom"
        self.rng = np.random.default_rng(seed)
        self.max_steps = int(EPISODE_S[task] / DT_CTRL)
        self.n_sub = int(round(DT_CTRL / rm.DT))        # one agent step = 20 physics steps
        self.b, self.fc = b, rm.F_C * friction          # model-error knobs for the sim-to-real check

    def _deriv(self, th, w, v, u):
        running = abs(v) >= rm.SPEED_MIN                # below this the stepper does not turn
        acc = u if running else 0.0
        phi_dot = v if running else 0.0
        s, c = math.sin(th), math.cos(th)
        w_dot = (rm.A_G * s + self.b * c * acc + s * c * phi_dot ** 2
                 - rm.C_D * w - self.fc * math.tanh(w / 0.01))
        return w, w_dot, phi_dot, u

    def _obs(self):
        th, w, phi, v = self.x
        phi_dot = v if abs(v) >= rm.SPEED_MIN else 0.0
        return np.array([math.cos(th), math.sin(th), w, phi, phi_dot], np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        r, o = self.rng, (options or {})
        if self.task == "balance":
            th0 = r.uniform(-0.2, 0.2)
        elif r.random() < 0.7:
            th0 = math.pi + r.uniform(-0.1, 0.1)        # hanging, where the rig actually starts
        else:
            th0 = r.uniform(-math.pi, math.pi)          # the rest spread out so upright is seen early
        self.x = [o.get("theta0", th0), o.get("theta_dot0", r.uniform(-0.5, 0.5)),
                  o.get("phi0", r.uniform(-0.2, 0.2)), 0.0]
        self.k = 0
        return self._obs(), {}

    def step(self, action):
        a = float(np.clip(np.ravel(action)[0], -1.0, 1.0))
        u = a * U_MAX
        th, w, phi, v = self.x
        h, d = rm.DT, self._deriv
        for _ in range(self.n_sub):                     # RK4, command held for the whole 20 ms
            k1 = d(th, w, v, u)
            k2 = d(th + h / 2 * k1[0], w + h / 2 * k1[1], v + h / 2 * k1[3], u)
            k3 = d(th + h / 2 * k2[0], w + h / 2 * k2[1], v + h / 2 * k2[3], u)
            k4 = d(th + h * k3[0], w + h * k3[1], v + h * k3[3], u)
            th += h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            w += h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            phi += h / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
            v += h / 6 * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])
            v = max(-rm.SPEED_MAX, min(rm.SPEED_MAX, v))
        th = wrap(th)
        self.x = [th, w, phi, v]
        self.k += 1

        r_up, r_arm, r_eff = reward_terms(self.cfg, th, phi, a)
        terminated = abs(phi) > ARM_LIM or (self.task == "balance" and abs(th) > FALL)
        reward = r_up + r_arm + r_eff + (self.cfg["term"] if terminated else 0.0)
        truncated = self.k >= self.max_steps
        info = dict(u=u, r_up=r_up, r_arm=r_arm, r_eff=r_eff,
                    why="arm" if abs(phi) > ARM_LIM else ("fall" if terminated else ""))
        return self._obs(), reward, terminated, truncated, info


# ---------------------------------------------------------------- rollout + scoring
def rollout(act, task="swingup", reward="full", theta0=None, theta_dot0=0.0, phi0=0.0,
            seconds=None, b=rm.B_U, friction=1.0, seed=0):
    """One deterministic episode. `act` maps obs -> a in [-1, 1]. Returns a dict of arrays."""
    env = RotaryEnv(task, reward, seed=seed, b=b, friction=friction)
    if seconds:
        env.max_steps = int(seconds / DT_CTRL)
    opt = dict(theta_dot0=theta_dot0, phi0=phi0)
    if theta0 is not None:
        opt["theta0"] = theta0
    obs, _ = env.reset(options=opt)
    keys = ("t", "th", "w", "phi", "u", "r", "r_up", "r_arm", "r_eff")
    log = {k: [] for k in keys}
    term = trunc = False
    while not (term or trunc):
        obs, r, term, trunc, info = env.step(act(obs))
        for k, v in zip(keys, (env.k * DT_CTRL, env.x[0], env.x[1], env.x[2],
                               info["u"], r, info["r_up"], info["r_arm"], info["r_eff"])):
            log[k].append(v)
    log = {k: np.asarray(v, np.float64) for k, v in log.items()}
    log["terminated"] = bool(term)
    log["why"] = info["why"]
    return log


# the same four tilts and three hanging starts for every agent, so runs are comparable
BALANCE_STARTS = [math.radians(d) for d in (5, -8, 11, -3)]
SWINGUP_STARTS = [math.pi, math.pi - 0.05, math.pi + 0.1]
TAIL_S = 2.0                    # the window at the end of the episode we score the hold on


def metrics(lg, tail_s=TAIL_S):
    """Closed-loop quality, in the same terms the imitation-learning study reported.

    bias and wobble are separated deliberately: a policy can be perfectly steady (small wobble)
    and still hold the rod several degrees off vertical (large bias), which is exactly what the
    cloned policies did. Scoring only "inside +-10 deg" hides that, so +-5 and +-2 are reported
    alongside it.
    """
    th = np.degrees(lg["th"])
    n = max(1, int(tail_s / DT_CTRL))
    tail = th[-n:]
    up = np.abs(th) < UPRIGHT_DEG
    return dict(
        ret=float(lg["r"].sum()),
        ret_up=float(lg["r_up"].sum()), ret_arm=float(lg["r_arm"].sum()),
        ret_eff=float(lg["r_eff"].sum()),
        steps=int(len(th)), terminated=bool(lg["terminated"]), why=lg["why"],
        t_up=float(lg["t"][np.argmax(up)]) if up.any() else float("nan"),
        bias_deg=float(tail.mean()), wobble_deg=float(tail.std()),
        hold10=float(np.mean(np.abs(tail) < 10.0) * 100.0),
        hold5=float(np.mean(np.abs(tail) < 5.0) * 100.0),
        hold2=float(np.mean(np.abs(tail) < 2.0) * 100.0),
        arm_max_deg=float(np.abs(np.degrees(lg["phi"])).max()),
        u_rms=float(np.sqrt(np.mean(lg["u"] ** 2))),
        sat_pct=float(np.mean(np.abs(lg["u"]) >= U_MAX - 1e-9) * 100.0),
    )


def evaluate(act, task="swingup", reward="full", **kw):
    """Mean metrics over the fixed start states for the task."""
    starts = BALANCE_STARTS if task == "balance" else SWINGUP_STARTS
    ms = [metrics(rollout(act, task, reward, theta0=t0, **kw)) for t0 in starts]
    out = {k: float(np.nanmean([m[k] for m in ms])) for k, v in ms[0].items()
           if isinstance(v, (int, float)) and not isinstance(v, bool)}
    out["failures"] = sum(m["terminated"] for m in ms)
    out["n_starts"] = len(starts)
    out["why"] = ",".join(sorted({m["why"] for m in ms if m["why"]})) or "-"
    return out


# ---------------------------------------------------------------- numpy inference
class NumpyMLP:
    """Forward pass of a trained net in plain numpy (same contract as il_common.NumpyMLP)."""

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


class NumpyActor:
    """Deterministic SAC actor from a .npz: obs -> a in [-1, 1].

    SAC's actor outputs the mean and log-std of a Gaussian and squashes a sample through tanh.
    Acting greedily means squashing the mean, so only the mean head is exported.
    """

    def __init__(self, ckpt):
        ck = dict(np.load(ckpt, allow_pickle=False))
        self.meta = json.loads(str(ck["meta_json"]))
        n = int(self.meta["n_layers"])
        self.net = NumpyMLP([ck[f"W{i}"] for i in range(n)], [ck[f"b{i}"] for i in range(n)],
                            self.meta.get("activation", "relu"))
        self.name = self.meta["name"]

    def __call__(self, obs):
        return float(np.tanh(self.net(np.asarray(obs, np.float64))[0]))

    # so the same object can drive rotary_model.Loop on the rig at 100 Hz (11절 sim-to-real)
    def as_rig_controller(self):
        def ctrl(z, t):
            th = wrap(z[0])
            obs = np.array([math.cos(th), math.sin(th), z[1], z[2], z[3]])
            return float(np.clip(self(obs) * U_MAX, -U_MAX, U_MAX))
        ctrl.reset = lambda *a, **kw: None
        ctrl.mode = "rl"
        return ctrl


def save_actor(path, weights, biases, meta):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ck = {f"W{i}": np.asarray(w, np.float64) for i, w in enumerate(weights)}
    ck.update({f"b{i}": np.asarray(b, np.float64) for i, b in enumerate(biases)})
    ck["meta_json"] = json.dumps(meta)
    np.savez_compressed(path, **ck)
    return Path(path)


def load_report(path):
    return json.loads(str(np.load(path, allow_pickle=False)["meta_json"]))
