"""Behavior cloning + DAgger (Lec05/Lec06) for the rotary pendulum SWING-UP, on the identified rig model.

A neural network learns to swing the rod up, catch it and balance it -- and to swing up again after a
fall -- by imitating our hand-designed controller (energy pumping + 4-state pole placement), the
"expert". The simulation is rotary_model.py: identified plant (wn 7.499, zeta 0.0117, b 0.730,
Coulomb friction), firmware limits (47.1 rad/s^2, 7.85 rad/s, 30 pps floor, 150 deg latch),
100 Hz loop with 4 ms delay and a 0.3 deg encoder.

Policy format = swing-up/swingup_rl_control.py (--kind bc), so the trained weights run on the rig:
    obs = [cos th, sin th, th', phi, phi']      th from upright (0 = balanced), as measured
    a   = tanh(net(obs)[0]) in [-1, 1]  ->  u = a * 47.1 rad/s^2
    net = nn.Sequential(Linear 5-128, ReLU, Linear 128-128, ReLU, Linear 128-out), keys 'net.N.*'
    recomputed every 20 ms (50 Hz) and held for two 100 Hz ticks.
Inputs are normalised while training; the normalisation is folded into the first layer when saving,
so the saved network takes raw observations.

Expert (stateless, so it can label ANY state -- DAgger needs that):
    near the top (|th| < 20 deg, |th'| < 5 rad/s):  u = -K x,  K = [236.23, 26.54, -4.16, -4.55]
    otherwise:  u = 30 clip(0.3 (1 - E), 0, 1) sign(th' cos th) - 5 phi' - 3 phi,   E = th'^2/2 + wn^2 (cos th - 1)

Lecture mapping
    version 0   deterministic policy, l2 regression to the expert action  (= Gaussian MLE, Lec05)
    version 1   Gaussian policy, maximise log N(a* | mu, sigma) with learned sigma, deploy mu
    compounding errors / covariate shift -> corrective data:
    DART        expert acts with injected noise, every state labelled with the clean expert action
    DAgger      student acts (mixed with the expert at rate beta), visited states labelled by the expert

    python3 bc_rotary_swingup.py train            # collect, BC, DART, DAgger, compare, save
    python3 bc_rotary_swingup.py eval             # re-evaluate the saved models
    python3 bc_rotary_swingup.py demo             # real-time window (student vs expert)
    python3 bc_rotary_swingup.py demo --model bc
    python3 bc_rotary_swingup.py gif              # record a demo to bc_models/swingup_demo.gif

Saved weights: simulator/bc_models/rotary_bc_<name>.pt (torch, for the rig) and .npz (numpy, this file).
On the rig:  python3 swing-up/swingup_rl_control.py --model BC_Pendulum/bc_models/rotary_bc_dagger.pt --kind bc

Demo keys (click the window first):
    up : knock it over (+5 rad/s)     left / right : kick (-/+ 1 rad/s)
    m  : next controller (dagger -> bc_dart -> bc -> bc_nll -> expert)
    c  : controller on / off          r : reset (hanging)          space : pause
If the window does not open under Wayland:  MPLBACKEND=TkAgg python3 ...
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rotary_model as rm  # noqa: E402

OUT = Path(__file__).resolve().parent / "bc_models"
K = rm.place4()
SW = rm.SWING
CATCH_TH, CATCH_W = math.radians(20.0), 5.0     # no arm gate: leaving the top because the arm
                                                  # passed 60 deg made the expert drop the rod
POLICY_EVERY = 2                        # 100 Hz loop ticks per policy step -> 50 Hz
ARM_LIMIT_DEG = 140.0
HIDDEN = 128
MODELS = ("dagger", "bc_dart", "bc", "bc_nll")


# ---------------------------------------------------------------- observation and expert
def obs(z):
    th, w, phi, phid = z
    return np.array([math.cos(th), math.sin(th), w, phi, phid])


def expert_u(z):
    """Stateless version of swing-up/swingup_control.py: same gains, no mode memory."""
    th, w, phi, phid = z
    if abs(th) < CATCH_TH and abs(w) < CATCH_W:
        return float(-K @ np.asarray(z, float))
    E = 0.5 * w * w + rm.A_G * (math.cos(th) - 1.0)
    scale = min(1.0, max(0.0, SW["ke"] * (SW["etarget"] - E)))
    return SW["pump"] * scale * (1.0 if w * math.cos(th) > 0 else -1.0) - SW["kda"] * phid - SW["kpa"] * phi


def expert_action(z):
    return float(np.clip(expert_u(z) / rm.ACCEL_MAX, -1.0, 1.0))


# ---------------------------------------------------------------- policy
class Policy:
    """MLP with ReLU hidden layers, numpy forward (no torch needed to run it)."""

    def __init__(self, layers, gaussian=False):
        self.layers, self.gaussian = layers, gaussian      # [(W, b), ...] taking RAW obs

    def __call__(self, o):
        h = np.asarray(o, float)
        for i, (W, b) in enumerate(self.layers):
            h = W @ h + b
            if i < len(self.layers) - 1:
                h = np.maximum(h, 0.0)
        return float(np.tanh(h[0]))

    def save(self, stem):
        """stem.npz (numpy) and stem.pt (torch state_dict, keys net.0/2/4) -- same weights."""
        stem = Path(stem)
        stem.parent.mkdir(parents=True, exist_ok=True)
        flat = {}
        for i, (W, b) in enumerate(self.layers):
            flat[f"W{i}"], flat[f"b{i}"] = W, b
        np.savez(stem.with_suffix(".npz"), gaussian=self.gaussian, **flat)
        try:
            import collections
            import torch
            sd = collections.OrderedDict()
            for i, (W, b) in enumerate(self.layers):
                sd[f"net.{2 * i}.weight"] = torch.tensor(W, dtype=torch.float32)
                sd[f"net.{2 * i}.bias"] = torch.tensor(b, dtype=torch.float32)
            torch.save(sd, stem.with_suffix(".pt"))
        except ImportError:
            pass

    @classmethod
    def load(cls, path):
        d = np.load(Path(path).with_suffix(".npz"))
        n = len([k for k in d.files if k.startswith("W")])
        return cls([(d[f"W{i}"], d[f"b{i}"]) for i in range(n)], bool(d["gaussian"]))


def train_policy(X, Y, gaussian=False, epochs=80, lr=1e-3, seed=0, verbose=True):
    """version 0: MSE on tanh(out0).  version 1: Gaussian NLL, sigma = exp(out1)."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    X, Y = np.asarray(X, np.float32), np.asarray(Y, np.float32)
    mean, std = X.mean(0), X.std(0) + 1e-6
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_val = max(1, len(X) // 10)
    Xt = torch.tensor((X[idx] - mean) / std)
    Yt = torch.tensor(Y[idx])[:, None]
    Xtr, Ytr, Xva, Yva = Xt[n_val:], Yt[n_val:], Xt[:n_val], Yt[:n_val]
    net = nn.Sequential(nn.Linear(5, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
                        nn.Linear(HIDDEN, 2 if gaussian else 1))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    def loss_fn(xb, yb):
        out = net(xb)
        mu = torch.tanh(out[:, :1])
        if not gaussian:
            return ((mu - yb) ** 2).mean()                              # -log N(a*|mu, s) with s fixed
        log_s = out[:, 1:].clamp(-6.0, 1.0)
        return (0.5 * ((yb - mu) / log_s.exp()) ** 2 + log_s).mean()     # -log N(a*|mu, s) + const

    best, best_state = float("inf"), None
    for _ in range(epochs):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 256):
            b = perm[i:i + 256]
            opt.zero_grad()
            loss_fn(Xtr[b], Ytr[b]).backward()
            opt.step()
        sched.step()
        with torch.no_grad():
            val = float(((torch.tanh(net(Xva)[:, :1]) - Yva) ** 2).mean())
        if val < best:
            best, best_state = val, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    if verbose:
        print(f"      {len(Xtr)} training samples, best validation MSE {best:.2e}")
    sd = {k: v.numpy().astype(np.float64) for k, v in net.state_dict().items()}
    W0, b0 = sd["0.weight"], sd["0.bias"]
    layers = [(W0 / std, b0 - W0 @ (mean / std)),                       # fold the normalisation in
              (sd["2.weight"], sd["2.bias"]), (sd["4.weight"], sd["4.bias"])]
    return Policy(layers, gaussian)


# ---------------------------------------------------------------- simulation
class Controller:
    """Policy step every 20 ms. Acts with `actor` (None = expert), optional noise / expert mixing,
    records (obs, expert action) at every policy step."""

    def __init__(self, actor=None, record=False, noise=0.0, beta=0.0, rng=None):
        self.actor, self.record, self.noise, self.beta = actor, record, noise, beta
        self.rng = rng or np.random.default_rng()
        self.X, self.Y = [], []
        self.k, self.u, self.a_expert = 0, 0.0, 0.0

    def __call__(self, z, t):
        if self.k % POLICY_EVERY == 0:
            o, a_exp = obs(z), expert_action(z)
            self.a_expert = a_exp
            if self.record:
                self.X.append(o); self.Y.append(a_exp)
            a = a_exp if (self.actor is None or self.rng.random() < self.beta) else self.actor(o)
            if self.noise:
                a += self.noise * self.rng.standard_normal()
            self.u = float(np.clip(a, -1.0, 1.0)) * rm.ACCEL_MAX
        self.k += 1
        return self.u


def make_loop(ctrl, th0, w0=0.0, b=rm.B_U):
    plant = rm.Plant(th0, b=b)
    plant.x[1] = w0
    return rm.Loop(plant, ctrl, rm.Sensor(True), arm_limit_deg=ARM_LIMIT_DEG)


def start_state(kind, rng):
    """'hang': hanging, small tilt -> swing-up.  'top': near upright, knocked over at 2 s -> recovery."""
    if kind == "hang":
        return math.pi + math.radians(rng.uniform(-3, 3)), rng.uniform(-0.2, 0.2), None
    return math.radians(rng.uniform(-3, 3)), 0.0, (2.0, rng.choice([-1, 1]) * rng.uniform(4.0, 6.0))


def run_episode(ctrl, kind, rng, duration=10.0, b=rm.B_U, delay=None, log=False):
    th0, w0, kick = start_state(kind, rng)
    d0 = rm.DELAY
    if delay is not None:
        rm.DELAY = delay
    try:
        lp = make_loop(ctrl, th0, w0, b)
        up_at, arm_max, kicked, rows = None, 0.0, False, []
        while lp.plant.t < duration:
            if kick and not kicked and lp.plant.t >= kick[0]:
                lp.plant.kick(kick[1]); kicked = True
            lp.advance(10)
            th = rm.wrap(lp.plant.x[0])
            arm_max = max(arm_max, abs(lp.plant.x[2]))
            if up_at is None and abs(th) < math.radians(3) and (kind == "hang" or kicked):
                up_at = lp.plant.t
            if log:
                rows.append((lp.plant.t, math.degrees(th), math.degrees(lp.plant.x[2]), lp.u,
                             ctrl.a_expert * rm.ACCEL_MAX))
    finally:
        rm.DELAY = d0
    th_end = abs(rm.wrap(lp.plant.x[0]))
    ok = not lp.stop and th_end < math.radians(3)
    out = dict(ok=ok, up=up_at, arm=math.degrees(arm_max), why=lp.stop)
    if log:
        out["log"] = np.array(rows)
    return out


def collect(n_eps, rng, actor=None, noise=0.0, beta=0.0, kinds=("hang", "top")):
    """Run episodes and return (obs, expert action) pairs from every policy step."""
    X, Y = [], []
    for i in range(n_eps):
        c = Controller(actor, record=True, noise=noise, beta=beta, rng=rng)
        run_episode(c, kinds[i % len(kinds)], rng)
        X += c.X; Y += c.Y
    return np.array(X), np.array(Y)


TESTS = {"swing-up": dict(kind="hang"),
         "knock-over": dict(kind="top"),
         "b -20%": dict(kind="hang", b=0.8 * rm.B_U),
         "delay 12 ms": dict(kind="hang", delay=0.012)}


def evaluate(actor, test, n=20, seed=123):
    rng = np.random.default_rng(seed)
    res = [run_episode(Controller(actor, rng=rng), rng=rng, **TESTS[test]) for _ in range(n)]
    ok = [r for r in res if r["ok"]]
    ups = [r["up"] for r in ok if r["up"]]
    return dict(rate=len(ok) / n, up=float(np.mean(ups)) if ups else float("nan"),
                arm=float(np.mean([r["arm"] for r in ok])) if ok else float("nan"))


def compare(models, n=20):
    names = list(TESTS)
    print(f"\n{'controller':<10}" + "".join(f"{t:>14}" for t in names) + f"{'swing-up time':>15}{'arm max':>9}")
    for name, m in [("expert", None)] + list(models.items()):
        cells, first = "", None
        for t in names:
            r = evaluate(m, t, n)
            first = first or r
            cells += f"{r['rate'] * 100:>13.0f}%"
        print(f"{name:<10}{cells}{first['up']:>14.2f}s{first['arm']:>8.0f}d")
    print(f"success = balanced (|theta| < 3 deg) at 10 s without hitting the arm limit, {n} runs each.\n"
          "knock-over: starts upright, kicked 4-6 rad/s at 2 s.  b -20% / delay 12 ms: model errors the rig could have.")


# ---------------------------------------------------------------- the full pipeline
def train_all(n_demo=20, dagger_rounds=5, dagger_eps=6, seed=0):
    rng = np.random.default_rng(seed)
    models = {}
    print(f"expert: energy pumping + pole placement (stateless), policy at {rm.CONTROL_HZ / POLICY_EVERY:.0f} Hz")

    print(f"\n[1] demonstrations: {n_demo} expert episodes (swing-up from hanging, recovery after a knock-over)")
    X0, Y0 = collect(n_demo, rng)
    print(f"      {len(X0)} (obs, action) pairs")
    print("[2] BC version 0: MSE regression")
    models["bc"] = train_policy(X0, Y0, seed=seed)

    print("[3] BC + DART: expert demos with action noise (sigma 0.2), clean labels")
    Xd, Yd = collect(n_demo, rng, noise=0.2)
    models["bc_dart"] = train_policy(np.vstack([X0, Xd]), np.concatenate([Y0, Yd]), seed=seed)

    print(f"[4] DAgger: from BC, {dagger_rounds} rounds x {dagger_eps} episodes, student acts, expert labels")
    X, Y, pol = X0.copy(), Y0.copy(), models["bc"]
    for i in range(dagger_rounds):
        beta = 0.5 ** (i + 1)
        Xn, Yn = collect(dagger_eps, rng, actor=pol, noise=0.05, beta=beta)
        X, Y = np.vstack([X, Xn]), np.concatenate([Y, Yn])
        pol = train_policy(X, Y, seed=seed, verbose=False)
        r = evaluate(pol, "swing-up", n=10, seed=1000 + i)
        print(f"      round {i + 1}: beta {beta:.3f}, +{len(Xn)} -> {len(X)} samples, swing-up {r['rate'] * 100:.0f}%")
    models["dagger"] = pol

    print("[5] BC version 1: Gaussian policy (NLL, learned sigma), deploy the mean")
    models["bc_nll"] = train_policy(X0, Y0, gaussian=True, seed=seed)

    for k, m in models.items():
        m.save(OUT / f"rotary_bc_{k}")
    print(f"\nsaved -> {OUT}/rotary_bc_*.pt / .npz")
    return models


def load_models():
    out = {k: Policy.load(OUT / f"rotary_bc_{k}") for k in MODELS if (OUT / f"rotary_bc_{k}.npz").exists()}
    if not out:
        raise SystemExit("no saved models -- run:  python3 bc_rotary_swingup.py train")
    return out


# ---------------------------------------------------------------- real-time demo
def demo(model="dagger", gif=None, gif_seconds=24.0):
    import matplotlib
    if gif:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    models = load_models()
    order = [k for k in MODELS if k in models] + ["expert"]
    S = {"name": model if model in order else order[0], "paused": False, "last": None}
    rng = np.random.default_rng()

    def build(state=None):
        ctrl = Controller(None if S["name"] == "expert" else models[S["name"]])
        th0, w0, ph0 = state or (math.pi + math.radians(1.0), 0.0, 0.0)
        lp = make_loop(ctrl, th0, w0)
        lp.plant.x[2] = ph0
        S.update(ctrl=ctrl, lp=lp, T=[], TH=[], PH=[], U=[], UE=[], E=[], last=None)

    build()
    fig = plt.figure(figsize=(13.5, 7))
    ax_side = fig.add_axes([0.03, 0.10, 0.36, 0.78])
    ax_top = fig.add_axes([0.28, 0.67, 0.11, 0.21])
    ax_th = fig.add_axes([0.47, 0.70, 0.50, 0.21])
    ax_ph = fig.add_axes([0.47, 0.39, 0.50, 0.21], sharex=ax_th)
    ax_u = fig.add_axes([0.47, 0.08, 0.50, 0.21], sharex=ax_th)

    lim = 1.25 * rm.L
    ax_side.set_aspect("equal"); ax_side.set_xlim(-lim, lim); ax_side.set_ylim(-lim, lim)
    ax_side.set_title("side view (up = balanced)", loc="left")
    ax_side.add_patch(plt.Circle((0, 0), rm.L, fill=False, color="0.85", ls="--"))
    rod, = ax_side.plot([], [], lw=8, solid_capstyle="round")
    ax_side.plot([0], [0], "ko", ms=7)
    title = ax_side.text(0, 1.12 * rm.L, "", ha="center", fontsize=13, weight="bold")
    info = ax_side.text(-0.98 * lim, -0.68 * lim, "", fontsize=8, family="monospace", va="top")

    lim_rad = math.radians(ARM_LIMIT_DEG)
    ax_top.set_aspect("equal"); ax_top.set_xlim(-1.3, 1.3); ax_top.set_ylim(-1.3, 1.3)
    ax_top.set_xticks([]); ax_top.set_yticks([]); ax_top.set_title("arm (top view)", fontsize=8)
    ax_top.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="0.7"))
    for s in (1, -1):
        ax_top.plot([0, s * math.sin(lim_rad)], [0, math.cos(lim_rad)], "r:", lw=0.8)
    arm, = ax_top.plot([], [], lw=4, color="#2774B8")

    line_th, = ax_th.plot([], [], color="#E97820")
    line_ph, = ax_ph.plot([], [], color="#2774B8")
    line_u, = ax_u.plot([], [], color="#C0392B", lw=1.2, label="network command")
    line_ue, = ax_u.plot([], [], color="0.2", lw=0.9, ls="--", label="expert would command")
    ax_th.set(ylabel="theta from up [deg]", ylim=(-190, 190), yticks=[-180, -90, 0, 90, 180])
    ax_th.axhspan(-20, 20, color="#2E8B57", alpha=0.12)
    ax_ph.set(ylabel="arm phi [deg]", ylim=(-160, 160))
    for s in (1, -1):
        ax_ph.axhline(s * ARM_LIMIT_DEG, color="r", ls=":", lw=0.8)
    ax_u.set(ylabel="u [rad/s^2]", xlabel="t [s]", ylim=(-52, 52)); ax_u.legend(fontsize=8, loc="upper right")
    for a in (ax_th, ax_ph, ax_u):
        a.grid(alpha=.3)
    fig.text(0.47, 0.95, "Behavior cloning: 5-128-128-1 ReLU network imitating energy swing-up + pole placement "
             "on the identified rig model (50 Hz policy, 4 ms delay, 0.3 deg encoder, 47 rad/s^2 limit)", fontsize=7.5)

    def on_key(key):
        lp = S["lp"]
        if key == "up": lp.plant.kick(5.0)
        elif key == "left": lp.plant.kick(-1.0)
        elif key == "right": lp.plant.kick(1.0)
        elif key == "c": lp.on = not lp.on
        elif key == " ": S["paused"] = not S["paused"]; S["last"] = None
        elif key == "r": build()
        elif key == "m":
            S["name"] = order[(order.index(S["name"]) + 1) % len(order)]
            x = lp.plant.x.copy()
            build((x[0], x[1], x[2]))
    fig.canvas.mpl_connect("key_press_event", lambda ev: on_key(ev.key))

    script = {8.0: "up", 15.0: "m", 17.0: "up"} if gif else {}
    clock = {"t": 0.0}

    def update(i):
        if gif:
            clock["t"] = i / 20.0
            for kt in [k for k in script if abs(clock["t"] - k) < 1e-6]:
                on_key(script.pop(kt))
        t_now = clock["t"] if gif else time.perf_counter()
        if S["last"] is None:
            S["last"] = t_now
        elapsed = min(t_now - S["last"], 0.1); S["last"] = t_now
        lp, ctrl = S["lp"], S["ctrl"]
        if not S["paused"]:
            lp.advance(int(round(elapsed / rm.DT)))
        th, ph = rm.wrap(lp.plant.x[0]), lp.plant.x[2]
        S["T"].append(lp.plant.t); S["TH"].append(math.degrees(th)); S["PH"].append(math.degrees(ph))
        S["U"].append(lp.u); S["UE"].append(ctrl.a_expert * rm.ACCEL_MAX)
        student = S["name"] != "expert"
        color = "#C0392B" if student else "#2E8B57"
        rod.set_data([0, rm.L * math.sin(th)], [0, rm.L * math.cos(th)]); rod.set_color(color)
        arm.set_data([0, math.sin(ph)], [0, math.cos(ph)])
        title.set_text(f"NETWORK  ({S['name']})" if student else "EXPERT  (energy + pole placement)")
        title.set_color(color)
        info.set_text(f"t={lp.plant.t:6.2f}s  theta={math.degrees(th):7.1f}  arm={math.degrees(ph):7.1f} deg\n"
                      f"u={lp.u:+6.1f} rad/s^2   status={lp.stop or 'OK'}   ctrl={'ON' if lp.on else 'OFF'}"
                      f"{'  PAUSED' if S['paused'] else ''}\n"
                      "keys: up knock over   <- -> kick   m next controller\n"
                      "      c ctrl on/off   r reset   space pause")
        t0 = max(0.0, lp.plant.t - 12.0)
        ax_th.set_xlim(t0, t0 + 12.0)
        line_th.set_data(S["T"], S["TH"]); line_ph.set_data(S["T"], S["PH"])
        line_u.set_data(S["T"], S["U"]); line_ue.set_data(S["T"], S["UE"])
        return rod, arm, line_th, line_ph, line_u, line_ue, info, title

    if gif:
        from matplotlib.animation import PillowWriter
        FuncAnimation(fig, update, frames=int(gif_seconds * 20), blit=False).save(
            gif, writer=PillowWriter(fps=20), dpi=65)
        print(f"gif -> {gif}")
        return
    S["ani"] = FuncAnimation(fig, update, interval=20, cache_frame_data=False)
    plt.show()


def plot_episode(actor, kind="hang", seed=0, ax=None):
    """Static plot of one episode (for Colab): theta, arm and network vs expert command."""
    import matplotlib.pyplot as plt
    r = run_episode(Controller(actor), kind, np.random.default_rng(seed), log=True)
    d = r["log"]
    fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    ax[0].plot(d[:, 0], d[:, 1], color="#E97820"); ax[0].axhspan(-20, 20, color="#2E8B57", alpha=0.12)
    who = "expert" if actor is None else "network"
    result = "balanced" if r["ok"] else "FAILED " + (r["why"] or "")
    up = "" if r["up"] is None else f", up at {r['up']:.2f} s"
    ax[0].set(ylabel="theta from up [deg]", ylim=(-190, 190), yticks=[-180, -90, 0, 90, 180],
              title=f"{who}: {result}{up}")
    ax[1].plot(d[:, 0], d[:, 2], color="#2774B8"); ax[1].set(ylabel="arm [deg]")
    ax[2].plot(d[:, 0], d[:, 3], color="#C0392B", lw=1.2, label="command")
    ax[2].plot(d[:, 0], d[:, 4], "k--", lw=0.8, label="expert would command")
    ax[2].set(ylabel="u [rad/s^2]", xlabel="t [s]"); ax[2].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=.3)
    fig.tight_layout()
    return fig


def animate_episode(actor, kind="hang", seed=0, duration=8.0, fps=20):
    """Animation of one episode for notebooks (Colab has no real-time window).
    Returns a matplotlib FuncAnimation; show it with IPython.display.HTML(anim.to_jshtml())."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    r = run_episode(Controller(actor), kind, np.random.default_rng(seed), duration=duration, log=True)
    d = r["log"]                                       # t, theta deg, arm deg, u, expert u  (every 10 ms)
    step = max(1, int(round(1.0 / (fps * 0.01))))
    frames = range(0, len(d), step)
    fig, (ax, ax_u) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw=dict(width_ratios=[1, 1.6]))
    lim = 1.25 * rm.L
    ax.set_aspect("equal"); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.axis("off")
    ax.add_patch(plt.Circle((0, 0), rm.L, fill=False, color="0.85", ls="--"))
    rod, = ax.plot([], [], lw=7, color="#C0392B" if actor is not None else "#2E8B57", solid_capstyle="round")
    ax.plot([0], [0], "ko", ms=6)
    txt = ax.text(0, -1.15 * rm.L, "", ha="center", fontsize=9, family="monospace")
    ax_u.plot(d[:, 0], d[:, 1], color="#E97820", lw=1.2)
    ax_u.axhspan(-20, 20, color="#2E8B57", alpha=0.12)
    ax_u.set(xlabel="t [s]", ylabel="theta from up [deg]", ylim=(-190, 190), yticks=[-180, -90, 0, 90, 180])
    ax_u.grid(alpha=.3)
    cursor = ax_u.axvline(0, color="k", lw=0.8)
    ax.set_title(f"{'expert' if actor is None else 'network'} — {'balanced' if r['ok'] else 'failed'}")

    def update(i):
        th = math.radians(d[i, 1])
        rod.set_data([0, rm.L * math.sin(th)], [0, rm.L * math.cos(th)])
        txt.set_text(f"t={d[i, 0]:4.1f}s  theta={d[i, 1]:6.1f}  arm={d[i, 2]:6.1f}")
        cursor.set_xdata([d[i, 0], d[i, 0]])
        return rod, txt, cursor

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=True)
    plt.close(fig)
    return anim


def compare_animation(actors, kind="top", seed=3, duration=8.0, fps=20):
    """Side-by-side animation of several controllers in the SAME episode (same start, same kick).
    actors: {"name": policy or None (= expert)}.  Returns a FuncAnimation."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    logs = {}
    for name, act in actors.items():
        r = run_episode(Controller(act), kind, np.random.default_rng(seed), duration=duration, log=True)
        logs[name] = (r, r["log"])
    n = len(actors)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 3.7))
    lim = 1.25 * rm.L
    rods, texts = [], []
    for ax, (name, (r, d)) in zip(np.atleast_1d(axes), logs.items()):
        ax.set_aspect("equal"); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.axis("off")
        ax.add_patch(plt.Circle((0, 0), rm.L, fill=False, color="0.85", ls="--"))
        rod, = ax.plot([], [], lw=6, solid_capstyle="round",
                       color="#2E8B57" if actors[name] is None else "#C0392B")
        ax.plot([0], [0], "ko", ms=5)
        ax.set_title(f"{name}\n{'balanced' if r['ok'] else 'FAILED'}", fontsize=10,
                     color="k" if r["ok"] else "#C0392B")
        rods.append((rod, d)); texts.append(ax.text(0, -1.18 * rm.L, "", ha="center", fontsize=8))
    step = max(1, int(round(1.0 / (fps * 0.01))))
    n_frames = min(len(d) for _, d in rods)

    def update(i):
        k = min(i * step, n_frames - 1)
        for (rod, d), txt in zip(rods, texts):
            th = math.radians(d[k, 1])
            rod.set_data([0, rm.L * math.sin(th)], [0, rm.L * math.cos(th)])
            txt.set_text(f"t={d[k, 0]:4.1f}s  arm {d[k, 2]:5.0f}°")
        return [r for r, _ in rods] + texts

    fig.suptitle({"top": "knocked over at 2 s (4-6 rad/s): does it swing up again?",
                  "hang": "swing-up from hanging"}[kind], fontsize=10, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    anim = FuncAnimation(fig, update, frames=n_frames // step, interval=1000 / fps, blit=True)
    plt.close(fig)
    return anim


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("train", "eval", "demo", "gif"))
    ap.add_argument("--model", default="dagger", choices=MODELS + ("expert",))
    ap.add_argument("--demos", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.cmd == "train":
        compare(train_all(n_demo=args.demos, seed=args.seed))
    elif args.cmd == "eval":
        compare(load_models())
    elif args.cmd == "demo":
        demo(args.model)
    else:
        demo(args.model, gif=str(OUT / "swingup_demo.gif"))


if __name__ == "__main__":
    main()
