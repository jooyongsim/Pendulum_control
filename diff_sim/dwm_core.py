"""Differentiable world models for the rotary inverted pendulum (STEVAL-EDUKIT01), in PyTorch.

One file shared by the four tutorial notebooks (01-04). Conventions follow Pendulum_control/simulator/rotary_model.py:

    x = [theta, theta', phi, phi']    theta: rod angle from UPRIGHT (0 = balanced, +-pi = hanging)
                                      phi:   arm angle,  u = commanded arm acceleration [rad/s^2]

World = actuator (firmware: u -> phi'') o pendulum physics (phi'' -> theta'').

    theta'' = a sin(theta) + b cos(theta) phi'' + k sin(theta) cos(theta) phi'^2 - c theta' - Fc tanh(theta'/eps)
    phi''   = sat(u)   (|u| <= 47.1 rad/s^2, |phi'| <= 7.85 rad/s, smoothed so that gradients exist)

Model families (same interface: model.step(x, u) -> next x, differentiable in x, u and in the parameters):
    PhysicsModel   white box : the ODE above, 5-6 physical parameters, RK4
    HybridModel    gray box  : PhysicsModel + small residual MLP on theta''   (APHYNITY / UDE style)
    MLPModel       black box : x_{t+1} = x_t + NN(features)                      (Nagabandi 2018 style)
    Ensemble       black box with uncertainty: B MLPModels                       (PETS style)
    GPModel        black box, GP mean (PILCO's model class; PILCO itself is in Pendulum_control/simulator)

The 'real system' used to test policies is rotary_model.Plant + Loop (1 kHz plant, 100 Hz loop, 4 ms delay,
encoder quantisation, motor dead band, 150 deg latch) -- NOT differentiable, and NOT identical to any model here.
"""
import glob
import math
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

DT = 0.01                                     # control period of the rig (100 Hz)
G = 9.81
# identified on our unit (Pendulum_control/simulator/rotary_model.py)
WN, ZETA, B_U = 7.499, 0.01172, 0.730
A_G, C_D = WN ** 2, 2 * ZETA * WN
F_C = math.radians(2.705) * WN ** 2 * (2 * math.pi / WN) / 4.0      # 0.556 rad/s^2
STEP_RAD = 2 * math.pi / 3200
U_MAX, V_MAX = 24000 * STEP_RAD, 4000 * STEP_RAD                    # 47.1 rad/s^2, 7.85 rad/s
ARM_LIM = math.radians(140.0)

HERE = Path(__file__).resolve().parent
REPO_RAW = "https://raw.githubusercontent.com/jooyongsim/Pendulum_control/main"


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def twrap(a):
    return torch.atan2(torch.sin(a), torch.cos(a))


def savgol(x, window=15, order=3, deriv=0, delta=DT):
    """Savitzky-Golay smoothing / derivative in plain numpy (scipy.signal is not needed)."""
    h = window // 2
    t = np.arange(-h, h + 1)
    c = np.linalg.pinv(np.vander(t, order + 1, increasing=True))[deriv] * math.factorial(deriv) / delta ** deriv
    xp = np.r_[np.full(h, x[0]), x, np.full(h, x[-1])]
    return np.convolve(xp, c[::-1], "valid")


# ======================================================================= data (real rig logs)
def find_pendulum_repo():
    """Pendulum_control folder: next to this course folder locally, or a clone on Colab."""
    for p in [HERE.parent.parent / "Pendulum_control", Path.cwd() / "Pendulum_control", Path("/content/Pendulum_control")]:
        if (p / "data").exists():
            return p
    return None


def load_real_logs(pattern="swingup_2026*.csv", min_len=150):
    """Real swing-up logs (100 Hz). Returns a list of dicts with t, theta, phi, theta_dot, phi_dot, u (rad units).

    phi and phi_dot are what the firmware reports (step count and its target speed); theta is the encoder.
    The arm acceleration that actually drove the rod is phi_ddot = diff(phi_dot)/dt -- NOT the command u,
    because the firmware clips u, stalls at reversals and (old firmware) flipped the speed sign at -max."""
    import pandas as pd
    repo = find_pendulum_repo()
    if repo is None:
        raise FileNotFoundError("Pendulum_control/data not found: clone github.com/jooyongsim/Pendulum_control "
                                "next to this folder (Colab: !git clone ... /content/Pendulum_control)")
    out = []
    for f in sorted((repo / "data").glob(pattern)):            # Path.glob: '[Courses]' in the path is not a pattern
        d = pd.read_csv(f)
        if len(d) < min_len:
            continue
        th = np.unwrap(np.radians(d.theta_deg.values))
        ph = np.radians(d.phi_deg.values)
        out.append(dict(name=Path(f).stem, t=d.t_s.values - d.t_s.values[0], theta=th, phi=ph,
                        theta_dot=savgol(th, 9, 3, 1), phi_dot=np.radians(d.phi_dot_dps.values),
                        u=d.u_rad_s2.values, latched=d.latched.values))
    return out


def arm_accel(log):
    """phi'' that acted on the rod during [t_k, t_k+1): finite difference of the reported arm speed."""
    pd_ = log["phi_dot"]
    return np.r_[np.diff(pd_) / DT, 0.0]


def windows(logs, length=30, stride=15, lag=0):
    """Cut logs into short windows for multiple-shooting identification.
    lag: the rod sees the reported arm motion `lag` samples late (stepper + driver), so the input is shifted.
    Returns tensors: X0 (N,4) initial states, A (N,L) arm accel inputs, TH (N,L+1) measured theta, PD (N,L+1)."""
    X0, A, TH, PD = [], [], [], []
    for lg in logs:
        a = arm_accel(lg)
        if lag:
            a = np.r_[np.zeros(lag), a[:-lag]]
        n = len(lg["theta"])
        for s in range(0, n - length - 1, stride):
            if lg["latched"][s:s + length + 1].any():
                continue
            X0.append([lg["theta"][s], lg["theta_dot"][s], lg["phi"][s], lg["phi_dot"][s]])
            A.append(a[s:s + length])
            TH.append(lg["theta"][s:s + length + 1])
            PD.append(lg["phi_dot"][s:s + length + 1])
    f = lambda v: torch.as_tensor(np.array(v), dtype=torch.float32)
    return f(X0), f(A), f(TH), f(PD)


# ======================================================================= white box: the ODE itself
class PhysicsModel(nn.Module):
    """The second-order ODE as a differentiable program.

    Parameters are stored as logs (positive by construction); the defaults are the hand-identified values
    (free-swing test + open-loop test). randomize(n, rel) draws one parameter set per rollout around the
    current values: model uncertainty without a GP (used for robust policy training, cf. domain randomization)."""

    NAMES = ("a", "b", "k", "c", "Fc")

    def __init__(self, a=A_G, b=B_U, k=1.0, c=C_D, Fc=F_C, eps=0.05, substeps=1):
        super().__init__()
        self.log_p = nn.Parameter(torch.log(torch.tensor([a, b, k, c, Fc], dtype=torch.float32)))
        self.eps, self.substeps = eps, substeps
        self.p_batch = None

    @property
    def p(self):
        return torch.exp(self.log_p) if self.p_batch is None else self.p_batch

    def randomize(self, n, rel=0.1, gen=None):
        """per-rollout parameters p_i = p * exp(rel * N(0,1)); call randomize(0) to switch off"""
        if n == 0:
            self.p_batch = None
            return
        z = torch.randn(n, len(self.NAMES), generator=gen)
        self.p_batch = torch.exp(self.log_p.detach() + rel * z)

    def params(self):
        return dict(zip(self.NAMES, [round(float(v), 4) for v in torch.exp(self.log_p).detach()]))

    # --- rod: theta'' given the arm motion (this is the part identified from data)
    def theta_acc(self, x, acc):
        a, b, k, c, Fc = self.p.unbind(-1)              # (5,) or (N,5) when randomised
        th, w, phid = x[..., 0], x[..., 1], x[..., 3]
        s, co = torch.sin(th), torch.cos(th)
        return a * s + b * co * acc + k * s * co * phid ** 2 - c * w - Fc * torch.tanh(w / self.eps)

    def f(self, x, acc):
        """continuous-time vector field dx/dt for a given arm acceleration"""
        return torch.stack([x[..., 1], self.theta_acc(x, acc), x[..., 3], acc], -1)

    def rk4(self, x, acc, dt):
        k1 = self.f(x, acc)
        k2 = self.f(x + dt / 2 * k1, acc)
        k3 = self.f(x + dt / 2 * k2, acc)
        k4 = self.f(x + dt * k3, acc)
        return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    # --- full world step: command u -> actuator -> rod
    def step(self, x, u, dt=DT):
        h = dt / self.substeps
        for _ in range(self.substeps):
            acc = actuator(u, x[..., 3])
            x = self.rk4(x, acc, h)
        return x

    def step_with_acc(self, x, acc, dt=DT):
        """identification: the arm acceleration is measured, the actuator is bypassed"""
        return self.rk4(x, acc, dt)


def actuator(u, phid, sharp=4.0):
    """Firmware, made smooth: |u| <= U_MAX (tanh), and the arm stops accelerating at the speed limit.
    Hard clipping would give zero gradient exactly where the policy most needs a signal."""
    a = U_MAX * torch.tanh(u / U_MAX)
    room = torch.sigmoid(sharp * (V_MAX - phid * torch.tanh(a / 1.0)))      # ~1 inside, ~0 at +-V_MAX in a's direction
    return a * room


# ======================================================================= gray box: physics + residual
class HybridModel(nn.Module):
    """theta'' = physics(x, phi'') + r_psi(x, phi'').  The residual starts at exactly zero (last layer zeroed),
    so before training the model IS the physics model; training only adds what the physics misses."""

    def __init__(self, physics=None, hidden=64):
        super().__init__()
        self.physics = physics or PhysicsModel()
        self.res = nn.Sequential(nn.Linear(6, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.res[-1].weight); nn.init.zeros_(self.res[-1].bias)

    def theta_acc(self, x, acc):
        feat = torch.stack([torch.sin(x[..., 0]), torch.cos(x[..., 0]), x[..., 1] / 10, x[..., 2], x[..., 3] / 5,
                            acc / 50], -1)
        return self.physics.theta_acc(x, acc) + 10.0 * self.res(feat)[..., 0]

    def f(self, x, acc):
        return torch.stack([x[..., 1], self.theta_acc(x, acc), x[..., 3], acc], -1)

    rk4 = PhysicsModel.rk4

    def step(self, x, u, dt=DT):
        return self.rk4(x, actuator(u, x[..., 3]), dt)

    def step_with_acc(self, x, acc, dt=DT):
        return self.rk4(x, acc, dt)


# ======================================================================= black box: discrete-time NN
def nn_features(x, acc):
    return torch.stack([torch.sin(x[..., 0]), torch.cos(x[..., 0]), x[..., 1] / 10, x[..., 2], x[..., 3] / 5, acc / 50], -1)


class MLPModel(nn.Module):
    """x_{t+1} = x_t + dt * [theta', NN, phi', acc]: the network learns the rod's angular acceleration only.
    Knows the kinematics (position integrates velocity) but no physics. Compare with a fully generic
    Delta-x network: that one must also learn the kinematics and is worse with little data."""

    def __init__(self, hidden=128, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(nn.Linear(6, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def theta_acc(self, x, acc):
        return 50.0 * self.net(nn_features(x, acc))[..., 0]

    def step_with_acc(self, x, acc, dt=DT):
        w = x[..., 1] + dt * self.theta_acc(x, acc)                      # semi-implicit Euler
        phid = x[..., 3] + dt * acc
        return torch.stack([x[..., 0] + dt * w, w, x[..., 2] + dt * phid, phid], -1)

    def step(self, x, u, dt=DT):
        return self.step_with_acc(x, actuator(u, x[..., 3]), dt)


class Ensemble(nn.Module):
    """B independently initialised MLPModels. Spread of the members = epistemic uncertainty (PETS).
    step() with member=None returns the mean; policy training samples one member per rollout (TS-inf)."""

    def __init__(self, B=5, **kw):
        super().__init__()
        self.members = nn.ModuleList([MLPModel(seed=i, **kw) for i in range(B)])

    def step_with_acc(self, x, acc, dt=DT, member=None):
        if member is not None:
            return self.members[member].step_with_acc(x, acc, dt)
        return torch.stack([m.step_with_acc(x, acc, dt) for m in self.members]).mean(0)

    def step(self, x, u, dt=DT, member=None):
        return self.step_with_acc(x, actuator(u, x[..., 3]), dt, member)

    def step_batch_members(self, x, u, idx, dt=DT):
        """x: (N,4); idx: (N,) member index per rollout -> each rollout follows its own member (trajectory sampling)"""
        acc = actuator(u, x[..., 3])
        out = torch.stack([m.step_with_acc(x, acc, dt) for m in self.members])        # (B,N,4)
        return out[idx, torch.arange(len(x))]


# ======================================================================= GP (PILCO's model class), mean only
class GPModel(nn.Module):
    """Exact GP on theta'' with an SE-ARD kernel; hyper-parameters by maximising the evidence.
    Used here as a deterministic mean model for comparison. PILCO additionally propagates the full
    predictive distribution by moment matching -- see Pendulum_control/simulator/pilco_core.py."""

    def __init__(self, max_points=400, seed=0):
        super().__init__()
        self.max_points, self.rng = max_points, np.random.default_rng(seed)

    def fit(self, F, y, iters=200, lr=0.05):
        if len(F) > self.max_points:
            k = self.rng.choice(len(F), self.max_points, replace=False)
            F, y = F[k], y[k]
        self.F, self.y = F.double(), y.double()
        self.ym = self.y.mean()
        E = F.shape[1]
        self.log_ell = nn.Parameter(torch.log(self.F.std(0).clamp_min(1e-2)))
        self.log_sf = nn.Parameter(torch.log(self.y.std()))
        self.log_sn = nn.Parameter(torch.log(self.y.std() / 10))
        opt = torch.optim.Adam([self.log_ell, self.log_sf, self.log_sn], lr=lr)
        for _ in range(iters):
            opt.zero_grad()
            K = self._k(self.F, self.F) + (torch.exp(2 * self.log_sn) + 1e-6) * torch.eye(len(self.F), dtype=torch.float64)
            L = torch.linalg.cholesky(K)
            r = (self.y - self.ym)[:, None]
            al = torch.cholesky_solve(r, L)
            nll = 0.5 * (r * al).sum() + torch.log(torch.diagonal(L)).sum()
            nll.backward(); opt.step()
        with torch.no_grad():
            K = self._k(self.F, self.F) + (torch.exp(2 * self.log_sn) + 1e-6) * torch.eye(len(self.F), dtype=torch.float64)
            self.L = torch.linalg.cholesky(K)
            self.alpha = torch.cholesky_solve((self.y - self.ym)[:, None], self.L)[:, 0]
        return self

    def _k(self, A, B):
        d = (A[:, None, :] - B[None, :, :]) / torch.exp(self.log_ell)
        return torch.exp(2 * self.log_sf) * torch.exp(-0.5 * (d ** 2).sum(-1))

    def predict(self, Fs):
        """mean and variance of theta'' at the inputs Fs (deterministic inputs)"""
        Ks = self._k(Fs.double(), self.F)
        mu = Ks @ self.alpha + self.ym
        v = torch.cholesky_solve(Ks.T, self.L)
        var = torch.exp(2 * self.log_sf) - (Ks * v.T).sum(1)
        return mu.float(), var.clamp_min(0).float()

    def theta_acc(self, x, acc):
        shp = x.shape[:-1]
        mu, _ = self.predict(nn_features(x.reshape(-1, 4), acc.reshape(-1)))
        return mu.reshape(shp)

    step_with_acc = MLPModel.step_with_acc
    step = MLPModel.step


# ======================================================================= identification by backprop through the solver
def simulate_windows(model, X0, A):
    """Open-loop simulation of every window, driven by the measured arm acceleration. Differentiable."""
    xs, x = [X0], X0
    for k in range(A.shape[1]):
        x = model.step_with_acc(x, A[:, k])
        xs.append(x)
    return torch.stack(xs, 1)                          # (N, L+1, 4)


def window_loss(model, X0, A, TH, PD, w0=None):
    """Mean squared angle error over the window (wrapped), plus arm-speed error (keeps phi' consistent)."""
    x0 = X0 if w0 is None else torch.cat([X0[:, :1], w0[:, None], X0[:, 2:]], 1)
    xs = simulate_windows(model, x0, A)
    e_th = twrap(xs[..., 0] - TH)
    return (e_th ** 2).mean() + 0.01 * ((xs[..., 3] - PD) ** 2).mean()


def identify(model, X0, A, TH, PD, iters=300, lr=0.02, free_w0=True, params=None, verbose=True, batch=None):
    """System identification = gradient descent on the simulation error (multiple shooting).
    free_w0: the initial theta' of each window is a free variable too (the measured one is a noisy derivative)."""
    w0 = nn.Parameter(X0[:, 1].clone()) if free_w0 else None
    ps = list(params if params is not None else model.parameters())
    opt = torch.optim.Adam(ps + ([w0] if free_w0 else []), lr=lr)
    hist = []
    for it in range(iters):
        idx = torch.randint(0, len(X0), (batch,)) if batch else slice(None)
        opt.zero_grad()
        loss = window_loss(model, X0[idx], A[idx], TH[idx], PD[idx], None if w0 is None else w0[idx])
        loss.backward()
        opt.step()
        hist.append(loss.item())
        if verbose and (it % max(1, iters // 6) == 0 or it == iters - 1):
            extra = f"  {model.params()}" if hasattr(model, "params") else ""
            print(f"  it {it:4d}  loss {loss.item():.5f}  (rms {math.degrees(math.sqrt(loss.item())):.2f} deg){extra}")
    return hist


def prediction_error(model, X0, A, TH, horizons=(5, 15, 30), warmup=0, iters=60):
    """open-loop RMS angle error [deg] h steps after the start (h * 10 ms).
    warmup=n: first estimate the initial theta' of each window from its first n samples with the model frozen
    (what a state estimator would do), then score the prediction n..n+h -- so the score measures the model,
    not the noise of a differentiated initial velocity. Returns {h: error} with h counted from the warm-up end."""
    if warmup:
        w0 = X0[:, 1].clone().requires_grad_(True)
        req = [p.requires_grad for p in model.parameters()]
        for p in model.parameters():
            p.requires_grad_(False)
        opt = torch.optim.Adam([w0], lr=0.2)
        for _ in range(iters):
            x0 = torch.cat([X0[:, :1], w0[:, None], X0[:, 2:]], 1)
            loss = (twrap(simulate_windows(model, x0, A[:, :warmup])[..., 0] - TH[:, :warmup + 1]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        for p, r in zip(model.parameters(), req):
            p.requires_grad_(r)
        X0 = torch.cat([X0[:, :1], w0.detach()[:, None], X0[:, 2:]], 1)
    with torch.no_grad():
        xs = simulate_windows(model, X0, A)
        e = twrap(xs[..., 0] - TH)
    return {h: float(torch.sqrt((e[:, warmup + h] ** 2).mean()) * 180 / math.pi) for h in horizons
            if warmup + h < e.shape[1]}


def window_features(X0, A, TH, PD, trim=4):
    """one-step regression data (features, theta'') from the identification windows -- for the GP, which
    (as in PILCO) is fitted to one-step targets rather than through the simulator"""
    F, Y = [], []
    for i in range(len(X0)):
        th = TH[i].numpy()
        thd, thdd = savgol(th, 9, 3, 1), savgol(th, 9, 3, 2)
        phd = PD[i].numpy()
        phi = X0[i, 2].item() + np.r_[0.0, np.cumsum(phd[:-1]) * DT]
        x = torch.as_tensor(np.c_[th, thd, phi, phd][:-1], dtype=torch.float32)
        F.append(nn_features(x, A[i])[trim:-trim]); Y.append(torch.as_tensor(thdd[:-1], dtype=torch.float32)[trim:-trim])
    return torch.cat(F), torch.cat(Y)


# ======================================================================= policy and BPTT through the world model
def lqr_gain(model, dt=DT, Q=(100.0, 1.0, 20.0, 1.0), R=0.02):
    """LQR at the upright from autograd Jacobians of ANY differentiable model (tutorial 01) -- zero rollouts.
    PhysicsModel: Coulomb friction is switched off for the linearisation (its smoothed slope would act as fake
    damping). Black-box models are linearised as they are -- only as good as their data near the upright.
    Returns None if the Riccati iteration does not converge (e.g. a learned model that is not stabilisable)."""
    from torch.autograd.functional import jacobian
    if isinstance(model, PhysicsModel):
        lin = PhysicsModel(*[float(v) for v in torch.exp(model.log_p.detach())[:4]], Fc=1e-9).double()
        dtype = torch.float64
    else:
        lin, dtype = model, torch.float32
    x0 = torch.zeros(4, dtype=dtype)
    u0 = torch.tensor(0.0, dtype=dtype)
    A = jacobian(lambda x: lin.step(x, u0, dt), x0).detach().double().numpy()
    B = jacobian(lambda u: lin.step(x0, u, dt), u0).detach().double().numpy().reshape(4, 1)
    Qm, Rm, P = np.diag(Q), np.array([[R]]), np.diag(Q)
    for _ in range(5000):
        K = np.linalg.solve(Rm + B.T @ P @ B, B.T @ P @ A)
        P = Qm + A.T @ P @ (A - B @ K)
    if not np.all(np.isfinite(K)) or np.abs(np.linalg.eigvals(A - B @ K)).max() >= 1:
        return None
    return K.ravel()


class Policy(nn.Module):
    """u = U_MAX * tanh( g(theta) * (-K x_up) / U_MAX + NN([sin theta, cos theta, theta'/10, phi, phi'/5]) )

    K given (lqr_gain): a physics prior inside the policy -- the LQR balancer, gated to the upright region by
    g = ((1 + cos theta)/2)^10 (1 upright, 0.2 at 45 deg, 0 hanging). BPTT then only has to learn the swing-up
    and the hand-off; the network starts at ~0 output so the initial policy IS the LQR near the top.
    K = None: plain network, everything learned from scratch.

    swing=True adds the rig's energy swing-up law (swing-up/swingup_control.py) away from the top:
        u_swing = pump * clip(kE (E* - E), 0, 1) * tanh(theta' cos theta / 0.3) - kda phi' - kpa phi
        prior   = (1 - g) u_swing + g u_lqr
    sign() is smoothed by tanh so the law is differentiable; its four gains start at the hardware values and are
    trained by BPTT together with the network (interpretable parameters + residual). The -kpa phi - kda phi' terms
    keep the arm centred during pumping -- the part a cost term alone did not teach.

    nn_gate=True multiplies the network by (1 - g): near the top only the LQR acts, so upright with the arm at 0
    stays an equilibrium. Without it the residual can learn a small offset at the top (the 4 s training rollouts
    barely see steady balancing) and the LQR then holds the arm off-centre (seen in diff_sim: arm parked at -60 deg)."""

    SWING0 = dict(pump=30.0, ke=0.3, kpa=3.0, kda=5.0)  # hardware defaults (rotary_model.SWING)
    E_TARGET = 1.0

    def __init__(self, hidden=64, seed=0, K=None, swing=False, nn_gate=False):
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(nn.Linear(5, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        nn.init.uniform_(self.net[-1].weight, -0.01, 0.01); nn.init.zeros_(self.net[-1].bias)
        self.register_buffer("K", torch.as_tensor(np.zeros(4) if K is None else np.asarray(K), dtype=torch.float32))
        self.swing = swing
        self.register_buffer("nn_gate", torch.tensor(float(nn_gate)))      # saved with the weights
        self.log_sw = nn.Parameter(torch.log(torch.tensor(list(self.SWING0.values()), dtype=torch.float32)),
                                   requires_grad=swing)

    def gains(self):
        return dict(zip(self.SWING0, [round(float(v), 3) for v in torch.exp(self.log_sw).detach()]))

    def forward(self, x):
        s, c = torch.sin(x[..., 0]), torch.cos(x[..., 0])
        f = torch.stack([s, c, x[..., 1] / 10, x[..., 2], x[..., 3] / 5], -1)
        x_up = torch.stack([s, x[..., 1], x[..., 2], x[..., 3]], -1)            # sin(theta) ~ theta near the top
        g = ((1 + c) / 2) ** 10
        prior = -(x_up * self.K).sum(-1) * g
        if self.swing:
            pump, ke, kpa, kda = torch.exp(self.log_sw)
            scale = torch.clamp(ke * (self.E_TARGET - rod_energy(x)), 0.0, 1.0)
            u_sw = pump * scale * torch.tanh(x[..., 1] * c / 0.3) - kda * x[..., 3] - kpa * x[..., 2]
            prior = prior + (1 - g) * u_sw
        res = self.net(f)[..., 0] * (1 - self.nn_gate * g)
        return U_MAX * torch.tanh(prior / U_MAX + res)

    def numpy_fn(self):
        W = [(l.weight.detach().double().numpy(), l.bias.detach().double().numpy()) for l in self.net if isinstance(l, nn.Linear)]
        K = self.K.double().numpy()
        pump, ke, kpa, kda = torch.exp(self.log_sw).detach().double().numpy()
        swing, et, gate = self.swing, self.E_TARGET, float(self.nn_gate)

        def pi(z):                                     # z = [theta, theta', phi, phi'] as the rig/Loop reports it
            s, c = math.sin(z[0]), math.cos(z[0])
            h = np.array([s, c, z[1] / 10, z[2], z[3] / 5])
            for i, (w, b) in enumerate(W):
                h = w @ h + b
                if i < len(W) - 1:
                    h = np.tanh(h)
            g = ((1 + c) / 2) ** 10
            prior = -float(K @ np.array([s, z[1], z[2], z[3]])) * g
            if swing:
                E = 0.5 * z[1] ** 2 + A_G * (c - 1)
                scale = min(1.0, max(0.0, ke * (et - E)))
                prior += (1 - g) * (pump * scale * math.tanh(z[1] * c / 0.3) - kda * z[3] - kpa * z[2])
            return float(U_MAX * math.tanh(prior / U_MAX + h[0] * (1 - gate * g)))
        return pi


def rod_energy(x):
    """rod energy per unit inertia: 0 upright at rest, -2 a hanging at rest (a = wn^2)"""
    return 0.5 * x[..., 1] ** 2 + A_G * (torch.cos(x[..., 0]) - 1)


def stage_cost(x, u, w_speed=0.0, w_energy=0.0, w_arm=0.05, w_wall=3.0):
    """bounded 'tip distance' cost (PILCO Eq. 25 style) + arm kept near the centre + a little effort.
    w_speed  > 0: charges rod speed near the top, so passing over the top fast (the 'keep spinning' local
                  optimum) costs more than arriving slowly and staying.
    w_energy > 0: charges the energy error (E / 2a)^2. The tip cost has zero gradient at the hanging state
                  (d(1-cos)/dtheta = sin = 0) and pays off only when the rod is high; the energy rises with
                  every pump, so even a short back-propagation window sees a signal -- a value function
                  supplied by physics (the role SHAC gives to a learned critic)."""
    up = (1 + torch.cos(x[..., 0])) / 2                 # 1 upright, 0 hanging (theta = 0 is UP)
    arm = (x[..., 2] / ARM_LIM) ** 2
    # soft wall before the 140 deg stop -- BOUNDED (at most 3 per step, like a termination penalty): an unbounded
    # wall lets a few runaway rollouts dominate J, and the optimum becomes 'never swing up' (seen in tutorial 03)
    wall = w_wall * torch.tanh(50 * torch.relu(x[..., 2].abs() - math.radians(120)) ** 2)
    c = (1 - up) + w_speed * up * (x[..., 1] / WN) ** 2 + w_arm * arm + wall + 1e-5 * u ** 2
    if w_energy:
        c = c + w_energy * (rod_energy(x) / (2 * A_G)) ** 2
    return c


def initial_states(n, spread=1.0, gen=None):
    """around hanging: theta = pi +- 0.3 spread, theta' = +-1 spread rad/s, phi = +-0.1 spread.
    Exactly at rest the gradient of any smooth cost vanishes (hanging is a stationary point), so training
    starts from small swings; the trained policy is then tested from rest."""
    g = gen or torch.Generator().manual_seed(0)
    th = math.pi + spread * 0.6 * (torch.rand(n, generator=g) - 0.5)
    w = spread * 2.0 * (torch.rand(n, generator=g) - 0.5)
    phi = spread * 0.2 * (torch.rand(n, generator=g) - 0.5)
    return torch.stack([th, w, phi, torch.zeros(n)], 1)


def rollout(model, policy, x0, H, trunc=None, member_idx=None, dt=DT, w_speed=0.0, w_energy=0.0, w_arm=0.05, w_wall=3.0):
    """Simulate H steps in the model, return per-step cost (N,H) and states. trunc=K detaches the state every
    K steps (short-horizon gradients, cf. SHAC): the cost still covers the full horizon, the gradient does not
    flow through more than K steps -> no exploding gradients through the unstable upright."""
    x, cs, xs = x0, [], [x0]
    for t in range(H):
        if trunc and t % trunc == 0 and t > 0:
            x = x.detach()
        u = policy(x)
        cs.append(stage_cost(x, u, w_speed, w_energy, w_arm, w_wall))
        x = model.step_batch_members(x, u, member_idx, dt) if member_idx is not None else model.step(x, u, dt)
        xs.append(x)
    return torch.stack(cs, 1), torch.stack(xs, 1)


def train_policy_bptt(model, policy, H=300, iters=500, batch=64, lr=3e-3, trunc=None, clip=1.0, ensemble=False,
                      randomize=0.0, dt=DT, w_speed=0.3, w_energy=0.0, w_arm=0.05, w_wall=3.0, x0_spread=1.0, verbose=True,
                      log_every=50, gen_seed=0, select_best=True, val_every=10):
    """Policy optimisation WITHOUT RL: J = mean cost of model rollouts, dJ/dtheta by backprop through time.
    The model is frozen; only the policy parameters move.
    select_best   : every val_every iterations score the policy on a FIXED validation batch (same starts, same
                    parameter draws) and keep the best; restore it at the end. Gradients through seconds of unstable
                    dynamics are noisy enough that a step can jump into a worse basin (seen in tutorial 03).
    ensemble=True : each rollout follows one randomly drawn ensemble member (PETS-style trajectory sampling)
    randomize=r   : PhysicsModel only -- each rollout gets its own physical parameters (relative spread r)
    dt            : policy period used in the model (0.02 halves the cost; the policy still runs at 100 Hz)"""
    for p in model.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(gen_seed)
    hist, t0 = [], time.time()
    vgen = torch.Generator().manual_seed(12345)
    vx0 = initial_states(batch, spread=x0_spread, gen=vgen)
    vp = None
    if randomize:
        model.randomize(batch, randomize, vgen)
        vp = model.p_batch.clone()
    vidx = torch.randint(0, len(model.members), (batch,), generator=vgen) if ensemble else None
    best = (float("inf"), None, -1)

    def validate():
        if randomize:
            model.p_batch = vp
        with torch.no_grad():
            cs, _ = rollout(model, policy, vx0, H, member_idx=vidx, dt=dt, w_speed=w_speed, w_energy=w_energy, w_arm=w_arm, w_wall=w_wall)
        return cs.mean().item()

    for it in range(iters):
        if select_best and it % val_every == 0:
            v = validate()
            if v < best[0]:
                best = (v, {k: t.clone() for k, t in policy.state_dict().items()}, it)
        x0 = initial_states(batch, spread=x0_spread, gen=gen)
        idx = torch.randint(0, len(model.members), (batch,), generator=gen) if ensemble else None
        if randomize:
            model.randomize(batch, randomize, gen)
        cs, xs = rollout(model, policy, x0, H, trunc=trunc, member_idx=idx, dt=dt, w_speed=w_speed, w_energy=w_energy,
                         w_arm=w_arm, w_wall=w_wall)
        J = cs.mean()
        opt.zero_grad()
        J.backward()
        gn = torch.nn.utils.clip_grad_norm_(policy.parameters(), clip)
        opt.step()
        up = (torch.cos(xs[:, -int(0.5 / dt):, 0]) > math.cos(math.radians(15))).float().mean().item()
        hist.append(dict(it=it, J=J.item(), grad=float(gn), up_last=up))
        if verbose and (it % log_every == 0 or it == iters - 1):
            print(f"  it {it:4d}  J {J.item():.4f}  |grad| {float(gn):9.3g}  upright(last 0.5 s, model) {100 * up:5.1f}%"
                  f"  [{time.time() - t0:5.0f} s]", flush=True)
    if select_best and iters:
        v = validate()
        if v < best[0]:
            best = (v, None, iters)
        elif best[1] is not None:
            policy.load_state_dict(best[1])
        if verbose:
            print(f"  kept the policy from iteration {best[2]} (validation J {best[0]:.4f})", flush=True)
        if hist:
            hist[-1]["best_it"], hist[-1]["best_val"] = best[2], best[0]
    for p in model.parameters():
        p.requires_grad_(True)
    if randomize:
        model.randomize(0)
    return hist


# ======================================================================= the 'real system' (not differentiable)
def import_rotary_model():
    """Pendulum_control/simulator/rotary_model.py: plant + firmware + sensor + loop timing of the real rig."""
    repo = find_pendulum_repo()
    if repo is not None:
        sys.path.insert(0, str(repo / "simulator"))
    else:
        dst = HERE / "rotary_model.py"
        if not dst.exists():
            urllib.request.urlretrieve(f"{REPO_RAW}/simulator/rotary_model.py", dst)
        sys.path.insert(0, str(HERE))
    import rotary_model
    return rotary_model


def run_real(policy_fn, seconds=5.0, theta0=math.pi, realistic=True, seed=0, plant_kw=None):
    """One trial on the stand-in real system. Returns dict with t, x (true plant state), u, stop reason.
    The policy sees what the rig's host sees (quantised encoder, filtered theta', 4 ms delay) when realistic."""
    rm = import_rotary_model()
    rng = np.random.default_rng(seed)
    plant = rm.Plant(theta0=theta0 + rng.uniform(-0.03, 0.03), **(plant_kw or {}))
    ctrl = lambda z, t: policy_fn(z)
    loop = rm.Loop(plant, ctrl, rm.Sensor(realistic=realistic))
    T, X, U = [], [], []
    for _ in range(int(round(seconds * rm.CONTROL_HZ))):
        loop.advance(int(round(1000 / rm.CONTROL_HZ)))
        T.append(plant.t); X.append(plant.x.copy()); U.append(loop.u)
        if loop.stop:
            break
    return dict(t=np.array(T), x=np.array(X), u=np.array(U), stop=loop.stop)


def score(trial, hold=1.0):
    """swing-up success: |theta| < 15 deg for the last `hold` seconds without a stop"""
    th = wrap(trial["x"][:, 0])
    n = int(hold / DT)
    up = np.abs(th) < math.radians(15)
    first = int(np.argmax(up)) * DT if up.any() else None
    ok = (not trial["stop"]) and len(th) >= n and bool(up[-n:].all())
    return dict(success=ok, first_upright=first, upright_frac_last=float(up[-n:].mean()) if len(up) >= n else 0.0,
                stop=trial["stop"])


def trial_to_log(trial):
    """real trial -> same dict format as load_real_logs (so the identification code is reused unchanged)"""
    x = trial["x"]
    th = np.unwrap(x[:, 0])
    return dict(name="sim", t=trial["t"], theta=th, phi=x[:, 2], theta_dot=savgol(th, 9, 3, 1),
                phi_dot=np.array([v if abs(v) >= 30 * STEP_RAD else 0.0 for v in x[:, 3]]), u=trial["u"],
                latched=np.zeros(len(th)))
