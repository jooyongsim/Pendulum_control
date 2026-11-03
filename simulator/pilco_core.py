"""PILCO (Deisenroth & Rasmussen, ICML 2011) for the rotary inverted pendulum, in PyTorch.

Same algorithm as the paper (Alg. 1): GP dynamics model on state differences (Sec. 2.1), Gaussian state
distributions propagated through policy and GP by exact moment matching (Sec. 2.2, Eqs. 10-23), expected
saturating cost (Eq. 25), gradient-based policy search (Sec. 2.3). The analytic gradients of Eqs. 26-30 are
obtained here by automatic differentiation of the same closed-form expressions, instead of hand-derived.
Like the original MATLAB code, angles enter the GP and the policy as (sin, cos), with moments computed in
closed form, and the RBF policy (Eqs. 31-32) is squashed by u = u_max sin(.) to respect the actuator limit.

State x = [theta', phi, phi', theta]   theta: rod angle from UPRIGHT (hanging = +-pi, kept unwrapped)
                                       phi: arm angle, u = phi'' [rad/s^2] (arm acceleration command)
Augmented z = [x, sin theta, cos theta];  policy input  = z[[0, 1, 2, 4, 5]]
                                          GP input      = [policy input, u],  GP target = x_{t+1} - x_t

Used by simulator/pilco_swingup_colab.py (simulation) and swing-up/swingup_pilco_control.py (real rig).
"""
import math
import time

import numpy as np
import torch

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)               # small matrices: one thread is much faster (12 s -> 0.15 s per GP step here)
D, IDX_TH = 4, 3                       # state size, index of theta
P_IN = [0, 1, 2, 4, 5]                 # policy input inside z = [x, sin, cos]
G_IN = [0, 1, 2, 4, 5, 6]              # GP input inside [z, u]


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ------------------------------------------------------------------ moment matching building blocks
def trig_moments(m, S, i=IDX_TH):
    """x ~ N(m, S) -> moments of z = [x, sin x_i, cos x_i] (exact)."""
    mu, s2 = m[i], S[i, i]
    e1, e2 = torch.exp(-s2 / 2), torch.exp(-2 * s2)
    Es, Ec = e1 * torch.sin(mu), e1 * torch.cos(mu)
    Vss = (1 - e2 * torch.cos(2 * mu)) / 2 - Es ** 2
    Vcc = (1 + e2 * torch.cos(2 * mu)) / 2 - Ec ** 2
    Vsc = e2 * torch.sin(2 * mu) / 2 - Es * Ec
    Cxs, Cxc = S[:, i] * Ec, -S[:, i] * Es                      # cov[x, sin], cov[x, cos] (Stein's lemma)
    M = torch.cat([m, Es[None], Ec[None]])
    top = torch.cat([S, Cxs[:, None], Cxc[:, None]], 1)
    bot = torch.stack([torch.cat([Cxs, Vss[None], Vsc[None]]), torch.cat([Cxc, Vsc[None], Vcc[None]])])
    return M, torch.cat([top, bot], 0)


def kernel_moments(m, S, X, beta, ell, alpha2, iK=None):
    """Moment matching of f_a(x) = sum_i beta_ia alpha2_a exp(-1/2 |x - X_i|^2_{Lambda_a}) at x ~ N(m, S)
    (Eqs. 14-16, 20-23). iK given -> it is a GP: adds E_x[var_f] (Eq. 23) on the diagonal.
    Returns M (K), V (K x K), and G (E x K) with cov[y, f] = cov[y, x] @ G for any y jointly Gaussian with x."""
    n, E = X.shape
    I = torch.eye(E)
    nu = X - m                                                   # (n, E), Eq. 16
    iL = 1.0 / ell ** 2                                          # (K, E) = diag(Lambda_a^-1)
    # ---- mean (Eqs. 14-16), all outputs at once
    A = S[None] + torch.diag_embed(ell ** 2)                     # S + Lambda_a
    t = torch.linalg.solve(A, nu.T.expand(len(ell), E, n)).transpose(1, 2)   # (K, n, E): (S+Lambda)^-1 nu_i
    c = alpha2 / torch.sqrt(torch.linalg.det(S[None] * iL[:, None, :] + I))
    q = c[:, None] * torch.exp(-0.5 * (nu[None] * t).sum(-1))   # (K, n), Eq. 15
    bq = beta.T * q                                              # (K, n)
    M = bq.sum(1)                                                # Eq. 14
    G = torch.einsum("kne,kn->ek", t, bq)                        # cov[x, f] = S G
    # ---- covariance (Eqs. 17-23), all output pairs at once
    logk = torch.log(alpha2)[:, None] - 0.5 * (nu[None] ** 2 * iL[:, None, :]).sum(-1)   # (K, n): log k_a(X_i, m)
    Z = nu[None] * iL[:, None, :]                                # (K, n, E): Lambda_a^-1 nu_i
    R = S[None, None] * (iL[:, None, :] + iL[None, :, :])[:, :, None, :] + I          # (K, K, E, E)
    T = torch.linalg.solve(R, S.expand_as(R))                    # R^-1 S (symmetric)
    T = (T + T.transpose(-1, -2)) / 2
    ZT = torch.einsum("ane,abef->abnf", Z, T)                    # (K, K, n, E)
    qa = 0.5 * (ZT * Z[:, None]).sum(-1)                         # z_a^T T z_a       (K, K, n)
    qb = 0.5 * (torch.einsum("bne,abef->abnf", Z, T) * Z[None]).sum(-1)              # z_b^T T z_b
    ex = (logk[:, None, :] + qa)[..., :, None] + (logk[None, :, :] + qb)[..., None, :] \
        + torch.einsum("abnf,bmf->abnm", ZT, Z)
    Q = torch.exp(ex) / torch.sqrt(torch.linalg.det(R))[..., None, None]             # Eq. 22
    V = torch.einsum("na,abnm,mb->ab", beta, Q, beta) - M[:, None] * M[None, :]       # Eqs. 18, 20
    if iK is not None:                                           # Eq. 23: E_x[var_f] on the diagonal
        Qd = torch.diagonal(Q, dim1=0, dim2=1).permute(2, 0, 1)  # (K, n, n)
        V = V + torch.diag(alpha2 - (iK * Qd).sum((1, 2)))
    return M, (V + V.T) / 2, G


# ------------------------------------------------------------------ GP dynamics model (Sec. 2.1)
class GPModel:
    """D independent GPs, SE-ARD kernel, zero prior mean; hyper-parameters by evidence maximisation."""

    def __init__(self, max_points=250, seed=0):
        self.max_points, self.rng = max_points, np.random.default_rng(seed)
        self.log_ell = self.log_alpha = self.log_sn = None

    def fit(self, X, Y, iters=150, lr=0.05, verbose=False):
        X, Y = np.asarray(X, float), np.asarray(Y, float)
        if len(X) > self.max_points:                             # subset of data (keeps cost bounded)
            keep = np.r_[np.arange(len(X) - self.max_points // 3, len(X)),
                         self.rng.choice(len(X) - self.max_points // 3, self.max_points - self.max_points // 3,
                                         replace=False)]
            X, Y = X[keep], Y[keep]
        self.X, self.Y = torch.as_tensor(X), torch.as_tensor(Y)
        n, E = X.shape
        if self.log_ell is None:                                 # init from data spread
            sx = np.maximum(X.std(0), 1e-3)
            sy = np.maximum(Y.std(0), 1e-4)
            self.log_ell = torch.tensor(np.log(np.tile(sx, (D, 1))), requires_grad=True)
            self.log_alpha = torch.tensor(np.log(sy), requires_grad=True)
            self.log_sn = torch.tensor(np.log(sy / 10), requires_grad=True)
        opt = torch.optim.Adam([self.log_ell, self.log_alpha, self.log_sn], lr=lr)
        for it in range(iters):
            opt.zero_grad()
            nll = self._nll()
            nll.backward()
            opt.step()
            if verbose and (it % 50 == 0 or it == iters - 1):
                print(f"   GP it {it:3d}  -log evidence {nll.item():9.2f}")
        with torch.no_grad():
            self._precompute()
        return self

    def _K(self, a, X1, X2):
        d = (X1[:, None, :] - X2[None, :, :]) / torch.exp(self.log_ell[a])
        return torch.exp(2 * self.log_alpha[a]) * torch.exp(-0.5 * (d ** 2).sum(-1))

    def _nll(self):
        n = len(self.X)
        d = (self.X[None, :, None, :] - self.X[None, None, :, :]) / torch.exp(self.log_ell)[:, None, None, :]
        K = torch.exp(2 * self.log_alpha)[:, None, None] * torch.exp(-0.5 * (d ** 2).sum(-1)) \
            + (torch.exp(2 * self.log_sn) + 1e-6)[:, None, None] * torch.eye(n)
        Lc = torch.linalg.cholesky(K)                            # (D, n, n), all outputs at once
        Yt = self.Y.T[:, :, None]
        al = torch.cholesky_solve(Yt, Lc)
        nll = 0.5 * (Yt * al).sum() + torch.log(torch.diagonal(Lc, dim1=1, dim2=2)).sum()
        # mild prior (as in the PILCO toolbox): signal-to-noise ratio below ~500
        return nll + (((self.log_alpha - self.log_sn) / math.log(500)) ** 30).sum()

    def _precompute(self):
        n = len(self.X)
        self.ell = torch.exp(self.log_ell).detach()
        self.alpha2 = torch.exp(2 * self.log_alpha).detach()
        iK, beta = [], []
        for a in range(D):
            K = self._K(a, self.X, self.X).detach() + (torch.exp(2 * self.log_sn[a]).detach() + 1e-6) * torch.eye(n)
            ik = torch.cholesky_inverse(torch.linalg.cholesky(K))
            iK.append(ik); beta.append(ik @ self.Y[:, a])
        self.iK, self.beta = torch.stack(iK), torch.stack(beta, 1)

    def predict_mean(self, Xs):
        """deterministic test inputs (for checking the model): mean of Delta"""
        Xs = torch.as_tensor(np.atleast_2d(Xs))
        with torch.no_grad():
            return torch.stack([self._K(a, Xs, self.X) @ self.beta[:, a] for a in range(D)], 1).numpy()


# ------------------------------------------------------------------ policy (Eqs. 31-32 + sin squashing)
class RBFPolicy(torch.nn.Module):
    """u = u_max sin( sum_i w_i exp(-1/2 |p - c_i|^2_Lambda) ),  p = [theta', phi, phi', sin theta, cos theta]"""

    def __init__(self, n_basis=50, u_max=47.12, seed=0):
        super().__init__()
        g = np.random.default_rng(seed)
        lo, hi = np.array([-8, -1.5, -4, -1, -1.0]), np.array([8, 1.5, 4, 1, 1.0])
        self.c = torch.nn.Parameter(torch.as_tensor(g.uniform(lo, hi, (n_basis, 5))))
        self.w = torch.nn.Parameter(torch.as_tensor(0.1 * g.standard_normal((n_basis, 1))))
        self.log_ell = torch.nn.Parameter(torch.log(torch.tensor([[4.0, 1.0, 2.5, 0.7, 0.7]])))
        self.u_max = u_max

    def moments(self, m, S):
        """p ~ N(m, S) -> mean, variance of u and G with cov[y, u] = cov[y, p] @ G"""
        M, V, G = kernel_moments(m, S, self.c, self.w, torch.exp(self.log_ell), torch.ones(1))
        mu, s2 = M[0], V[0, 0]
        e1, e2 = torch.exp(-s2 / 2), torch.exp(-2 * s2)
        Mu = self.u_max * e1 * torch.sin(mu)
        Vu = self.u_max ** 2 * (1 - e2 * torch.cos(2 * mu)) / 2 - Mu ** 2
        return Mu, Vu, G[:, 0] * self.u_max * e1 * torch.cos(mu)

    def numpy_params(self):
        return dict(c=self.c.detach().numpy().copy(), w=self.w.detach().numpy()[:, 0].copy(),
                    ell=torch.exp(self.log_ell).detach().numpy()[0].copy(), u_max=np.array(self.u_max))


def policy_numpy(params):
    """torch-free policy for the real-time loop: x = [theta', phi, phi', theta] -> u [rad/s^2]"""
    c, w, ell, umax = params["c"], params["w"], params["ell"], float(params["u_max"])

    def pi(x):
        p = np.array([x[0], x[1], x[2], math.sin(x[3]), math.cos(x[3])])
        return umax * math.sin(float(w @ np.exp(-0.5 * (((p - c) / ell) ** 2).sum(1))))
    return pi


# ------------------------------------------------------------------ long-term prediction + cost (Sec. 2.2)
def propagate(m, S, policy, gp):
    """one step: p(x_t) -> p(x_t+1) by moment matching (Eqs. 10-12)"""
    Mz, Vz = trig_moments(m, S)                                  # z = [x, sin, cos]
    Mu, Vu, Gu = policy.moments(Mz[P_IN], Vz[P_IN][:, P_IN])
    Czu = Vz[:, P_IN] @ Gu                                       # cov[z, u]
    Mj = torch.cat([Mz, Mu[None]])
    Vj = torch.cat([torch.cat([Vz, Czu[:, None]], 1), torch.cat([Czu, Vu[None]])[None, :]], 0)
    Mg, Vg = Mj[G_IN], Vj[G_IN][:, G_IN]
    Md, Vd, Gd = kernel_moments(Mg, Vg, gp.X, gp.beta, gp.ell, gp.alpha2, gp.iK)
    Cxd = Vj[:D][:, G_IN] @ Gd                                   # cov[x, Delta]
    S1 = S + Vd + Cxd + Cxd.T
    return m + Md, (S1 + S1.T) / 2, (Mu, Vu)


class Cost:
    """saturating cost (Eq. 25) on y = [phi, sin theta, cos theta], target upright with the arm centred.
    arm_quad > 0 adds an (unbounded) quadratic arm term -- keep it 0: with an uncertain model it outweighs the
    bounded cost and the optimum becomes "do not move" (checked: the policy then never leaves the hanging state)."""

    def __init__(self, sigma_c=0.5, arm_scale=2.0, arm_quad=0.0, arm_lim=math.radians(140)):
        self.W = torch.diag(torch.tensor([1 / arm_scale ** 2, 1 / sigma_c ** 2, 1 / sigma_c ** 2]))
        self.target = torch.tensor([0.0, 0.0, 1.0])
        self.arm_quad, self.arm_lim = arm_quad, arm_lim

    def __call__(self, m, S):
        Mz, Vz = trig_moments(m, S)
        idx = [1, 4, 5]
        d, Sy = Mz[idx] - self.target, Vz[idx][:, idx]
        IpSW = torch.eye(3) + Sy @ self.W
        sat = 1 - torch.exp(-0.5 * d @ self.W @ torch.linalg.solve(IpSW, d)) / torch.sqrt(torch.linalg.det(IpSW))
        return sat + self.arm_quad * (m[1] ** 2 + S[1, 1]) / self.arm_lim ** 2

    def point(self, x):
        """cost of one observed state (for reporting)"""
        y = np.array([x[1], math.sin(x[3]), math.cos(x[3])]) - self.target.numpy()
        return 1 - math.exp(-0.5 * y @ self.W.numpy() @ y) + self.arm_quad * x[1] ** 2 / self.arm_lim ** 2


def predict(policy, gp, cost, m0, S0, H):
    m, S = torch.as_tensor(m0), torch.as_tensor(S0)
    ms, Ss, cs = [m], [S], [cost(m, S)]
    for _ in range(H):
        m, S, _ = propagate(m, S, policy, gp)
        ms.append(m); Ss.append(S); cs.append(cost(m, S))
    return torch.stack(ms), torch.stack(Ss), torch.stack(cs)


def optimize_policy(policy, gp, cost, m0, S0, H, iters=100, lr=0.02, verbose=True):
    """minimise J = sum_t E[c(x_t)] (Eq. 2) with gradients through the moment-matching chain (Sec. 2.3)"""
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    best, best_state, t0 = float("inf"), None, time.time()
    for it in range(iters):
        opt.zero_grad()
        _, _, cs = predict(policy, gp, cost, m0, S0, H)
        J = cs.sum()
        if not torch.isfinite(J):
            if best_state is not None:
                policy.load_state_dict(best_state)
            break
        J.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
        opt.step()
        if J.item() < best:
            best, best_state = J.item(), {k: v.detach().clone() for k, v in policy.state_dict().items()}
        if verbose and (it % 20 == 0 or it == iters - 1):
            print(f"   policy it {it:3d}  J = {J.item():7.3f}  (best {best:7.3f}, {time.time() - t0:5.1f} s)", flush=True)
    policy.load_state_dict(best_state)
    return best


# ------------------------------------------------------------------ data handling
def transitions(X, U):
    """one trial: states X (T+1, 4) and actions U (T,) -> GP inputs (T, 6) and targets (T, 4)"""
    X, U = np.asarray(X, float), np.asarray(U, float)
    inp = np.c_[X[:-1, :3], np.sin(X[:-1, 3]), np.cos(X[:-1, 3]), U]
    dX = X[1:] - X[:-1]
    dX[:, IDX_TH] = wrap(dX[:, IDX_TH])
    return inp, dX
