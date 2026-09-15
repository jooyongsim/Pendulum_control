"""Walk through the envelope fit step by step, with the real numbers.

The fit answers: given the sequence of swing amplitudes A_k at times t_k, what
viscous rate sigma and Coulomb term c reproduce them?
"""
import glob
import sys

import numpy as np

from _common import (ENC_RES_DEG, alternating_extrema, load_log, log_paths, split_runs)  # noqa: E402



def amplitudes(path):
    t, theta = load_log(path)
    k = alternating_extrema(theta, np.median(theta))
    eq = np.median(0.5 * (theta[k][:-1] + theta[k][1:]))
    tp_all, amp_all = t[k], np.abs(theta[k] - eq)
    for a, b in split_runs(tp_all, amp_all):
        tp, ap = tp_all[a:b], amp_all[a:b]
        m = ap > 2 * np.deg2rad(ENC_RES_DEG)
        tp, ap = tp[m], ap[m]
        if len(tp) >= 8 and ap[-1] < ap[0]:
            yield tp - tp[0], ap


# the 93.5 deg release: the longest run, 30 extrema
run = None
for p in sorted(log_paths()):
    for t, A in amplitudes(p):
        if run is None or len(t) > len(run[0]):
            run = (t, A)
t, A = run
print(f"run used: {len(t)} extrema, {np.rad2deg(A[0]):.1f} -> {np.rad2deg(A[-1]):.1f} deg, "
      f"{t[-1]:.1f} s\n")

print("=" * 76)
print("STEP 1  the model")
print("=" * 76)
print("  Per half swing, viscous friction removes a FRACTION of the amplitude and")
print("  Coulomb friction removes a FIXED amount. Over time:")
print("      dA/dt = -(sigma*A + c)")
print("  Solving that linear ODE:")
print("      A(t) = (A0 + c/sigma) exp(-sigma t) - c/sigma")
print("  sigma=0 gives a straight line (pure Coulomb); c=0 gives a pure exponential.\n")

print("=" * 76)
print("STEP 2  why it is not a plain least-squares problem")
print("=" * 76)
print("  A(t) = P exp(-s t) + Q  with  P = A0 + c/s,  Q = -c/s")
print("  For a FIXED s this is linear in (P, Q). Only s is nonlinear.\n")

print("=" * 76)
print("STEP 3  scan s, solve (P, Q) exactly at each s")
print("=" * 76)
print("  Normal equations for the 2-parameter linear fit, e = exp(-s t):")
print("      [ e.e   sum(e) ] [P]   [ e.A  ]")
print("      [ sum(e)   n   ] [Q] = [ sum(A)]")
print("  2x2, so it is solved in closed form -- no iteration, no optimiser.\n")


def fit_at(s):
    e = np.exp(-s * t)
    n = len(t)
    s11, s12 = float(e @ e), float(e.sum())
    b1, b2 = float(e @ A), float(A.sum())
    det = s11 * n - s12 * s12
    P = (b1 * n - b2 * s12) / det
    Q = (s11 * b2 - s12 * b1) / det
    resid = A - (P * e + Q)
    return float(resid @ resid), P, Q


grid = np.linspace(1e-3, 3.0, 4000)
ss = np.array([fit_at(s)[0] for s in grid])
best = int(np.argmin(ss))
s_hat = grid[best]
_, P, Q = fit_at(s_hat)
sigma, c = s_hat, -Q * s_hat
A0 = P + Q
tot = float(np.sum((A - A.mean()) ** 2))
r2 = 1 - ss[best] / tot

print(f"  scanned {len(grid)} values of s in [{grid[0]:.3f}, {grid[-1]:.1f}] 1/s")
print(f"  minimum at s = {s_hat:.4f} 1/s\n")
print("  nearby values of the sum of squared residuals:")
for j in range(max(0, best - 2), min(len(grid), best + 3)):
    mark = "  <-- min" if j == best else ""
    print(f"      s = {grid[j]:.4f}   SSE = {ss[j]:.6e}{mark}")

print("\n" + "=" * 76)
print("STEP 4  read the physical parameters back out")
print("=" * 76)
print(f"  P = {P:.6f} rad,  Q = {Q:.6f} rad")
print(f"  sigma = s          = {sigma:.4f} 1/s          (viscous decay rate)")
print(f"  c     = -Q * sigma = {np.rad2deg(c):.3f} deg/s   (Coulomb amplitude loss)")
print(f"  A0    = P + Q      = {np.rad2deg(A0):.2f} deg")
print(f"  R^2                = {r2:.4f}")

print("\n" + "=" * 76)
print("STEP 5  compare with the single-mechanism fits")
print("=" * 76)
pv = np.polyfit(t, np.log(A), 1)
r2v = 1 - np.var(np.log(A) - np.polyval(pv, t)) / np.var(np.log(A))
pl = np.polyfit(t, A, 1)
r2l = 1 - np.var(A - np.polyval(pl, t)) / np.var(A)
print(f"  viscous only  (log A linear in t) : sigma = {-pv[0]:.4f} 1/s, R^2 = {r2v:.4f}")
print(f"  Coulomb only  (A linear in t)     : rate  = {np.rad2deg(-pl[0]):.3f} deg/s, "
      f"R^2 = {r2l:.4f}")
print(f"  joint                              : R^2 = {r2:.4f}")
print("\n  The log-decrement method IS the viscous-only fit, so using it here would")
print(f"  report sigma = {-pv[0]:.4f} instead of {sigma:.4f} -- a {100*abs(-pv[0]-sigma)/sigma:.0f} % error.")

print("\n" + "=" * 76)
print("STEP 6  why sigma alone is not trustworthy")
print("=" * 76)
print("  Walk sigma away from the optimum and refit c each time:")
print("      sigma [1/s]   c [deg/s]    SSE          R^2")
for s in (0.02, 0.05, sigma, 0.15, 0.25):
    sse, P2, Q2 = fit_at(s)
    print(f"      {s:9.4f}   {np.rad2deg(-Q2*s):8.3f}   {sse:.3e}   {1-sse/tot:.4f}")
print("\n  c moves to absorb sigma and R^2 barely changes: the SUM of the two is")
print("  well determined, the SPLIT is not. That is why the per-run sigma spans 8x.")
