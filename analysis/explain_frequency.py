"""Show exactly how sigma, omega_d and omega_n were obtained from the logs.

Three questions:
  1. what does the peak sequence actually measure -- omega_n or omega_d?
  2. is the period just (elapsed time) / (number of swings)?
  3. where does sigma come from, and how well is it determined?
"""
import glob
import sys

import numpy as np
import pandas as pd

from _common import (ENC_RES_DEG, G, alternating_extrema, envelope_decay, load_log, log_paths, split_runs)  # noqa: E402


rows = []
# the project path contains [brackets], which glob reads as a character class
for path in sorted(log_paths()):
    t, theta = load_log(path)
    k = alternating_extrema(theta, np.median(theta))
    eq = np.median(0.5 * (theta[k][:-1] + theta[k][1:]))
    tp_all, amp_all = t[k], np.abs(theta[k] - eq)
    for a, b in split_runs(tp_all, amp_all):
        tp, ap = tp_all[a:b], amp_all[a:b]
        m = ap > 2 * np.deg2rad(ENC_RES_DEG)
        tp, ap = tp[m], ap[m]
        if len(tp) < 8 or ap[-1] >= ap[0]:
            continue
        small = ap < np.deg2rad(25)
        ts = tp[small] if small.sum() >= 6 else tp
        n = len(ts)

        # (a) the naive count: peaks are HALF periods apart
        T_count = 2 * (ts[-1] - ts[0]) / (n - 1)
        # (b) least squares on peak time vs peak index
        T_fit = 2 * np.polyfit(np.arange(n), ts, 1)[0]

        env = envelope_decay(tp - tp[0], ap)
        rows.append(dict(release=np.rad2deg(ap[0]), n_small=n,
                         span=ts[-1] - ts[0], half_swings=n - 1,
                         T_count=T_count, T_fit=T_fit,
                         sigma=env["sigma"], n_peaks=len(tp)))

df = pd.DataFrame(rows)
pd.set_option("display.width", 200)

print("=" * 88)
print("1. THE PERIOD: is it just elapsed time / number of swings?")
print("=" * 88)
print("   Peaks alternate max, min, max, ... so consecutive peaks are HALF a period apart.")
print("   naive : T = 2 * (t_last - t_first) / (n_peaks - 1)")
print("   used  : T = 2 * slope of a least-squares fit of peak TIME on peak INDEX\n")
print(df[["release", "n_small", "span", "half_swings", "T_count", "T_fit"]].to_string(
    index=False, float_format=lambda v: f"{v:9.4f}"))
diff = 100 * (df.T_fit - df.T_count).abs() / df.T_count
print(f"\n   the two agree to {diff.max():.3f} % -- the fit only removes polling jitter")

T = float((df.T_fit * df.n_peaks).sum() / df.n_peaks.sum())
w_meas = 2 * np.pi / T
print(f"\n   weighted period  T = {T:.5f} s   ->   2*pi/T = {w_meas:.4f} rad/s")

print("\n" + "=" * 88)
print("2. WHAT THAT FREQUENCY IS: omega_d, not omega_n")
print("=" * 88)
sigma = float((df.sigma * df.n_peaks).sum() / df.n_peaks.sum())
print("   A decaying oscillation crosses its peaks at the DAMPED frequency, so")
print("   the measured 2*pi/T is omega_d. The undamped omega_n is then")
print("       omega_n = omega_d / sqrt(1 - zeta^2),   zeta = sigma / omega_n")
print("   which is implicit in zeta; solve it directly:")
print("       omega_n = sqrt(omega_d^2 + sigma^2)")
wn = np.sqrt(w_meas ** 2 + sigma ** 2)
zeta = sigma / wn
wd_back = wn * np.sqrt(1 - zeta ** 2)
print(f"\n   sigma    = {sigma:.4f} 1/s      (from the envelope fit, section 3)")
print(f"   omega_d  = {w_meas:.4f} rad/s   (measured directly)")
print(f"   omega_n  = {wn:.4f} rad/s   = sqrt(omega_d^2 + sigma^2)")
print(f"   zeta     = {zeta:.6f}")
print(f"   check    : omega_n*sqrt(1-zeta^2) = {wd_back:.4f} rad/s  (= omega_d)")
print(f"\n   difference omega_n - omega_d = {wn - w_meas:.6f} rad/s "
      f"({100 * (wn - w_meas) / w_meas:.4f} %)")
print("   The run-to-run spread is 0.14 %, i.e. ~300x larger, so the distinction")
print("   does not change any number on the slides -- but the labels should be right.")

print("\n" + "=" * 88)
print("3. SIGMA: from the envelope fit, not from a log decrement")
print("=" * 88)
print("   A(t) = (A0 + c/sigma) exp(-sigma t) - c/sigma     [viscous + Coulomb]")
print("   For a fixed sigma this is linear in the other two, so sigma is scanned")
print("   and the 2x2 normal equations are solved in closed form.\n")
print(df[["release", "n_peaks", "sigma"]].to_string(index=False,
                                                    float_format=lambda v: f"{v:9.4f}"))
print(f"\n   weighted mean sigma = {sigma:.4f} 1/s")
print(f"   BUT the run-to-run spread is {df.sigma.min():.4f} .. {df.sigma.max():.4f} 1/s "
      f"({df.sigma.max() / df.sigma.min():.1f}x)")
print("   sigma and the Coulomb term trade off against each other in the fit, so")
print("   the TOTAL decay is well determined while the split is not.")

print("\n" + "=" * 88)
print("4. TERMINOLOGY")
print("=" * 88)
for sym, eng, kor, unit in [
    ("sigma = zeta*omega_n", "decay rate / decay constant", "감쇠율 (감쇠 상수)", "1/s"),
    ("zeta", "damping ratio", "감쇠비", "dimensionless"),
    ("c", "damping coefficient", "감쇠 계수", "N m s / rad"),
    ("omega_n", "(undamped) natural frequency", "고유 진동수", "rad/s"),
    ("omega_d", "damped natural frequency", "감쇠 고유 진동수", "rad/s"),
    ("tau = 1/sigma", "time constant", "시상수", "s"),
]:
    print(f"   {sym:22s} {eng:30s} {kor:18s} [{unit}]")
print(f"\n   tau = 1/sigma = {1 / sigma:.1f} s")
