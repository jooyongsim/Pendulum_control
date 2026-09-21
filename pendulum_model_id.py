"""Identify the pendulum model from the free-swing logs of encoder_logging_60s.ipynb.

The rotor is held stopped during those runs, so the pendulum is an unforced
second-order system and its parameters can be read straight off the decay.

Everything works on the sequence of swing extrema rather than on the raw samples,
which makes it robust to the manual re-pushes in a 60 s log: a push shows up as
an amplitude that jumps back up, and each decaying run is fitted on its own.

What is identified
------------------
natural frequency
    From the peak times, regressed on the peak index. Differencing adjacent
    peaks would fold the ~20 ms polling jitter straight into a ~0.42 s half
    period; regressing on the index averages it out instead. Only peaks below
    SMALL_ANGLE_DEG are used, so the large-angle lengthening
    T(A) ~= T0 (1 + A^2/16) stays out of the estimate.

friction
    Viscous and Coulomb decay differently -- exponential vs linear amplitude
    envelope -- so both are fitted jointly:

        dA/dt = -(sigma*A + c)   =>   A(t) = (A0 + c/sigma) e^(-sigma t) - c/sigma

    Reported alongside the two single-mechanism fits, so the split is visible.

validation
    The identified model is integrated forward from a measured extremum and
    compared with the measurement. Amplitude and phase error are reported
    separately: for an oscillator a pointwise RMS is dominated by accumulated
    phase and says little about whether the damping is right.

Usage
-----
    python pendulum_model_id.py                     # all data/encoder_log_*.csv
    python pendulum_model_id.py --glob "data/encoder_log_2026*.csv"
    python pendulum_model_id.py --no-plots
"""

import argparse
import glob as globmod
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

G = 9.80665
ENC_COUNTS_PER_REV = 1200                 # ENC_STEPS_PER_ROTATION in the firmware
ENC_RES_DEG = 360.0 / ENC_COUNTS_PER_REV

SMALL_ANGLE_DEG = 25.0    # peaks below this define the linear-regime period
PUSH_RATIO = 1.3          # amplitude jump that marks a new manual release
PUSH_MIN_DEG = 3.0        # ... and the size it must exceed to count
MIN_PEAKS = 8             # a run needs this many peaks to be worth fitting
LINEAR_REGIME_DEG = 50.0  # validation is reported separately below this


# --------------------------------------------------------------------------
# signal -> swing extrema
# --------------------------------------------------------------------------
def load_log(path):
    """Return (t, theta) with theta unwrapped about the hanging equilibrium."""
    d = pd.read_csv(path)
    t = d["host_elapsed_s"].to_numpy(float)
    # The firmware reports 0..360 deg with 180 = upright, so hanging is 0/360.
    theta = np.unwrap(np.deg2rad(d["pendulum_angle_deg"].to_numpy(float)))
    return t, theta


def alternating_extrema(y, baseline):
    """Indices of the swing extrema, one (the most extreme) per half swing."""
    candidates = [i for i in range(1, len(y) - 1)
                  if (y[i] - y[i - 1]) * (y[i + 1] - y[i]) <= 0]
    kept = []
    for i in candidates:
        if not kept:
            kept.append(i)
        elif np.sign(y[i] - baseline) == np.sign(y[kept[-1]] - baseline):
            # same side of the equilibrium: keep only the larger excursion
            if abs(y[i] - baseline) > abs(y[kept[-1]] - baseline):
                kept[-1] = i
        else:
            kept.append(i)
    return np.array(kept, int)


def split_runs(t_peak, amplitude):
    """Split the peak sequence wherever a manual push makes amplitude rise."""
    starts = [0]
    for i in range(1, len(amplitude)):
        if (amplitude[i] > PUSH_RATIO * amplitude[i - 1]
                and amplitude[i] > np.deg2rad(PUSH_MIN_DEG)):
            starts.append(i)
    starts.append(len(amplitude))
    return list(zip(starts[:-1], starts[1:]))


# --------------------------------------------------------------------------
# parameter fits
# --------------------------------------------------------------------------
def small_angle_period(t_peak, amplitude):
    """Linear-regime period, from peak time regressed on peak index."""
    small = amplitude < np.deg2rad(SMALL_ANGLE_DEG)
    ts = t_peak[small] if small.sum() >= 6 else t_peak
    return 2.0 * np.polyfit(np.arange(len(ts)), ts, 1)[0]


def envelope_decay(t, A):
    """Fit A(t) = P e^(-s t) + Q, i.e. viscous sigma = s and Coulomb c = -Q s.

    For a fixed s the model is linear in (P, Q), so s is scanned and the 2x2
    normal equations are solved in closed form. This deliberately avoids
    scipy.optimize.curve_fit: its bounded path crashes the scipy build used
    here (Windows fatal exception 0xc06d007f inside LAPACK _compute_lwork).
    """
    n = len(t)
    best = None
    for s in np.linspace(1e-3, 3.0, 4000):
        e = np.exp(-s * t)
        s11, s12 = float(e @ e), float(e.sum())
        b1, b2 = float(e @ A), float(A.sum())
        det = s11 * n - s12 * s12
        if abs(det) < 1e-18:
            continue
        P = (b1 * n - b2 * s12) / det
        Q = (s11 * b2 - s12 * b1) / det
        resid = A - (P * e + Q)
        ss = float(resid @ resid)
        if best is None or ss < best[0]:
            best = (ss, s, P, Q)
    ss, sigma, P, Q = best
    r2 = 1.0 - ss / float(np.sum((A - A.mean()) ** 2))
    return dict(A0=P + Q, sigma=sigma, coulomb=-Q * sigma, r2=r2)


def single_mechanism_fits(t, A):
    """Fit each mechanism on its own, and return the parameters, not just R^2.

        viscous only : A = A0 exp(-sigma t)   -- a straight line in log A
        Coulomb only : A = A0 - rate * t      -- a straight line in A

    Each is fitted where it is linear, so `r2` below is measured on log A for
    the viscous model and on A for the Coulomb one. Those two numbers are NOT
    directly comparable with each other, nor with envelope_decay()'s r2, which
    is measured on A. `r2_on_A` is the same fit scored on A for all of them,
    which is the comparison to quote when the models are put side by side.
    """
    pv = np.polyfit(t, np.log(A), 1)
    r2v = 1 - np.var(np.log(A) - np.polyval(pv, t)) / np.var(np.log(A))
    pl = np.polyfit(t, A, 1)
    r2l = 1 - np.var(A - np.polyval(pl, t)) / np.var(A)

    def r2_on_A(pred):
        return float(1.0 - np.sum((A - pred) ** 2) / np.sum((A - A.mean()) ** 2))

    visc = dict(A0=float(np.exp(pv[1])), sigma=float(-pv[0]), r2=float(r2v))
    coul = dict(A0=float(pl[1]), rate=float(-pl[0]), r2=float(r2l))
    visc["r2_on_A"] = r2_on_A(visc["A0"] * np.exp(-visc["sigma"] * t))
    coul["r2_on_A"] = r2_on_A(np.polyval(pl, t))
    return dict(viscous=visc, coulomb=coul)


def single_mechanism_r2(t, A):
    """R^2 of the viscous-only and Coulomb-only envelopes, for comparison."""
    fits = single_mechanism_fits(t, A)
    return fits["viscous"]["r2"], fits["coulomb"]["r2"]


def analyse_file(path):
    """Return one record per decaying run found in the log."""
    t, theta = load_log(path)
    k = alternating_extrema(theta, np.median(theta))
    if len(k) < MIN_PEAKS:
        return [], []

    # Equilibrium = midpoint of successive opposite extrema. This picks up the
    # static offset Coulomb friction leaves behind (the pendulum stops anywhere
    # inside the stiction deadband, not necessarily at zero).
    equilibrium = np.median(0.5 * (theta[k][:-1] + theta[k][1:]))
    t_peak, amplitude = t[k], np.abs(theta[k] - equilibrium)

    records, traces = [], []
    for a, b in split_runs(t_peak, amplitude):
        tp, ap = t_peak[a:b], amplitude[a:b]
        usable = ap > 2 * np.deg2rad(ENC_RES_DEG)      # above encoder noise
        tp, ap = tp[usable], ap[usable]
        if len(tp) < MIN_PEAKS or ap[-1] >= ap[0]:     # decaying runs only
            continue

        rel = tp - tp[0]
        T0 = small_angle_period(tp, ap)
        w0 = 2 * np.pi / T0
        env = envelope_decay(rel, ap)
        r2v, r2l = single_mechanism_r2(rel, ap)

        records.append(dict(
            file=Path(path).name, t_release=tp[0], n_peaks=len(tp),
            A0_deg=np.rad2deg(ap[0]), A1_deg=np.rad2deg(ap[-1]),
            equilibrium_deg=np.rad2deg(equilibrium),
            T0=T0, f0=1 / T0, w0=w0, L_eff_mm=1000 * G / w0 ** 2,
            sigma=env["sigma"], zeta=env["sigma"] / w0,
            coulomb_dps=np.rad2deg(env["coulomb"]),
            r2_joint=env["r2"], r2_viscous=r2v, r2_coulomb=r2l))
        traces.append(dict(t=rel, A=ap, T0=T0, env=env, r2v=r2v, r2l=r2l,
                           label=f"{Path(path).name}\nrelease {np.rad2deg(ap[0]):.0f} deg"))
    return records, traces


# --------------------------------------------------------------------------
# validation by forward simulation
# --------------------------------------------------------------------------
def simulate(theta0, w0, sigma, coulomb_rad, duration, dt=1e-4):
    """Integrate theta'' = -w0^2 sin(theta) - 2 sigma theta' - F sgn(theta')."""
    T0 = 2 * np.pi / w0
    F = coulomb_rad * w0 ** 2 * T0 / 4.0     # amplitude loss 4F/w0^2 per period
    n = int(duration / dt) + 2
    ts, ys = np.empty(n), np.empty(n)
    th, om = theta0, 0.0
    for i in range(n):
        acc = -w0 ** 2 * np.sin(th) - 2 * sigma * om - F * np.tanh(om / 1e-3)
        om += acc * dt
        th += om * dt
        ts[i], ys[i] = (i + 1) * dt, th
    return ts, ys


def validate(path, w0, sigma, coulomb_dps, window=None):
    """Compare a forward simulation with the measurement, amplitude vs phase."""
    t, theta = load_log(path)
    if window:
        m = (t >= window[0]) & (t <= window[1])
        t, theta = t[m], theta[m]
    theta = theta - 0.5 * (theta.max() + theta.min())

    big = np.deg2rad(20)
    ext = [i for i in range(1, len(theta) - 1)
           if (theta[i] - theta[i - 1]) * (theta[i + 1] - theta[i]) <= 0
           and abs(theta[i]) > big]
    if not ext:
        return None
    i0 = ext[0]                       # start at an extremum: velocity is ~0 there
    rel, meas = t[i0:] - t[i0], theta[i0:]

    st, sy = simulate(meas[0], w0, sigma, np.deg2rad(coulomb_dps), rel[-1])

    def peaks(tt, yy):
        idx = [i for i in range(1, len(yy) - 1)
               if (yy[i] - yy[i - 1]) * (yy[i + 1] - yy[i]) <= 0
               and abs(yy[i]) > np.deg2rad(1)]
        out = []
        for i in idx:
            if out and np.sign(yy[i]) == np.sign(yy[out[-1]]):
                if abs(yy[i]) > abs(yy[out[-1]]):
                    out[-1] = i
            else:
                out.append(i)
        return tt[out], np.abs(yy[out])

    tm, am = peaks(rel, meas)
    ts_, as_ = peaks(st, sy)
    # Compare peak k with peak k: the model drifts in phase, so matching on time
    # would pair peaks from different cycles once the drift exceeds a half period.
    n = min(len(tm), len(ts_))
    if n == 0:
        return None
    dA = np.rad2deg(as_[:n] - am[:n])
    dT = 1000 * (ts_[:n] - tm[:n])
    # Ignore the first swings: the start is not exactly the release point, and
    # that initial-condition mismatch would masquerade as a frequency error.
    lo = min(4, n - 2)
    span = tm[n - 1] - tm[lo]
    # Split by amplitude, not by index: the model is expected to be weakest at
    # large angles, where sin(theta) and the drag are furthest from the fit.
    linear = np.rad2deg(am[:n]) < LINEAR_REGIME_DEG
    return dict(n_peaks=n, A_first=np.rad2deg(am[0]), A_last=np.rad2deg(am[n - 1]),
                dA_rms=float(np.sqrt(np.mean(dA ** 2))), dA_max=float(np.abs(dA).max()),
                n_linear=int(linear.sum()),
                dA_rms_linear=float(np.sqrt(np.mean(dA[linear] ** 2))) if linear.any() else np.nan,
                dA_max_linear=float(np.abs(dA[linear]).max()) if linear.any() else np.nan,
                freq_err_pct=float(100 * (dT[n - 1] - dT[lo]) / 1000 / span) if span > 0 else np.nan,
                rel=rel, meas=meas, sim_t=st, sim_y=sy)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--glob", default="data/encoder_log_*.csv")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=Path("."))
    args = ap.parse_args()

    if args.no_plots:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records, traces = [], []
    for path in sorted(globmod.glob(args.glob)):
        r, tr = analyse_file(path)
        records += r
        traces += tr
    if not records:
        raise SystemExit(f"no usable decay runs in {args.glob}")

    df = pd.DataFrame(records)
    pd.set_option("display.width", 220)
    print("=" * 100)
    print("DECAY RUNS")
    print("=" * 100)
    print(df[["file", "t_release", "n_peaks", "A0_deg", "A1_deg", "equilibrium_deg"]]
          .to_string(index=False, float_format=lambda v: f"{v:9.2f}"))

    print("\n" + "=" * 100)
    print("PER-RUN PARAMETERS")
    print("=" * 100)
    print(df[["A0_deg", "f0", "T0", "w0", "L_eff_mm", "sigma", "zeta", "coulomb_dps",
              "r2_joint", "r2_viscous", "r2_coulomb"]]
          .to_string(index=False, float_format=lambda v: f"{v:9.4f}"))

    wgt = df["n_peaks"] / df["n_peaks"].sum()       # longer runs are better determined
    w0 = float((df.w0 * wgt).sum())
    sigma = float((df.sigma * wgt).sum())
    coulomb = float((df.coulomb_dps * wgt).sum())
    L = G / w0 ** 2

    print("\n" + "=" * 100)
    print("CONSOLIDATED MODEL")
    print("=" * 100)
    print(f"  w0 = {w0:.3f} rad/s    f0 = {w0 / 2 / np.pi:.4f} Hz    T0 = {2 * np.pi / w0:.4f} s"
          f"    (run spread {100 * df.w0.std() / df.w0.mean():.2f} %)")
    print(f"  L_eff = g/w0^2 = {1000 * L:.0f} mm"
          f"   -> uniform rod pivoted at end: {1500 * L:.0f} mm, point mass: {1000 * L:.0f} mm")
    print(f"  viscous sigma = {sigma:.4f} 1/s   zeta = {sigma / w0:.5f}   Q = {w0 / (2 * sigma):.0f}")
    print(f"  Coulomb c     = {coulomb:.3f} deg/s of amplitude")
    print(f"  envelope R2: joint {float((df.r2_joint * wgt).sum()):.4f}   "
          f"viscous-only {float((df.r2_viscous * wgt).sum()):.4f}   "
          f"Coulomb-only {float((df.r2_coulomb * wgt).sum()):.4f}")

    print("\n" + "=" * 100)
    print("LINEARISED CONTROL MODEL      theta measured from the equilibrium")
    print("=" * 100)
    print(f"  hanging   theta'' = -{w0 ** 2:6.2f} theta - {2 * sigma:.3f} theta'"
          f"    poles {-sigma:+.3f} +- j{w0:.3f}   stable, lightly damped")
    print(f"  UPRIGHT   theta'' = +{w0 ** 2:6.2f} theta - {2 * sigma:.3f} theta'"
          f"    poles {w0 - sigma:+.3f}, {-w0 - sigma:+.3f}   UNSTABLE")
    print(f"  unstable pole +{w0 - sigma:.2f} rad/s -> e-fold {1000 / (w0 - sigma):.0f} ms, "
          f"double {1000 * np.log(2) / (w0 - sigma):.0f} ms")
    print(f"  at 50 Hz that is {50 / (w0 - sigma):.1f} samples per e-folding")

    # validation on the largest release available
    biggest = df.loc[df.A0_deg.idxmax()]
    v = validate(str(Path(args.glob).parent / biggest.file), w0, sigma, coulomb,
                 window=(biggest.t_release - 1.0, biggest.t_release + 14.0))
    if v:
        print("\n" + "=" * 100)
        print(f"VALIDATION -- forward simulation vs {biggest.file} ({biggest.A0_deg:.0f} deg release)")
        print("=" * 100)
        print(f"  peaks compared      {v['n_peaks']}  ({v['A_first']:.0f} -> {v['A_last']:.0f} deg)")
        print(f"  amplitude error     RMS {v['dA_rms']:.2f} deg, max {v['dA_max']:.2f} deg  (all peaks)")
        print(f"  amplitude error     RMS {v['dA_rms_linear']:.2f} deg, max {v['dA_max_linear']:.2f} deg"
              f"  ({v['n_linear']} peaks below {LINEAR_REGIME_DEG:.0f} deg)")
        print(f"  frequency error     {v['freq_err_pct']:+.2f} %")

    if not args.no_plots:
        n = len(traces)
        fig, axes = plt.subplots(n, 2, figsize=(12, 3.4 * n), squeeze=False)
        for r, tr in enumerate(traces):
            env, t, A = tr["env"], tr["t"], tr["A"]
            model = (env["A0"] + env["coulomb"] / env["sigma"]) * np.exp(-env["sigma"] * t) \
                - env["coulomb"] / env["sigma"]
            ax = axes[r][0]
            ax.plot(t, np.rad2deg(A), "o", ms=3, label="measured peaks")
            ax.plot(t, np.rad2deg(model), "r-", label=f"viscous+Coulomb R2={env['r2']:.3f}")
            ax.set_xlabel("Time since release (s)"); ax.set_ylabel("Amplitude (deg)")
            ax.set_title(tr["label"]); ax.grid(alpha=.3); ax.legend(fontsize=8)

            ax = axes[r][1]
            ax.semilogy(t, np.rad2deg(A), "o", ms=3)
            ax.semilogy(t, np.rad2deg(model), "r-", label=f"joint R2={env['r2']:.3f}")
            ax.semilogy(t, np.rad2deg(A[0]) * np.exp(-env["sigma"] * t), "g--",
                        label=f"viscous only R2={tr['r2v']:.3f}")
            ax.set_xlabel("Time since release (s)"); ax.set_ylabel("Amplitude (deg, log)")
            ax.set_title("straight line here = pure viscous")
            ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8)
        fig.tight_layout()
        out = args.out_dir / "pendulum_model_id.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"\nfigure -> {out}")

        if v:
            pred = np.interp(v["rel"], v["sim_t"], v["sim_y"])
            fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
            ax[0].plot(v["rel"], np.rad2deg(v["meas"]), lw=.9, label="measured")
            ax[0].plot(v["rel"], np.rad2deg(pred), lw=.9, ls="--", label="identified model")
            ax[0].set_ylabel("Angle (deg)"); ax[0].legend(); ax[0].grid(alpha=.3)
            ax[0].set_title(f"Forward-simulation validation: w0={w0:.3f} rad/s, "
                            f"sigma={sigma:.4f} 1/s, Coulomb={coulomb:.2f} deg/s")
            ax[1].plot(v["rel"], np.rad2deg(pred - v["meas"]), lw=.8, color="r")
            ax[1].set_ylabel("Error (deg)"); ax[1].set_xlabel("Time since release (s)")
            ax[1].grid(alpha=.3)
            fig.tight_layout()
            out = args.out_dir / "pendulum_model_validation.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            print(f"figure -> {out}")


if __name__ == "__main__":
    main()
