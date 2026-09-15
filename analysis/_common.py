"""Shared helpers for the analysis scripts.

Keeps the data loading and extremum bookkeeping in one place so the explanation
scripts and pendulum_model_id.py cannot drift apart.
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT)

from pendulum_model_id import (ENC_RES_DEG, G, SMALL_ANGLE_DEG,  # noqa: E402
                               alternating_extrema, envelope_decay, load_log,
                               single_mechanism_r2, split_runs)

DATA_DIR = os.path.join(PROJECT, "data")


def log_paths(pattern="encoder_log_*.csv", data_dir=None):
    """Find logs. glob.escape matters: the project path contains [brackets],
    which glob would otherwise read as a character class and match nothing."""
    d = data_dir or DATA_DIR
    return sorted(glob.glob(os.path.join(glob.escape(d), pattern)))


def extrema_of(path):
    """Return (t, theta, idx, equilibrium) for one log.

    idx indexes BOTH maxima and minima, alternating about the equilibrium, so
    consecutive entries are half a period apart -- that is what makes
    T = 2 * (t_N - t_1)/(N-1) correct.
    """
    t, theta = load_log(path)
    idx = alternating_extrema(theta, np.median(theta))
    equilibrium = np.median(0.5 * (theta[idx][:-1] + theta[idx][1:]))
    return t, theta, idx, equilibrium


def decay_runs(path, min_extrema=8):
    """Yield (t_rel, amplitude, t_abs) for each decaying run in a log."""
    t, theta, idx, eq = extrema_of(path)
    t_ext, amp = t[idx], np.abs(theta[idx] - eq)
    for a, b in split_runs(t_ext, amp):
        te, ae = t_ext[a:b], amp[a:b]
        usable = ae > 2 * np.deg2rad(ENC_RES_DEG)
        te, ae = te[usable], ae[usable]
        if len(te) >= min_extrema and ae[-1] < ae[0]:
            yield te - te[0], ae, te


def longest_run(paths=None):
    """The run with the most extrema -- the 93.5 deg release in these logs."""
    best = None
    for p in (paths or log_paths()):
        for t_rel, amp, t_abs in decay_runs(p):
            if best is None or len(t_rel) > len(best[0]):
                best = (t_rel, amp, t_abs, p)
    if best is None:
        raise SystemExit("no decay runs found in " + DATA_DIR)
    return best
