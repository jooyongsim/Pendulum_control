"""Identified plant for the swing-up scripts, read from the free-swing logs.

Keeps this folder self-contained: it re-identifies from data/encoder_log_*.csv with
pendulum_model_id, so a new measurement changes the controller design with no edits.
"""

import glob
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pendulum_sim as ps                      # noqa: E402
from pendulum_model_id import analyse_file     # noqa: E402

DEFAULT_GLOB = str(ROOT / "data" / "encoder_log_*.csv")

# Measured on this rig (step_response, 2026-09-22): the driver reaches 104 % of the
# configured 10000 pps^2, tracks the commanded rate, and a POSITIVE pps command tips
# the pendulum the other way -- hence the sign flip in the firmware.
MEASURED = {"accel_pps2": 10411.0, "pivot_accel": 2.45, "delay_ms": 11.0,
            "coupling_sign": -1.0, "min_speed_pps": 300.0}

PPS2_MAX, PPS_MAX, PULSES_PER_REV = 10000.0, 4000.0, 200 * 16


def rotor_limits(arm_radius_m):
    alpha_ddot_max = PPS2_MAX * 2 * np.pi / PULSES_PER_REV
    alpha_dot_max = PPS_MAX * 2 * np.pi / PULSES_PER_REV
    return {"alpha_ddot_max": alpha_ddot_max, "alpha_dot_max": alpha_dot_max,
            "accel_max": arm_radius_m * alpha_ddot_max,
            "vel_max": arm_radius_m * alpha_dot_max}


def identify(pattern=DEFAULT_GLOB, arm_radius_m=0.120, quiet=False):
    """Weighted consolidation of every decay run found, as PendulumParams."""
    records = []
    for path in sorted(glob.glob(pattern)):
        try:
            r, _ = analyse_file(path)
            records += r
        except Exception as exc:
            if not quiet:
                print(f"  [skip] {Path(path).name}: {exc}")
    if not records:
        if not quiet:
            print("  [warn] no decay runs found -> pendulum_sim defaults")
        return ps.PendulumParams(arm_radius_m=arm_radius_m)
    w = np.array([r["n_peaks"] for r in records], float)
    w /= w.sum()
    avg = lambda key: float(np.sum(w * np.array([r[key] for r in records], float)))
    wn = avg("w0")
    p = ps.PendulumParams(wn=wn, zeta=avg("sigma") / wn, coulomb_dps=avg("coulomb_dps"),
                          arm_radius_m=arm_radius_m)
    if not quiet:
        print(f"identified from {len(records)} decay run(s)")
        print("  " + p.summary().replace("
", "
  "))
    return p


def add_plant_args(ap):
    ap.add_argument("--glob", default=DEFAULT_GLOB, help="free-swing logs to identify from")
    ap.add_argument("--arm", type=float, default=0.120, help="rotor arm radius [m]")
    return ap
