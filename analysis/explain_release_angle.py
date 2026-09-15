"""What exactly is the 'release angle' quoted for each run?

It is NOT a commanded setpoint -- nothing sets it. It is a measurement: the
amplitude of the FIRST extremum of the run, i.e. how far the pendulum was
pulled aside before the hand let go, read back from the encoder.
"""
import glob
import sys

import numpy as np

from _common import (ENC_RES_DEG, alternating_extrema, load_log, log_paths, split_runs)  # noqa: E402


for path in sorted(log_paths()):
    t, theta = load_log(path)
    k = alternating_extrema(theta, np.median(theta))
    eq = np.median(0.5 * (theta[k][:-1] + theta[k][1:]))
    tp_all, amp_all = t[k], np.abs(theta[k] - eq)

    print("=" * 76)
    print(path.split("\\")[-1])
    print(f"  equilibrium (rest position) = {np.rad2deg(eq):+.2f} deg from the encoder zero")
    for a, b in split_runs(tp_all, amp_all):
        tp, ap = tp_all[a:b], amp_all[a:b]
        m = ap > 2 * np.deg2rad(ENC_RES_DEG)
        tp, ap = tp[m], ap[m]
        if len(tp) < 8 or ap[-1] >= ap[0]:
            continue
        # raw angle of the first extremum, before subtracting the equilibrium
        i0 = int(np.argmin(np.abs(t - tp[0])))
        raw = np.rad2deg(theta[i0])
        print(f"\n  run released at t = {tp[0]:.2f} s")
        print(f"    first extremum, raw encoder angle      : {raw:+.2f} deg")
        print(f"    minus equilibrium {np.rad2deg(eq):+.2f} deg            "
              f": {np.rad2deg(ap[0]):+.2f} deg   <-- quoted as 'release angle'")
        print(f"    so it is the AMPLITUDE of the first swing, measured from rest,")
        print(f"    not a number anybody commanded.")
        # how quickly does it decay: first few amplitudes
        print(f"    first 5 amplitudes [deg]: "
              f"{np.round(np.rad2deg(ap[:5]), 1).tolist()}")

print("\n" + "=" * 76)
print("The pendulum is pulled aside by hand and let go, so the release angle is")
print("whatever the hand happened to choose. Three different values (40.7, 93.5,")
print("43.7 deg) is exactly what makes the run-to-run agreement meaningful: the")
print("identified wn has to come out the same from all three.")
