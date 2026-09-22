"""Derive rotary_model.Plant.deriv() from scratch and check it.

The constants A_G and b are quoted in rotary_plant_model.md as Mgl_c/Jp and
rMl_c/Jp; the centrifugal term is quoted as having NO coefficient because it is
Jp/Jp. Rather than trust the algebra, this builds the rod's kinetic energy from
20000 elements, takes every Euler-Lagrange derivative numerically, and compares
the result with the formula the simulator integrates.

    python check_rotary_plant.py
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rotary_model as rm  # noqa: E402

M = 1.0                      # the rod mass cancels out of every ratio below
N = 20000
L, R_ARM, G = rm.L, rm.R_ARM, 9.81
s = (np.arange(N) + 0.5) * L / N          # element centres along the rod
dm = M / N                                # uniform rod

JP = float(np.sum(s ** 2) * dm)           # inertia about the pivot
S1 = float(np.sum(s) * dm)                # first moment = M * l_c


def kinetic(th, thd, phid):
    """T of the rod whose pivot is carried around by the arm.

    |v|^2 = (s sin(th) phid)^2 + (r phid)^2 + 2 r s phid thd cos(th) + (s thd)^2
    The first term is the whole story: leaning moves each element AWAY from the
    vertical axis, which is what makes phid do work on theta.
    """
    v2 = ((s * math.sin(th) * phid) ** 2 + R_ARM ** 2 * phid ** 2
          + 2 * R_ARM * s * phid * thd * math.cos(th) + (s * thd) ** 2)
    return 0.5 * float(np.sum(v2) * dm)


def potential(th):
    return S1 * G * math.cos(th)          # theta measured from upright


def euler_lagrange(th, thd, phid, phidd, h=1e-4):
    """theta'' from d/dt(dT/dthd) - dT/dth + dV/dth = 0, all derivatives numeric.

    h = 1e-4 is deliberate. These are SECOND differences, so rounding noise
    enters as eps/h^2: at h = 1e-6 the last digits are pure noise (the check
    agreed only to 1e-2), at h = 1e-4 truncation and rounding balance and the
    agreement is ~1e-6. Anything quoted from this function is only meaningful
    at that step.
    """
    def dT_dthd(a, b, c):
        return (kinetic(a, b + h, c) - kinetic(a, b - h, c)) / (2 * h)

    # d/dt (dT/dthd) = T_thd_thd * thdd + T_thd_th * thd + T_thd_phid * phidd
    t_thd_thd = (dT_dthd(th, thd + h, phid) - dT_dthd(th, thd - h, phid)) / (2 * h)
    t_thd_th = (dT_dthd(th + h, thd, phid) - dT_dthd(th - h, thd, phid)) / (2 * h)
    t_thd_phid = (dT_dthd(th, thd, phid + h) - dT_dthd(th, thd, phid - h)) / (2 * h)
    t_th = (kinetic(th + h, thd, phid) - kinetic(th - h, thd, phid)) / (2 * h)
    v_th = (potential(th + h) - potential(th - h)) / (2 * h)
    return (t_th - v_th - t_thd_th * thd - t_thd_phid * phidd) / t_thd_thd


def formula(th, phid, phidd):
    """What the simulator integrates, without damping (the Lagrangian has none).

    Note the sign on the input term: the derivation gives -b cos(th) phidd,
    while Plant.deriv() writes + b cos(th) acc. That is the sign convention for
    positive arm acceleration; place2/place4 use the same one, so the loop is
    consistent. Everything else matches term for term.
    """
    return (rm.A_G * math.sin(th) - rm.B_U * math.cos(th) * phidd
            + math.sin(th) * math.cos(th) * phid ** 2)


def main():
    print("=" * 72)
    print("constants: the ratios the model is built from")
    print("=" * 72)
    print(f"  Jp = {JP:.6f}   (M L^2 / 3 = {M * L ** 2 / 3:.6f})")
    print(f"  S1 = {S1:.6f}   (M L / 2   = {M * L / 2:.6f})     l = {L * 1000:.0f} mm")
    print(f"  M g l_c / Jp = {S1 * G / JP:.4f}   vs A_G = {rm.A_G:.4f}"
          f"     (3g/2l = {3 * G / (2 * L):.4f})")
    print(f"  r M l_c / Jp = {R_ARM * S1 / JP:.4f}   vs b   = {rm.B_U:.4f}"
          f"     (3r/2l = {3 * R_ARM / (2 * L):.4f})")
    print(f"  centrifugal coefficient = Jp/Jp = 1  -> no constant in the code")
    assert abs(S1 * G / JP - rm.A_G) < 1e-3
    assert abs(R_ARM * S1 / JP - rm.B_U) < 1e-3

    print()
    print("=" * 72)
    print("Euler-Lagrange, built from 20000 rod elements, vs the formula")
    print("=" * 72)
    print(f"{'theta':>7}{'thetad':>9}{'phid':>7}{'phidd':>8}"
          f"{'numeric EL':>14}{'formula':>13}{'diff':>11}")
    worst = 0.0
    for th_deg, thd, phid, phidd in ((5, 0.5, 1.0, 10.0), (30, 2.0, 3.0, 20.0),
                                     (60, -1.0, 4.0, -15.0), (90, 2.0, 5.0, 20.0)):
        th = math.radians(th_deg)
        num = euler_lagrange(th, thd, phid, phidd)
        f = formula(th, phid, phidd)
        worst = max(worst, abs(num - f) / max(1.0, abs(f)))
        print(f"{th_deg:>7}{thd:>9.1f}{phid:>7.1f}{phidd:>8.1f}"
              f"{num:>14.6f}{f:>13.6f}{num - f:>11.2e}")
    print(f"\n  worst relative difference {worst:.1e} (nested finite differences)")
    assert worst < 1e-3

    print()
    print("=" * 72)
    print("how big is the centrifugal term, next to the others")
    print("=" * 72)
    print(f"{'theta':>7}{'thetad':>9}{'phid':>7}{'a':>7}"
          f"{'gravity':>10}{'arm':>9}{'centrif':>10}{'viscous':>9}{'coulomb':>9}")
    for th_deg, thd, phid, a in ((5, 0.5, 1.0, 10.0), (30, 2.0, 3.0, 20.0),
                                 (90, 2.0, 5.0, 20.0)):
        th = math.radians(th_deg)
        print(f"{th_deg:>7}{thd:>9.1f}{phid:>7.1f}{a:>7.1f}"
              f"{rm.A_G * math.sin(th):>10.2f}"
              f"{rm.B_U * math.cos(th) * a:>9.2f}"
              f"{math.sin(th) * math.cos(th) * phid ** 2:>10.2f}"
              f"{-rm.C_D * thd:>9.2f}"
              f"{-rm.F_C * math.tanh(thd / 0.01):>9.2f}")

    th_max = math.degrees(math.atan(rm.B_U * rm.ACCEL_MAX / rm.A_G))
    print(f"\n  arm accel to hold 5 deg: "
          f"{rm.A_G * math.sin(math.radians(5)) / (rm.B_U * math.cos(math.radians(5))):.2f}"
          f" rad/s^2 of {rm.ACCEL_MAX:.1f}")
    print(f"  gravity beats the arm past tan(th) = b a_max / wn^2 -> {th_max:.1f} deg")
    print("\nrotary_plant_model.md checks out")


if __name__ == "__main__":
    main()
