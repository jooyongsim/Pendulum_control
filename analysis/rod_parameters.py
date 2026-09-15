"""Re-derive the identified parameters under the uniform-rod assumption.

A rod of mass m and length l, hinged at one end:

    J   = (1/3) m l^2          (moment of inertia about the pivot)
    l_c = l / 2                (centre of mass)
    wn^2 = m g l_c / J = 3g / (2l)      ->  l = 3g / (2 wn^2)
    zeta = c / (2 J wn)                 ->  c = (2/3) zeta wn m l^2

The point-mass reading of the same data gives l = g/wn^2, which is 2/3 of the
rod length -- that ratio is exactly L_eff = 2l/3.
"""
import numpy as np

G = 9.80665
WN = 7.508          # rad/s, consolidated
SIGMA = 0.0798      # 1/s  = zeta * wn
ZETA = SIGMA / WN
RUNS = [("40.7°", 7.505), ("93.5°", 7.514), ("43.7°", 7.502)]

print(f"wn   = {WN:.4f} rad/s      zeta = {ZETA:.6f}")
print(f"2*zeta*wn = c/J = {2 * SIGMA:.4f} 1/s")
print()

L_eff = G / WN ** 2
l_rod = 3 * G / (2 * WN ** 2)
print("point-mass reading :  l = g/wn^2          = %.2f mm" % (1000 * L_eff))
print("uniform-rod reading:  l = 3g/(2 wn^2)     = %.2f mm" % (1000 * l_rod))
print("ratio L_eff / l_rod = %.4f   (expected 2/3 = %.4f)" % (L_eff / l_rod, 2 / 3))
print()

print("per-run rod length")
for name, wn in RUNS:
    print("  %-7s wn=%.3f -> l = %.2f mm" % (name, wn, 1000 * 3 * G / (2 * wn ** 2)))
ls = [3 * G / (2 * wn ** 2) for _, wn in RUNS]
print("  spread: %.2f mm (%.2f %%)" % (1000 * np.std(ls), 100 * np.std(ls) / np.mean(ls)))
print()

# mass-normalised quantities: with the rod assumption only m is left unknown
J_over_m = l_rod ** 2 / 3
c_over_m = (2 / 3) * ZETA * WN * l_rod ** 2
mgl_over_m = G * l_rod / 2
print("mass-normalised (multiply by the measured rod mass m):")
print("  J / m      = l^2/3            = %.6f m^2" % J_over_m)
print("  c / m      = (2/3) zeta wn l^2 = %.6e N m s / (rad kg)" % c_over_m)
print("  m g l_c/m  = g l / 2          = %.6f N m / kg" % mgl_over_m)
print()

print("worked values for a few rod masses")
print("   m [kg]      J [kg m^2]       c [N m s/rad]     m g l/2 [N m]")
for m in (0.030, 0.050, 0.100, 0.300):
    print("   %.3f     %.4e      %.4e        %.4e"
          % (m, m * J_over_m, m * c_over_m, m * mgl_over_m))
print()

# the input gain is unchanged: it depends only on J/(m l_c) = L_eff
b_point = -1 / L_eff
b_rod = -(l_rod / 2) / (l_rod ** 2 / 3)
print("input gain b = -(m l_c)/J:")
print("  point mass : %.4f" % b_point)
print("  uniform rod: %.4f   -> identical, because J/(m l_c) = L_eff either way"
      % b_rod)
print()
print("check wn from the rod length: sqrt(3g/2l) = %.4f rad/s" % np.sqrt(3 * G / (2 * l_rod)))
