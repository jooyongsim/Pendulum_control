# daeun — swing-up and balance, run on the board

Swing-up from hanging and 4-state balancing for the STEVAL-EDUKIT01, with the
control loop **on the MCU at 500 Hz**. The host only starts it, tunes gains and logs.

Why on the board: the upright error doubles every ~94 ms, and one serial round
trip over the ST-Link VCP costs 3–15 ms. A host loop measured 66 Hz and lost the
pendulum right after catching it; moving the loop onto the MCU fixed that.

## Files

| File | What it is |
| --- | --- |
| `SwingUpController/SwingUpController.ino` | firmware: state estimate, energy pumping, 4-state balance, safety |
| `swingup_run.py` | start / tune / log one run |
| `swingup.py` | the same control law in simulation, on the identified plant |
| `plant_local.py` | re-identifies the plant from `../data/encoder_log_*.csv` |

## Run

```bash
# firmware: upload SwingUpController.ino  (Upload method: STM32CubeProgrammer (SWD))
python daeun/swingup_run.py --port COM40 --duration 120 --calibrate
python daeun/swingup_run.py --port COM40 --mode balance --calibrate   # hand it the pendulum
```

Start with the rod **hanging still**: the script pins the reference to it, then
measures the true vertical before balancing.

## Control

Input is the pivot acceleration `u` [m/s²]; the arm converts it and the firmware
integrates it into the step rate the L6474 accepts.

```
swing-up   E = ½θ̇² + ωₙ²cos θ                      (upright +ωₙ², hanging −ωₙ²)
           u = pump·(E − E_up)·sign(θ̇·cos θ) − k_rot·φ − k_rot_d·φ̇
balance    u = −(k₁θ + k₂θ̇ + k₃(φ − φ_ref) + k₄φ̇)
both       α̈ = u / r ;  v += α̈·dt ;  pps = SIGN·v·3200/2π
catch      |θ| < 8°, |θ̇| < 90°/s, |φ| < 60°   ·   release beyond |θ| > 25°
```

Gains come from pole placement on the identified plant (ωₙ 7.521 rad/s, σ 0.145,
L_eff 173 mm, arm 120 mm): `k = (−25.37, −3.08, −0.397, −0.459)` for ω_des 6,
ζ 0.8 and rotor poles −2, −2.6.

## Three things that cost a day, kept here so they are not repeated

**The upright reference is not the hanging position.** Dry friction lets the rod
rest anywhere in its stiction band, and the balancer holds the arm ~64° away per
degree of error (k₁/|k₃|). The arm then walks into its limit and the catch is
lost. Two fixes are in: `--calibrate` gives the hanging rod a zero-mean doublet
and takes the centre of the free swing as the true vertical, and while balancing
a slow adaptation corrects the rest — subtracting, not adding: a positive arm
offset means the assumed vertical is too large.

**The encoder count is not the rod angle.** It counts from power-up, so a rod
that went over the top in an earlier run leaves the zero half a turn out and the
board tries to balance at the bottom. `CMD_SET_HANG_REF` pins the reference to
the rod as it hangs now, and the host sends it at every start.

**A sudden loss of performance is usually mechanical.** Swing-up stopped working
for two hours; gain tuning did nothing. A 40 s free-swing measurement showed σ
had gone from 0.145 to 0.855 1/s (ζ 0.019 → 0.114) while ωₙ was unchanged — the
encoder body had worked loose and was dragging. Measure the decay before touching
the controller:

```bash
python daeun/swingup_run.py --port COM40 --mode idle --duration 40   # release the rod by hand
```

## Results on this rig

| | |
| --- | --- |
| balance | 0.26–0.52° rms, 0 % of the command saturated, arm within ±30° |
| longest hold | 107 s (limited by the run length, not by a fall) |
| swing-up | 4.5–13 s to the first catch, with occasional missed catches |
| disturbance | recovered from a 24° knock; re-swings by itself after a fall |

Open: the swing-up time varies a lot because the rod arrives at the top too fast
to catch. The pumping saturates all the way up (the energy error is ~−113 at the
bottom), so the arrival speed is not controlled. `--ke` enables an energy-scaled
push that eases off near the top; it is off by default and needs repeated runs to
tune, since single attempts vary too much to compare.
