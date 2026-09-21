# Analysis

Visualisation and explanation scripts for the free-swing identification. The
identification itself lives one level up in `../pendulum_model_id.py`; these
scripts open up the steps inside it so each one can be checked on its own.

Every script imports `_common.py`, which re-exports the real functions from
`pendulum_model_id.py`. Nothing here re-implements the maths, so the explanation
cannot drift away from what actually produced the numbers.

---

## The pipeline

```
measurement/log_free_swing.py     hardware  ->  data/encoder_log_*.csv
        |
        v
analysis/plot_extrema_detection.py        did the extremum detection work?
        |
        v
../pendulum_model_id.py           CSV  ->  wn, zeta, sigma, rod length, figures
        |
        v
../pendulum_sim.py                identified plant  ->  controller simulation
```

---

## Scripts

| Script | Answers |
| --- | --- |
| `plot_extrema_detection.py` | Were the maxima and minima found correctly, and where were the releases? |
| `plot_extrema_all.py` | The same check, drawn exhaustively: every log, every decay run, with the viscous-only and viscous+Coulomb envelope fits overlaid |
| `damping_mechanisms.py` | Why is viscous decay exponential and Coulomb decay linear? |
| `explain_envelope_fit.py` | How is the envelope actually fitted, step by step? |
| `explain_frequency.py` | Where do the period, ω_d, ωₙ and ζ come from? |
| `explain_release_angle.py` | What is the "release angle" — a setting or a measurement? |
| `rod_parameters.py` | What follows from J = ⅓ m ℓ² ? |
| `realtime_sim.py` | **Interactive**: run the identified plant live and poke at the controller |
| `design_pid.py` | Design a PID from the identified coefficients, and does it survive? |

```bash
cd analysis
python plot_extrema_detection.py --no-show     # writes extrema_detection.png
python plot_extrema_all.py                     # save + open 9 windows
python plot_extrema_all.py --no-show --out-dir figs_extrema   # save only
python plot_extrema_all.py --select viscous    # viscous-only model, its own page
python plot_extrema_all.py --select viscous,joint --no-save   # one page per model
python damping_mechanisms.py --no-show         # writes damping_mechanisms.png
python explain_envelope_fit.py                 # prints the fit, step by step
python explain_frequency.py
python explain_release_angle.py
python rod_parameters.py
python realtime_sim.py                         # interactive window
python design_pid.py --no-show                 # writes design_pid.png
```

`realtime_sim.py` runs the plant against the wall clock and draws it live:
space pauses, `c` turns the controller off (watch it fall), `1/2/3` switch
design, arrows disturb it, `d` cycles the damping model, `v` swaps an exact
velocity for one differenced from the quantised angle.

`--no-show` saves the figure without opening a window; `plot_extrema_all.py`
also takes `--select full,runs,summary` to draw a subset and `--no-save` to
open the windows without writing anything. `plot_extrema_detection.py`
also takes `--zoom T0 T1` and `--glob`.

---

## 1. Extremum detection

`plot_extrema_detection.py` draws the raw trace with **maxima as red triangles
and minima as blue triangles**, marks the releases, and zooms into the largest
swing with the measured half period annotated.

This is the step worth inspecting by eye. Everything after it — period, envelope,
σ, ωₙ, the rod length — is computed from this sequence, and if the detection is
wrong nothing downstream will say so.

Two things the figure is there to confirm:

- the extrema **alternate** in sign about the rest position, which is what makes
  adjacent gaps a *half* period rather than a full one;
- the **rest position is not zero** (−1.65° in one log, +2.25° in the other).
  Dry friction lets the pendulum stop anywhere inside its stiction band, so the
  equilibrium is measured from the data rather than assumed.

---

## 2. Viscous vs Coulomb damping

Both remove energy, but not in the same proportion, and the difference shows up
in the shape of the envelope.

**Viscous** (air, oil, eddy currents): the torque is proportional to speed,
`tau = -c*theta_dot`. A big swing is also a fast swing, so it loses
proportionally more. That gives

```
dA/dt = -sigma*A        ->      A(t) = A0 * exp(-sigma*t)
```

Each cycle keeps the same **fraction** of the previous amplitude: a straight line
on a log axis, and the motion never quite stops.

**Coulomb** (dry / bearing friction): the torque has constant magnitude and only
changes sign, `tau = -F*sign(theta_dot)`. It costs the same whether the swing is
big or small. Over a quarter cycle the pendulum sweeps `A`, so the work is `F*A`
and a full cycle costs `4*F*A`. With `E = 1/2 k A^2`:

```
k*A*dA = -4*F*A per cycle    ->    dA = -4F/k per cycle, independent of A
```

Each cycle loses the same **absolute** amount: a straight line on a *linear*
axis, reaching zero in finite time — the pendulum stops dead.

Together:

```
dA/dt = -(sigma*A + c_coulomb)
```

### What this pendulum does

| Model | Envelope shape | R² |
| --- | --- | --- |
| Viscous only | exponential | 0.943 (0.928 scored on A) |
| Coulomb only | linear | 0.960 |
| **Viscous + Coulomb** | mixed | **0.997** |

Each model is fitted where it is linear, so the viscous R² above is measured on
log A while the other two are measured on A. Scored on A like the others, the
viscous-only fit gives 0.928 — the ranking is unchanged, but only the second
column is a like-for-like comparison. `single_mechanism_fits()` returns both.

Neither alone fits. The cleanest single indicator is the ratio between
successive same-sign extrema: pure viscous damping holds it **constant**, and
here it falls from 0.88 to 0.55 as the swing dies out — the fingerprint of a
fixed absolute loss per cycle.

Two practical consequences:

- **The classical log-decrement method is the viscous-only fit.** Applied to this
  data it returns σ = 0.2154 instead of 0.1187 — an 81 % error, because it is
  forced to explain Coulomb losses with an exponential.
- **The pendulum rests off-centre.** It stopped at 2.4°, not 0°, which a purely
  viscous system could never do.

---

## 3. A caveat worth repeating

σ and the Coulomb term trade off against each other in the fit. Forcing σ
anywhere between 0.02 and 0.25 1/s and refitting the Coulomb term keeps R² above
0.96. The **total** decay is well determined; the **split** is not, which is why
the per-run σ spans a factor of 8. Quote the envelope, not σ on its own.

`design_pid.py` closes the loop with the identified numbers: closed-form PID
gains, the bandwidth the ±6 m/s² limit allows, what the integrator is and is
not for, and the cascade that keeps the rotor from walking away. Written up in
`../docs/pendulum_pid_design.md`.

See `../docs/pendulum_model_identification.md` for the full method and results.
