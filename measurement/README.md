# Free-swing measurement

Everything needed to reproduce the pendulum identification, from hardware to CSV
to identified parameters.

The measurement itself is deliberately trivial: the motor is held in hard stop
and the host only polls the encoder. Nothing drives the pendulum. You pull it
aside by hand, let go, and record the decay.

---

## 1. What you need

| Item | Notes |
| --- | --- |
| STEVAL-EDUKIT01 rotary inverted pendulum | or any pendulum on the same encoder |
| NUCLEO-F401RE + X-NUCLEO-IHM01A1 (L6474) | connected over USB (ST-Link VCP) |
| Firmware `PendulumController.ino` | must be **uploaded**, not merely compiled |
| Python 3.10+ with `requirements.txt` | pyserial, pandas, numpy, matplotlib |

Firmware and wiring: see [`../PendulumController/`](../PendulumController/) and the
[board setup guide](../docs/arduino_board_setup/README.md).

### Check the firmware is really on the board

```bash
python ../diagnose_manual.py
```

It must report **4 observations**. Two means the original vendor example is still
flashed; the identification pipeline will produce nothing usable from that.

---

## 2. Record

```bash
cd measurement
python log_free_swing.py                 # 60 s at 50 Hz, port auto-detected
python log_free_swing.py --seconds 120   # longer record
python log_free_swing.py --rate 100      # faster polling
```

Procedure:

1. Let the pendulum hang still. The script sets home and clears the safety latch.
2. Start the script.
3. Pull the pendulum aside **by hand** and let go cleanly — do not push it.
4. Let it decay. You can release it again later in the same recording; the
   analysis splits the log into independent decay runs automatically.
5. `Ctrl+C` stops early and keeps whatever was collected.

The CSV lands in `../data/encoder_log_<timestamp>.csv`.

> **The release angle is not something you set.** It is whatever the hand chose,
> read back afterwards from the encoder as the amplitude of the first extremum.
> Using visibly different release angles across runs is a feature: the identified
> natural frequency has to come out the same from all of them, which is the only
> real check that the model fits.

---

## 3. What is in the CSV

| Column | Meaning |
| --- | --- |
| `host_elapsed_s` | host clock since the start of the run — the time base used |
| `host_dt_s` | interval to the previous sample (polling jitter shows up here) |
| `mcu_timestamp_ms` | firmware `millis()`, for cross-checking the host clock |
| `pendulum_angle_deg` | encoder angle, 0..360, **180 = upright**, 0 = hanging |
| `pendulum_error_deg` | error from upright, wrapped to ±180 |
| `pendulum_velocity_dps_est` | host-side difference, wrapped — noisy, for inspection only |
| `rotor_angle_deg` | arm angle (should stay put during a free swing) |
| `motor_speed_pps_observed` | L6474 reported step rate (0 throughout) |
| `l6474_status_raw` | driver status register |
| `response_status`, `terminated` | firmware status / safety latch |

Encoder resolution is 1200 counts/rev = **0.3°/count**, which sets the noise
floor on small swings.

---

## 4. Identify the model

```bash
cd ..
python pendulum_model_id.py                              # all logs in data/
python pendulum_model_id.py --glob "data/encoder_log_2026*.csv"
```

Prints the decay runs, the per-run parameters, the consolidated model, the
linearised control model, and a forward-simulation validation; writes
`pendulum_model_id.png` and `pendulum_model_validation.png`.

Then simulate a controller against the identified plant:

```bash
python -c "import pendulum_sim as ps; print(ps.PendulumParams().summary())"
```

---

## 5. Things that go wrong

| Symptom | Cause |
| --- | --- |
| No reply at all | Firmware not uploaded, or another process holds the port |
| Port busy | A notebook kernel or serial monitor still has it open |
| 2 observations instead of 4 | Old vendor firmware still flashed |
| Port keeps changing | The Nucleo re-enumerates; `find_board_port()` handles it |
| Rotor angle drifts during the run | Something is driving the motor — it should be in hard stop |
| Only one decay run found | The pendulum was re-pushed too gently to register as a release |

---

## 6. Files

- `log_free_swing.py` — the measurement script (this folder)
- `../control_comms.py` — newline-delimited JSON serial interface
- `../pendulum_model_id.py` — identification
- `../pendulum_sim.py` — controller simulator on the identified plant
- `../docs/pendulum_model_identification.md` — method and results
- `../docs/pendulum_ivp_and_simulator.md` — the two parameter forms, IVP, simulator
- `../encoder_logging_60s.ipynb` — the original notebook this script came from
