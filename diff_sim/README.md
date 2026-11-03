# diff_sim — differentiable-model policy on the rig simulator

The swing-up controller learned **through a differentiable world model** (BPTT through a PyTorch RK4 model of the
pendulum, no RL), run on the real-time rig simulator `../simulator/rotary_model.py` (1 kHz nonlinear plant,
firmware motor limits, 100 Hz loop, 4 ms delay, 0.3° encoder). Same window as
`../simulator/rotary_swingup_realtime.py`; only the controller is different, and `p` switches to the rig's
hand-designed controller for comparison.

Standalone copy of `08_Claude/02_differentiable_world_policy/diff_sim` (the lecture package, tutorials 01–05).

| File | |
|---|---|
| `diff_swingup_realtime.py` | real-time window (`p` switches controller), `--check` robustness table, `--plot` → `swingup_compare.png` |
| `diff_controller.py` | wraps a trained policy as a rig controller (`controller(z, t)`) |
| `train_diff_policy.py` | trains the policy through the differentiable model, ~8 min CPU → `policy.pt` |
| `dwm_core.py` | differentiable physics model, policy, BPTT training (copy from the lecture package) |
| `world_models.pt` | physical parameters identified from the real logs (tutorial 02), used by the training |
| `policy.pt` | trained policy (energy swing-up + LQR priors, gated NN residual) — used by default |
| `policy_03_colab.pt` | tutorial 03 policy (Colab run), used when `policy.pt` is missing |

```powershell
conda activate control2025              # torch, numpy, matplotlib
cd -LiteralPath "...\Pendulum_control\diff_sim"
python diff_swingup_realtime.py         # window
python diff_swingup_realtime.py --check
python diff_swingup_realtime.py --plot
python train_diff_policy.py             # optional: retrain -> policy.pt
```

Keys (click the window first): `p` switch controller · `↑` knock over · `←`/`→` kick · `x` realistic/ideal sensor ·
`o` encoder +1° · `c` controller on/off · `r` reset · space pause.

**The policy** is one smooth law, not a state machine:
u = U_max·tanh(((1−g)·u_swing + g·u_LQR)/U_max + (1−g)·NN(x)), g = ((1+cos θ)/2)¹⁰.
u_swing is the rig's energy-pumping law (its 4 gains are trained), u_LQR comes from autodiff linearisation of the
model (zero rollouts), and the NN residual is trained by BPTT with ±10 % parameter randomisation. The NN is switched
off near the top so that upright with the arm at 0 stays the equilibrium.
The “SWING / BALANCE” label uses the rig controller's catch (12°) and fall (30°) thresholds, for display only.
