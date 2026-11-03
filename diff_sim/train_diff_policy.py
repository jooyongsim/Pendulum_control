"""Train the swing-up policy THROUGH the differentiable world model (no RL), then test it on the rig simulator.

Recipe from tutorial 03 (best variant, 10/10 on the rig simulator):
    policy = energy swing-up prior + autodiff-LQR prior (gated) + NN residual
    BPTT through the differentiable RK4 model, 0.5 s back-propagation windows, 4 s horizon, dt 20 ms
    physical parameters randomised +-10 % per rollout (model uncertainty without a GP), lr 3e-4,
    best iterate on a fixed validation batch.
Wall weight 0.5 instead of 3 (tutorial 04: a heavy wall cost can make 'never swing up' the cheapest policy).
NN residual gated off near the top (nn_gate): without it the residual learned an offset at upright and the
balancer parked the arm at -60 deg (the 4 s training rollouts barely see steady balancing).

    python train_diff_policy.py                 # ~6 min on a CPU -> policy.pt
    python train_diff_policy.py --quick         # 30 iterations, just to see it run
    python train_diff_policy.py --hand          # hand-identified parameters instead of tutorial 02's
"""
import math
import sys
import time

import numpy as np
import torch

from diff_controller import HERE, WORLD_MODEL_CANDIDATES, dw

QUICK = "--quick" in sys.argv
ITERS = 30 if QUICK else 300
torch.set_num_threads(1)
torch.manual_seed(0)

# ---------------------------------------------------------------- 1. world model
model = dw.PhysicsModel()
src = next((p for p in WORLD_MODEL_CANDIDATES if p.exists()), None)
if src is not None and "--hand" not in sys.argv:
    model.load_state_dict(torch.load(src)["physics"])
    print("world model: identified from the real logs (tutorial 02)", model.params())
else:
    print("world model: hand-identified parameters", model.params())
for p in model.parameters():
    p.requires_grad_(False)

# ---------------------------------------------------------------- 2. policy with physics priors
K = dw.lqr_gain(model)                          # LQR from the model's own autodiff linearisation: zero rollouts
print("autodiff LQR K =", np.round(K, 2))
policy = dw.Policy(seed=0, swing=True, K=K, nn_gate=True)   # NN off near the top: upright + arm 0 stays an equilibrium

# ---------------------------------------------------------------- 3. BPTT through the model
t0 = time.time()
hist = dw.train_policy_bptt(model, policy, H=200, iters=ITERS, batch=64, lr=3e-4, trunc=25, dt=0.02,
                            randomize=0.1, w_energy=0.0, w_arm=0.3, w_wall=0.5, x0_spread=0.3,
                            verbose=True, log_every=max(1, ITERS // 10), gen_seed=0)
print(f"trained in {(time.time() - t0) / 60:.1f} min, swing-up gains {policy.gains()}")
torch.save(policy.state_dict(), HERE / "policy.pt")
print("saved", HERE / "policy.pt")

# ---------------------------------------------------------------- 4. rig simulator (not differentiable, not the model)
res = [dw.score(dw.run_real(policy.numpy_fn(), seconds=8.0, seed=s)) for s in range(10)]
ups = [r["first_upright"] for r in res if r["success"]]
print(f"rig simulator: {sum(r['success'] for r in res)}/10 balanced after 8 s"
      + (f", upright after {np.mean(ups):.2f} s on average" if ups else ""))
