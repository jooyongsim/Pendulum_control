"""A policy trained THROUGH the differentiable world model, wrapped as a controller for the rig simulator.

The rig simulator (Pendulum_control/simulator/rotary_model.py: 1 kHz plant, firmware motor limits, 100 Hz loop,
4 ms delay, quantised encoder) calls `controller(z, t)` with z = [theta, theta', phi, phi'] (theta = 0 upright),
which is exactly the state convention of dwm_core, so the trained policy plugs in unchanged.

The policy is ONE smooth law (no state machine):
    u = U_max tanh( ((1-g) u_swing + g u_LQR) / U_max + NN(x) ),   g = ((1 + cos theta)/2)^10
`mode`, `t_catch` and `falls` below are only labels for the display, computed with the rig controller's
catch / fall thresholds so the two controllers can be compared on the same terms.
"""
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
# two layouts: 02_differentiable_world_policy/diff_sim (dwm_core in ../code) or a standalone copy with dwm_core
# beside this file (Pendulum_control/diff_sim, rotary_model in ../simulator)
CODE = HERE if (HERE / "dwm_core.py").exists() else HERE.parent / "code"
sys.path.insert(0, str(CODE))
if (HERE.parent / "simulator" / "rotary_model.py").exists():
    sys.path.insert(0, str(HERE.parent / "simulator"))
import dwm_core as dw                                   # noqa: E402

rm = dw.import_rotary_model()

POLICY_CANDIDATES = [
    HERE / "policy.pt",                                                   # train_diff_policy.py
    HERE / "policy_03_colab.pt",                                          # standalone copy of the one below
    CODE / "results" / "03" / "policy_swing_lqr_nn_trunc_rand.pt",        # tutorial 03 (Colab run, 10/10)
]
WORLD_MODEL_CANDIDATES = [HERE / "world_models.pt", CODE / "results" / "02" / "world_models.pt"]   # tutorial 02


def load_policy(path=None):
    """dw.Policy with the saved weights, gains and LQR prior. Returns (policy, path)."""
    paths = [Path(path)] if path else POLICY_CANDIDATES
    for p in paths:
        if p.exists():
            pol = dw.Policy(seed=0, swing=True)
            sd = torch.load(p, map_location="cpu")
            sd.setdefault("nn_gate", torch.tensor(0.0))           # policies saved before the nn_gate option
            pol.load_state_dict(sd)
            pol.eval()
            return pol, p
    raise FileNotFoundError("no trained policy found -- run train_diff_policy.py first, or pass --policy")


class DiffPolicyController:
    name = "differentiable policy"

    def __init__(self, policy, catch_deg=rm.SWING["catch"], fall_deg=rm.SWING["fall"]):
        self.fn = policy.numpy_fn()
        self.catch, self.fall = catch_deg, fall_deg
        self.reset()

    def reset(self, offset_guess=0.0):
        self.mode = "swing"
        self.t_start, self.t_catch = None, None
        self.catches, self.falls = 0, 0
        self.b_hat = 0.0                                # (the rig controller estimates an encoder offset; this one does not)
        self.E = float("nan")

    def __call__(self, z, t):
        if self.t_start is None:
            self.t_start = t
        th = math.degrees(abs(z[0]))
        self.E = rm.energy(z[0], z[1])
        if self.mode == "swing" and th < self.catch:
            self.mode = "balance"
            self.catches += 1
            if self.t_catch is None:
                self.t_catch = t - self.t_start
        elif self.mode == "balance" and th > self.fall:
            self.mode = "swing"
            self.falls += 1
        return float(self.fn(z))


def rig_controller():
    """The hand-designed controller running on the real rig (energy pumping + catch + pole-placement balancer)."""
    c = rm.SwingUp(rm.place4())
    c.name = "rig swing-up (hand-designed)"
    return c
