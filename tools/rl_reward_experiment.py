"""Run simulator/rl_pendulum_tutorial.py headless (sections 1-6.1) with patched source,
to compare design variants before editing the tutorial itself.

    python tools/rl_reward_experiment.py <variant>

Variants are lists of (old, new) source replacements; see VARIANTS below.
Prints the eval curve and the step-response metrics (steady-state error)."""
import sys
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

ENT = [("self.target_entropy = -float(act_dim)", "self.target_entropy = -4.0 * act_dim")]
ESCALE = [("                         e, self.ref, self.a_prev]", "                         5 * e, self.ref, self.a_prev]"),
          ("        e, w = float(obs[3]), float(obs[2]) * 10.0", "        e, w = float(obs[3]) / 5, float(obs[2]) * 10.0")]
INTEG = [("    obs_dim, act_dim = 6, 1", "    obs_dim, act_dim = 7, 1"),
         ("                         5 * e, self.ref, self.a_prev]", "                         5 * e, self.ref, self.a_prev, 2 * self.ie]"),
         ("        self.a_prev, self.k, self.t = 0.0, 0, 0.0", "        self.a_prev, self.k, self.t, self.ie = 0.0, 0, 0.0, 0.0"),
         ("        self.a_prev = a\n", "        self.a_prev = a\n        self.ie = float(np.clip(self.ie + e * self.dt_ctrl, -0.5, 0.5))\n")]
FF = [("        u = a * U_MAX\n", "        u = float(np.clip(a * U_MAX + K * np.sin(self.ref), -U_MAX, U_MAX))\n"),
      ("        return np.clip(u_raw / U_MAX, -1, 1)", "        return np.clip(u_raw / U_MAX, -1, 1) - K * np.sin(obs[4]) / U_MAX")]

VARIANTS = {
    "base": [],
    "ent": ENT + ESCALE,
    "ent_int": ENT + ESCALE + INTEG,
    "ent_ff": ENT + ESCALE + FF,
}

tag = sys.argv[1]
sim_dir = Path(__file__).resolve().parents[1] / "simulator"
src = (sim_dir / "rl_pendulum_tutorial.py").read_text(encoding="utf-8")
src = src.split("# %% [markdown]\n# ### 6.2")[0]                 # stop before policy map / save
for old, new in VARIANTS[tag]:
    assert src.count(old) == 1, old
    src = src.replace(old, new)
src = src.replace("print(f\"step", f"print(\"[{tag}]\", f\"step")
os.chdir(sim_dir)
exec(compile(src, "tutorial", "exec"), {"__name__": "exp", "__file__": str(sim_dir / "rl_pendulum_tutorial.py")})
