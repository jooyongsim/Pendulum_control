"""Run a trained policy on the real STEVAL-EDUKIT01 rig.

    python Yeeun/pipeline_yeen/run_rig.py --check                 simulate only, no hardware
    python Yeeun/pipeline_yeen/run_rig.py --dry-run               read the board, command nothing
    python Yeeun/pipeline_yeen/run_rig.py                         drive the motor
    python Yeeun/pipeline_yeen/run_rig.py --model <path.npz> --port COM10 --max-pps 1200

Everything else in this folder runs in simulation. This is the one file that moves hardware, so
read the three gaps it has to bridge before changing it.

1. The firmware speaks a different language from the policy.

       firmware observation   [pendulum_deg 0..360, rotor_deg, motor_pps, l6474_status]
       policy observation     [cos th, sin th, th', phi, phi']     th: upright = 0
       firmware command       CMD_SET_VELOCITY, signed pps
       policy action          a in [-1, 1]  ->  arm ACCELERATION

   So the host converts units, shifts the pendulum zero from "hanging" to "upright", and
   integrates acceleration into a velocity exactly the way rotary_model.Plant does.

2. The firmware does not measure th'. It is differenced from successive encoder readings and
   low-pass filtered with rotary_model.TAU_V, the same filter rotary_model.Sensor applies in
   simulation -- otherwise the policy sees a much noisier th' than it ever trained on.

3. The policy trained against rotary_model.SPEED_MAX = 4000 pps, while the hand-control script
   (manual_keyboard_balance.py) keeps the rig between 200 and 1000 pps. --max-pps defaults to
   the cautious end. Raising it toward 4000 moves the rig closer to the training conditions and
   also makes every mistake faster; raise it in steps, with a hand on the power.

Safety:
  * |phi| beyond --arm-limit stops the run, as the firmware's own rotor limit does.
  * HARD_STOP is sent on every exit path, including Ctrl-C and any exception.
  * --dry-run does the whole loop and prints the command it would have sent, without sending it.
  * CONTROL_SIGN flips left/right if the rig moves the wrong way. Fix it here, never in the
    firmware (project rule).

Before any of this, --check runs the same policy in the simulator, which is the project rule for
anything that moves hardware.
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse              # noqa: E402
import math                  # noqa: E402
import sys                   # noqa: E402
import time                  # noqa: E402
from pathlib import Path     # noqa: E402

import numpy as np           # noqa: E402

import arena                 # noqa: E402  -- resolves simulator/, rl_yeen/ on sys.path
import rotary_model as rm    # noqa: E402
import rl_common as rl       # noqa: E402

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "checkpoints" / "sac_warm_swingup.npz"

# --- firmware protocol (PendulumController.ino) ---------------------------------------------
CMD_SET_HOME, CMD_SET_VELOCITY, CMD_HARD_STOP = 0, 4, 5
CMD_QUERY, CMD_RESET_SAFETY = 6, 7
BAUD = 500_000                      # 115200 does not work; this is the firmware's rate

# --- host-side limits -----------------------------------------------------------------------
CONTROL_HZ = 100.0
ARM_LIMIT_DEG = 140.0
MIN_PPS = 200.0                     # the firmware turns anything slower into a hard stop
MAX_PPS = 1000.0                    # cautious default; the policy trained against 4000
STEP_RAD = rm.STEP_RAD              # rad per microstep: pps * STEP_RAD = rad/s
CONTROL_SIGN = 1.0                  # set to -1.0 if the arm moves the wrong way


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def load_policy(path):
    """Return ctrl(obs5) -> a in [-1, 1] for a SAC checkpoint, or a BC checkpoint's equivalent."""
    path = Path(path)
    name = path.stem
    if name.startswith(("sac_", "bc_swingup", "dagger_swingup")):
        actor = rl.NumpyActor(path)
        return actor, lambda o: actor(o)
    raise SystemExit(f"{path.name}: expected an rl_yeen/pipeline_yeen actor (.npz). "
                     "The behaviour-cloning checkpoints in behavior_cloning_yeen use a "
                     "different observation (5 frames stacked) and need their own runner.")


class ThetaDot:
    """Differentiate the encoder and filter it the way rotary_model.Sensor does in simulation."""

    def __init__(self, tau=rm.TAU_V):
        self.tau, self.prev, self.v = tau, None, 0.0

    def __call__(self, th, dt):
        if self.prev is None or dt <= 0.0:
            self.prev = th
            return 0.0
        raw = rm.wrap(th - self.prev) / dt
        self.prev = th
        alpha = dt / (self.tau + dt)
        self.v += alpha * (raw - self.v)
        return self.v


# ---------------------------------------------------------------- simulation check
def check(policy_path, seconds, offset_deg=0.0):
    """The project rule: validate in the simulator before touching the rig."""
    ctrl = rl.NumpyActor(policy_path).as_rig_controller()
    for label, kw in (("clean", {}), ("encoder off 2 deg", dict(offset_deg=2.0)),
                      ("2x friction", dict(coulomb=2.0))):
        lg = arena.run(ctrl, "swingup", seconds=seconds, **kw)
        s = arena.score(lg)
        print(f"  {label:<20s} hold2 {s['hold2']:5.1f}%  bias {s['bias_deg']:+6.2f} deg"
              f"  t_up {s['t_up']:5.2f} s  arm {s['arm_max_deg']:6.1f} deg"
              f"  {s['stop'] or 'ok'}")
    print("\n  'arm' close to 140 means little margin left on the real rig.")


# ---------------------------------------------------------------- the rig
def zero_encoder(cc, samples=40, settle=0.5):
    """Average the encoder while the rod hangs still -- that reading becomes theta = pi."""
    print(f"  hold the rod still (hanging) -- averaging {samples} samples")
    time.sleep(settle)
    s = c = 0.0
    got = 0
    while got < samples:
        r = cc.step(CMD_QUERY, [0.0])
        if r is None:
            continue
        deg = r[3][0]
        s += math.sin(math.radians(deg))
        c += math.cos(math.radians(deg))
        got += 1
        time.sleep(0.01)
    zero = math.degrees(math.atan2(s, c)) % 360.0
    print(f"  hanging zero = {zero:.2f} deg")
    return zero


def run(policy_path, port=None, duration=25.0, max_pps=MAX_PPS, arm_limit=ARM_LIMIT_DEG,
        sign=CONTROL_SIGN, dry_run=False, log=None):
    sys.path.insert(0, str(arena.REPO))
    from control_comms import ControlComms, StatusCode, find_board_port  # noqa: E402

    actor, act = load_policy(policy_path)
    print(f"model {Path(policy_path).name}")
    rep = actor.meta.get("report", {})
    if rep:
        print(f"  trained for '{rep.get('task')}' | {rep.get('steps', '?')} steps"
              f" | final hold2 {rep.get('final', {}).get('hold2', float('nan')):.1f}% (in sim)")

    cc = ControlComms(timeout=0.2)
    port = port or find_board_port()
    if cc.connect(port, BAUD) is not StatusCode.OK:
        raise SystemExit(f"could not open {port} at {BAUD}")
    print(f"port {port} @ {BAUD}")

    dt = 1.0 / CONTROL_HZ
    v_max = max_pps * STEP_RAD                   # rad/s the host will allow itself to command
    rows = []
    try:
        cc.step(CMD_HARD_STOP, [0.0])
        cc.step(CMD_RESET_SAFETY, [0.0])
        cc.step(CMD_SET_HOME, [0.0])             # rotor angle starts at 0
        zero = zero_encoder(cc)

        thd = ThetaDot()
        v = 0.0                                  # commanded arm velocity [rad/s], host-integrated
        t0 = time.perf_counter()
        next_t = 0.0
        last_t = 0.0
        print(f"\n  running {duration:.0f} s"
              + ("  [DRY RUN -- no motor commands]" if dry_run else "")
              + f"  | max {max_pps:.0f} pps | arm stop {arm_limit:.0f} deg\n")

        while True:
            t = time.perf_counter() - t0
            if t > duration:
                break
            if t < next_t:
                time.sleep(max(0.0, next_t - t))
                continue
            next_t += dt

            r = cc.step(CMD_QUERY, [0.0])
            if r is None:
                continue
            _status, _ms, terminated, obs = r
            if terminated:
                print("  firmware reports terminated (safety latch) -- stopping")
                break

            # rig frame -> policy frame: hanging reads `zero`, and the policy wants upright = 0
            th = rm.wrap(math.radians(wrap_deg(obs[0] - zero)) + math.pi)
            phi = math.radians(obs[1])
            phi_dot = obs[2] * STEP_RAD
            w = thd(th, t - last_t)
            last_t = t

            if abs(math.degrees(phi)) > arm_limit:
                print(f"  ARM LIMIT ({math.degrees(phi):+.1f} deg) -- stopping")
                break

            a = act(np.array([math.cos(th), math.sin(th), w, phi, phi_dot]))
            u = float(np.clip(a, -1.0, 1.0)) * rl.U_MAX * sign

            v = float(np.clip(v + u * dt, -v_max, v_max))     # same integration as Plant
            pps = v / STEP_RAD
            if abs(pps) < MIN_PPS:               # the firmware hard-stops below this anyway
                pps = 0.0

            if not dry_run:
                cc.step(CMD_SET_VELOCITY if pps else CMD_HARD_STOP, [float(pps)])

            rows.append((t, math.degrees(th), w, math.degrees(phi), phi_dot, a, pps))
            if len(rows) % 50 == 0:
                print(f"    t {t:5.2f}  th {math.degrees(th):+7.1f}  w {w:+6.2f}"
                      f"  phi {math.degrees(phi):+7.1f}  a {a:+5.2f}  {pps:+7.0f} pps")
    except KeyboardInterrupt:
        print("\n  interrupted")
    finally:
        cc.step(CMD_HARD_STOP, [0.0])
        cc.close()
        print("  hard stop sent, port closed")

    if rows:
        th = np.array([r[1] for r in rows])
        tail = th[-int(5.0 / dt):] if len(th) > int(5.0 / dt) else th
        print(f"\n  {len(rows)} control steps"
              f"  | last 5 s: mean {tail.mean():+.2f} deg, sd {tail.std():.2f} deg"
              f"  | within +-10 deg {100 * np.mean(np.abs(tail) < 10):.1f}%"
              f"  | arm max {max(abs(r[3]) for r in rows):.1f} deg")
    if log and rows:
        import csv
        with open(log, "w", newline="", encoding="utf-8") as f:
            w_ = csv.writer(f)
            w_.writerow(["t", "theta_deg", "theta_dot", "phi_deg", "phi_dot", "action", "pps"])
            w_.writerows(rows)
        print(f"  log -> {log}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--port", default=None, help="e.g. COM10; found automatically if omitted")
    ap.add_argument("--duration", type=float, default=25.0)
    ap.add_argument("--max-pps", type=float, default=MAX_PPS,
                    help=f"host speed cap (firmware floor {MIN_PPS:.0f}; "
                         f"training used {rm.SPEED_MAX / STEP_RAD:.0f})")
    ap.add_argument("--arm-limit", type=float, default=ARM_LIMIT_DEG)
    ap.add_argument("--sign", type=float, default=CONTROL_SIGN, choices=(1.0, -1.0),
                    help="-1 if the arm moves the wrong way")
    ap.add_argument("--check", action="store_true", help="simulate only, no hardware")
    ap.add_argument("--dry-run", action="store_true", help="read the board, command nothing")
    ap.add_argument("--log", default=None)
    a = ap.parse_args()

    if not Path(a.model).is_file():
        raise SystemExit(f"{a.model} not found")

    if a.check:
        print(f"simulation check: {Path(a.model).name}\n")
        check(a.model, a.duration)
        return

    run(a.model, port=a.port, duration=a.duration, max_pps=a.max_pps,
        arm_limit=a.arm_limit, sign=a.sign, dry_run=a.dry_run, log=a.log)


if __name__ == "__main__":
    main()
