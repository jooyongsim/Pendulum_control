"""Retrain the data-size behaviour-cloning sweep, then film it.

    python Yeeun/pipeline_yeen/sweep_bc.py                  collect + train + video + table
    python Yeeun/pipeline_yeen/sweep_bc.py --no-train       just film an existing sweep

The data-size study trained these in a notebook and wrote them to /content/sweep, which was
never committed -- so the checkpoints do not exist anywhere and have to be made again. This does
it in one command, with the same settings as that notebook:

    SIZES = [40, 60, 80] collected episodes, plus the committed bc_h5.npz as the 206-episode one

The kept count is lower than the requested one. collect_expert.py throws away an episode the
teacher itself never got upright, because there is no correct answer to learn from it; roughly
80% survive, so 40 -> ~31. That is why the models are labelled by what they actually trained on.

Two things this guards that the notebook did not:

  * train_bc.py always writes checkpoints/bc_h5.npz. Run it here and it would destroy the
    committed 206-episode model, so that file is backed up first and restored at the end.
  * Collection is numpy-only; only the training step needs torch. They are separate subprocess
    calls, so a missing torch fails loudly at the right step instead of halfway through.
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import argparse              # noqa: E402
import json                  # noqa: E402
import shutil                # noqa: E402
import subprocess            # noqa: E402
import sys                   # noqa: E402
import time                  # noqa: E402
from pathlib import Path     # noqa: E402

import numpy as np           # noqa: E402

import arena                 # noqa: E402  -- resolves simulator/, rl_yeen/, behavior_cloning_yeen/
import gallery               # noqa: E402

ROOT = Path(__file__).resolve().parent
SIZES = [40, 60, 80]
HISTORY = 5                  # the sweep's models all stack 5 frames; only the data size varies


def _report(path):
    m = json.loads(str(np.load(path, allow_pickle=False)["meta_json"]))
    return m, m.get("report", {})


def rename(path, newname):
    """Set the display name inside a checkpoint, so evaluation output can tell them apart."""
    d = dict(np.load(path, allow_pickle=False))
    m = json.loads(str(d["meta_json"]))
    m["name"] = newname
    d["meta_json"] = json.dumps(m)
    np.savez_compressed(path, **d)


def run(cmd, env):
    print("    $ " + " ".join(str(c) for c in cmd))
    r = subprocess.run([str(c) for c in cmd], env=env)
    if r.returncode:
        raise SystemExit(f"failed ({r.returncode}): {' '.join(str(c) for c in cmd)}")


def train_sweep(sizes, out_dir, bc_dir, epochs=80, seed=0):
    out_dir.mkdir(parents=True, exist_ok=True)
    ck = bc_dir / "checkpoints"
    live = ck / f"bc_h{HISTORY}.npz"
    if not live.is_file():
        raise SystemExit(f"{live} not found -- the 206-episode model is the sweep's top end")

    backup = out_dir / "_bc_h5_original.npz"
    shutil.copy(live, backup)                    # train_bc.py overwrites this file
    shutil.copy(live, out_dir / "bc_206.npz")
    rename(out_dir / "bc_206.npz", "ep206")
    print(f"  kept the committed model: {_report(out_dir / 'bc_206.npz')[1].get('episodes')} ep")

    env = dict(os.environ, PYTHONPATH=str(bc_dir))
    try:
        for n in sizes:
            t0 = time.perf_counter()
            print(f"\n  --- {n} episodes " + "-" * 40)
            data = bc_dir / "data" / f"expert_{n}.npz"
            run([sys.executable, bc_dir / "collect_expert.py",
                 "--episodes", n, "--out", data, "--seed", seed], env)
            run([sys.executable, bc_dir / "train_bc.py",
                 "--data", data, "--history", HISTORY, "--epochs", epochs, "--seed", seed], env)
            dst = out_dir / f"bc_{n}.npz"
            shutil.copy(live, dst)
            kept = _report(dst)[1].get("episodes")
            rename(dst, f"ep{kept}")
            print(f"  -> {dst.name}  requested {n}, kept {kept}"
                  f"  [{time.perf_counter() - t0:.0f}s]")
    finally:
        shutil.copy(backup, live)                # put the committed model back, even on failure
        backup.unlink(missing_ok=True)
        print(f"\n  restored {live.name}")

    print("\n  sweep:")
    for p in sorted(out_dir.glob("bc_*.npz"), key=lambda q: _report(q)[1].get("episodes", 0)):
        _, r = _report(p)
        print(f"    {p.name:<12} {r.get('episodes'):>4} ep   rmse {r.get('rmse', 0):.3f}"
              f"   swing {r.get('rmse_swing', 0):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    ap.add_argument("--out", default=str(ROOT / "sweep"))
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-train", action="store_true", help="film an existing sweep folder")
    ap.add_argument("--with-dagger", action="store_true",
                    help="add the DAgger model beside the sweep as a reference")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--seconds", type=float, default=gallery.VIDEO_S)
    ap.add_argument("--suite", default="robust", choices=list(arena.SUITES))
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    out_dir = Path(a.out)
    if arena.BC_DIR is None:
        raise SystemExit("behavior_cloning_yeen not found next to this folder")

    if not a.no_train:
        train_sweep(a.sizes, out_dir, arena.BC_DIR, epochs=a.epochs, seed=a.seed)

    # the sweep alone: adding DAgger here would mix "how much data" with "which data"
    entries = gallery.imitation(str(out_dir), with_dagger=a.with_dagger)
    if not entries:
        raise SystemExit(f"no checkpoints in {out_dir} -- run without --no-train first")
    print("\nline-up:", [c.label for c in entries])

    rows = gallery.rows_for(entries, task="swingup", suite=a.suite, group="sweep")
    arena.print_table([r for r in rows if r["cond"] == "clean"],
                      "behaviour cloning by training-set size")
    if a.suite != "clean":
        import collections
        conds = [c["label"] for c in arena.SUITES[a.suite]]
        wide = collections.defaultdict(dict)
        for r in rows:
            wide[r["agent"]][r["cond"]] = r["hold2"]
        print("\n  hold2 [%] by condition")
        print(f"    {'agent':<14s}" + "".join(f"{c:>10s}" for c in conds))
        for ag, d in wide.items():
            print(f"    {ag:<14s}" + "".join(f"{d.get(c, float('nan')):10.1f}" for c in conds))

    if a.csv:
        import csv
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\n  table -> {a.csv}")

    if not a.no_video:
        p, _ = gallery.render(entries, ROOT / "gallery" / "sweep_swingup.mp4",
                              task="swingup", seconds=a.seconds,
                              title="behaviour cloning -- more training episodes")
        print(f"  video -> {p}  ({p.stat().st_size / 1e6:.1f} MB)")
        p2, _ = gallery.render(entries, ROOT / "gallery" / "sweep_offset2.mp4",
                               task="swingup", seconds=a.seconds, offset_deg=2.0,
                               title="same models, encoder zero off by 2 deg")
        print(f"  video -> {p2}  ({p2.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
