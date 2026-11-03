"""Behaviour cloning: fit an MLP to the expert's (observation -> arm acceleration) pairs.

    python behavior_cloning_yeen/train_bc.py              history 5 (default) -> checkpoints/bc_h5.npz
    python behavior_cloning_yeen/train_bc.py --history 1  memoryless baseline -> checkpoints/bc_h1.npz
    python behavior_cloning_yeen/train_bc.py --compare    trains h1 and h5 and prints them side by side

The train/validation split is by *episode*, not by sample. Consecutive samples 10 ms apart are
almost identical, so a random sample split would put near-duplicates on both sides and report a
validation error that means nothing.

The checkpoint is a plain .npz of weights plus the normalisation constants, so inference needs
numpy only (il_common.BCPolicy) -- the host loop that drives the real rig does not need torch.
"""
import il_common as il       # first: sets the OpenMP env vars torch needs (see il_common)

import argparse              # noqa: E402
import json                  # noqa: E402
import time                  # noqa: E402

import numpy as np           # noqa: E402
import torch                 # noqa: E402
import torch.nn as nn        # noqa: E402


def load(path, history, val_frac=0.1, seed=0):
    d, meta = il.load_dataset(path)
    feat, act, mode, ep = d["feat"], d["act"], d["mode"], d["ep"]

    X, Y, M, E = [], [], [], []
    for e in np.unique(ep):
        s = ep == e
        X.append(il.stack_episode(feat[s].astype(np.float64), history))
        Y.append(act[s].astype(np.float64))
        M.append(mode[s])
        E.append(np.full(s.sum(), e))
    X, Y, M, E = (np.concatenate(a) for a in (X, Y, M, E))

    eps = np.unique(ep)
    rng = np.random.default_rng(seed)
    val_eps = set(rng.permutation(eps)[:max(1, int(round(val_frac * len(eps))))].tolist())
    v = np.isin(E, list(val_eps))
    return dict(Xtr=X[~v], Ytr=Y[~v], Mtr=M[~v], Xva=X[v], Yva=Y[v], Mva=M[v],
                n_episodes=len(eps), n_val_episodes=len(val_eps), meta=meta)


class MLP(nn.Module):
    def __init__(self, n_in, hidden=(256, 256)):
        super().__init__()
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train(path, history, hidden, epochs, batch, lr, swing_weight, seed, quiet=False, name=None):
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    ds = load(path, history, seed=seed)
    Xtr, Ytr, Xva, Yva = ds["Xtr"], ds["Ytr"], ds["Xva"], ds["Yva"]

    x_mu, x_sd = Xtr.mean(0), np.maximum(Xtr.std(0), 1e-6)
    a_mu, a_sd = float(Ytr.mean()), float(max(Ytr.std(), 1e-6))
    xt = torch.tensor((Xtr - x_mu) / x_sd, dtype=torch.float32)
    yt = torch.tensor((Ytr - a_mu) / a_sd, dtype=torch.float32)
    xv = torch.tensor((Xva - x_mu) / x_sd, dtype=torch.float32)
    yv = torch.tensor((Yva - a_mu) / a_sd, dtype=torch.float32)
    # swing is the rarer and harder mode; upweighting it trades balance precision for swing-up
    w = torch.tensor(np.where(ds["Mtr"] == 0, swing_weight, 1.0), dtype=torch.float32)

    net = MLP(Xtr.shape[1], hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = len(xt)
    hist, best = [], None
    t0 = time.perf_counter()
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = (w[idx] * (net(xt[idx]) - yt[idx]) ** 2).sum() / w[idx].sum()
            loss.backward()
            opt.step()
            tot += loss.detach().item() * len(idx)
        sched.step()
        net.eval()
        with torch.no_grad():
            vl = float(((net(xv) - yv) ** 2).mean())
        hist.append((tot / n, vl))
        if best is None or vl < best[0]:
            best = (vl, {k: v.detach().clone() for k, v in net.state_dict().items()})
        if not quiet and (ep + 1) % max(1, epochs // 10) == 0:
            print(f"    epoch {ep + 1:4d}/{epochs}  train {tot / n:.5f}  val {vl:.5f}")
    net.load_state_dict(best[1])                      # early stopping on the episode-held-out set

    # report in physical units (rad/s^2), per mode
    net.eval()
    with torch.no_grad():
        pv = net(xv).numpy() * a_sd + a_mu
    err = pv - Yva
    rep = dict(history=history, n_in=int(Xtr.shape[1]), params=sum(p.numel() for p in net.parameters()),
               n_train=int(n), n_val=int(len(xv)), episodes=ds["n_episodes"],
               val_episodes=ds["n_val_episodes"], train_s=time.perf_counter() - t0,
               rmse=float(np.sqrt(np.mean(err ** 2))),
               rmse_swing=float(np.sqrt(np.mean(err[ds["Mva"] == 0] ** 2))),
               rmse_balance=float(np.sqrt(np.mean(err[ds["Mva"] == 1] ** 2))),
               r2=float(1 - np.mean(err ** 2) / np.var(Yva)),
               expert_std=float(np.std(Yva)))

    weights = [p.detach().numpy().astype(np.float64) for nm, p in net.state_dict().items()
               if nm.endswith("weight")]
    biases = [p.detach().numpy().astype(np.float64) for nm, p in net.state_dict().items()
              if nm.endswith("bias")]
    meta = dict(name=name or f"bc_h{history}", history=history, hidden=list(hidden),
                activation="relu",
                n_layers=len(weights), feat_names=list(il.FEAT_NAMES), u_max=il.U_MAX,
                control_hz=il.rm.CONTROL_HZ, dataset=str(path), report=rep)
    ck = {f"W{i}": w_ for i, w_ in enumerate(weights)}
    ck.update({f"b{i}": b_ for i, b_ in enumerate(biases)})
    ck.update(x_mu=x_mu, x_sd=x_sd, a_mu=np.float64(a_mu), a_sd=np.float64(a_sd),
              meta_json=json.dumps(meta))
    rep["export_err"] = check_numpy_matches_torch(net, ck, Xva)
    return ck, rep, np.asarray(hist)


def check_numpy_matches_torch(net, ck, X, tol=1e-4):
    """The deployed policy runs on NumpyMLP, not on torch. Verify the two agree.

    Catches both a weight-ordering mistake in the export and any numerical damage from the
    duplicated OpenMP runtime.
    """
    n = int(json.loads(ck["meta_json"])["n_layers"])
    mlp = il.NumpyMLP([ck[f"W{i}"] for i in range(n)], [ck[f"b{i}"] for i in range(n)], "relu")
    xs = (X[:512] - ck["x_mu"]) / ck["x_sd"]
    net.eval()
    with torch.no_grad():
        a = net(torch.tensor(xs, dtype=torch.float32)).numpy()
    b = mlp(xs)[:, 0]
    err = float(np.max(np.abs(a - b)))         # torch runs float32, NumpyMLP float64
    if not err < tol:
        raise SystemExit(f"numpy inference disagrees with torch by {err:.2e} -- export is broken")
    return err


def save(ck, out):
    il.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **ck)
    return out


def line(r):
    return (f"h={r['history']:<2d} in={r['n_in']:<3d} params={r['params']:<7d} "
            f"RMSE {r['rmse']:6.3f}  swing {r['rmse_swing']:6.3f}  balance {r['rmse_balance']:6.3f}"
            f"  R2 {r['r2']:6.3f}   [{r['train_s']:.0f}s]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--history", type=int, default=5, help="frames stacked into the observation")
    ap.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--swing-weight", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compare", action="store_true", help="also train the memoryless baseline")
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()

    data = a.data or (il.DATA_DIR / "expert.npz")
    hs = sorted({1, a.history}) if a.compare else [a.history]
    kw = dict(hidden=tuple(a.hidden), epochs=a.epochs, batch=a.batch, lr=a.lr,
              swing_weight=a.swing_weight, seed=a.seed)

    out = []
    for h in hs:
        print(f"\n--- history {h} " + "-" * 50)
        ck, rep, hist = train(data, h, **kw)
        p = save(ck, il.CKPT_DIR / f"bc_h{h}.npz")
        print(f"  {line(rep)}")
        print(f"  expert action std on val: {rep['expert_std']:.3f} rad/s^2   "
              f"numpy/torch export agreement: {rep['export_err']:.1e}")
        print(f"  -> {p}")
        out.append((h, rep, hist))

    if len(out) > 1:
        print("\nopen-loop fit (validation episodes, rad/s^2):")
        for _, rep, _ in out:
            print(f"  {line(rep)}")
        print("\n  Open-loop RMSE only says how well each policy copies single actions.")
        print("  What matters is whether it swings up: run eval_bc.py.")

    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        il.FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        for h, _, hist in out:
            ax.plot(hist[:, 0], lw=1, ls="--", color=f"C{h % 10}")
            ax.plot(hist[:, 1], lw=1.5, color=f"C{h % 10}", label=f"history {h} (val)")
        ax.set(xlabel="epoch", ylabel="MSE (normalised action)", yscale="log")
        ax.grid(alpha=.3); ax.legend(title="dashed = train")
        fig.tight_layout()
        fig.savefig(il.FIG_DIR / "bc_training.png", dpi=140)
        print(f"\n  curve -> {il.FIG_DIR / 'bc_training.png'}")


if __name__ == "__main__":
    main()
