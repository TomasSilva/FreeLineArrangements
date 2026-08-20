"""
surrogate.py — learned proposal ranker for the swap search (opt-in).

A feature-MLP trained on this project's own evaluated candidates predicts,
for a PROPOSED swap result, (a) the log10 penalized Saito loss and (b) the
probability that the raw loss falls below the 1e-6 certification-gate
threshold.  Engines use it only to ORDER wide candidate sets so the true
evaluations are spent on the most promising swaps.

Discipline (tested in tests/test_surrogate.py):
  * surrogate scores are never logged as losses;
  * acceptance/energy decisions use only true raw evaluations;
  * the certification gate and exact certification are untouched;
  * with no ranker, every code path is byte-identical to production.

Features are computed from the trial arrangement's EXACT intersection
lattice (ms-scale even at n = 20) — no evaluator call is needed.  The same
`extract_features` serves the dataset builder and the runtime ranker, so
train/serve cannot drift.  GNN over the incidence graph is deferred to v2.
"""

import hashlib
import json
import math
import os

import numpy as np

from arrangement import LineArrangement

FEATURE_SCHEMA_VERSION = "surrogate-features-v1"
FIELD_SLOTS = ("QQ", 2, 3, 5, -1, -3)
GATE = 1e-6                       # certification-gate label threshold

FEATURE_NAMES = (
    ["n", "d1", "d2", "pair_spread", "b2_rel_gap", "m_max",
     "m_max_minus_minfeas", "n_points_norm"]
    + [f"t{r}_norm" for r in range(2, 9)]
    + ["line_pts_min", "line_pts_max", "line_pts_mean", "line_pts_std",
       "lines_t_d1", "lines_t_d2", "log_height"]
    + [f"field_{s}" for s in FIELD_SLOTS]
)
N_FEATURES = len(FEATURE_NAMES)


def extract_features(arr: LineArrangement, d1: int, d2: int,
                     height=None) -> np.ndarray:
    """Feature vector for one arrangement at a target pair (float64)."""
    from swap_search import min_feasible_m
    n = len(arr)
    pts = arr._structure()
    mults = [len(v) for v in pts.values()]
    m_max = max(mults) if mults else 0
    b2 = sum(m - 1 for m in mults)
    b2_star = (n - 1) + d1 * d2
    per_line = [0] * n
    for p, ls in pts.items():
        for i in ls:
            per_line[i] += 1
    tvec = [0] * 7                      # t2..t8 (t>=8 folded into t8)
    for m in mults:
        tvec[min(m, 8) - 2] += 1
    npairs = max(1, n * (n - 1) // 2)
    if height is None:
        from novelty import coordinate_height
        height = coordinate_height(arr)
    pl = np.array(per_line, dtype=np.float64)
    x = ([float(n), float(d1), float(d2), float(d2 - d1),
          (b2 - b2_star) / (1.0 + b2_star), float(m_max),
          float(m_max - min_feasible_m(n, d1)),
          len(pts) / npairs]
         + [t / npairs for t in tvec]
         + [float(pl.min()), float(pl.max()), float(pl.mean()),
            float(pl.std()),
            float(sum(1 for t in per_line if t - 1 == d1)),
            float(sum(1 for t in per_line if t - 1 == d2)),
            math.log10(max(1.0, float(height)))])
    fd = None
    for line in arr.lines:
        f = getattr(line, "field", None)
        if f is not None:
            fd = f.d
            break
    for s in FIELD_SLOTS:
        x.append(1.0 if (s == "QQ" and fd is None) or (s == fd) else 0.0)
    return np.array(x, dtype=np.float64)


# ── model ────────────────────────────────────────────────────────────────


def _build_mlp(torch, n_in):
    # dtype-explicit (other components set the torch global default to
    # float64; the surrogate is float32 end to end regardless)
    return torch.nn.Sequential(
        torch.nn.Linear(n_in, 256), torch.nn.ReLU(), torch.nn.Dropout(0.1),
        torch.nn.Linear(256, 256), torch.nn.ReLU(), torch.nn.Dropout(0.1),
        torch.nn.Linear(256, 256), torch.nn.ReLU(),
        torch.nn.Linear(256, 2),      # [log10-loss, certifiable-logit]
    ).to(torch.float32)


def train_surrogate(dataset_npz, out_path, epochs=30, batch=4096,
                    lr=1e-3, seed=0, val_frac=0.1, verbose=True):
    """Train the two-head MLP; returns metrics dict (also embedded in the
    checkpoint together with data/commit provenance)."""
    import torch
    torch.manual_seed(seed)
    data = np.load(dataset_npz, allow_pickle=True)
    X, y_log, y_cls = data["X"], data["y_log"], data["y_cls"]
    holdout_mask = data["holdout"].astype(bool)
    Xtr_all, Xho = X[~holdout_mask], X[holdout_mask]
    ytr_log, yho_log = y_log[~holdout_mask], y_log[holdout_mask]
    ytr_cls, yho_cls = y_cls[~holdout_mask], y_cls[holdout_mask]

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(Xtr_all))
    n_val = max(1, int(val_frac * len(idx)))
    val_i, tr_i = idx[:n_val], idx[n_val:]
    mu, sd = Xtr_all[tr_i].mean(0), Xtr_all[tr_i].std(0) + 1e-9

    def T(a):
        return torch.tensor((a - mu) / sd, dtype=torch.float32)

    Xtr, Xval = T(Xtr_all[tr_i]), T(Xtr_all[val_i])
    yl_tr = torch.tensor(ytr_log[tr_i], dtype=torch.float32)
    yc_tr = torch.tensor(ytr_cls[tr_i], dtype=torch.float32)
    yl_val = torch.tensor(ytr_log[val_i], dtype=torch.float32)
    yc_val = torch.tensor(ytr_cls[val_i], dtype=torch.float32)

    model = _build_mlp(torch, X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = torch.nn.BCEWithLogitsLoss()
    pos_w = max(1.0, float((yc_tr == 0).sum()) / max(1.0,
                float((yc_tr == 1).sum())))
    bce_w = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(min(pos_w, 20.0)))
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        tot = 0.0
        for k in range(0, len(Xtr), batch):
            b = perm[k:k + batch]
            opt.zero_grad()
            out = model(Xtr[b])
            loss = (torch.nn.functional.mse_loss(out[:, 0], yl_tr[b])
                    + bce_w(out[:, 1], yc_tr[b]))
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            model.eval()
            with torch.no_grad():
                ov = model(Xval)
                mse = float(torch.nn.functional.mse_loss(ov[:, 0], yl_val))
            print(f"  epoch {ep:3d}: train {tot/len(Xtr):.4f} "
                  f"val-mse {mse:.4f}", flush=True)

    # held-out-cell metrics (the go/no-go numbers)
    model.eval()

    def _metrics(Xa, yl, yc):
        import torch
        with torch.no_grad():
            o = model(torch.tensor((Xa - mu) / sd, dtype=torch.float32))
            pred_l = o[:, 0].numpy()
            logit = o[:, 1].numpy()
        ss_res = float(((pred_l - yl) ** 2).sum())
        ss_tot = float(((yl - yl.mean()) ** 2).sum()) + 1e-12
        r2 = 1.0 - ss_res / ss_tot
        # AUC by rank statistic
        order = np.argsort(logit)
        ranks = np.empty(len(order)); ranks[order] = np.arange(len(order))
        pos, neg = ranks[yc == 1], ranks[yc == 0]
        auc = ((pos.mean() - (len(pos) - 1) / 2.0) / max(1, len(neg))
               if len(pos) and len(neg) else float("nan"))
        return {"r2_logloss": r2, "auc_certifiable": float(auc),
                "n": int(len(Xa)), "n_pos": int((yc == 1).sum())}
    holdout_metrics = (_metrics(Xho, yho_log, yho_cls)
                       if len(Xho) else {"n": 0})

    ckpt = {"state_dict": model.state_dict(), "mu": mu, "sd": sd,
            "feature_schema": FEATURE_SCHEMA_VERSION,
            "feature_names": FEATURE_NAMES,
            "dataset_manifest_hash": str(data.get("manifest_hash", "")),
            "holdout_metrics": holdout_metrics, "epochs": epochs,
            "seed": seed}
    import torch as _t
    _t.save(ckpt, out_path)
    return holdout_metrics


class SurrogateRanker:
    """Runtime ranker: higher score = more promising (predicted-low loss,
    high certifiable probability).  Scores are ORDERING DEVICES ONLY."""

    def __init__(self, model, mu, sd, provenance=None):
        self.model = model
        self.mu, self.sd = mu, sd
        self.provenance = provenance or {}
        self.n_ranked = 0

    @classmethod
    def load(cls, path):
        import torch
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if ckpt.get("feature_schema") != FEATURE_SCHEMA_VERSION:
            raise ValueError(
                f"surrogate checkpoint feature schema "
                f"{ckpt.get('feature_schema')!r} != {FEATURE_SCHEMA_VERSION!r}")
        model = _build_mlp(torch, len(ckpt["feature_names"]))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return cls(model, ckpt["mu"], ckpt["sd"],
                   {"path": path,
                    "dataset_manifest_hash": ckpt.get(
                        "dataset_manifest_hash"),
                    "holdout_metrics": ckpt.get("holdout_metrics")})

    def rank(self, arrangements, d1, d2):
        import torch
        X = np.stack([extract_features(a, d1, d2) for a in arrangements])
        with torch.no_grad():
            o = self.model(torch.tensor((X - self.mu) / self.sd,
                                        dtype=torch.float32))
        self.n_ranked += len(arrangements)
        # score: predicted certifiability logit minus predicted log-loss
        return (o[:, 1] - o[:, 0]).numpy()
