"""
calibration.py

RL reward-calibration layer on top of the RAW penalized Saito loss.

The mathematical loss is NEVER changed: reports, certificates, thresholds
and diagnostics always use the raw S_hat.  For RL shaping only, the raw
loss is passed through the strictly monotone squash

    calibrated_loss(s; tau) = (1 + tau) * s / (s + tau),      tau > 0,
    freeness_potential      = 1 - calibrated_loss,

which maps [0, 1] -> [0, 1] with endpoints fixed (0 -> 0, 1 -> 1) and puts
the half-way point at s = tau.  Strict monotonicity means action ORDERING
under the potential is identical to ordering under the raw loss; the layer
only reshapes reward magnitudes.

tau is chosen BEFORE training as the median raw loss over a fixed
generic-arrangement calibration set for the exact tuple
(n, d1, d2, lambda, beta, field, optimizer profile), then FROZEN for the
whole run and recorded in manifests and cache keys.  Shaping:

    reward_shape = eta * (discount * potential(next) - potential(current))

plus the separate large terminal reward issued only on exact Saito
certification.
"""

import json
import os

import numpy as np

from penalized_saito import (DEFAULT_LAMBDA, DEFAULT_BETA, PROFILES,
                             FUNCTIONAL_VERSION)

__all__ = ["calibrated_loss", "freeness_potential", "compute_tau",
           "calibration_key"]


def calibrated_loss(s: float, tau: float) -> float:
    """(1 + tau) * s / (s + tau); strictly increasing on [0, 1] with
    calibrated(0) = 0 and calibrated(1) = 1."""
    if tau <= 0:
        raise ValueError("tau must be positive")
    s = min(max(float(s), 0.0), 1.0)
    return (1.0 + tau) * s / (s + tau)


def freeness_potential(s: float, tau: float) -> float:
    return 1.0 - calibrated_loss(s, tau)


def calibration_key(n, d1, d2, lam=DEFAULT_LAMBDA, beta=DEFAULT_BETA,
                    field="real", profile="rl"):
    return (f"n{n}_d{d1}_{d2}_lam{lam}_beta{beta}_{field}_{profile}_"
            f"fv{FUNCTIONAL_VERSION}")


def compute_tau(n, d1, d2, lam=DEFAULT_LAMBDA, beta=DEFAULT_BETA,
                field="real", profile="rl", n_samples=24, seed=12345,
                coord_range=3, cache_path=None):
    """Median raw loss over a fixed generic-arrangement calibration set for
    the exact tuple.  Deterministic for fixed arguments; cached to JSON when
    cache_path is given."""
    key = calibration_key(n, d1, d2, lam, beta, field, profile)
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)
        if key in cache:
            return float(cache[key]["tau"])
    from swap_search import random_valid_seed
    from penalized_saito import penalized_saito_loss
    rng = np.random.default_rng(seed)
    losses = []
    while len(losses) < n_samples:
        try:
            arr = random_valid_seed(n, rng, coord_range=coord_range,
                                    nontrivial=(d1 >= 2))
        except RuntimeError:
            break
        try:
            losses.append(penalized_saito_loss(
                arr, d1, d2, lam=lam, beta=beta, profile=profile,
                seed=seed))
        except Exception:
            continue
    if not losses:
        raise RuntimeError("calibration set could not be built")
    tau = float(np.median(losses))
    tau = max(tau, 1e-6)          # guard against a degenerate zero median
    if cache_path:
        cache = {}
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                cache = json.load(f)
        cache[key] = {"tau": tau, "n_samples": len(losses),
                      "seed": seed, "coord_range": coord_range,
                      "losses_median": tau,
                      "losses_min": float(min(losses)),
                      "losses_max": float(max(losses))}
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=1)
    return tau
