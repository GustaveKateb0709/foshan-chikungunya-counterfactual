#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""foshan_r0_ceiling.py — is the R0 ceiling real, or an optimiser artefact?

Context
-------
`foshan_r0_bounded.py` confined every free parameter to a physically defensible
interval and profiled the post-window fit against the intrinsic R0.  The result
was a sharp transition:

    R0 = 1.24 .. 3.00 -> loss_post 0.21 .. 1.79   (fitted)
    R0 = 4.00         -> loss_post 10.35          (degraded)
    R0 >= 5.00        -> loss_post 4.4e3 .. 4.8e6 (fitted not at all)

But at R0 >= 4 every parameter sat ON its bound (s_min = 0.2000, r_base = 0.0500,
seed = its lower limit) and the across-start loss spread exploded (R0 = 4:
88.8 vs 11.3; R0 = 5: 4.1e4 vs 4.4e3).  Sitting on a bound while a local
optimiser flails is exactly what a *failed search* looks like, not necessarily
what infeasibility looks like.

This script settles it with two independent devices:

  1. GLOBAL SEARCH.  Differential evolution (bounded, population-based) followed
     by a Nelder-Mead polish, at the decisive transmissibility values.  If a
     population-based global search also fails to reach an acceptable loss, the
     ceiling is a property of the constraint set rather than of the starting
     point.
  2. CONSTRAINT SENSITIVITY.  The same profile under three constraint sets
     (tight / baseline / loose).  A headline number of the form "the observed
     plateau supports R0 <= X" is only meaningful together with how much X moves
     when the constraints move.

Outputs: results/r0_ceiling_de<de_maxiter>_p<de_popsize>.csv
(any console log is produced by the caller redirecting stdout).
Usage:   python foshan_r0_ceiling.py [de_maxiter] [de_popsize]
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import foshan_hv2 as hv2                       # noqa: E402
import foshan_r0_bounded as B                  # noqa: E402
from foshan_validation import ngm_r0, model_growth_rate    # noqa: E402
from foshan_r0_anchor import (beta_scale_for_r0, scaled_theta,  # noqa: E402
                              _ORIG_ODE, at_bound_flags, bounds_record,
                              fmt_seed)

RES = B.RES

BOUND_SETS = {
    "baseline": [("cpa", 0.05, 0.95), ("cpb", 0.05, 0.95),
                 ("s_min", 0.20, 0.95), ("r_base", 0.05, 0.90),
                 ("r_off", 3.0, 25.0), ("esc_off", 5.2, 24.0),
                 ("stag_b", 0.2, 6.8), ("seed", 1e-6, 1e-2)],
    "loose": [("cpa", 0.01, 0.99), ("cpb", 0.01, 0.99),
              ("s_min", 0.05, 0.95), ("r_base", 0.01, 0.95),
              ("r_off", 1.0, 30.0), ("esc_off", 2.0, 28.0),
              ("stag_b", 0.05, 6.95), ("seed", 1e-10, 1e-2)],
    "tight": [("cpa", 0.10, 0.90), ("cpb", 0.10, 0.90),
              ("s_min", 0.35, 0.95), ("r_base", 0.10, 0.90),
              ("r_off", 5.0, 20.0), ("esc_off", 6.0, 20.0),
              ("stag_b", 0.5, 6.0), ("seed", 1e-4, 1e-2)],
}

AGENDA = [("baseline", [3.5, 4.0, 4.5]),
          ("loose", [3.5, 4.0, 4.5]),
          ("tight", [2.0, 2.5])]

DE_MAXITER = 35
DE_POPSIZE = 8
Z_LIM = 14.0


def run_one(bound_set, r0_target, th_base, daily_dates, pfi, t_daily, n,
            detect_idx, norm_idx, tag=""):
    B.BOUNDS = BOUND_SETS[bound_set]
    s, _ = beta_scale_for_r0(th_base, r0_target)
    th_anchor = scaled_theta(th_base, s)
    obj, evaluate = B.make_objective(daily_dates, pfi, t_daily, n,
                                     detect_idx, norm_idx, th_anchor)
    z_bounds = [(-Z_LIM, Z_LIM)] * len(B.BOUNDS)
    de = differential_evolution(obj, z_bounds, seed=20260915,
                                maxiter=DE_MAXITER, popsize=DE_POPSIZE,
                                tol=1e-8, polish=False,
                                mutation=(0.4, 1.0), recombination=0.7,
                                init="sobol", disp=False)
    lm = minimize(obj, de.x, method="Nelder-Mead",
                  options={"maxiter": 1800, "xatol": 1e-4, "fatol": 1e-8})
    out = evaluate(lm.x)
    if out is None:
        return None
    th_fit, seed, v = B.to_theta(th_anchor, lm.x)
    r0_chk, _ = ngm_r0(th_fit)
    g, td = model_growth_rate(th_fit)
    cf = B.counterfactuals(th_fit, seed, pfi, daily_dates, t_daily, n,
                           detect_idx, norm_idx)
    rec = {"bound_set": bound_set, "R0_target": r0_target,
           "R0_check": round(r0_chk, 3),
           "loss_total": round(out["loss"], 4),
           "loss_post": round(out["loss_post"], 4),
           "loss_early": round(out["loss_early"], 4),
           "de_loss": round(float(de.fun), 4),
           "growth_rate": round(g, 4) if np.isfinite(g) else None,
           "doubling_days": round(td, 2) if np.isfinite(td) else None,
           **{k: (fmt_seed(v[k]) if k == "seed" else round(v[k], 6))
              for k, _, _ in B.BOUNDS},
           "bounds": bounds_record(B.BOUNDS),
           **at_bound_flags(v, B.BOUNDS)}
    for k in ("none", "actual", "early", "late", "rule"):
        rec[f"tru_{k}"] = round(cf[k][1]) if np.isfinite(cf[k][1]) else None
    b_ = cf["actual"][1]
    for k in ("none", "early", "late", "rule"):
        rec[f"tru_{k}_x"] = (round(cf[k][1] / b_, 3)
                             if b_ and np.isfinite(b_) and b_ > 0 else None)
    return rec


def main(de_maxiter=None, de_popsize=None):
    global DE_MAXITER, DE_POPSIZE
    if de_maxiter:
        DE_MAXITER = int(de_maxiter)
    if de_popsize:
        DE_POPSIZE = int(de_popsize)

    daily_dates, pfi, t_daily, n, detect_idx, norm_idx = hv2.build_daily()
    with open(os.path.join(RES, "foshan_hv2_multistart.json")) as f:
        base_thetas = [np.array(t) for t in json.load(f)["kept_thetas"]]

    rows = []
    for bound_set, r0_list in AGENDA:
        for r0_target in r0_list:
            rec = run_one(bound_set, r0_target, base_thetas[0], daily_dates,
                          pfi, t_daily, n, detect_idx, norm_idx)
            if rec is None:
                print(f"{bound_set:9s} R0={r0_target:5.2f}  -> no solution",
                      flush=True)
                continue
            rows.append(rec)
            print(f"{bound_set:9s} R0={rec['R0_target']:5.2f} "
                  f"chk={rec['R0_check']:5.2f}  DE={rec['de_loss']:10.3f} -> "
                  f"NM loss_post={rec['loss_post']:10.4f}  "
                  f"s_min={rec['s_min']:.3f} r_base={rec['r_base']:.3f} "
                  f"seed={rec['seed']} n_at_bound={rec['n_params_at_bound']} | "
                  f"early_x={rec['tru_early_x']} "
                  f"late_x={rec['tru_late_x']}", flush=True)
            pd.DataFrame(rows).to_csv(
                os.path.join(RES,
                             f"r0_ceiling_de{DE_MAXITER}_p{DE_POPSIZE}.csv"),
                index=False, encoding="utf-8-sig")
    hv2.ode_zone = _ORIG_ODE
    print(f"\nsaved r0_ceiling_de{DE_MAXITER}_p{DE_POPSIZE}.csv")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else None
    b = sys.argv[2] if len(sys.argv) > 2 else None
    main(a, b)
