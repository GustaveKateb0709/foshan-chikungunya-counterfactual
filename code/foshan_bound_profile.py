#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""foshan_bound_profile.py — the centrepiece evidence for the identifiability paper.

Finding that motivates it
-------------------------
`foshan_r0_ceiling.py` showed that the apparent ceiling on the intrinsic R0 is
NOT a property of the data.  Restricting the isolation efficacy to
s_min >= 0.20 (baseline) pushed the ceiling to about R0 = 3.5; loosening it to
s_min >= 0.05 (loose) let R0 = 4.5 fit BETTER than R0 = 3.0 did under the
baseline set (loss_post 0.502 vs 1.793).  At every R0 target above the lowest
one the optimiser drove s_min onto that constraint set's floor; at the lowest
target (R0 = 1.24) s_min is still well above it (0.43 tight / 0.62 baseline /
0.55 loose), so the floor binds as R0 rises rather than in every single fit.

So the binding object is the s_min <-> R0 trade-off: the plateau can be
explained by (low R0 + weak isolation) or by (high R0 + strong isolation), and
the surveillance data cannot tell the two apart.  Any "R0 <= X" statement is
therefore a restatement of the assumed parameter bounds, not a measurement.

This script produces the profile that shows it: for each constraint set, sweep
R0 and record the best achievable fit together with which parameters sit on
their bounds and how far apart the multi-start solutions are.

Outputs: results/bound_profile_i{maxiter}.csv, results/bound_profile_starts_i{maxiter}.csv
Usage:   python foshan_bound_profile.py [maxiter]
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import foshan_hv2 as hv2                       # noqa: E402
import foshan_r0_bounded as B                  # noqa: E402
from foshan_r0_ceiling import BOUND_SETS       # noqa: E402
from foshan_validation import ngm_r0           # noqa: E402
from foshan_r0_anchor import (beta_scale_for_r0, scaled_theta,   # noqa: E402
                              _ORIG_ODE)

RES = B.RES

AGENDA = [
    ("tight",    [1.24, 1.5, 2.0, 2.5, 3.0]),
    ("baseline", [1.24, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]),
    ("loose",    [1.24, 2.0, 3.0, 4.0, 4.5, 5.0, 6.0]),
]


def main(maxiter=600):
    daily_dates, pfi, t_daily, n, detect_idx, norm_idx = hv2.build_daily()
    with open(os.path.join(RES, "foshan_hv2_multistart.json")) as f:
        base_thetas = [np.array(t) for t in json.load(f)["kept_thetas"]]

    rows, start_rows = [], []
    for set_name, r0_list in AGENDA:
        B.BOUNDS = BOUND_SETS[set_name]
        for r0_target in r0_list:
            sols = []
            for si, s_vals in enumerate(B.STARTS):
                th_base = base_thetas[min(si, len(base_thetas) - 1)]
                s, _ = beta_scale_for_r0(th_base, r0_target)
                th_anchor = scaled_theta(th_base, s)
                obj, evaluate = B.make_objective(
                    daily_dates, pfi, t_daily, n, detect_idx, norm_idx,
                    th_anchor)
                z0 = B.z0_for(s_vals)
                res = minimize(obj, z0, method="Nelder-Mead",
                               options={"maxiter": maxiter, "xatol": 1e-3,
                                        "fatol": 1e-6})
                out = evaluate(res.x)
                if out is None:
                    continue
                th_fit, seed, v = B.to_theta(th_anchor, res.x)
                r0_chk, _ = ngm_r0(th_fit)
                cf = B.counterfactuals(th_fit, seed, pfi, daily_dates,
                                       t_daily, n, detect_idx, norm_idx)
                rec = {"bound_set": set_name, "R0_target": r0_target,
                       "start": si, "R0_check": round(r0_chk, 3),
                       "loss_total": round(out["loss"], 4),
                       "loss_post": round(out["loss_post"], 4),
                       "loss_early": round(out["loss_early"], 4),
                       "seed_val": float(f"{seed:.3e}"),
                       **{k: v[k] for k, _, _ in B.BOUNDS},
                       **B.at_bound_flags(v, B.BOUNDS)}
                for k in ("none", "actual", "early", "late", "rule"):
                    rec[f"tru_{k}"] = (round(cf[k][1])
                                       if np.isfinite(cf[k][1]) else None)
                b_ = cf["actual"][1]
                for k in ("none", "early", "late", "rule"):
                    rec[f"tru_{k}_x"] = (round(cf[k][1] / b_, 3)
                                         if b_ and np.isfinite(b_) and b_ > 0
                                         else None)
                sols.append(rec)
                start_rows.append(rec)
            if not sols:
                continue
            sols.sort(key=lambda r: r["loss_post"])
            best = dict(sols[0])
            best["n_starts"] = len(sols)
            best["loss_post_range"] = (
                f"{sols[-1]['loss_post']:.4f}..{sols[0]['loss_post']:.4f}")
            rows.append(best)
            print(f"{set_name:9s} R0={r0_target:5.2f} chk={best['R0_check']:5.2f} "
                  f"loss_post={best['loss_post']:9.4f} "
                  f"at_bound={best['n_params_at_bound']} "
                  f"s_min={best['s_min']:.4f}{'*' if best.get('s_min_at_bound') else ' '} "
                  f"r_base={best['r_base']:.4f}{'*' if best.get('r_base_at_bound') else ' '} "
                  f"seed={best['seed_val']} | range {best['loss_post_range']} "
                  f"| early_x={best['tru_early_x']} late_x={best['tru_late_x']}",
                  flush=True)
            pd.DataFrame(rows).to_csv(
                os.path.join(RES, f"bound_profile_i{maxiter}.csv"), index=False,
                encoding="utf-8-sig")
            pd.DataFrame(start_rows).to_csv(
                os.path.join(RES, f"bound_profile_starts_i{maxiter}.csv"),
                index=False, encoding="utf-8-sig")
    hv2.ode_zone = _ORIG_ODE
    print(f"\nsaved bound_profile_i{maxiter}.csv")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 600)
