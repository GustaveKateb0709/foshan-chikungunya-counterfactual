#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""foshan_r0_grid.py — admissible-transmissibility mapping.

Why
---
`foshan_r0_anchor.py` showed that the observed post-saturation city plateau
does NOT discriminate between an intrinsic R0 of 1.24 and 7.28: both reach a
post-window loss of ~0.175 with different compensating parameters, while
R0 = 16.3 fails catastrophically.  The counterfactual *magnitudes* moved by
orders of magnitude across those fits.

This script maps the admissible region properly:

  1. a finer R0 grid, 2 multi-starts each, same objective as
     `foshan_hv2.multistart` (post-window city anchors) plus one early anchor
     (Shunde 2025-07-15 = 478 reported), so the fits are comparable;
  2. for every fit, the five counterfactual arms in BOTH currencies:
        reported cases  (confounded by the detection surge through rho(t))
        true infections (ascertainment-free; isolates the epidemiological
                         effect of response timing)
     so we can tell whether the instability of the original "7.94x" is a
     transmission problem or an ascertainment artefact;
  3. growth rate and doubling time of every fit, to compare with the
     published early-phase estimates.

Outputs: results/r0_grid_scan_s<n_starts>_i<maxiter>.csv
(any console log is produced by the caller redirecting stdout, not by this
script).  The filename encodes n_starts and maxiter so that a short smoke
run cannot silently overwrite the full run.
Usage:   python foshan_r0_grid.py [n_starts] [maxiter]
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import foshan_hv2 as hv2                       # noqa: E402
from foshan_validation import ngm_r0, model_growth_rate   # noqa: E402
from foshan_r0_anchor import (ode_zone_seeded, beta_scale_for_r0,   # noqa: E402
                              scaled_theta, at_bound_flags,
                              bounds_record, fmt_seed, PARAM_BOUNDS)

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.abspath(os.path.join(BASE, "..")), "results")

R0_GRID = [1.24, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.28, 8.5, 10.0, 12.0, 16.3]
ANCHOR_0715_SHUNDE = 478.0
_ORIG_ODE = hv2.ode_zone


def make_loss(daily_dates, pfi, t_daily, n, detect_idx, norm_idx, theta_base):
    a_tC = np.array([(pd.Timestamp(d) - pd.Timestamp(hv2.T0)).days
                     for d, _ in hv2.ANCHORS_CITY])
    a_vC = np.array([v for _, v in hv2.ANCHORS_CITY])
    post = np.array([pd.Timestamp(d) >= pd.Timestamp("2025-08-16")
                     for d, _ in hv2.ANCHORS_CITY])
    i715 = int(np.argmax(daily_dates >= pd.Timestamp("2025-07-15")))

    def build_free(free):
        th = np.array(theta_base, dtype=float).copy()
        th[6], th[7], th[8], th[9] = free[0], free[1], free[2], free[3]
        th[10], th[11], th[12] = free[4], free[5], free[6]
        return th, float(np.exp(free[7]))

    def evaluate(free):
        th, seed = build_free(free)
        hv2.ode_zone = ode_zone_seeded(seed)
        try:
            r = hv2.run_two_zone(th, daily_dates, pfi, t_daily, n,
                                 detect_idx, norm_idx)
        except Exception:
            return None
        cumC = np.cumsum(r["onsets_A"] + r["onsets_B"])
        if not np.all(np.isfinite(cumC)):
            return None
        pred = cumC[np.clip(a_tC, 0, n - 1)]
        rel = (pred[post] - a_vC[post]) / np.maximum(a_vC[post], 1.0)
        loss_post = float(np.sum(rel ** 2))
        rho = r["r_base"] + (1 - r["r_base"]) * r["report_step"]
        pred_early = float(np.cumsum(rho * r["onsets_A"])[i715])
        loss_early = ((pred_early - ANCHOR_0715_SHUNDE)
                      / ANCHOR_0715_SHUNDE) ** 2
        return {"loss": loss_post + loss_early, "loss_post": loss_post,
                "loss_early": float(loss_early), "pred_early": pred_early,
                "r": r}

    def loss(free):
        out = evaluate(free)
        return 1e18 if out is None else out["loss"]
    return loss, evaluate, build_free


def counterfactuals(theta, seed, pfi, daily_dates, t_daily, n, detect_idx,
                    norm_idx):
    """Five arms -> (reported final, true-infection final)."""
    hv2.ode_zone = ode_zone_seeded(seed)
    p = hv2.unpack(theta)
    bi_none = pfi * np.exp(theta[4])
    trig = int(np.argmax(bi_none > 5.0))
    out = {}
    for arm, shift in (("none", 0), ("actual", 0), ("early", -14),
                       ("late", 14), ("rule", 0)):
        try:
            r = hv2.run_two_zone(theta, daily_dates, pfi, t_daily, n,
                                 detect_idx, norm_idx, shift_weeks=shift,
                                 arm=arm,
                                 detect_idx_override=(trig if arm == "rule"
                                                      else None))
        except Exception:
            out[arm] = (np.nan, np.nan)
            continue
        rho = r["r_base"] + (1 - r["r_base"]) * r["report_step"]
        tot = r["onsets_A"] + r["onsets_B"]
        out[arm] = (float(np.cumsum(rho * tot)[-1]), float(np.cumsum(tot)[-1]))
    hv2.ode_zone = _ORIG_ODE
    return out


def main(n_starts=2, maxiter=700):
    daily_dates, pfi, t_daily, n, detect_idx, norm_idx = hv2.build_daily()
    with open(os.path.join(RES, "foshan_hv2_multistart.json")) as f:
        base_thetas = [np.array(t) for t in json.load(f)["kept_thetas"]]

    rows = []
    for r0_target in R0_GRID:
        best = None
        for th_base in base_thetas[:n_starts]:
            s, _ = beta_scale_for_r0(th_base, r0_target)
            th_anchor = scaled_theta(th_base, s)
            loss, evaluate, build_free = make_loss(
                daily_dates, pfi, t_daily, n, detect_idx, norm_idx, th_anchor)
            free0 = np.array(list(th_anchor[6:13]) + [np.log(1e-4)])
            res = minimize_safe(loss, free0, maxiter)
            out = evaluate(res.x)
            if out is None:
                continue
            th_fit, seed = build_free(res.x)
            r0_chk, _ = ngm_r0(th_fit)          # confirm the anchor held
            g, td = model_growth_rate(th_fit)
            cf = counterfactuals(th_fit, seed, pfi, daily_dates, t_daily, n,
                                 detect_idx, norm_idx)
            pp = hv2.unpack(th_fit)
            vals = {**pp, "seed": seed}
            rep = {k: v[0] for k, v in cf.items()}
            tru = {k: v[1] for k, v in cf.items()}
            rec = {
                "R0_target": r0_target, "R0_check": round(r0_chk, 3),
                "loss_total": round(out["loss"], 4),
                "loss_post": round(out["loss_post"], 4),
                "loss_early": round(out["loss_early"], 4),
                "pred_repA_0715": round(out["pred_early"], 1),
                "growth_rate": round(g, 4) if np.isfinite(g) else None,
                "doubling_days": round(td, 2) if np.isfinite(td) else None,
                "cpa": round(pp["cpa"], 6), "cpb": round(pp["cpb"], 6),
                "s_min": round(pp["s_min"], 6), "r_base": round(pp["r_base"], 6),
                "r_off": round(pp["r_off"], 4), "esc_off": round(pp["esc_off"], 4),
                "stag_b": round(pp["stag_b"], 4),
                "seed": fmt_seed(seed),
                "bounds": bounds_record(PARAM_BOUNDS),
                **at_bound_flags(vals, PARAM_BOUNDS),
            }
            for tag, d in (("rep", rep), ("tru", tru)):
                for k in ("none", "actual", "early", "late", "rule"):
                    rec[f"{tag}_{k}"] = round(d[k]) if np.isfinite(d[k]) else None
                base = d["actual"] if d.get("actual") else np.nan
                for k in ("none", "early", "late", "rule"):
                    rec[f"{tag}_{k}_x"] = (round(d[k] / base, 3)
                                           if base and np.isfinite(base)
                                           and base > 0 else None)
            if best is None or rec["loss_total"] < best["loss_total"]:
                best = rec
        if best:
            rows.append(best)
            print(f"R0={r0_target:5.2f} chk={best['R0_check']:5.2f} "
                  f"loss={best['loss_total']:10.4f} "
                  f"(post {best['loss_post']:8.4f} / early {best['loss_early']:.4f}) "
                  f"r_base={best['r_base']:.2f} seed={best['seed']} "
                  f"n_at_bound={best['n_params_at_bound']} "
                  f"| TRUE early_x={best['tru_early_x']} late_x={best['tru_late_x']} "
                  f"none_x={best['tru_none_x']} rule_x={best['tru_rule_x']} "
                  f"| REP early_x={best['rep_early_x']} late_x={best['rep_late_x']} "
                  f"none_x={best['rep_none_x']}", flush=True)
            df_out = pd.DataFrame(rows)
            df_out.to_csv(
                os.path.join(RES, f"r0_grid_scan_s{n_starts}_i{maxiter}.csv"),
                index=False, encoding="utf-8-sig")
    hv2.ode_zone = _ORIG_ODE
    print(f"\nsaved r0_grid_scan_s{n_starts}_i{maxiter}.csv")
    return pd.DataFrame(rows)


def minimize_safe(loss, x0, maxiter):
    from scipy.optimize import minimize
    return minimize(loss, x0, method="Nelder-Mead",
                    options={"maxiter": maxiter, "xatol": 1e-3,
                             "fatol": 1e-6})


if __name__ == "__main__":
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    mi = int(sys.argv[2]) if len(sys.argv) > 2 else 700
    main(ns, mi)
