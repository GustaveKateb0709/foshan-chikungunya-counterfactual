#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""foshan_r0_bounded.py — physically bounded re-fit across the R0 grid.

Motivation
----------
The unbounded refits (`foshan_r0_anchor.py`, `foshan_r0_grid.py`) showed that
the post-saturation plateau can be fitted at essentially any intrinsic R0 up
to >=10, but only by driving three parameters to numerical boundaries:

    R0 =  1.24 ->  s_min = 0.578  r_base = 0.097  seed = 2.4e-04
    R0 =  7.28 ->  s_min = 0.007  r_base = 0.176  seed = 6.0e-32
    R0 = 10.00 ->  s_min = 0.002  r_base = 0.000  seed = 3.8e-175

`s_min` is the residual susceptibility under isolation, so s_min -> 0.007 means
a claimed 99.3% isolation efficacy; `r_base` -> 0 is a zero baseline detection
rate; `seed` is the undisclosed initial vector infection fraction (hard-coded
at 5e-4 in `foshan_hv2.py:141`).  Those are not credible parameterisations, so
"the plateau also supports R0 = 7.28" is not a safe statement.

This script asks the quantitative question instead: **with every free
parameter confined to a physically defensible interval, what range of R0 can
the observed plateau still support?**  Parameters are fitted in an unconstrained
z-space and mapped through a logistic transform into the bounds, so the
optimiser can never reach the boundaries.  Each R0 target is fitted from three
structurally different starting points and the across-start spread is recorded,
because a wide spread is itself evidence of non-identification.

Outputs: results/r0_bounded_scan_i<maxiter>[_<tag>].csv,
         results/r0_bounded_starts_i<maxiter>[_<tag>].csv
Usage:   python foshan_r0_bounded.py [maxiter] [tag]
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import foshan_hv2 as hv2                       # noqa: E402
from foshan_validation import ngm_r0, model_growth_rate    # noqa: E402
from foshan_r0_anchor import (ode_zone_seeded, beta_scale_for_r0,   # noqa: E402
                              scaled_theta, at_bound_flags,
                              bounds_record, fmt_seed)

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.abspath(os.path.join(BASE, "..")), "results")

R0_GRID = [1.24, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.28, 8.5, 10.0, 12.0, 16.3]
ANCHOR_0715_SHUNDE = 478.0
_ORIG_ODE = hv2.ode_zone

# --- defensible parameter intervals -----------------------------------------
# seed: at least ~1 infected vector in the seeded vector pool (N_m ~ 2e5 ->
#       1 vector is ~5e-6) up to 1 % of the vector population.
# s_min: residual susceptibility under isolation; 0.20 = an already optimistic
#        80 % isolation efficacy, 0.95 = near-useless isolation.
# r_base: baseline detection rate; 0.05 = 95 % of infections missed,
#        0.90 = near-complete baseline ascertainment.
# cpa/cpb: peak vector-control coverage; 0.05 .. 0.95.
# r_off / esc_off / stag_b: operational delays, in days.
BOUNDS = [("cpa", 0.05, 0.95), ("cpb", 0.05, 0.95),
          ("s_min", 0.20, 0.95), ("r_base", 0.05, 0.90),
          ("r_off", 3.0, 25.0), ("esc_off", 5.2, 24.0),
          ("stag_b", 0.2, 6.8), ("seed", 1e-6, 1e-2)]
LOG_KEYS = {"seed"}

# three structurally different starts (original-ish / mid / low-ascertainment)
STARTS = [
    {"cpa": 0.774, "cpb": 0.307, "s_min": 0.398, "r_base": 0.852,
     "r_off": 9.25, "esc_off": 13.0, "stag_b": 2.0, "seed": 5.0e-4},
    {"cpa": 0.500, "cpb": 0.300, "s_min": 0.600, "r_base": 0.300,
     "r_off": 12.0, "esc_off": 10.0, "stag_b": 2.0, "seed": 5.0e-5},
    {"cpa": 0.300, "cpb": 0.300, "s_min": 0.800, "r_base": 0.100,
     "r_off": 15.0, "esc_off": 8.0, "stag_b": 3.0, "seed": 5.0e-5},
]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def unbound(z, key, lo, hi):
    s = sigmoid(z)
    if key in LOG_KEYS:
        return float(np.exp(np.log(lo) + (np.log(hi) - np.log(lo)) * s))
    return float(lo + (hi - lo) * s)


def z0_for(vals):
    z = []
    for k, lo, hi in BOUNDS:
        x = min(max(vals[k], lo * 1.0001), hi * 0.9999)
        if k in LOG_KEYS:
            t = (np.log(x) - np.log(lo)) / (np.log(hi) - np.log(lo))
        else:
            t = (x - lo) / (hi - lo)
        t = min(max(t, 1e-3), 1 - 1e-3)
        z.append(np.log(t / (1 - t)))
    return np.array(z)


def to_theta(theta_base, z):
    v = {}
    for i, (k, lo, hi) in enumerate(BOUNDS):
        v[k] = unbound(z[i], k, lo, hi)
    th = np.array(theta_base, dtype=float).copy()
    th[6] = np.log(v["cpa"] / (1 - v["cpa"]))
    th[7] = np.log(v["cpb"] / (1 - v["cpb"]))
    th[8] = np.log(1.0 / v["s_min"] - 1.0)
    th[9] = np.log(1.0 / v["r_base"] - 1.0)
    th[10] = -np.log(25.0 / v["r_off"] - 1.0)
    th[11] = np.log((v["esc_off"] - 5.0) / (25.0 - v["esc_off"]))
    th[12] = -np.log(7.0 / v["stag_b"] - 1.0)
    return th, v["seed"], v


def make_objective(daily_dates, pfi, t_daily, n, detect_idx, norm_idx,
                   theta_base):
    a_tC = np.array([(pd.Timestamp(d) - pd.Timestamp(hv2.T0)).days
                     for d, _ in hv2.ANCHORS_CITY])
    a_vC = np.array([v for _, v in hv2.ANCHORS_CITY])
    post = np.array([pd.Timestamp(d) >= pd.Timestamp("2025-08-16")
                     for d, _ in hv2.ANCHORS_CITY])
    i715 = int(np.argmax(daily_dates >= pd.Timestamp("2025-07-15")))

    def evaluate(z):
        th, seed, _ = to_theta(theta_base, z)
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
                "loss_early": float(loss_early), "pred_early": pred_early}

    def obj(z):
        out = evaluate(z)
        return 1e18 if out is None else out["loss"]
    return obj, evaluate


def counterfactuals(theta, seed, pfi, daily_dates, t_daily, n, detect_idx,
                    norm_idx):
    """Five arms -> (reported final, true-infection final)."""
    hv2.ode_zone = ode_zone_seeded(seed)
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
            out[arm] = (np.nan, np.nan, np.nan)
            continue
        rho = r["r_base"] + (1 - r["r_base"]) * r["report_step"]
        tot = r["onsets_A"] + r["onsets_B"]
        peak = float(np.max(tot))
        out[arm] = (float(np.cumsum(rho * tot)[-1]), float(np.cumsum(tot)[-1]),
                    peak)
    hv2.ode_zone = _ORIG_ODE
    return out


def main(maxiter=900, tag=""):
    daily_dates, pfi, t_daily, n, detect_idx, norm_idx = hv2.build_daily()
    with open(os.path.join(RES, "foshan_hv2_multistart.json")) as f:
        base_thetas = [np.array(t) for t in json.load(f)["kept_thetas"]]

    rows, start_rows = [], []
    suffix = f"_{tag}" if tag else ""
    scan_name = f"r0_bounded_scan_i{maxiter}{suffix}.csv"
    starts_name = f"r0_bounded_starts_i{maxiter}{suffix}.csv"
    for r0_target in R0_GRID:
        solutions = []
        for si, s_vals in enumerate(STARTS):
            th_base = base_thetas[min(si, len(base_thetas) - 1)]
            s, _ = beta_scale_for_r0(th_base, r0_target)
            th_anchor = scaled_theta(th_base, s)
            obj, evaluate = make_objective(
                daily_dates, pfi, t_daily, n, detect_idx, norm_idx, th_anchor)
            z0 = z0_for(s_vals)
            res = minimize(obj, z0, method="Nelder-Mead",
                           options={"maxiter": maxiter, "xatol": 1e-3,
                                    "fatol": 1e-6})
            out = evaluate(res.x)
            if out is None:
                continue
            th_fit, seed, v = to_theta(th_anchor, res.x)
            r0_chk, _ = ngm_r0(th_fit)
            g, td = model_growth_rate(th_fit)
            cf = counterfactuals(th_fit, seed, pfi, daily_dates, t_daily, n,
                                 detect_idx, norm_idx)
            rep = {k: x[0] for k, x in cf.items()}
            tru = {k: x[1] for k, x in cf.items()}
            pk = {k: x[2] for k, x in cf.items()}
            rec = {"R0_target": r0_target, "start": si, "R0_check": round(r0_chk, 3),
                   "loss_total": round(out["loss"], 4),
                   "loss_post": round(out["loss_post"], 4),
                   "loss_early": round(out["loss_early"], 4),
                   "pred_repA_0715": round(out["pred_early"], 1),
                   "growth_rate": round(g, 4) if np.isfinite(g) else None,
                   "doubling_days": round(td, 2) if np.isfinite(td) else None,
                   **{k: (fmt_seed(v[k]) if k == "seed" else round(v[k], 6))
                      for k, _, _ in BOUNDS},
                   "bounds": bounds_record(BOUNDS),
                   **at_bound_flags(v, BOUNDS)}
            for t_, d in (("rep", rep), ("tru", tru), ("peak", pk)):
                for k in ("none", "actual", "early", "late", "rule"):
                    rec[f"{t_}_{k}"] = round(d[k]) if np.isfinite(d[k]) else None
                b = d.get("actual")
                for k in ("none", "early", "late", "rule"):
                    rec[f"{t_}_{k}_x"] = (round(d[k] / b, 3)
                                          if b and np.isfinite(b) and b > 0
                                          else None)
            solutions.append(rec)
            start_rows.append(rec)
        if solutions:
            solutions.sort(key=lambda r: r["loss_total"])
            best = dict(solutions[0])
            best["n_starts"] = len(solutions)
            best["loss_range"] = (f"{solutions[-1]['loss_total']:.4f}"
                                  f"..{solutions[0]['loss_total']:.4f}")
            for k in ("tru_early_x", "tru_late_x", "tru_none_x", "tru_rule_x",
                      "r_base", "s_min"):
                vals = [r[k] for r in solutions if r[k] is not None]
                if vals:
                    best[f"{k}_min"] = min(vals)
                    best[f"{k}_max"] = max(vals)
            rows.append(best)
            print(f"R0={r0_target:5.2f} chk={best['R0_check']:5.2f} "
                  f"loss={best['loss_total']:9.4f} "
                  f"(post {best['loss_post']:8.4f} / early {best['loss_early']:.4f}) "
                  f"[{best['n_starts']} starts, range {best['loss_range']}] "
                  f"r_base={best['r_base']:.3f} s_min={best['s_min']:.3f} "
                  f"seed={best['seed']} n_at_bound={best['n_params_at_bound']} | "
                  f"TRUE none_x={best['tru_none_x']} "
                  f"early_x={best['tru_early_x']} late_x={best['tru_late_x']} "
                  f"rule_x={best['tru_rule_x']}", flush=True)
            pd.DataFrame(rows).to_csv(
                os.path.join(RES, scan_name), index=False,
                encoding="utf-8-sig")
            pd.DataFrame(start_rows).to_csv(
                os.path.join(RES, starts_name), index=False,
                encoding="utf-8-sig")
    hv2.ode_zone = _ORIG_ODE
    print(f"\nsaved {scan_name} / {starts_name}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    mi = int(sys.argv[1]) if len(sys.argv) > 1 else 900
    tg = sys.argv[2] if len(sys.argv) > 2 else ""
    main(mi, tg)
