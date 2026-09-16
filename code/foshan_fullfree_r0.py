#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""foshan_fullfree_r0.py — the decisive test: with EVERYTHING free but R0 pinned,
can the observed plateau still be reproduced?

Why this is needed
------------------
Every earlier scan freed only eight parameters (the response-coverage,
ascertainment, delay and seeding group) and held the other five — the initial
human caseload i0, the zone-A population share s_a, and the two zone density
calibrations calib_a / calib_b — at their calibrated values.  So statements
like "R0 = 16.3 cannot reproduce the plateau" were only true inside that
eight-dimensional subspace, and a referee would be right to say the remaining
five were never given a chance to compensate.

Here all twelve substantive parameters are free (the two betas are not free by
construction: for every candidate vector the beta scale is re-solved so that the
next-generation R0 equals the target exactly).  The search intervals are
deliberately WIDE — wider than the "loose" set used elsewhere — so the test asks
the strongest available version of the question:

    with generously wide but finite parameters, is any R0 excluded by the data?

If R0 = 16.3 still cannot be fitted here, the plateau genuinely bounds the
intrinsic transmissibility from above within this model structure.  If it can,
then the surveillance data do not identify R0 at all.

Outputs: results/fullfree_r0_i{maxiter}.csv, results/fullfree_r0_starts_i{maxiter}.csv
Usage:   python foshan_fullfree_r0.py [maxiter] [n_starts]
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
from foshan_validation import ngm_r0           # noqa: E402
from foshan_r0_anchor import (ode_zone_seeded, scaled_theta, _ORIG_ODE)  # noqa: E402

RES = B.RES
R0_GRID = [1.24, 3.0, 7.28, 16.3]
ANCHOR_0715_SHUNDE = 478.0

# name, transform, lo, hi   (transform: "lin" -> logistic, "log" -> log-scale)
SPEC = [
    ("i0",      "log", 0.05, 4.0e2),
    ("s_a",     "lin", 0.05, 0.95),
    ("calib_a", "log", 0.05, 5.0),
    ("calib_b", "log", 0.05, 5.0),
    ("cpa",     "lin", 0.01, 0.99),
    ("cpb",     "lin", 0.01, 0.99),
    ("s_min",   "lin", 0.02, 0.99),
    ("r_base",  "lin", 0.01, 0.99),
    ("r_off",   "lin", 0.50, 25.0),
    ("esc_off", "lin", 5.05, 24.9),
    ("stag_b",  "lin", 0.05, 6.99),
    ("seed",    "log", 1e-8, 1e-2),
]

BETA0 = (np.log(0.78), np.log(0.54))     # unchanging base for the two betas


def val(z, kind, lo, hi):
    s = B.sigmoid(z)
    return float(np.exp(np.log(lo) + (np.log(hi) - np.log(lo)) * s)) \
        if kind == "log" else float(lo + (hi - lo) * s)


def z0_of(vals):
    z = []
    for (k, kind, lo, hi) in SPEC:
        x = min(max(vals[k], lo * 1.001), hi * 0.999)
        t = ((np.log(x) - np.log(lo)) / (np.log(hi) - np.log(lo))
             if kind == "log" else (x - lo) / (hi - lo))
        t = min(max(t, 1e-3), 1 - 1e-3)
        z.append(np.log(t / (1 - t)))
    return np.array(z)


def theta_from_z(z):
    v = {k: val(z[i], kind, lo, hi) for i, (k, kind, lo, hi) in enumerate(SPEC)}
    th = np.zeros(13)
    th[2] = np.log(v["i0"])
    th[3] = np.log(v["s_a"] / (1 - v["s_a"]))
    th[4] = np.log(v["calib_a"])
    th[5] = np.log(v["calib_b"])
    th[6] = np.log(v["cpa"] / (1 - v["cpa"]))
    th[7] = np.log(v["cpb"] / (1 - v["cpb"]))
    th[8] = np.log(1.0 / v["s_min"] - 1.0)
    th[9] = np.log(1.0 / v["r_base"] - 1.0)
    th[10] = -np.log(25.0 / v["r_off"] - 1.0)
    th[11] = np.log((v["esc_off"] - 5.0) / (25.0 - v["esc_off"]))
    th[12] = -np.log(7.0 / v["stag_b"] - 1.0)
    return th, v


def solve_beta_scale(th, r0_target, date="2025-07-15", tol=1e-7, maxit=80):
    """s such that scaling both betas of `th` gives NGM R0 = r0_target."""
    th0 = np.array(th, dtype=float)
    th0[0], th0[1] = BETA0
    r0_now, _ = ngm_r0(th0, date=date)
    if not np.isfinite(r0_now) or r0_now <= 0:
        return None
    s = float(r0_target / r0_now)
    for _ in range(maxit):
        r0_s, _ = ngm_r0(scaled_theta(th0, s), date=date)
        if not np.isfinite(r0_s) or r0_s <= 0:
            return None
        if abs(r0_s - r0_target) <= tol * max(1.0, r0_target):
            break
        s *= r0_target / r0_s
    return float(s)


def make_objective(daily_dates, pfi, t_daily, n, detect_idx, norm_idx,
                   r0_target):
    a_tC = np.array([(pd.Timestamp(d) - pd.Timestamp(hv2.T0)).days
                     for d, _ in hv2.ANCHORS_CITY])
    a_vC = np.array([v for _, v in hv2.ANCHORS_CITY])
    post = np.array([pd.Timestamp(d) >= pd.Timestamp("2025-08-16")
                     for d, _ in hv2.ANCHORS_CITY])
    i715 = int(np.argmax(daily_dates >= pd.Timestamp("2025-07-15")))

    def evaluate(z):
        th, v = theta_from_z(z)
        s = solve_beta_scale(th, r0_target)
        if s is None:
            return None
        th[0] = BETA0[0] + np.log(s)
        th[1] = BETA0[1] + np.log(s)
        hv2.ode_zone = ode_zone_seeded(v["seed"])
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
                "loss_early": float(loss_early), "th": th, "v": v,
                "scale": s}

    def obj(z):
        out = evaluate(z)
        return 1e18 if out is None else out["loss"]
    return obj, evaluate


def main(maxiter=1500, n_starts=2):
    daily_dates, pfi, t_daily, n, detect_idx, norm_idx = hv2.build_daily()
    with open(os.path.join(RES, "foshan_hv2_params.json")) as f:
        pj = json.load(f)
    p = hv2.unpack(np.array(list(pj["theta"]) + [-0.4055, -0.9229]))
    central = {"i0": p["i0"], "s_a": p["s_a"], "calib_a": p["calib_a"],
               "calib_b": p["calib_b"], "cpa": p["cpa"], "cpb": p["cpb"],
               "s_min": p["s_min"], "r_base": p["r_base"], "r_off": p["r_off"],
               "esc_off": 13.0, "stag_b": 2.0, "seed": 5.0e-4}
    alt = dict(central, s_min=0.6, r_base=0.25, cpa=0.4, cpb=0.4,
               seed=5.0e-5, calib_a=central["calib_a"], r_off=12.0)

    rows, start_rows = [], []
    for r0_target in R0_GRID:
        obj, evaluate = make_objective(daily_dates, pfi, t_daily, n,
                                       detect_idx, norm_idx, r0_target)
        best = None
        for si, sv in enumerate([central, alt][:n_starts]):
            res = minimize(obj, z0_of(sv), method="Nelder-Mead",
                           options={"maxiter": maxiter, "xatol": 1e-3,
                                    "fatol": 1e-6})
            out = evaluate(res.x)
            if out is None:
                continue
            th = out["th"]
            r0_chk, _ = ngm_r0(th)
            cf = B.counterfactuals(th, out["v"]["seed"], pfi, daily_dates,
                                   t_daily, n, detect_idx, norm_idx)
            rec = {"R0_target": r0_target, "start": si,
                   "R0_check": round(r0_chk, 3),
                   "loss_total": round(out["loss"], 4),
                   "loss_post": round(out["loss_post"], 4),
                   "loss_early": round(out["loss_early"], 4),
                   "beta_scale": round(out["scale"], 4),
                   **{k: out["v"][k] for k in out["v"]}}
            for k in ("none", "actual", "early", "late", "rule"):
                rec[f"tru_{k}"] = (round(cf[k][1])
                                   if np.isfinite(cf[k][1]) else None)
            b_ = cf["actual"][1]
            for k in ("none", "early", "late", "rule"):
                rec[f"tru_{k}_x"] = (round(cf[k][1] / b_, 3)
                                     if b_ and np.isfinite(b_) and b_ > 0
                                     else None)
            start_rows.append(rec)
            if best is None or rec["loss_post"] < best["loss_post"]:
                best = rec
        if best:
            rows.append(best)
            print(f"R0={r0_target:5.2f} chk={best['R0_check']:5.2f} "
                  f"loss_post={best['loss_post']:12.4f} "
                  f"(total {best['loss_total']:12.4f}) "
                  f"i0={best['i0']:.2f} s_a={best['s_a']:.3f} "
                  f"calib_a={best['calib_a']:.3f} calib_b={best['calib_b']:.3f} "
                  f"s_min={best['s_min']:.4f} r_base={best['r_base']:.4f} "
                  f"seed={best['seed']:.2e} | early_x={best['tru_early_x']} "
                  f"late_x={best['tru_late_x']}", flush=True)
            pd.DataFrame(rows).to_csv(
                os.path.join(RES, f"fullfree_r0_i{maxiter}.csv"), index=False,
                encoding="utf-8-sig")
            pd.DataFrame(start_rows).to_csv(
                os.path.join(RES, f"fullfree_r0_starts_i{maxiter}.csv"),
                index=False, encoding="utf-8-sig")
    hv2.ode_zone = _ORIG_ODE
    print(f"\nsaved fullfree_r0_i{maxiter}.csv")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    mi = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    ns = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    main(mi, ns)
