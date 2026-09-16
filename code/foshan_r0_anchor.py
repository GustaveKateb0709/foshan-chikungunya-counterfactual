#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""foshan_r0_anchor.py — exploratory external-anchor experiment.

Question addressed
------------------
The paper's calibration used only post-saturation reported anchors and never
constrained (or reported) the model's intrinsic transmissibility. Two
independent published estimates for the same outbreak give R0 = 7.28
(Zhao 2025, Infect Dis Poverty 14:106) and R0 = 16.3 (Zhang 2025, Infect Dis
Poverty, doi 10.1186/s40249-025-01364-y), while the original fit implies
R0 ~ 1.2. This script asks what happens when the model is re-fitted under an
*externally anchored* transmissibility:

  for each R0 target in a grid,
      scale beta_mp and beta_pm so that the model's next-generation R0 at
      pre-response July conditions equals the target,
      re-fit the response / ascertainment / seeding parameters against
      (a) the 13 post-saturation city anchors and
      (b) the 2025-07-15 Shunde early anchor (478 reported),
      record the best achievable loss, the required control coverage and the
      five counterfactual final sizes.

The loss is the test: if the observed plateau cannot be reproduced at
R0 >= 7.28 with any admissible parameters, the published early-phase
estimates are incompatible with the observed final size.

Status: EXPLORATORY.  This script motivated the later, bounded main
experiment (foshan_r0_bounded.py); its own output is retained for completeness.  Its helper functions
(ode_zone_seeded, beta_scale_for_r0, scaled_theta) ARE reused by
foshan_r0_grid / foshan_r0_bounded / foshan_r0_ceiling and must not be
removed.

Outputs: results/r0_anchor_experiment_s<n_starts>_i<maxiter>.csv
(console log, if wanted, is produced by the caller redirecting stdout).
Usage:  python foshan_r0_anchor.py [n_starts] [maxiter]
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import foshan_hv2 as hv2                       # noqa: E402
from foshan_validation import ngm_r0           # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.abspath(os.path.join(BASE, "..")), "results")

R0_GRID = [1.24, 3.0, 7.28, 16.3]
PUBLISHED = {"zhao_2025": 7.28, "zhang_2025": 16.3}
ANCHOR_0715_SHUNDE = 478.0          # cumulative reported, Shunde, 2025-07-15
LAMBDA_EARLY = 1.0                  # weight of the single early anchor

_ORIG_ODE = hv2.ode_zone

# --- shared bound bookkeeping (imported by grid / bounded / ceiling) --------
# Physical reference intervals (identical to foshan_r0_bounded.BOUNDS).
PARAM_BOUNDS = [("cpa", 0.05, 0.95), ("cpb", 0.05, 0.95),
                ("s_min", 0.20, 0.95), ("r_base", 0.05, 0.90),
                ("r_off", 3.0, 25.0), ("esc_off", 5.2, 24.0),
                ("stag_b", 0.2, 6.8), ("seed", 1e-6, 1e-2)]
BOUND_REL_TOL = 0.01


def fmt_seed(seed):
    """Seed as a scientific-notation string so 1e-6 stays readable (instead
    of being rounded to 0.0 by a fixed-decimal formatter)."""
    return f"{float(seed):.3e}"


def at_bound_flags(vals, bounds=PARAM_BOUNDS, tol=BOUND_REL_TOL):
    """Map param -> 'pinned to a bound' boolean (plus n_params_at_bound).

    A parameter counts as at-bound when it is within `tol` (1 %) of its
    lower or upper bound, or strictly outside the interval.  The 'outside'
    case covers the unbounded arm (foshan_r0_grid), whose solutions can run
    past the physical reference interval used for comparability.
    """
    flags, n = {}, 0
    for k, lo, hi in bounds:
        x = float(vals[k])
        hit = (abs(x - lo) / max(abs(lo), 1e-300) < tol
               or abs(x - hi) / max(abs(hi), 1e-300) < tol
               or x < lo or x > hi)
        flags[f"{k}_at_bound"] = bool(hit)
        n += int(hit)
    flags["n_params_at_bound"] = n
    return flags


def bounds_record(bounds=PARAM_BOUNDS):
    """Compact string of the active constraint set, written into every CSV
    so an at-bound flag is interpretable without reading the source."""
    return ";".join(f"{k}:[{lo},{hi}]" for k, lo, hi in bounds)


# --------------------------------------------------------------- machinery --
def ode_zone_seeded(seed_frac):
    """hv2.ode_zone with the vector infection seeding made explicit.

    Original code hard-codes 0.05 % exposed + 0.05 % infectious vectors
    (foshan_hv2.py:141); here it is a free parameter.
    """
    def f(beta_mp, beta_pm, i0, e0, n_p_zone, n_m, t_daily, dn, sigma):
        idx = np.arange(len(n_m))

        def rhs(t, y):
            s_p, e_p, i_p, a_p, r_p, s_m, e_m, i_m = y
            w_m = 1.0 / hv2.eip50_days(t_daily[int(t)])
            sig = sigma[int(t)]
            nm = max(n_m[int(t)], 1.0)
            rec = (hv2.A_VEC * (nm - hv2.N_VERT * i_m)
                   - beta_pm * (a_p + i_p) * s_m / n_p_zone
                   - hv2.B_VEC * s_m - dn[int(t)] * (s_m / nm))
            return [
                -sig * beta_mp * i_m * s_p / n_p_zone,
                sig * beta_mp * i_m * s_p / n_p_zone
                - (1 - hv2.P_ASYM) * hv2.W_INC * e_p
                - hv2.P_ASYM * hv2.W_LAT * e_p,
                (1 - hv2.P_ASYM) * hv2.W_INC * e_p - hv2.G_SYM * i_p,
                hv2.P_ASYM * hv2.W_LAT * e_p - hv2.G_ASYM * a_p,
                hv2.G_ASYM * a_p + hv2.G_SYM * i_p,
                rec,
                beta_pm * (a_p + i_p) * s_m / n_p_zone - w_m * e_m
                - hv2.B_VEC * e_m - dn[int(t)] * (e_m / nm),
                w_m * e_m + hv2.N_VERT * hv2.A_VEC * i_m - hv2.B_VEC * i_m
                - dn[int(t)] * (i_m / nm),
            ]
        y0 = [n_p_zone - i0 - e0, e0, i0, 0.0, 0.0,
              n_m[0] * (1 - 2 * seed_frac), n_m[0] * seed_frac,
              n_m[0] * seed_frac]
        sol = solve_ivp(rhs, [0, len(n_m) - 1], y0, t_eval=idx,
                        method="RK45", rtol=1e-6, atol=1.0)
        onset = (1 - hv2.P_ASYM) * hv2.W_INC * sol.y[1]
        return np.cumsum(onset), onset
    return f


def beta_scale_for_r0(theta, r0_target, date="2025-07-15", tol=1e-7, maxit=80):
    """Scale factor s such that scaling both betas by s gives R0 = target.

    R0 is NOT exactly linear in the beta scale.  The next-generation matrix
    contains a beta-independent vertical-transmission entry
    (K[4,4] = N_VERT * A_VEC), so a one-shot scale s = target / current
    undershoots as s grows: measured shortfall grows from 0.7 % at a target of
    1.5 to 6.6 % at 7.28 and 7.0 % at 16.3.  The scale is therefore solved by
    fixed-point iteration until the ACHIEVED R0 matches the target, so the
    grid axis is the true intrinsic R0 rather than the requested one.
    """
    r0_now, _ = ngm_r0(theta, date=date)
    s = float(r0_target / r0_now)
    for _ in range(maxit):
        r0_s, _ = ngm_r0(scaled_theta(theta, s), date=date)
        if abs(r0_s - r0_target) <= tol * max(1.0, r0_target):
            break
        s *= r0_target / r0_s
    return s, r0_now


def scaled_theta(theta, s):
    th = np.array(theta, dtype=float).copy()
    th[0] = th[0] + np.log(s)      # l_bmp
    th[1] = th[1] + np.log(s)      # l_bpm
    return th


# ------------------------------------------------------------------- driver --
def make_loss(daily_dates, pfi, t_daily, n, detect_idx, norm_idx,
              theta_base, seed_frac_default):
    a_tC = np.array([(pd.Timestamp(d) - pd.Timestamp(hv2.T0)).days
                     for d, _ in hv2.ANCHORS_CITY])
    a_vC = np.array([v for _, v in hv2.ANCHORS_CITY])
    post = np.array([pd.Timestamp(d) >= pd.Timestamp("2025-08-16")
                     for d, _ in hv2.ANCHORS_CITY])
    i715 = int(np.argmax(daily_dates >= pd.Timestamp("2025-07-15")))

    def build_free(free):
        """free = [l_cpa, l_cpb, l_sraw, l_rraw, l_roff, l_esc, l_stag,
                  l_seed]

        theta index map of hv2.unpack():
          0 l_bmp  1 l_bpm  2 l_i0  3 l_sa  4 l_ca  5 l_cb
          6 l_cpa  7 l_cpb  8 l_sraw  9 l_rraw  10 l_roff  11 l_esc  12 l_stag
        """
        th = np.array(theta_base, dtype=float).copy()
        th[6] = free[0]     # l_cpa
        th[7] = free[1]     # l_cpb
        th[8] = free[2]     # l_sraw
        th[9] = free[3]     # l_rraw
        th[10] = free[4]    # l_roff
        th[11] = free[5]    # l_esc
        th[12] = free[6]    # l_stag
        seed = float(np.exp(free[7]))
        return th, seed

    def evaluate(free):
        th, seed = build_free(free)
        hv2.ode_zone = ode_zone_seeded(seed)
        try:
            r = hv2.run_two_zone(th, daily_dates, pfi, t_daily, n,
                                 detect_idx, norm_idx)
        except Exception:
            return None
        cumC = np.cumsum(r["onsets_A"] + r["onsets_B"])
        pred = cumC[np.clip(a_tC, 0, n - 1)]
        rel = (pred[post] - a_vC[post]) / np.maximum(a_vC[post], 1.0)
        loss_post = float(np.sum(rel ** 2))
        rho = r["r_base"] + (1 - r["r_base"]) * r["report_step"]
        repA = np.cumsum(rho * r["onsets_A"])
        pred_early = float(repA[i715])
        loss_early = LAMBDA_EARLY * ((pred_early - ANCHOR_0715_SHUNDE)
                                     / ANCHOR_0715_SHUNDE) ** 2
        return {"loss": loss_post + loss_early, "loss_post": loss_post,
                "loss_early": loss_early, "pred_early": pred_early,
                "r": r, "cumC": cumC}

    def loss(free):
        out = evaluate(free)
        return 1e18 if out is None else out["loss"]
    return loss, evaluate, build_free


def counterfactuals(theta, seed, pfi, daily_dates, t_daily, n, detect_idx,
                    norm_idx):
    """Five arms at a given parameter set -> final reported-city sizes."""
    hv2.ode_zone = ode_zone_seeded(seed)
    out = {}
    for arm, shift in (("none", 0), ("actual", 0), ("early", -14),
                       ("late", 14), ("rule", 0)):
        p = hv2.unpack(theta)
        bi_none = pfi * np.exp(theta[4])
        trig = int(np.argmax(bi_none > 5.0))
        r = hv2.run_two_zone(theta, daily_dates, pfi, t_daily, n,
                             detect_idx, norm_idx, shift_weeks=shift,
                             arm=arm,
                             detect_idx_override=(trig if arm == "rule"
                                                  else None))
        rho = r["r_base"] + (1 - r["r_base"]) * r["report_step"]
        rep = np.cumsum(rho * (r["onsets_A"] + r["onsets_B"]))
        out[arm] = float(rep[-1])
    return out


def main(n_starts=2, maxiter=700):
    daily_dates, pfi, t_daily, n, detect_idx, norm_idx = hv2.build_daily()
    with open(os.path.join(RES, "foshan_hv2_multistart.json")) as f:
        base_thetas = [np.array(t) for t in json.load(f)["kept_thetas"]]
    with open(os.path.join(RES, "foshan_hv2_params.json")) as f:
        pj = json.load(f)
    central = hv2.centre_theta(pj)

    rows = []

    # ---- as-published reference point (central fit, seeding 0.0005) --------
    loss_ref, ev_ref, bf_ref = make_loss(daily_dates, pfi, t_daily, n,
                                         detect_idx, norm_idx, central, 0.0005)
    out_ref = ev_ref(np.array([central[6], central[7], central[8],
                               central[9], central[10], central[11],
                               central[12], np.log(0.0005)]))
    r0_ref, _ = ngm_r0(central)
    if out_ref is not None:
        th_r, seed_r = bf_ref(np.array([central[6], central[7], central[8],
                                       central[9], central[10], central[11],
                                       central[12], np.log(0.0005)]))
        cf = counterfactuals(th_r, seed_r, pfi, daily_dates, t_daily, n,
                             detect_idx, norm_idx)
        ref = cf["actual"] or 1.0
        rec = {"R0_target": "as-published", "R0_base_fit": round(r0_ref, 3),
               "loss_total": round(out_ref["loss"], 4),
               "loss_post": round(out_ref["loss_post"], 4),
               "loss_early": round(out_ref["loss_early"], 4),
               "pred_repA_0715": round(out_ref["pred_early"], 1),
               "obs_0715": ANCHOR_0715_SHUNDE,
               "cpa": round(hv2.unpack(th_r)["cpa"], 3),
               "cpb": round(hv2.unpack(th_r)["cpb"], 3),
               "s_min": round(hv2.unpack(th_r)["s_min"], 3),
               "r_base": round(hv2.unpack(th_r)["r_base"], 3),
               "seed_frac": float(f"{seed_r:.2e}"),
               **{k: round(v) for k, v in cf.items()}}
        for k in ("none", "early", "late", "rule"):
            rec[k + "_x"] = round(cf[k] / ref, 2)
        rec["required_reduction"] = round(1 - cf["actual"] / cf["none"], 4)
        rows.append(rec)
        print(f"as-published R0={r0_ref:.2f} loss={rec['loss_total']:.4f} "
              f"none={rec['none']:,} actual={rec['actual']:,}", flush=True)
    hv2.ode_zone = _ORIG_ODE

    for r0_target in R0_GRID:
        best = None
        for th_base in base_thetas[:n_starts]:
            s, r0_now = beta_scale_for_r0(th_base, r0_target)
            th_anchor = scaled_theta(th_base, s)
            loss, evaluate, build_free = make_loss(
                daily_dates, pfi, t_daily, n, detect_idx, norm_idx,
                th_anchor, 0.0005)
            free0 = np.array([th_anchor[6], th_anchor[7], th_anchor[8],
                              th_anchor[9], th_anchor[10], th_anchor[11],
                              th_anchor[12], np.log(1e-4)])
            res = minimize(loss, free0, method="Nelder-Mead",
                           options={"maxiter": maxiter, "xatol": 1e-3,
                                    "fatol": 1e-6})
            out = evaluate(res.x)
            if out is None:
                continue
            th_fit, seed = build_free(res.x)
            cf = counterfactuals(th_fit, seed, pfi, daily_dates, t_daily, n,
                                 detect_idx, norm_idx)
            rec = {
                "R0_target": r0_target, "R0_base_fit": round(r0_now, 3),
                "loss_total": round(out["loss"], 4),
                "loss_post": round(out["loss_post"], 4),
                "loss_early": round(out["loss_early"], 4),
                "pred_repA_0715": round(out["pred_early"], 1),
                "obs_0715": ANCHOR_0715_SHUNDE,
                "cpa": round(hv2.unpack(th_fit)["cpa"], 3),
                "cpb": round(hv2.unpack(th_fit)["cpb"], 3),
                "s_min": round(hv2.unpack(th_fit)["s_min"], 3),
                "r_base": round(hv2.unpack(th_fit)["r_base"], 3),
                "seed_frac": float(f"{seed:.2e}"),
                **{k: round(v) for k, v in cf.items()},
            }
            ref = rec["actual"] if rec["actual"] else 1.0
            for k in ("none", "early", "late", "rule"):
                rec[k + "_x"] = round(rec[k] / ref, 2)
            rec["required_reduction"] = (
                round(1 - rec["actual"] / rec["none"], 4)
                if rec["none"] else None)
            if best is None or rec["loss_total"] < best["loss_total"]:
                best = rec
            hv2.ode_zone = _ORIG_ODE
        if best:
            rows.append(best)
            print(f"R0={r0_target:5.2f}  loss={best['loss_total']:9.4f}  "
                  f"(post {best['loss_post']:.4f} / early {best['loss_early']:.4f})"
                  f"  cpa={best['cpa']:.2f} cpb={best['cpb']:.2f} "
                  f"r_base={best['r_base']:.2f} seed={best['seed_frac']:.1e}  "
                  f"none={best['none']:,} actual={best['actual']:,}  "
                  f"none/actual={best['none_x']}x", flush=True)

    df = pd.DataFrame(rows)
    out = os.path.join(
        RES, f"r0_anchor_experiment_s{n_starts}_i{maxiter}.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\nsaved {os.path.basename(out)}")
    hv2.ode_zone = _ORIG_ODE
    return df


if __name__ == "__main__":
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    mi = int(sys.argv[2]) if len(sys.argv) > 2 else 700
    main(ns, mi)
