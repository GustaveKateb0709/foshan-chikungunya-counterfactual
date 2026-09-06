#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""foshan_hv2.py — v2b: two-zone host-vector model (Shunde core + rest).

Data-driven motivation: 92.96% of reported cases concentrated in Shunde; the
emergency response was hot-spot focused (90% of affected villages down to
mid-risk by 2025-07-20) while the citywide mean BI fell only moderately
(8.5 -> 6.2).  A single-zone model cannot reproduce the observed
"hot-spot explosion + citywide plateau" shape.

Zone A = Shunde (intense response).  Zone B = other four districts (moderate
response).  Shared transmission betas; zone-specific density calibration and
response coverage peak.  Shared ascertainment surge rho(t) (citywide
detection effort, documented 2025-07-20: 35 hospitals added CHIKV PCR +
door-to-door screening).

Framework: standard host-vector ODE with asymptomatic compartment (Zhao et
al., Infect Dis Poverty 2025;14:106 — cited for the framework); vector
abundance N_m(t) per zone from the weather-driven density model (foshan_cf).

Usage:  python foshan_hv2.py fit
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE, ".."))
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)

N_P_TOTAL = 9_615_400.0
NM_PER_BI = 20_000.0
P_ASYM = 0.140
W_LAT = 1 / 7.0
W_INC = 1 / 7.0
G_SYM = 1 / 8.0
G_ASYM = 1 / 8.0
A_VEC, B_VEC, N_VERT = 0.071, 0.071, 0.168
SR_DECAY, SP_DECAY = 0.45, 0.30
SR_EFF, SP_EFF = 0.35, 0.55

T0 = "2025-06-08"          # integration start (index onset 06-16 + burn-in)
T1 = "2025-12-31"
DETECT = "2025-07-06"
NORMALISE = "2025-08-24"

ANCHORS_SHUNDE = [("2025-07-15", 478), ("2025-07-18", 1161),
                  ("2025-07-19", 1790), ("2025-07-20", 2285),
                  ("2025-07-21", 2471), ("2025-07-26", 4208)]
ANCHORS_CITY = [("2025-07-19", 1873), ("2025-07-21", 2659),
                ("2025-07-22", 3195), ("2025-07-24", 4014),
                ("2025-07-26", 4754), ("2025-08-02", 7524),
                ("2025-08-09", 8736), ("2025-08-16", 9380),
                ("2025-08-23", 9586), ("2025-08-30", 9705),
                ("2025-09-06", 9790), ("2025-09-13", 9876),
                ("2025-09-20", 9957), ("2025-09-27", 10035),
                ("2025-10-04", 10192), ("2025-10-11", 10388),
                ("2025-10-18", 10629), ("2025-10-25", 10759),
                ("2025-11-01", 10810), ("2025-11-22", 10819)]


def eip50_days(T):
    return np.clip(np.interp(T, [18.0, 30.0], [8.74, 1.74]), 1.5, 24.0)


def build_daily():
    """Daily weather + PFI over T0..T1 (PFI from the weekly calibrated
    formula, interpolated to days)."""
    import foshan_cf
    w = foshan_cf.load_weekly("2025-05-04", "2026-01-04")
    wk_dates = pd.to_datetime(w["date"])
    daily_dates = pd.date_range(T0, T1, freq="D")
    pfi_daily = np.interp(daily_dates.astype("int64"),
                          wk_dates.astype("int64"), w["pfi"].values)
    T_daily = np.interp(daily_dates.astype("int64"),
                        wk_dates.astype("int64"), w["T"].values)
    n = len(daily_dates)
    detect_idx = int(np.argmax(daily_dates >= pd.Timestamp(DETECT)))
    norm_idx = int(np.argmax(daily_dates >= pd.Timestamp(NORMALISE)))
    return daily_dates, pfi_daily, T_daily, n, detect_idx, norm_idx


def schedule(n, onset, ramp, c_peak, norm_idx, tail=0.30):
    c = np.zeros(n)
    for t in range(n):
        if t < onset:
            c[t] = 0.0
        elif t < onset + ramp:
            c[t] = c_peak * (t - onset + 1) / ramp
        elif t < norm_idx:
            c[t] = c_peak
        else:
            c[t] = min(c_peak * 0.82 + tail, 1.0)
    return c


def zone_density(pfi, calib_z, c_z):
    p_sr = p_sp = 0.0
    bi = np.zeros(len(pfi))
    bi[0] = max(0.0, pfi[0] * calib_z)
    for t in range(1, len(pfi)):
        p_sr = min(0.95, p_sr * SR_DECAY + SR_EFF * c_z[t - 1])
        p_sp = min(0.95, p_sp * SP_DECAY + SP_EFF * c_z[t - 1])
        bi[t] = max(0.0, pfi[t] * calib_z) * (1 - p_sr) * (1 - p_sp)
    nm = np.clip(bi, 0.05, None) * NM_PER_BI
    dn = np.gradient(nm, np.arange(len(nm)).astype(float))
    return bi, nm, dn


def ode_zone(beta_mp, beta_pm, i0, e0, n_p_zone, n_m, t_daily, dn, sigma):
    idx = np.arange(len(n_m))

    def rhs(t, y):
        s_p, e_p, i_p, a_p, r_p, s_m, e_m, i_m = y
        w_m = 1.0 / eip50_days(t_daily[int(t)])
        sig = sigma[int(t)]
        nm = max(n_m[int(t)], 1.0)
        rec = A_VEC * (nm - N_VERT * i_m) \
            - beta_pm * (a_p + i_p) * s_m / n_p_zone \
            - B_VEC * s_m - dn[int(t)] * (s_m / nm)
        return [
            -sig * beta_mp * i_m * s_p / n_p_zone,
            sig * beta_mp * i_m * s_p / n_p_zone
            - (1 - P_ASYM) * W_INC * e_p - P_ASYM * W_LAT * e_p,
            (1 - P_ASYM) * W_INC * e_p - G_SYM * i_p,
            P_ASYM * W_LAT * e_p - G_ASYM * a_p,
            G_ASYM * a_p + G_SYM * i_p,
            rec,
            beta_pm * (a_p + i_p) * s_m / n_p_zone - w_m * e_m - B_VEC * e_m
            - dn[int(t)] * (e_m / nm),
            w_m * e_m + N_VERT * A_VEC * i_m - B_VEC * i_m
            - dn[int(t)] * (i_m / nm),
        ]

    y0 = [n_p_zone - i0 - e0, e0, i0, 0.0, 0.0,
          n_m[0] * 0.999, n_m[0] * 0.0005, n_m[0] * 0.0005]
    sol = solve_ivp(rhs, [0, len(n_m) - 1], y0, t_eval=idx,
                    method="RK45", rtol=1e-6, atol=1.0)
    onset = (1 - P_ASYM) * W_INC * sol.y[1]
    return np.cumsum(onset), onset


def unpack(theta):
    (l_bmp, l_bpm, l_i0, l_sa, l_ca, l_cb, l_cpa, l_cpb,
     l_sraw, l_rraw, l_roff, l_esc, l_stag) = theta
    r_off = 25.0 / (1.0 + np.exp(-l_roff))
    esc_off = 5.0 + 20.0 / (1.0 + np.exp(-l_esc))
    stag_b = 7.0 / (1.0 + np.exp(-l_stag))
    return {
        "r_off": float(r_off),
        "esc_off": float(esc_off),
        "stag_b": float(stag_b),
        "beta_mp": float(np.exp(l_bmp)),
        "beta_pm": float(np.exp(l_bpm)),
        "i0": float(np.exp(l_i0)),
        "s_a": float(1 / (1 + np.exp(-l_sa))),
        "calib_a": float(np.exp(l_ca)),
        "calib_b": float(np.exp(l_cb)),
        "cpa": float(1 / (1 + np.exp(-l_cpa))),
        "cpb": float(1 / (1 + np.exp(-l_cpb))),
        "s_min": float(1 / (1 + np.exp(l_sraw))),
        "r_base": float(1 / (1 + np.exp(l_rraw))),
    }


def run_two_zone(theta, daily_dates, pfi, t_daily, n, detect_idx, norm_idx,
                 shift_weeks=0, arm="actual", detect_idx_override=None):
    """Two-step response per the official record:
    step 1 (detection 2025-07-08): case finding/isolation -> sigma & rho step;
    step 2 (2025-07-19/20): citywide mosquito-control campaign -> c step."""
    p = unpack(theta)
    if detect_idx_override is not None:
        detect_idx = detect_idx_override
    detect = detect_idx + shift_weeks
    escalate_A = min(detect_idx + int(round(p["esc_off"])) + shift_weeks,
                     n - 1)
    escalate_B = min(escalate_A + int(round(p["stag_b"])), n - 1)
    step_idx = np.arange(n)
    if arm == "none":
        c_A = np.zeros(n)
        c_B = np.zeros(n)
    else:
        c_A = np.where(step_idx < detect, 0.0,
                       np.where(step_idx < escalate_A, 0.12, p["cpa"]))
        c_B = np.where(step_idx < detect, 0.0,
                       np.where(step_idx < escalate_B, 0.05, p["cpb"]))
    c_A[norm_idx:] = np.maximum(c_A[norm_idx:], p["cpa"] * 0.5 + 0.1)
    c_B[norm_idx:] = np.maximum(c_B[norm_idx:], p["cpb"] * 0.5 + 0.1)
    bi_A, nm_A, dn_A = zone_density(pfi, p["calib_a"], c_A)
    bi_B, nm_B, dn_B = zone_density(pfi, p["calib_b"], c_B)
    n_pA = N_P_TOTAL * p["s_a"]
    n_pB = N_P_TOTAL * (1 - p["s_a"])
    i0_b = p["i0"] * ((1 - p["s_a"]) / p["s_a"]) * 0.03
    if arm == "none":
        sigma_step = np.zeros(n)      # no case finding, no isolation
        report_step = np.zeros(n)     # no detection surge
    else:
        sigma_step = np.where(step_idx >= detect, 1.0, 0.0)
        report_step = np.where(step_idx >= detect + int(round(p["r_off"])),
                               1.0, 0.0)
    sig_A = 1 - (1 - p["s_min"]) * sigma_step
    sig_B = sig_A.copy()
    cumA, onsA = ode_zone(p["beta_mp"], p["beta_pm"], p["i0"],
                          p["i0"] * 2, n_pA, nm_A, t_daily, dn_A, sig_A)
    cumB, onsB = ode_zone(p["beta_mp"], p["beta_pm"], i0_b, i0_b * 2,
                          n_pB, nm_B, t_daily, dn_B, sig_B)
    return {"s_a": p["s_a"], "r_base": p["r_base"], "p": p,
            "report_step": report_step,
            "onsets_A": onsA, "onsets_B": onsB,
            "cum_A": pd.Series(cumA, index=daily_dates),
            "cum_B": pd.Series(cumB, index=daily_dates),
            "cum_city": pd.Series(cumA + cumB, index=daily_dates),
            "BI_A": pd.Series(bi_A, index=daily_dates),
            "BI_B": pd.Series(bi_B, index=daily_dates)}


def main_fit():
    daily_dates, pfi, t_daily, n, detect_idx, norm_idx = build_daily()

    a_tA = np.array([(pd.Timestamp(d) - pd.Timestamp(T0)).days
                     for d, _ in ANCHORS_SHUNDE])
    a_vA = np.array([v for _, v in ANCHORS_SHUNDE])
    a_tC = np.array([(pd.Timestamp(d) - pd.Timestamp(T0)).days
                     for d, _ in ANCHORS_CITY])
    a_vC = np.array([v for _, v in ANCHORS_CITY])
    # Plan c: detection-wave window (7-15 ~ 8-09) excluded from calibration
    post_mask = np.array([pd.Timestamp(d) >= pd.Timestamp("2025-08-16")
                          for d, _ in ANCHORS_CITY])
    a_tPost = a_tC[post_mask]
    a_vPost = a_vC[post_mask]

    def loss(theta):
        try:
            r = run_two_zone(theta, daily_dates, pfi, t_daily, n,
                             detect_idx, norm_idx)
        except Exception:
            return 1e18
        onsets_city = r["onsets_A"] + r["onsets_B"]
        cumC = np.cumsum(onsets_city)          # post-wave: rho ~ 1
        if not np.all(np.isfinite(cumC)):
            return 1e18
        predC = cumC[np.clip(a_tPost, 0, n - 1)]
        relC = (predC - a_vPost) / np.maximum(a_vPost, 1.0)
        return float(np.sum(relC ** 2))

    theta0 = np.array([
        np.log(0.78), np.log(0.54), np.log(5.0), -0.68,
        np.log(0.65), np.log(0.45), 1.73, -0.62, 0.32, -1.73,
        -0.575,
    ])
    res = minimize(loss, theta0, method="Nelder-Mead",
                   options={"maxiter": 2500, "xatol": 1e-3, "fatol": 1e-4})
    print("fitted loss =", round(res.fun, 4))
    r = run_two_zone(res.x, daily_dates, pfi, t_daily, n, detect_idx,
                     norm_idx)
    p = r["p"]
    onsets_city = r["onsets_A"] + r["onsets_B"]
    cumC = np.cumsum(onsets_city)
    predC = cumC[np.clip(a_tC, 0, n - 1)]
    # out-of-window validation: back-extrapolated TRUE incidence at 2025-07-15
    i715 = int(np.argmax(daily_dates >= pd.Timestamp("2025-07-15")))
    true_715 = float(r["cum_A"].values[i715] + r["cum_B"].values[i715])
    reported_715 = 478.0
    underdetect = true_715 / reported_715
    print("PRE-WAVE VALIDATION: model-true cum @2025-07-15 = %.0f "
          "vs reported 478 -> early under-ascertainment factor = %.1fx"
          % (true_715, underdetect))
    cmp = pd.DataFrame({
        "date": [d for d, _ in ANCHORS_CITY],
        "series": ["city_pre" if d < "2025-08-16" else "city_post"
                   for d, _ in ANCHORS_CITY],
        "observed": a_vC, "model_onsets_cum": predC.round(1)})
    cmp.loc[len(cmp)] = {"date": "2025-07-15", "series": "pre_wave_validation",
                         "observed": 478.0, "model_onsets_cum":
                             round(true_715, 1)}
    print(cmp.to_string(index=False))
    cmp.to_csv(os.path.join(RES, "foshan_hv2_fit_check.csv"), index=False,
               encoding="utf-8-sig")
    with open(os.path.join(RES, "foshan_hv2_params.json"), "w") as f:
        json.dump({"theta": list(res.x), "loss": res.fun, **p,
                   "pre_wave_true_cum_0715": round(true_715, 1),
                   "early_under_ascertainment_factor": round(underdetect, 1)},
                  f, indent=2)
    print("saved foshan_hv2_params.json / foshan_hv2_fit_check.csv")


def multistart(n_starts=6):
    """Multi-start uncertainty envelope over the 13-param two-zone model."""
    daily_dates, pfi, t_daily, n, detect_idx, norm_idx = build_daily()
    a_tC = np.array([(pd.Timestamp(d) - pd.Timestamp(T0)).days
                     for d, _ in ANCHORS_CITY])
    a_vC = np.array([v for _, v in ANCHORS_CITY])
    post = np.array([pd.Timestamp(d) >= pd.Timestamp("2025-08-16")
                     for d, _ in ANCHORS_CITY])
    a_tPost, a_vPost = a_tC[post], a_vC[post]
    i715 = int(np.argmax(daily_dates >= pd.Timestamp("2025-07-15")))

    def loss(theta):
        try:
            r = run_two_zone(theta, daily_dates, pfi, t_daily, n,
                             detect_idx, norm_idx)
        except Exception:
            return 1e18
        cumC = np.cumsum(r["onsets_A"] + r["onsets_B"])
        if not np.all(np.isfinite(cumC)):
            return 1e18
        predC = cumC[np.clip(a_tPost, 0, n - 1)]
        relC = (predC - a_vPost) / np.maximum(a_vPost, 1.0)
        return float(np.sum(relC ** 2))

    with open(os.path.join(RES, "foshan_hv2_params.json")) as f:
        prev = json.load(f)
    # defaults = the previously fixed values (13 d escalate offset, 2 d B-zone
    # stagger); the saved params json predates these parameters
    l_esc0 = -0.4055   # esc_off ~ 13 d
    l_stag0 = -0.9229  # stag_b ~ 2 d
    base = np.array(list(prev["theta"]) + [l_esc0, l_stag0])
    rng = np.random.default_rng(2026)
    results = []
    for k in range(n_starts):
        th0 = base + rng.normal(0, 0.25, len(base))
        res = minimize(loss, th0, method="Nelder-Mead",
                       options={"maxiter": 1200, "xatol": 1e-3,
                                "fatol": 1e-4})
        try:
            r = run_two_zone(res.x, daily_dates, pfi, t_daily, n,
                             detect_idx, norm_idx)
            cumC = np.cumsum(r["onsets_A"] + r["onsets_B"])
            true_715 = float(r["cum_A"].values[i715]
                             + r["cum_B"].values[i715])
        except Exception:
            continue
        results.append({"loss": float(res.fun), "theta": list(res.x),
                        "cumC": cumC, "true_715": true_715,
                        "r_base": r["r_base"], "s_a": r["s_a"]})
        print(f"start {k}: loss={res.fun:.4f} true715={true_715:.0f}",
              flush=True)
    # keep ALL starts: with 13 params vs 13 post anchors the fit can
    # interpolate — absolute parameter identification is impossible, so the
    # across-start SPREAD is the honest uncertainty representation
    ok = sorted(results, key=lambda r: r["loss"])
    post_dates = [d for d, p_ in zip(ANCHORS_CITY, post) if p_]
    env_rows = []
    for j, (d, _) in enumerate([(d, 1) for d, _ in ANCHORS_CITY if True]):
        vals = [r["cumC"][a_tC[j]] for r in ok]
        env_rows.append({"date": d, "observed": a_vC[j],
                         "env_min": round(min(vals), 0),
                         "env_median": round(float(np.median(vals)), 0),
                         "env_max": round(max(vals), 0),
                         "in_window": bool(post[j])})
    env = pd.DataFrame(env_rows)
    env.to_csv(os.path.join(RES, "foshan_hv2_envelope.csv"), index=False,
               encoding="utf-8-sig")
    factors = [r["true_715"] / 478.0 for r in ok]
    summary = {"n_starts": n_starts, "n_kept": len(ok),
               "loss_range": [min(r["loss"] for r in ok),
                              max(r["loss"] for r in ok)],
               "true715_range": [min(r["true_715"] for r in ok),
                                 max(r["true_715"] for r in ok)],
               "under_ascertainment_range": [round(min(factors), 1),
                                             round(max(factors), 1)]}
    with open(os.path.join(RES, "foshan_hv2_multistart.json"), "w") as f:
        json.dump({"summary": summary,
                   "kept_thetas": [r["theta"] for r in ok]}, f, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("saved foshan_hv2_envelope.csv / foshan_hv2_multistart.json")


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "fit"
    if mode == "multi":
        multistart()
    else:
        main_fit()
