#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""foshan_validation.py — independent-validation add-ons for the Foshan
chikungunya counterfactual.

Modes (these two are the only implemented entry points)
  r0        model-implied basic reproduction number (next-generation matrix)
            evaluated under no-response conditions, per multi-start fit;
            compared with two independently published estimates.
  growth    model-implied early exponential growth rate and doubling time.

The held-out comparison against the pre-saturation anchors is implemented in
`foshan_out_of_sample.py`, not in this module.

Usage:  python foshan_validation.py r0
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import foshan_hv2 as hv2      # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.abspath(os.path.join(BASE, "..")), "results")

# ---------------------------------------------------------------- helpers ---
def load_thetas():
    with open(os.path.join(RES, "foshan_hv2_multistart.json")) as f:
        ms = json.load(f)
    return [np.asarray(t) for t in ms["kept_thetas"]], ms["summary"]


def ngm_r0(theta, date="2025-07-15", p_asym=None, eip_scale=1.0):
    """Basic reproduction number of the two-zone model via the next-generation
    matrix, evaluated at `date` under no-response conditions (zone A).

    Compartments of the infected subsystem: E_p, I_p, A_p, E_m, I_m.
    """
    daily_dates, pfi, t_daily, n, detect_idx, norm_idx = hv2.build_daily()
    i = int(np.argmax(daily_dates >= pd.Timestamp(date)))
    p = hv2.unpack(theta)

    p_asym = hv2.P_ASYM if p_asym is None else p_asym
    # no-response density (coverage 0 -> no protection stocks)
    bi_a = max(0.0, float(pfi[i]) * p["calib_a"])
    n_m = bi_a * hv2.NM_PER_BI
    n_p = hv2.N_P_TOTAL * p["s_a"]
    assert n_p < hv2.N_P_TOTAL        # zone A is a subset of the population
    ratio = n_m / n_p

    eip50 = hv2.eip50_days(t_daily[i]) * eip_scale
    w_m = 1.0 / eip50
    v = hv2.B_VEC                      # adult vector mortality
    sig = 1.0                          # no isolation (no response)

    F = np.zeros((5, 5))
    F[0, 4] = sig * p["beta_mp"]                      # vector -> human
    F[3, 1] = F[3, 2] = p["beta_pm"] * ratio          # human -> vector
    F[4, 4] = hv2.N_VERT * hv2.A_VEC                  # vertical transmission

    V = np.zeros((5, 5))
    V[0, 0] = (1 - p_asym) * hv2.W_INC + p_asym * hv2.W_LAT
    V[1, 0] = -(1 - p_asym) * hv2.W_INC
    V[1, 1] = hv2.G_SYM
    V[2, 0] = -p_asym * hv2.W_LAT
    V[2, 2] = hv2.G_ASYM
    V[3, 3] = w_m + v
    V[4, 3] = -w_m
    V[4, 4] = v

    K = F @ np.linalg.inv(V)
    return float(max(abs(np.linalg.eigvals(K)))), {
        "bi_a": bi_a, "n_m": n_m, "n_p_zone": n_p, "n_m_per_n_p": ratio,
        "temp": float(t_daily[i]), "eip50_days": eip50, "w_m": w_m,
    }


def model_growth_rate(theta, tstart="2025-06-16", tend="2025-07-15"):
    """Early exponential growth rate of TRUE onsets under no response."""
    daily_dates, pfi, t_daily, n, detect_idx, norm_idx = hv2.build_daily()
    p = hv2.unpack(theta)
    bi_a = pfi * p["calib_a"]
    nm_a, dn_a = hv2.zone_density(pfi, p["calib_a"], np.zeros(n))[1:]
    nm_b, dn_b = hv2.zone_density(pfi, p["calib_b"], np.zeros(n))[1:]
    n_pA = hv2.N_P_TOTAL * p["s_a"]
    n_pB = hv2.N_P_TOTAL * (1 - p["s_a"])
    i0_b = p["i0"] * ((1 - p["s_a"]) / p["s_a"]) * 0.03
    sig1 = np.ones(n)
    _, onsA = hv2.ode_zone(p["beta_mp"], p["beta_pm"], p["i0"],
                           p["i0"] * 2, n_pA, nm_a, t_daily, dn_a, sig1)
    _, onsB = hv2.ode_zone(p["beta_mp"], p["beta_pm"], i0_b, i0_b * 2,
                           n_pB, nm_b, t_daily, dn_b, sig1)
    tot = onsA + onsB
    i0_ = int(np.argmax(daily_dates >= pd.Timestamp(tstart)))
    i1_ = int(np.argmax(daily_dates >= pd.Timestamp(tend)))
    seg = tot[i0_:i1_][tot[i0_:i1_] > 0]
    if len(seg) < 5:
        return np.nan, np.nan
    t = np.arange(len(seg), dtype=float)
    r = float(np.polyfit(t, np.log(seg), 1)[0])
    return r, float(np.log(2) / r) if r > 0 else np.inf


def cmd_r0():
    thetas, _ = load_thetas()
    rows = []
    for k, th in enumerate(thetas):
        r0, info = ngm_r0(th)
        r, td = model_growth_rate(th)
        rows.append({"start": k + 1, "R0": round(r0, 3),
                     "growth_rate": round(r, 4), "doubling_days": round(td, 2),
                     "BI_zoneA": round(info["bi_a"], 2),
                     "vectors_per_person": round(info["n_m_per_n_p"], 5),
                     "eip50_days": round(info["eip50_days"], 2)})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print("\nR0   median = %.3f   range = %.3f .. %.3f"
          % (df.R0.median(), df.R0.min(), df.R0.max()))
    print("doubling time median = %.2f d (range %.2f .. %.2f)"
          % (df.doubling_days.median(), df.doubling_days.min(),
             df.doubling_days.max()))
    print("\n--- independently published estimates ---")
    print("Zhang 2025 (Infect Dis Poverty, doi 10.1186/s40249-025-01364-y):")
    print("    R0 = 16.3 (95%CI 15.0-17.5), r = 0.20/d, Td = 3.5 d")
    print("Zhao 2025 (Infect Dis Poverty 14:106): R0 = 7.28 "
          "(IQR 7.28-7.28)")
    df.to_csv(os.path.join(RES, "validation_r0.csv"), index=False,
              encoding="utf-8-sig")
    print("\nsaved validation_r0.csv")


def cmd_growth():
    thetas, _ = load_thetas()
    for k, th in enumerate(thetas):
        r, td = model_growth_rate(th)
        print(f"start {k+1}: true-incidence growth {r:.4f}/d, "
              f"doubling {td:.2f} d")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "r0"
    {"r0": cmd_r0, "growth": cmd_growth}[mode]()
