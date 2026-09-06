#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""foshan_cf.py — Foshan 2025 chikungunya outbreak policy counterfactual (v1).

Case-curve-anchored outbreak-window counterfactual:
  weather-driven PFI vector density (Delatte 2009 / Briere 1999 literature
  anchors) + documented response timeline (official quantification) ->
  five arms -> weekly transmission potential (Tegar et al. 2026 traits).

Design contract (see manuscript Methods):
  - every numeric anchor traced to anchors.json (sources included);
  - counterfactual results are RELATIVE comparisons, not absolute predictions;
  - known limitation: citywide BI calibration anchors are sparse (2025-07 8.5,
    2025-08 6.2, both response-affected) -> density scale x response intensity
    handled by an explicit 2-D sensitivity scan.

Arms
  none       no response
  actual     response schedule anchored to the official timeline
             (detection 2025-07-08; escalation through 2025-07-20; III-level
             lifted 2025-08-26; normalised thereafter)
  early/late the actual schedule shifted by -/+ 2 weeks
  rule       BI-threshold-triggered ramp (trigger at BI > 5, national guidance)

Usage
  python foshan_cf.py run                     # five arms with current params
  python foshan_cf.py calibrate               # grid search over (scale, c_peak)
"""
import json
import os

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE, ".."))
WEATHER = os.path.join(ROOT, "data", "era5_foshan_daily_2024-2026.csv")
ANCHORS = os.path.join(ROOT, "data", "anchors.json")
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

# ------------------------------------------------------------------ PFI model
# Ported verbatim from the Guangzhou benchmark pipeline (aedes-benchmark-
# guangzhou, pipeline.py); literature anchors: Delatte 2009 (PMID 19198515),
# Briere 1999.  Constants are species-level literature values, not city fits.
T_DEV0, T_OPT, T_MAX = 10.0, 27.0, 39.0
SURV_MAX, SURV_K = 0.95, 0.004
RAIN_THRESH, RAIN_GAIN = 5.0, 2.5


def briere_dev_rate(T, a=0.000113, Tmin=T_DEV0, Tmax=T_MAX):
    T = np.asarray(T, dtype=float)
    out = np.zeros_like(T)
    m = (T > Tmin) & (T < Tmax)
    out[m] = a * T[m] * (T[m] - Tmin) * np.sqrt(np.maximum(Tmax - T[m], 0))
    return out


def temp_survival(T):
    return SURV_MAX * np.exp(-SURV_K * (np.asarray(T, dtype=float) - T_OPT) ** 2)


def rainfall_brood_factor(rain_mm):
    eff = np.maximum(0.0, np.asarray(rain_mm, dtype=float) - RAIN_THRESH)
    return 1.0 + RAIN_GAIN * (1.0 - np.exp(-eff / 15.0))


def compute_pfi_weekly(T_wk, rain_wk, hum_wk):
    dev = briere_dev_rate(T_wk)
    surv = temp_survival(T_wk)
    hum = np.clip(0.5 + 0.5 * np.clip(hum_wk / 70.0, 0, 1.6), 0, 1)
    return dev * surv * hum * rainfall_brood_factor(rain_wk) * 28.0


# ------------------------------------------------------- transmission (Tegar)
def eip50(t):
    return float(np.clip(np.interp(t, [18.0, 30.0], [8.74, 1.74]), 1.5, 24.0))


def vc(t):
    return np.where(
        (t >= 13.8) & (t <= 31.8),
        0.96 * np.exp(-((t - 25.6) ** 2) / (2 * 2.48 ** 2)),
        0.0,
    )


# ------------------------------------------------------------------- weather
def load_weekly(start="2025-05-04", end="2026-01-04"):
    """ERA5 daily -> weekly (W-SUN) over [start, end]."""
    df = pd.read_csv(WEATHER, skiprows=3)
    df.columns = ["date", "T", "Tmax", "Tmin", "rain", "hum"]
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    wk = df.resample("W-SUN").agg(T=("T", "mean"), Rain=("rain", "sum"),
                                  Hum=("hum", "mean"))
    wk = wk.loc[start:end].reset_index()
    iso = wk["date"].dt.isocalendar()
    wk["year"] = iso["year"].values
    wk["week_in_year"] = iso["week"].values
    wk["pfi"] = compute_pfi_weekly(wk["T"].values, wk["Rain"].values,
                                   wk["Hum"].values)
    return wk


# ------------------------------------------------------------------ arms
def response_schedule(n_weeks, onset_idx, ramp_weeks, c_peak, c_sustain,
                      sustain_from_idx, tail=0.35):
    """Effective citywide response coverage per week (0-1).

    Encodes: zero before detection; linear ramp over `ramp_weeks`; peak
    `c_peak` during the emergency phase; `c_sustain` from `sustain_from_idx`
    (III-level lifted -> normalised); `tail` afterwards.  The schedule is a
    parameterisation of the documented timeline (anchors.response_timeline),
    fitted -- not measured -- at citywide aggregate level.
    """
    c = np.zeros(n_weeks)
    for t in range(n_weeks):
        if t < onset_idx:
            c[t] = 0.0
        elif t < onset_idx + ramp_weeks:
            c[t] = c_peak * (t - onset_idx + 1) / ramp_weeks
        elif t < sustain_from_idx:
            c[t] = c_peak
        else:
            c[t] = c_sustain * (c_peak / max(c_peak, 1e-9)) + tail
            c[t] = min(c[t], 1.0)
    return np.clip(c, 0.0, 1.0)


# ------------------------------------------------------- aggregate decays
# NOTE: the #2 grid-level stock model used slow decay (0.85) for a treated
# GRID.  Here we model the CITYWIDE AGGREGATE: only a fraction of the city is
# removed each week and untreated sites rebound quickly ("2-3 weeks to
# rebound", Foshan CDC via media) -> fast effective decay.  Documented as an
# aggregate-encoding assumption; perturbed in sensitivity analysis.
SR_DECAY, SP_DECAY = 0.45, 0.30


def simulate(wk, calib, c_sr, c_sp, seed=1000):
    """Run one arm.  Returns weekly BI trajectory."""
    rng = np.random.default_rng(seed)
    pfi = wk["pfi"].values * calib
    p_sr = p_sp = 0.0
    bi = np.zeros(len(wk))
    bi[0] = max(0.0, pfi[0] + rng.normal(0, 0.3))
    for t in range(1, len(wk)):
        p_sr = min(0.95, p_sr * SR_DECAY + 0.35 * c_sr[t - 1])
        p_sp = min(0.95, p_sp * SP_DECAY + 0.55 * c_sp[t - 1])
        base = max(0.0, pfi[t] + rng.normal(0, 0.3))
        bi[t] = base * (1.0 - p_sr) * (1.0 - p_sp)
    return bi


def transmission_potential(wk, bi, p_surv=0.90):
    T = wk["T"].values
    return bi * vc(T) * (p_surv ** np.array([eip50(t) for t in T]))


# ------------------------------------------------------------------ main
def build_arms(wk, params):
    """Return dict arm -> (c_sr, c_sp) coverage schedules."""
    n = len(wk)
    dates = wk["date"].values
    detect_idx = int(np.argmax(dates >= np.datetime64("2025-07-06")))
    normalise_idx = int(np.argmax(dates >= np.datetime64("2025-08-24")))
    ramp = params.get("ramp_weeks", 2)
    c_peak = params.get("c_peak", 0.55)
    c_sustain = params.get("c_sustain", 0.45)

    def shift(schedule, weeks):
        out = np.zeros(n)
        if weeks == 0:
            return schedule.copy()
        if weeks > 0:
            out[weeks:] = schedule[: n - weeks]
        else:
            w = -weeks
            out[: n - w] = schedule[w:]
            out[n - w:] = schedule[-1]
        return out

    actual = response_schedule(n, detect_idx, ramp, c_peak, c_sustain,
                               normalise_idx)
    sp = np.where((dates >= np.datetime64("2025-07-13"))
                  & (dates <= np.datetime64("2025-08-10")), 0.45, 0.0)
    arms = {
        "none": (np.zeros(n), np.zeros(n)),
        "actual": (actual, sp),
        "early": (shift(actual, -2), shift(sp, -2)),
        "late": (shift(actual, 2), shift(sp, 2)),
    }
    # rule-trigger: ramp starts the first week simulated BI (no response) > 5
    base_bi = simulate(wk, params.get("calib", 0.3476), np.zeros(n),
                       np.zeros(n), seed=1000)
    trig = int(np.argmax(base_bi > 5.0))
    arms["rule"] = (response_schedule(n, trig, ramp, c_peak, c_sustain,
                                      normalise_idx), sp)
    return arms, detect_idx, trig


def run(params=None, verbose=True):
    params = params or {}
    wk = load_weekly()
    arms, detect_idx, trig = build_arms(wk, params)
    calib = params.get("calib", 0.3476)
    rows, tp_tot = [], {}
    for arm, (c_sr, c_sp) in arms.items():
        bi = simulate(wk, calib, c_sr, c_sp, seed=1000)
        tp = transmission_potential(wk, bi)
        tp_tot[arm] = float(np.sum(tp))
        for t in range(len(wk)):
            rows.append({"arm": arm, "date": str(wk["date"][t].date()),
                         "week_in_year": int(wk["week_in_year"][t]),
                         "T": round(float(wk["T"][t]), 2),
                         "PFI_raw": round(float(wk["pfi"][t]), 3),
                         "BI": round(float(bi[t]), 3),
                         "c_sr": round(float(c_sr[t]), 3),
                         "c_sp": round(float(c_sp[t]), 3),
                         "TP": round(float(tp[t]), 3)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "foshan_arms_weekly_v1.csv"),
              index=False, encoding="utf-8-sig")
    ref = tp_tot["none"]
    summary = pd.DataFrame([
        {"arm": a, "cumulative_TP": round(v, 1),
         "pct_of_none": round(100 * v / ref, 1),
         "TP_ratio_vs_actual": round(v / tp_tot["actual"], 3)}
        for a, v in tp_tot.items()])
    summary.to_csv(os.path.join(RESULTS, "foshan_arms_summary_v1.csv"),
                   index=False, encoding="utf-8-sig")
    if verbose:
        print(summary.to_string(index=False))
        jul = df[(df["arm"] == "actual") & (df["date"].str.startswith("2025-07"))]
        aug = df[(df["arm"] == "actual") & (df["date"].str.startswith("2025-08"))]
        print(f"\nactual-arm July mean BI  = {jul['BI'].mean():.2f}  (anchor 8.5)")
        print(f"actual-arm Aug   mean BI  = {aug['BI'].mean():.2f}  (anchor 6.2)")
        print(f"detection week idx = {detect_idx}, rule trigger week idx = {trig}")
    return df, summary


def calibrate():
    """2-D grid: density scale x response peak coverage, scored against
    the BI anchors (2025-07 8.5, 2025-08 6.2)."""
    wk = load_weekly()
    best = []
    for calib in np.arange(0.25, 0.71, 0.05):
        for c_peak in np.arange(0.35, 0.96, 0.05):
            params = {"calib": float(calib), "c_peak": float(c_peak),
                      "c_sustain": float(c_peak) * 0.82}
            arms, _, _ = build_arms(wk, params)
            c_sr, c_sp = arms["actual"]
            bi = simulate(wk, params["calib"], c_sr, c_sp, seed=1000)
            dates = wk["date"].values
            jul = bi[(dates >= np.datetime64("2025-07-01"))
                     & (dates < np.datetime64("2025-08-01"))].mean()
            aug = bi[(dates >= np.datetime64("2025-08-01"))
                     & (dates < np.datetime64("2025-09-01"))].mean()
            loss = (jul - 8.5) ** 2 / 8.5 + (aug - 6.2) ** 2 / 6.2
            best.append((round(loss, 4), round(float(calib), 2),
                         round(float(c_peak), 2), round(float(jul), 2),
                         round(float(aug), 2)))
    best.sort(key=lambda x: x[0])
    print("top-5 (loss, calib, c_peak, Jul BI, Aug BI):")
    for row in best[:5]:
        print(" ", row)
    return best[0]


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode == "calibrate":
        calibrate()
    else:
        run()
