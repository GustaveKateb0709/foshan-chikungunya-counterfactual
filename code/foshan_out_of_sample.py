#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""foshan_out_of_sample.py — held-out early-anchor check (Foshan chikungunya analysis).

Purpose
-------
The Methods section of the manuscript states that the calibration used only
the 13 city-level anchors dated 2025-08-16 or later, and that all earlier
anchors -- including the earliest district-level anchor -- were reserved for
"out-of-sample comparison".  This script performs that comparison.

What it does
------------
1. Replays every archived, *faithfully reproducible* calibrated solution of
   the two-zone model (see SOLUTIONS below).
2. Evaluates each solution under its own fitted "actual" response scenario,
   over the daily grid the model is defined on (2025-06-08 .. 2025-12-31;
   the model has no grid earlier than 2025-06-08, so the hold-out interval
   2025-06-01 .. 2025-08-16 is evaluated at every held-out anchor date).
3. For every held-out early anchor (all ANCHORS_SHUNDE / ANCHORS_CITY entries
   dated before 2025-08-16) reports the model's cumulative ONsets, the
   observed reported count and their ratio (the implied early
   under-ascertainment factor).

Reconstruction fidelity
-----------------------
Every parameter vector used here is read from an archived results file; no
value is invented and nothing is optimised.  Where an archived file stores a
parameter only at reduced precision this is stated in the `note` column of
the output CSV.  A solution is emitted with replayed=True only when
its full 13-element theta can be rebuilt exactly from archive (the recovered
R0 is cross-checked against the archived R0_check where one exists).

Scope convention
----------------
The model's zone A is Shunde (the early hotspot); zone B is the rest of the
city.  For a Shunde (district-level) anchor the model cumulative is zone A;
for a city-level anchor it is zone A + zone B.  The implied factor is
therefore always "model infections in the scope of the observation, divided
by the reported count in that same scope".

Red-line hygiene: the script and its products carry no personal identifiers,
no absolute home paths and no author metadata; every number in the output CSV
is produced here; un-replayable solutions are flagged, never approximated.

Usage:  python foshan_out_of_sample.py [out_tag]
        (out_tag, if given, suffixes the output filename so a smoke run
         cannot overwrite the canonical product)
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import foshan_hv2 as hv2                        # noqa: E402
import foshan_r0_anchor as A                    # noqa: E402
from foshan_validation import ngm_r0            # noqa: E402

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(CODE, ".."))
RES = os.path.join(ROOT, "results")

# run setting encoded into the archived copy filename
RUN_TAG = "hv2params+multistart+boundprofile_baseline_loose+fullfree_1.24_7.28"
RUN_SLUG = "hv2params_multistart_boundprofile_bl_fullfree_1.24_7.28"

CALIB_START = pd.Timestamp("2025-08-16")   # anchors >= this date were fitted
BOUND_KEYS = ["cpa", "cpb", "s_min", "r_base", "r_off", "esc_off", "stag_b"]
FF_KEYS = ["i0", "s_a", "calib_a", "calib_b", "cpa", "cpb",
           "s_min", "r_base", "r_off", "esc_off", "stag_b"]
FF_BETA0 = (float(np.log(0.78)), float(np.log(0.54)))   # foshan_fullfree_r0.BETA0


def held_out_anchors():
    """All pre-calibration-window anchors, tagged by geographic scope."""
    rows = []
    for scope, lst in (("shunde", hv2.ANCHORS_SHUNDE),
                       ("city", hv2.ANCHORS_CITY)):
        for d, v in lst:
            if pd.Timestamp(d) < CALIB_START:
                rows.append({"scope": scope, "date": d, "observed": float(v)})
    return rows


def theta_from_free_values(v):
    """Rebuild the 11 theta entries (indices 2..12) from named free values.

    Identical arithmetic to foshan_fullfree_r0.theta_from_z, but fed the
    archived *values* instead of the unconstrained z coordinates.  Betas
    (indices 0,1) are set by the caller.
    """
    th = np.zeros(13)
    th[2] = np.log(v["i0"])
    th[3] = np.log(v["s_a"] / (1.0 - v["s_a"]))
    th[4] = np.log(v["calib_a"])
    th[5] = np.log(v["calib_b"])
    th[6] = np.log(v["cpa"] / (1.0 - v["cpa"]))
    th[7] = np.log(v["cpb"] / (1.0 - v["cpb"]))
    th[8] = np.log(1.0 / v["s_min"] - 1.0)
    th[9] = np.log(1.0 / v["r_base"] - 1.0)
    th[10] = -np.log(25.0 / v["r_off"] - 1.0)
    th[11] = np.log((v["esc_off"] - 5.0) / (25.0 - v["esc_off"]))
    th[12] = -np.log(7.0 / v["stag_b"] - 1.0)
    return th


def apply_free_values(theta_base, v):
    """Copy of foshan_r0_bounded.to_theta for indices 6..12 (the free group).

    Uses the archived values directly, so no round-trip through z is needed.
    """
    th = np.array(theta_base, dtype=float).copy()
    th[6] = np.log(v["cpa"] / (1.0 - v["cpa"]))
    th[7] = np.log(v["cpb"] / (1.0 - v["cpb"]))
    th[8] = np.log(1.0 / v["s_min"] - 1.0)
    th[9] = np.log(1.0 / v["r_base"] - 1.0)
    th[10] = -np.log(25.0 / v["r_off"] - 1.0)
    th[11] = np.log((v["esc_off"] - 5.0) / (25.0 - v["esc_off"]))
    th[12] = -np.log(7.0 / v["stag_b"] - 1.0)
    return th


def iter_solutions():
    """Yield every archived solution that can be faithfully rebuilt."""
    # --- (1) original centre calibration (11-elt theta + 2 fixed delays) ----
    with open(os.path.join(RES, "foshan_hv2_params.json")) as f:
        pj = json.load(f)
    yield {"run_tag": "hv2params_actual_1",
           "source_file": "foshan_hv2_params.json",
           "solution_id": "centre_theta",
           "theta": hv2.centre_theta(pj), "seed": None,
           "ref_r0": None, "oos": True,
           "note": "11-elt theta + [L_ESC0,L_STAG0]; default ode_zone (seed 5e-4)"}

    # --- (2) six multi-start kept vectors (13-elt, frozen) ------------------
    with open(os.path.join(RES, "foshan_hv2_multistart.json")) as f:
        ms = json.load(f)
    base_thetas = [np.asarray(t, dtype=float) for t in ms["kept_thetas"]]
    for k, th in enumerate(base_thetas, 1):
        yield {"run_tag": f"multistart_actual_{k}",
               "source_file": "foshan_hv2_multistart.json",
               "solution_id": f"kept_theta_{k}",
               "theta": th, "seed": None, "ref_r0": None, "oos": True,
               "note": "13-elt frozen; default ode_zone (seed 5e-4)"}

    # --- (3) bounded profile, baseline + loose sets -------------------------
    bp = pd.read_csv(os.path.join(RES, "bound_profile_i600.csv"))
    bp = bp[bp["bound_set"].isin(["baseline", "loose"])].reset_index(drop=True)
    for idx, row in bp.iterrows():
        si = int(row["start"])
        r0t = float(row["R0_target"])
        th_base = base_thetas[min(si, len(base_thetas) - 1)]
        s, _ = A.beta_scale_for_r0(th_base, r0t)
        th_anchor = A.scaled_theta(th_base, s)
        v = {k: float(row[k]) for k in BOUND_KEYS}
        th = apply_free_values(th_anchor, v)
        yield {"run_tag": f"boundprofile_{row['bound_set']}_R0{r0t:g}_s{si}",
               "source_file": "bound_profile_i600.csv",
               "solution_id": f"{row['bound_set']}|R0={r0t:g}|start={si}",
               "theta": th, "seed": float(row["seed"]),
               "ref_r0": float(row["R0_check"]), "oos": False,
               "note": (f"th_anchor(multistart start {si}) + archived 8 free "
                        "params; ode_zone_seeded")}

    # --- (4) all-free solutions at R0 = 1.24 and 7.28 -----------------------
    ff = pd.read_csv(os.path.join(RES, "fullfree_r0_i1500.csv"))
    ff = ff[ff["R0_target"].isin([1.24, 7.28])].reset_index(drop=True)
    for idx, row in ff.iterrows():
        v = {k: float(row[k]) for k in FF_KEYS}
        th = theta_from_free_values(v)
        th[0] = FF_BETA0[0] + float(np.log(float(row["beta_scale"])))
        th[1] = FF_BETA0[1] + float(np.log(float(row["beta_scale"])))
        yield {"run_tag": f"fullfree_R0{float(row['R0_target']):g}"
                          f"_s{int(row['start'])}",
               "source_file": "fullfree_r0_i1500.csv",
               "solution_id": f"R0={float(row['R0_target']):g}"
                              f"|start={int(row['start'])}",
               "theta": th, "seed": float(row["seed"]),
               "ref_r0": float(row["R0_check"]), "oos": False,
               "note": "12 free params from archive; beta_scale stored at 4 d.p.; "
                       "ode_zone_seeded"}


def evaluate_solution(sol, anchors, grid):
    """Run one solution and return per-anchor (date, obs, model_cum)."""
    daily_dates, pfi, t_daily, n, detect_idx, norm_idx = grid
    try:
        hv2.ode_zone = (A._ORIG_ODE if sol["seed"] is None
                        else A.ode_zone_seeded(sol["seed"]))
        r = hv2.run_two_zone(sol["theta"], daily_dates, pfi, t_daily,
                             n, detect_idx, norm_idx)
        cum_city = r["cum_A"].values + r["cum_B"].values
        cum_a = r["cum_A"].values
        r0_chk, _ = ngm_r0(sol["theta"])
    except Exception as exc:                    # pragma: no cover
        return None, None, f"replay failed: {type(exc).__name__}: {exc}"
    finally:
        hv2.ode_zone = A._ORIG_ODE
    out = []
    for a in anchors:
        i = int(np.argmax(daily_dates >= pd.Timestamp(a["date"])))
        model_cum = float(cum_a[i] if a["scope"] == "shunde" else cum_city[i])
        out.append((a["scope"], a["date"], a["observed"], model_cum))
    return out, float(r0_chk), None


def main(out_tag=""):
    grid = hv2.build_daily()
    anchors = held_out_anchors()
    print(f"held-out early anchors (pre {CALIB_START.date()}): "
          f"{len(anchors)}  "
          f"[{sum(a['scope'] == 'shunde' for a in anchors)} shunde / "
          f"{sum(a['scope'] == 'city' for a in anchors)} city]")

    rows = []
    n_ok = n_bad = 0
    for sol in iter_solutions():
        res, r0_chk, err = evaluate_solution(sol, anchors, grid)
        fidelity = sol["note"]
        if err is not None:
            replayed = False
            note = f"{fidelity}; {err}"
            n_bad += 1
            # still emit rows so the failure is explicit and countable
            res = [(a["scope"], a["date"], a["observed"], np.nan)
                   for a in anchors]
        else:
            replayed = True
            if sol["ref_r0"] is not None:
                dev = abs(r0_chk - sol["ref_r0"]) / max(abs(sol["ref_r0"]), 1e-9)
                if dev > 0.01:
                    replayed = False
                    note = (f"{fidelity}; R0 replay {r0_chk:.3f} vs archived "
                            f"{sol['ref_r0']:.3f} (dev {dev:.1%}) -> rejected")
                    n_bad += 1
                else:
                    note = f"{fidelity}; R0 replay ok ({r0_chk:.3f})"
                    n_ok += 1
            else:
                note = f"{fidelity}; R0 replay {r0_chk:.3f}"
                n_ok += 1
        for scope, dstr, obs, mcum in res:
            ratio = (mcum / obs) if (obs and np.isfinite(mcum)) else np.nan
            calib = "no" if sol["oos"] else "yes"
            rows.append({
                "run_tag": sol["run_tag"],
                "source_file": sol["source_file"],
                "solution_id": sol["solution_id"],
                "anchor_date": dstr,
                "observed": obs,
                "model_cum": (round(mcum, 1) if np.isfinite(mcum) else None),
                "ratio_model_over_observed":
                    (round(ratio, 3) if np.isfinite(ratio) else None),
                "replayed": bool(replayed),
                "note": (f"scope={scope}; early_anchor_in_calib={calib}; "
                         f"{note}"),
            })

    df = pd.DataFrame(rows, columns=[
        "run_tag", "source_file", "solution_id", "anchor_date", "observed",
        "model_cum", "ratio_model_over_observed", "replayed", "note"])
    suffix = f"__{out_tag}" if out_tag else ""
    primary = os.path.join(RES, f"out_of_sample_early{suffix}.csv")
    df.to_csv(primary, index=False, encoding="utf-8-sig")
    # run-setting-encoded archive copy (guards against accidental overwrite)
    archived = os.path.join(RES, f"out_of_sample_early__{RUN_SLUG}{suffix}.csv")
    df.to_csv(archived, index=False, encoding="utf-8-sig")

    # --- script-generated headline table (see the reproducibility contract)
    grp = df["note"].str.contains("early_anchor_in_calib=no")
    df2 = df.assign(group=np.where(grp, "holdout_no_early_anchor",
                                   "in_sample_478_in_calib"))
    head = (df2.groupby(["group", "anchor_date", "observed"])
                .agg(n_solutions=("run_tag", "nunique"),
                     model_cum_min=("model_cum", "min"),
                     model_cum_median=("model_cum", "median"),
                     model_cum_max=("model_cum", "max"),
                     ratio_min=("ratio_model_over_observed", "min"),
                     ratio_median=("ratio_model_over_observed", "median"),
                     ratio_max=("ratio_model_over_observed", "max"))
                .round(3).reset_index())
    head_path = os.path.join(RES, f"out_of_sample_early__headline{suffix}.csv")
    head.to_csv(head_path, index=False, encoding="utf-8-sig")

    n_sol = df["run_tag"].nunique()
    print(f"solutions emitted: {n_sol}  (faithfully replayed: {n_ok}, "
          f"rejected: {n_bad})")
    print(f"rows: {len(df)}")
    print(f"saved {os.path.basename(primary)}")
    print(f"saved {os.path.basename(archived)}")
    print(f"saved {os.path.basename(head_path)}")
    print("\n--- headline: model_cum and model/observed by anchor & group ---")
    print(head.to_string(index=False))
    return df


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else ""
    main(tag)
