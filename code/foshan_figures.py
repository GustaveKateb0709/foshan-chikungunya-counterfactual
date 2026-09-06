#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""foshan_figures.py — publication figures for the Foshan counterfactual
paper.  300 dpi PNG + vector PDF; English labels; single palette; titles
follow "Fig. N. <clause>." and match filenames figN_* (three-way rule).

  fig1_workflow        study schematic
  fig2_reconstruction  reported curve + model reconstruction + detection wave
  fig3_counterfactual  five arms, citywide reported cumulative, multistart band
  fig4_validation      under-ascertainment envelope + density anchors
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import foshan_cf
import foshan_hv2 as hv2

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE, ".."))
RES = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "figures")
os.makedirs(FIGS, exist_ok=True)

INK = "#2c3e50"
C_NONE = "#95a5a6"
C_THRESH = "#5d6d7e"
C_CAL = "#e67e22"
C_PPO = "#c0392b"
C_SR = "#2e86c1"
C_SHADE = "#fdebd0"
ARM_COLOR = {"none": C_NONE, "actual": C_PPO, "early": C_SR,
             "late": C_CAL, "rule": C_THRESH}
ARM_LABEL = {"none": "No response", "actual": "Actual response (fitted)",
             "early": "Response 2 weeks earlier",
             "late": "Response 2 weeks later",
             "rule": "BI-threshold trigger (BI > 5)"}

_DAILY = None


def build_daily_args():
    """Cached daily grid (dates, pfi, T, n, detect_idx, norm_idx)."""
    global _DAILY
    if _DAILY is None:
        _DAILY = hv2.build_daily()
    return _DAILY


def load_thetas():
    with open(os.path.join(RES, "foshan_hv2_multistart.json")) as f:
        ms = json.load(f)
    return ms["kept_thetas"], ms["summary"]


def finish(fig, name, title):
    fig.suptitle(title, x=0.01, y=0.985, ha="left", va="top",
                 fontsize=13, fontweight="bold", color=INK)
    for ext in (".png", ".pdf"):
        fig.savefig(os.path.join(FIGS, name + ext), dpi=300,
                    facecolor="white")
    plt.close(fig)
    print("  saved", name, flush=True)


def date_ax(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def collect_curves():
    """Five arms x multistart thetas -> citywide reported-cumulative."""
    daily_dates, pfi, t_daily, n, detect_idx, norm_idx = build_daily_args()
    thetas, _ = load_thetas()
    arms = ("none", "actual", "early", "late", "rule")
    curves = {a: [] for a in arms}
    for th in thetas:
        th = np.asarray(th)
        calib_a = float(np.exp(th[4]))
        bi_none = pfi * calib_a              # zone A density, no response
        trigger_idx = int(np.argmax(bi_none > 5.0))
        for arm, shift, dover in (
                ("none", 0, None),
                ("actual", 0, None),
                ("early", -14, None),
                ("late", 14, None),
                ("rule", 0, trigger_idx)):
            r = hv2.run_two_zone(th, daily_dates, pfi, t_daily, n,
                                 detect_idx, norm_idx, shift_weeks=shift,
                                 arm=arm, detect_idx_override=dover)
            rho = r["r_base"] + (1 - r["r_base"]) * r["report_step"]
            rep = np.cumsum(rho * (r["onsets_A"] + r["onsets_B"]))
            curves[arm].append(pd.Series(rep, index=daily_dates))
    return curves, len(thetas)


def fig1_workflow():
    fig, ax = plt.subplots(figsize=(11.4, 4.9))
    ax.set_xlim(0, 124)
    ax.set_ylim(1, 44)
    ax.axis("off")
    W, H, GAP, M = 27.5, 17.5, 3.5, 3.5
    xs = [M + W / 2 + i * (W + GAP) for i in range(4)]
    CY = 31.0
    stages = [
        (xs[0], "Weather & density",
         "ERA5 reanalysis (Foshan);\nPFI mechanistic model",
         C_NONE, "#f4f6f7"),
        (xs[1], "Response timeline",
         "case finding 07-08;\ncitywide campaign 07-19",
         C_CAL, "#fdf2e9"),
        (xs[2], "Transmission ODE",
         "8 compartments;\ntemperature-dependent\nEIP & competence",
         C_SR, "#eaf2f8"),
        (xs[3], "Counterfactual arms",
         "none / actual /\n±2 weeks / BI trigger",
         C_PPO, "#fdedec"),
    ]
    for x, head, body, ec, fc in stages:
        ax.add_patch(FancyBboxPatch((x - W / 2, CY - H / 2), W, H,
                                    boxstyle="round,pad=0.02,"
                                             "rounding_size=1.2",
                                    lw=1.5, edgecolor=ec, facecolor=fc))
        ax.text(x, CY + H / 2 - 3.4, head, ha="center", va="center",
                fontsize=11.5, fontweight="bold", color=INK)
        ax.text(x, CY - 3.0, body, ha="center", va="center", fontsize=9.4,
                color=INK, linespacing=1.5)
    for i in range(3):
        ax.add_patch(FancyArrowPatch((xs[i] + W / 2 + 0.4, CY),
                                     (xs[i + 1] - W / 2 - 0.4, CY),
                                     arrowstyle="-|>", mutation_scale=17,
                                     lw=1.8, color=C_SR))
    RB_W, RB_H, RB_CY = 82.0, 14.0, 10.0
    ax.add_patch(FancyBboxPatch((63 - RB_W / 2, RB_CY - RB_H / 2), RB_W,
                                RB_H,
                                boxstyle="round,pad=0.02,rounding_size=1.2",
                                lw=1.4, edgecolor=C_NONE, facecolor="#f4f6f7"))
    ax.text(63, RB_CY + 2.8, "Calibration & validation anchors",
            ha="center", va="center", fontsize=10.5, fontweight="bold",
            color=INK)
    ax.text(63, RB_CY - 1.4,
            "official case curve (15+ anchors) · citywide BI (8.5 / 6.2)\n"
            "response quantification (90% / 33% / 78% / 95%)",
            ha="center", va="center", fontsize=9.2, color=INK,
            linespacing=1.4)
    ax.set_ylim(1.5, 46)
    finish(fig, "fig1_workflow",
           "Fig. 1. Study workflow.")


def fig2_reconstruction(daily_dates, curves):
    stack = pd.concat(curves["actual"], axis=1)
    med = stack.median(axis=1)
    lo, hi = stack.min(axis=1), stack.max(axis=1)
    fig, ax = plt.subplots(figsize=(10.4, 5.4))
    ax.axvspan(pd.Timestamp("2025-07-15"), pd.Timestamp("2025-08-09"),
               color=C_SHADE, alpha=0.55, lw=0)
    ax.plot(daily_dates, med.values, color=C_PPO, lw=2.0,
            label="Model reconstruction (reported)")
    ax.fill_between(daily_dates, lo.values, hi.values, color=C_PPO,
                    alpha=0.14, lw=0, label="Multi-start envelope")
    obs = hv2.ANCHORS_CITY
    ax.plot([pd.Timestamp(d) for d, _ in obs], [v for _, v in obs], "o",
            ms=4.5, color=INK, label="Official reported anchors")
    ax.plot([pd.Timestamp("2025-07-15")], [478], "D", ms=6, mfc="white",
            mec=C_THRESH, mew=1.6, label="Shunde only (pre-wave)")
    ax.axvline(pd.Timestamp("2025-07-19"), color=C_THRESH, ls="--", lw=1.1)
    ax.text(pd.Timestamp("2025-07-20"), 300, "citywide campaign\n(07-19/20)",
            fontsize=8.2, color=INK)
    ax.text(pd.Timestamp("2025-08-06"), 11500,
            "detection wave\n(active screening)", fontsize=8.2,
            color=C_THRESH)
    date_ax(ax)
    ax.set_ylabel("Cumulative reported cases")
    ax.set_ylim(0, 12500)
    ax.legend(loc="lower right", fontsize=9.4, frameon=False,
              handlelength=1.6)
    finish(fig, "fig2_reconstruction",
           "Fig. 2. Reported cases, model reconstruction and the "
           "detection wave.")


def fig3_counterfactual(daily_dates, curves):
    fig, ax = plt.subplots(figsize=(10.4, 5.4))
    for arm in ("none", "late", "actual", "early", "rule"):
        stack = pd.concat(curves[arm], axis=1)
        med = stack.median(axis=1)
        lo, hi = stack.min(axis=1), stack.max(axis=1)
        lw = 2.2 if arm == "actual" else 1.5
        ax.plot(daily_dates, med.values, color=ARM_COLOR[arm], lw=lw,
                label=f"{ARM_LABEL[arm]}  (final {med.iloc[-1]:,.0f})")
        ax.fill_between(daily_dates, lo.values, hi.values,
                        color=ARM_COLOR[arm], alpha=0.10, lw=0)
    obs = [(d, v) for d, v in hv2.ANCHORS_CITY if d >= "2025-07-19"]
    ax.plot([pd.Timestamp(d) for d, _ in obs], [v for _, v in obs], "o",
            ms=4, color=INK, label="Observed (actual response)")
    date_ax(ax)
    ax.set_ylim(0, 132000)
    import matplotlib.ticker as mticker
    ax.set_yticks([0, 20000, 40000, 60000, 80000, 100000, 120000])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _p: f"{v / 10000:g}"))
    ax.set_ylabel("Cumulative reported cases ($\\times 10^4$)")
    ax.legend(loc="upper left", fontsize=9.8, frameon=False,
              handlelength=1.6)
    finish(fig, "fig3_counterfactual",
           "Fig. 3. Counterfactual reported-case trajectories under five "
           "response scenarios.")


def fig4_validation(daily_dates, curves, thetas):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.8),
                                   gridspec_kw={"width_ratios": [1, 1.3]})
    factors = []
    for th in thetas:
        th = np.asarray(th)
        i715 = int(np.argmax(daily_dates >= pd.Timestamp("2025-07-15")))
        r = hv2.run_two_zone(th, daily_dates, *build_daily_args()[1:3],
                             *build_daily_args()[3:], arm="actual")
        true715 = float(r["cum_A"].values[i715] + r["cum_B"].values[i715])
        factors.append(true715 / 478.0)
    ax1.bar(np.arange(len(factors)), factors,
            color=[C_PPO if k == 0 else C_SR for k in range(len(factors))],
            width=0.6)
    ax1.set_xticks(np.arange(len(factors)))
    ax1.set_xticklabels([f"S{k}" for k in range(len(factors))], fontsize=8)
    ax1.set_ylabel("True-to-reported ratio\nat 2025-07-15")
    ax1.set_xlabel("Multi-start fit")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    central = np.asarray(thetas[0])
    daily_dates2, pfi2, t_daily2, n2, di2, ni2 = build_daily_args()
    r = hv2.run_two_zone(central, daily_dates2, pfi2, t_daily2, n2, di2,
                         ni2, arm="actual")
    ax2.plot(daily_dates2, r["BI_A"].values, color=C_PPO, lw=1.8,
             label="Shunde (hot-spot zone)")
    ax2.plot(daily_dates2, r["BI_B"].values, color=C_SR, lw=1.8,
             label="Other districts")
    for period, val, lab, axpos in (
            ("2025-07", 8.5, "Jul citywide 8.5", (0.02, 1.04)),
            ("2025-08", 6.2, "Aug citywide 6.2", (0.30, 1.04))):
        ax2.scatter([pd.Timestamp(period + "-15")], [val], s=42, zorder=5,
                    color=INK)
        ax2.annotate(lab, (pd.Timestamp(period + "-15"), val),
                     xytext=axpos, textcoords="axes fraction", fontsize=8,
                     color=INK,
                     arrowprops=dict(arrowstyle="-", lw=0.7, color=INK))
    fig.subplots_adjust(top=0.84)
    ax2.axhline(5.0, color=INK, ls=":", lw=1.0)
    ax2.text(daily_dates2[-1], 5.35, "BI = 5 risk threshold", ha="right",
             fontsize=8, color=INK)
    ax2.set_xlim(pd.Timestamp("2025-06-01"), pd.Timestamp("2025-12-31"))
    ax2.set_ylim(0, 14)
    date_ax(ax2)
    ax2.set_ylabel("Breteau index")
    ax2.legend(loc="upper right", fontsize=8.2, frameon=False)
    ax1.text(0.5, -0.20, "(a) Early under-ascertainment (multi-start envelope)",
             transform=ax1.transAxes, ha="center", va="top", fontsize=9.5,
             color=INK)
    ax2.text(0.5, -0.20, "(b) Response-adjusted BI vs surveillance anchors",
             transform=ax2.transAxes, ha="center", va="top", fontsize=9.5,
             color=INK)
    fig.subplots_adjust(bottom=0.20, wspace=0.24)
    finish(fig, "fig4_validation",
           "Fig. 4. Under-ascertainment envelope and density-model "
           "validation.")


def main():
    daily_dates, pfi, t_daily, n, detect_idx, norm_idx = build_daily_args()
    thetas, ms_summary = load_thetas()
    print("multistart thetas:", len(thetas), "| summary:", ms_summary)
    fig1_workflow()
    curves, n_th = collect_curves()
    fig2_reconstruction(daily_dates, curves)
    fig3_counterfactual(daily_dates, curves)
    fig4_validation(daily_dates, curves, thetas)
    rows = []
    act_med = float(pd.concat(curves["actual"], axis=1).iloc[-1].median())
    for arm in ("none", "actual", "early", "late", "rule"):
        stack = pd.concat(curves[arm], axis=1)
        finals = stack.iloc[-1].values
        rows.append({"arm": arm,
                     "final_median": round(float(np.median(finals))),
                     "final_min": round(float(finals.min())),
                     "final_max": round(float(finals.max())),
                     "ratio_vs_actual": round(float(np.median(finals))
                                              / act_med, 3)})
    tbl = pd.DataFrame(rows)
    tbl.to_csv(os.path.join(RES, "foshan_arms_cases_v2.csv"), index=False,
               encoding="utf-8-sig")
    print(tbl.to_string(index=False))
    print("all figures + arm table done ->", FIGS)


if __name__ == "__main__":
    main()
