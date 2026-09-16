#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""foshan_eid_figures.py — identifiability figures for the Foshan chikungunya analysis.

Produces (both .png and .pdf, 300 dpi, white background, colour-blind-safe):

  figures/fig1_identifiability.png/.pdf
      (a) achieved NGM R0 (x) vs isolation residual susceptibility s_min (y),
          one line per constraint set; horizontal dashed lines mark each set's
          s_min lower bound -> shows each line descending onto its own lower
          bound as R0 rises (at the lowest target s_min is still above it).
      (b) achieved NGM R0 (x) vs post-saturation loss (y, log scale), same
          three sets; vertical dashed lines mark each set's apparent R0
          ceiling -> shows where the fit collapses.

  figures/fig2_counterfactual_factors.png/.pdf
      three panels (early / late / none) showing every ACCEPTABLE solution's
      counterfactual factor (true-infection final size relative to the
      actual-response arm), log x-axis, vertical baseline at 1.0
      -> early factors all < 1, late factors all > 1, no-response factors
         far > 1 (direction is robust) while the horizontal spread within an
         arm is orders of magnitude (magnitude is not identified). Points are
         vertically offset only to separate overlapping solutions; the
         vertical axis carries no quantitative meaning.

  results/fig_points_used.csv
      one row per point ACTUALLY DRAWN, with the source file, the plotted
      columns and a boolean `used` flag:
        panel = fig1a / fig1b : every row of bound_profile_i600.csv
                                (y = s_min / loss_post, all drawn -> used=True)
        panel = fig2          : the candidate acceptable rows, `used` = kept
                                after de-duplication.

Data sources (read locally, no network):
  results/bound_profile_i600.csv        best fit per (bound_set, R0_target)
  results/fullfree_r0_i1500.csv         unbounded "full-free" refits, best per R0
  results/fullfree_r0_starts_i1500.csv  the same refits, one row per start

ACCEPTABLE-SOLUTION RULE (Fig 2; stated verbatim for auditing)
--------------------------------------------------------------
  a solution is accepted iff it is a row of
      bound_profile_i600.csv         with loss_post <= 2.0, OR
      fullfree_r0_i1500.csv          with loss_post <= 1.0, OR
      fullfree_r0_starts_i1500.csv   with loss_post <= 1.0.
  Exact duplicate solutions between the fullfree "best" and "start" files
  (the best row is always one of the start rows) are counted ONCE; every
  candidate row is written to fig_points_used.csv with `used` = True/False.

Usage:  python foshan_eid_figures.py
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE, ".."))
RES = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "figures")
os.makedirs(FIGS, exist_ok=True)

# --- style -----------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.9,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

# colour-blind-safe palette (Okabe-Ito)
SET_STYLE = {
    "tight":    {"color": "#0072B2", "marker": "o", "ls": "-",  "label": "tight"},
    "baseline": {"color": "#E69F00", "marker": "s", "ls": "--", "label": "baseline"},
    "loose":    {"color": "#009E73", "marker": "^", "ls": ":",  "label": "loose"},
    "free":     {"color": "#CC79A7", "marker": "D", "ls": "",   "label": "full-free"},
}
S_MIN_BOUND = {"tight": 0.35, "baseline": 0.20, "loose": 0.05}
R0_CEILING = {"tight": 2.0, "baseline": 3.5, "loose": 5.5}
SET_ORDER = ["tight", "baseline", "loose"]

LOSS_POST_THRESH_BOUND = 2.0
LOSS_POST_THRESH_FREE = 1.0


# --- data ------------------------------------------------------------------
def load_data():
    b = pd.read_csv(os.path.join(RES, "bound_profile_i600.csv"))
    ff = pd.read_csv(os.path.join(RES, "fullfree_r0_i1500.csv"))
    fs = pd.read_csv(os.path.join(RES, "fullfree_r0_starts_i1500.csv"))
    return b, ff, fs


def acceptable_rows(b, ff, fs):
    """Apply the documented rule and return the unique accepted solutions."""
    cand = []
    for _, r in b[b.loss_post <= LOSS_POST_THRESH_BOUND].iterrows():
        cand.append({"source_file": "bound_profile_i600.csv",
                     "bound_set": r.bound_set, "start": r.get("start", ""),
                     "R0_check": r.R0_check, "s_min": r.s_min,
                     "loss_post": r.loss_post,
                     "tru_early_x": r.tru_early_x, "tru_late_x": r.tru_late_x,
                     "tru_none_x": r.tru_none_x})
    for _, r in ff[ff.loss_post <= LOSS_POST_THRESH_FREE].iterrows():
        cand.append({"source_file": "fullfree_r0_i1500.csv",
                     "bound_set": "free", "start": r.get("start", ""),
                     "R0_check": r.R0_check, "s_min": r.s_min,
                     "loss_post": r.loss_post,
                     "tru_early_x": r.tru_early_x, "tru_late_x": r.tru_late_x,
                     "tru_none_x": r.tru_none_x})
    for _, r in fs[fs.loss_post <= LOSS_POST_THRESH_FREE].iterrows():
        cand.append({"source_file": "fullfree_r0_starts_i1500.csv",
                     "bound_set": "free", "start": r.get("start", ""),
                     "R0_check": r.R0_check, "s_min": r.s_min,
                     "loss_post": r.loss_post,
                     "tru_early_x": r.tru_early_x, "tru_late_x": r.tru_late_x,
                     "tru_none_x": r.tru_none_x})

    seen, used = set(), []
    for c in cand:
        key = (round(float(c["loss_post"]), 6), round(float(c["R0_check"]), 4),
               round(float(c["tru_early_x"]), 6),
               round(float(c["tru_late_x"]), 6),
               round(float(c["tru_none_x"]), 6))
        c["used"] = key not in seen
        if c["used"]:
            seen.add(key)
            used.append(c)
    return pd.DataFrame(cand), pd.DataFrame(used)


# --- figure 1 --------------------------------------------------------------
def fig1(b):
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.2, 3.15))

    for s in SET_ORDER:
        d = b[b.bound_set == s].sort_values("R0_check")
        st = SET_STYLE[s]
        axa.plot(d.R0_check, d.s_min, color=st["color"], marker=st["marker"],
                 ls=st["ls"], lw=1.5, ms=5, mfc="white", mew=1.3,
                 label=st["label"], zorder=3)
        axb.plot(d.R0_check, d.loss_post, color=st["color"],
                 marker=st["marker"], ls=st["ls"], lw=1.5, ms=5,
                 mfc="white", mew=1.3, label=st["label"], zorder=3)

    # (a) per-set s_min lower bounds
    for s in SET_ORDER:
        axa.axhline(S_MIN_BOUND[s], color=SET_STYLE[s]["color"], ls="--",
                    lw=1.0, alpha=0.75, zorder=1)
        axa.annotate(f"{s} bound\n{S_MIN_BOUND[s]:.2f}",
                     xy=(axa.get_xlim()[1], S_MIN_BOUND[s]),
                     xytext=(-4, 3), textcoords="offset points",
                     ha="right", va="bottom", fontsize=7,
                     color=SET_STYLE[s]["color"])
    axa.set_xlabel("Achieved $R_0$ (next-generation matrix)")
    axa.set_ylabel("Isolation residual susceptibility  $s_{min}$")
    axa.set_ylim(0, 1.0)
    axa.legend(frameon=False, fontsize=8, loc="upper right", handlelength=1.8)

    # (b) apparent R0 ceilings
    for s in SET_ORDER:
        axb.axvline(R0_CEILING[s], color=SET_STYLE[s]["color"], ls="--",
                    lw=1.0, alpha=0.7, zorder=1)
        axb.annotate(f"{s}\n≈{R0_CEILING[s]:g}", xy=(R0_CEILING[s], 6e3),
                     xytext=(3, 0), textcoords="offset points",
                     ha="left", va="top", fontsize=7,
                     color=SET_STYLE[s]["color"])
    axb.set_yscale("log")
    axb.set_xlabel("Achieved $R_0$ (next-generation matrix)")
    axb.set_ylabel("Post-saturation loss  (log scale)")
    axb.set_ylim(1e-2, 1e4)
    # no legend here: panel (a) already carries the shared set legend

    for ax, lab in ((axa, "(a)"), (axb, "(b)")):
        ax.text(-0.02, 1.06, lab, transform=ax.transAxes, fontsize=10,
                fontweight="bold", va="top", ha="left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.16, top=0.9,
                        wspace=0.30)
    _save(fig, "fig1_identifiability")


# --- figure 2 --------------------------------------------------------------
def fig2(used):
    panels = [("early", "tru_early_x", "(a) Response advanced 14 days"),
              ("late", "tru_late_x", "(b) Response delayed 14 days"),
              ("none", "tru_none_x", "(c) No response")]
    # common log x-axis shared by all three panels so the 1.0 baseline is a
    # single reference across arms; limits span every plotted factor.
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), sharex=True)
    rng = np.random.default_rng(20260916)
    XLO, XHI = 0.02, 1.2e3

    for ax, (arm, col, lab) in zip(axes, panels):
        vals = used[col].astype(float).values
        # deterministic vertical jitter so identical values stay visible
        y = rng.uniform(-0.16, 0.16, size=len(vals))
        for s in ["tight", "baseline", "loose", "free"]:
            m = (used.bound_set == s).values
            if not m.any():
                continue
            st = SET_STYLE[s]
            ax.scatter(vals[m], y[m], s=32, color=st["color"],
                       marker=st["marker"], edgecolor="white", linewidth=0.5,
                       label=st["label"], zorder=3)
        ax.set_xscale("log")
        ax.set_xlim(XLO, XHI)
        ax.set_xticks([0.1, 1.0, 10.0, 100.0, 1000.0])
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, p: "%g" % v))
        ax.axvspan(XLO, 1.0, color="#000000", alpha=0.04, zorder=0)
        ax.axvline(1.0, color="#444444", ls="-", lw=1.1, zorder=1)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_title(lab, fontsize=8.5, pad=4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(labelsize=8)

    axes[0].text(1.6, 0.44, "1.0 = actual", fontsize=7, ha="left",
                 va="center", color="#444444")
    # single shared x-label (repeating it per panel would overlap)
    fig.text(0.512, 0.15, "Counterfactual factor vs actual response",
             ha="center", va="center", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, ncol=4,
               loc="lower center", bbox_to_anchor=(0.5, -0.02),
               handletextpad=0.2, columnspacing=1.1)
    # disclose the vertical-axis semantics: the offsets are pure de-overlap
    fig.text(0.512, 0.088,
             "Vertical offsets separate overlapping solutions; the vertical "
             "axis carries no quantitative meaning.",
             ha="center", va="center", fontsize=6.4, color="#555555")
    fig.subplots_adjust(left=0.035, right=0.99, bottom=0.30, top=0.86,
                        wspace=0.12)
    _save(fig, "fig2_counterfactual_factors")


def _save(fig, name):
    for ext in (".png", ".pdf"):
        fig.savefig(os.path.join(FIGS, name + ext), dpi=300,
                    facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("  saved", name, ".png/.pdf ->", FIGS, flush=True)


# --- main ------------------------------------------------------------------
def main():
    b, ff, fs = load_data()
    cand, used = acceptable_rows(b, ff, fs)

    # traceability table: every point ACTUALLY DRAWN + its source row.
    #   fig1a / fig1b : all rows of bound_profile_i600.csv (y = s_min / loss_post)
    #   fig2          : candidate acceptable rows (used = kept after de-dup)
    f1 = b[["bound_set", "R0_check", "s_min", "loss_post",
            "tru_early_x", "tru_late_x", "tru_none_x"]].copy()
    f1.insert(0, "source_file", "bound_profile_i600.csv")
    f1["used"] = True
    f1a = f1.copy(); f1a.insert(0, "panel", "fig1a")
    f1b = f1.copy(); f1b.insert(0, "panel", "fig1b")
    f2 = cand[["source_file", "bound_set", "R0_check", "s_min", "loss_post",
               "tru_early_x", "tru_late_x", "tru_none_x", "used"]].copy()
    f2.insert(0, "panel", "fig2")
    cols = ["panel", "source_file", "bound_set", "R0_check", "s_min",
            "loss_post", "tru_early_x", "tru_late_x", "tru_none_x", "used"]
    trace = pd.concat([f1a[cols], f1b[cols], f2[cols]], ignore_index=True)
    trace.to_csv(os.path.join(RES, "fig_points_used.csv"), index=False,
                 encoding="utf-8-sig")
    print("fig_points_used.csv: %d rows (fig1a %d, fig1b %d, fig2 %d)"
          % (len(trace), len(f1a), len(f1b), len(f2)))
    print("  fig2 candidates: %d rows, %d used (%d duplicates)"
          % (len(cand), int(cand.used.sum()), len(cand) - int(cand.used.sum())))

    fig1(b)
    fig2(used)

    print("\n-- panel counts --")
    for s in SET_ORDER:
        print("Fig 1a/1b %-8s : %d points" % (s, (b.bound_set == s).sum()))
    print("Fig 2 acceptable solutions: %d" % len(used))
    for s in ["tight", "baseline", "loose", "free"]:
        print("   Fig 2  %-8s : %d points" % (s, (used.bound_set == s).sum()))
    for arm, col, _ in (("early", "tru_early_x", ""),
                        ("late", "tru_late_x", ""),
                        ("none", "tru_none_x", "")):
        v = used[col].astype(float)
        print("   Fig 2  %-5s : n=%d, all %s 1.0 (min %.3f, max %.3f)"
              % (arm, len(v), "<" if (v < 1).all() else ">",
                 v.min(), v.max()))


if __name__ == "__main__":
    main()
