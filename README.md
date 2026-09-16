# Code and data: partial identification of transmission intensity and response
# timing in the 2025 Foshan chikungunya outbreak, China

This package reproduces every number, table and figure in the manuscript: all
archived results are shipped under `results/`, and both manuscript figures are
regenerated from the shipped code and data by `code/foshan_eid_figures.py`.

The study asks what the 2025 Foshan case record can and cannot identify. Three
independent analyses fitted the same reported case series and returned R0
values spanning nearly threefold; this package contains the model, the
constraint experiments and the archived outputs behind the finding that the
apparent bound on R0 is a property of the assumed parameter constraints rather
than a measurement, together with a held-out comparison against the
pre-saturation anchors that the calibration set aside.

## Layout

```
code/
  foshan_cf.py               weather-driven potential-fecundity-index vector
                             density (Briere development, temperature/humidity
                             survival, saturating rainfall); `run` / `calibrate`
  foshan_hv2.py              two-zone host-vector model (Shunde core + rest of
                             city); `fit` / `multi` / `verify`
  foshan_validation.py       NGM R0 and early exponential growth rate per
                             multi-start fit; `r0` / `growth`
  foshan_figures.py          outbreak figures (workflow, reconstruction,
                             counterfactual, validation) and foshan_arms_cases_v2.csv
  foshan_r0_anchor.py        exploratory externally anchored R0 refits; also the
                             shared helper library (beta scaling, seeded ODE,
                             bound-hit flags) reused by the scripts below
  foshan_r0_grid.py          unbounded R0 grid scan in both reporting currencies
  foshan_r0_bounded.py       physically bounded re-fit across the R0 grid
                             (main experiment; logistic z-space mapping)
  foshan_r0_ceiling.py       differential-evolution global check and the
                             tight/baseline/loose constraint sensitivity
  foshan_bound_profile.py    constraint-set profile: best fit, bound hits and
                             across-start loss spread per (set, R0) cell
  foshan_fullfree_r0.py      all-12-parameter-free refit with R0 pinned exactly
  foshan_out_of_sample.py    held-out comparison against the early anchors
                             (all anchors dated before 2025-08-16)
  foshan_eid_figures.py      manuscript Figures 1 and 2 plus the traceability
                             table fig_points_used.csv
figures/                    pre-generated outputs. fig1_identifiability and
                             fig2_counterfactual_factors are the manuscript
                             Figures 1-2 (regenerate with
                             code/foshan_eid_figures.py); workflow,
                             reconstruction, counterfactual_trajectories and
                             validation_anchors are calibration-pipeline
                             diagnostics (regenerate with code/foshan_figures.py)
data/
  anchors.json               case, response-milestone and density anchors,
                             each entry carrying its source
  era5_foshan_daily_2024-2026.csv
                             daily ERA5 fields for Foshan (mean/max/min
                             temperature, precipitation, relative humidity)
results/                     every archived output (19 files, one producer each)
  foshan_hv2_params.json     central fit as archived (11-parameter vintage)
  foshan_hv2_fit_check.csv   21 daily rows recomputed from params.json by `verify`
  foshan_hv2_multistart.json six multi-start fits and the kept parameter vectors
  foshan_hv2_envelope.csv    multi-start reported-case envelope (20 rows)
  validation_r0.csv          NGM R0, growth rate, doubling time per fit (6 rows)
  foshan_arms_cases_v2.csv   five response arms, citywide cumulative (5 rows)
  r0_anchor_experiment.csv   exploratory anchored refits (5 rows)
  r0_grid_scan.csv           unbounded grid, 12 R0 targets (12 rows)
  r0_bounded_scan.csv        bounded main experiment, 12 R0 targets (12 rows)
  r0_bounded_starts.csv      per-start rows behind the bounded scan (36 rows)
  r0_ceiling.csv             global-search check + constraint sensitivity (8 rows)
  bound_profile_i600.csv     constraint profile, 20 (set, R0) cells (20 rows)
  bound_profile_starts_i600.csv
                             per-start rows behind the profile (60 rows)
  fullfree_r0_i1500.csv      all-free refit, 4 R0 targets (4 rows)
  fullfree_r0_starts_i1500.csv
                             per-start rows behind it (8 rows)
  out_of_sample_early.csv    held-out early-anchor comparison (312 rows)
  out_of_sample_early__headline.csv
                             grouped headline of the same (26 rows)
  out_of_sample_early__hv2params_multistart_boundprofile_bl_fullfree_1.24_7.28.csv
                             the same replay tagged with its run slug (312 rows)
  fig_points_used.csv        every point drawn in Figures 1-2 with its source
                             row and a `used` flag (60 rows)
requirements.txt
LICENSE
.gitignore
```

## How to run

```bash
pip install -r requirements.txt
cd code

# quick check (seconds): NGM R0 per multi-start fit -> ../results/validation_r0.csv
python foshan_validation.py r0

# outbreak figures and the five-arm table
python foshan_figures.py

# manuscript Figures 1-2 and fig_points_used.csv (writes ../02_Figures/)
python foshan_eid_figures.py

# held-out early-anchor comparison
python foshan_out_of_sample.py

# heavier refits (minutes to hours each)
python foshan_hv2.py fit            # central fit; OVERWRITES foshan_hv2_params.json
python foshan_hv2.py multi          # six-start uncertainty envelope
python foshan_hv2.py verify         # rebuild foshan_hv2_fit_check.csv from the
                                    # archived 11-parameter params.json (safe path)
python foshan_r0_bounded.py 900     # bounded main experiment
python foshan_bound_profile.py 600
python foshan_fullfree_r0.py 1500 2
python foshan_r0_ceiling.py
python foshan_r0_grid.py 2 700
python foshan_r0_anchor.py 2 700
python foshan_cf.py run             # v1 counterfactual layer (legacy; superseded)
```

Notes.

- All paths are relative to the package root; no script reads or writes outside
  it. `foshan_figures.py` and `foshan_eid_figures.py` write into `figures/`.
- `foshan_hv2.py fit` rewrites the archived parameter file with a 13-parameter
  vector. `verify` is the reproduction path that consumes the archived
  11-parameter file as-is; downstream scripts adapt to either length.
- The R0 targets are imposed, not fitted: the transmission coefficient is
  re-solved by fixed-point iteration until the next-generation-matrix R0 equals
  its target to within 1e-6.

## Data provenance

- Weather input: ERA5 reanalysis fields for Foshan obtained through the
  Open-Meteo archive (https://open-meteo.com/), shipped as
  `data/era5_foshan_daily_2024-2026.csv` (974 daily rows, no missing values).
- Case anchors, response milestones and Breteau-index density anchors are
  compiled in `data/anchors.json`; every entry carries the briefing or
  publication it was taken from. The calibration used the 13 city-level anchors
  dated 2025-08-16 or later; all earlier anchors are reserved for the
  out-of-sample comparison produced by `code/foshan_out_of_sample.py`.

## Software

Tested with Python 3.13.12 and numpy 2.4.6, pandas 3.0.5, scipy 1.18.0,
matplotlib 3.11.1 (see `requirements.txt`).

## License

MIT — see `LICENSE`.
