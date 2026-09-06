# foshan-chikungunya-counterfactual

Code and data for the study:

> **Response timing and the cost of delay: a model-based policy counterfactual of the 2025 Foshan chikungunya outbreak in China**
> Tian-Yu Wang (ORCID: [0000-0002-5124-7965](https://orcid.org/0000-0002-5124-7965))

The repository contains everything needed to reproduce the model reconstruction, the counterfactual scenario analysis and all figures of the manuscript. No individual-level or human-subject data are involved; all inputs are publicly reported aggregate surveillance quantities and reanalysis weather.

## Repository structure

```
├── code/
│   ├── foshan_cf.py        Weather-driven mechanistic density model (PFI) + response encoding + five-arm screening
│   ├── foshan_hv2.py       Two-zone host-vector transmission model (fit / multi-start uncertainty envelope)
│   └── foshan_figures.py   Reproduces all four manuscript figures from the archived model outputs
├── data/
│   ├── anchors.json        Machine-readable anchor ledger: every quantitative claim with its source
│   └── era5_foshan_daily_2024-2026.csv   Daily ERA5 reanalysis weather (via Open-Meteo), 974 days
├── results/                Archived model outputs used in the manuscript (fit checks, multi-start envelope, parameters, scenario table)
└── figures/                Manuscript figures (PNG 300 dpi + PDF vector)
```

## Requirements

Python 3.11+ with `numpy`, `pandas`, `scipy`, `matplotlib`.

## Reproduction

All model outputs used in the manuscript are archived in `results/`, so the figures can be regenerated directly:

```bash
python code/foshan_figures.py
```

To refit everything from scratch, run in order:

```bash
python code/foshan_cf.py calibrate   # density-model calibration grid
python code/foshan_cf.py run         # density model + five response arms (screening layer)
python code/foshan_hv2.py fit        # two-zone host-vector model, single fit
python code/foshan_hv2.py multi      # six-start multi-start uncertainty envelope
python code/foshan_figures.py        # figures from the refitted outputs
```

Model design, calibration windows, scenario definitions and all parameter values are documented in the manuscript (Methods).

## Data provenance

- **Outbreak and response anchors**: compiled from municipal and provincial press briefings, official releases and published papers. Every value in `data/anchors.json` carries its source; values that failed verification are listed under `excluded_values` with reasons.
- **Weather**: ERA5 reanalysis retrieved through the [Open-Meteo archive](https://open-meteo.com/), point 23.02N 113.13E (Chancheng, Foshan).

## License

MIT — see [LICENSE](LICENSE).

## Citation

If you use this code or the anchor database, please cite the manuscript.
