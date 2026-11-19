# Leakage-controlled tree-ensemble forecasting of dengue incidence (Brazil, OpenDengue V1.3)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24358097.svg)](https://doi.org/10.5281/zenodo.24358097)

Code, data and results for the article

> B. A. Glay, "Leakage-Controlled Tree-Ensemble Forecasting of Dengue Incidence Under Distribution Shift,"
> *Latin-American Journal of Computing (LAJC)*, vol. 14, no. 1, 2027.

The study forecasts weekly dengue incidence for 26 Brazilian states one, two and four weeks ahead.
Persistence, seasonal-naive, ARIMA(2,1,1) and ridge-autoregression baselines are compared with Random Forest
and XGBoost under three validation regimes, with an outbreak-alert classifier and grouped SHAP/permutation attribution.

Python 3.10+ (tested 3.13); `pip install -r requirements.txt`.

## Contents

| Path | What it is |
|---|---|
| `panel_state_week.csv` | Harmonised panel: 26 states x 573 weeks (5 Jan 2014 - 22 Dec 2024), 14,898 state-weeks |
| `build_panel.py` | Rebuilds the panel from the OpenDengue V1.3 temporal extract |
| `run_experiment.py`, `arima_baseline.py` | Forecasting experiments (regimes A, B, C) |
| `evaluate.py` | Pooled and per-fold accuracy, block-bootstrap intervals |
| `alerts_shap.py` | Alert classifiers, calibration, grouped SHAP and permutation reliance |
| `error_analysis.py` | Error by state size and realised direction |
| `make_figures.py` | Figures |
| `supp.py`, `supp2.py` | Supplementary analyses requested in peer review |
| `results/` | Result tables and predictions used in the article |
| `supplementary/` | Supplementary tables (CSV) and `Supplementary_Results_Dengue.docx` |

## Data
`panel_state_week.csv`: weekly dengue case counts for 26 Brazilian states, 2014-2024, built from OpenDengue V1.3
using only the MOH-BRA municipal "Probable" rows (the WHO/PAHO national rows double-count the same cases).
Espirito Santo is excluded because it has no records after December 2020.
`build_panel.py` rebuilds the panel from `Temporal_extract_V1_3.csv` (OpenDengue V1.3 release); edit the `SRC` path at the top.
Implicit zero-filled state-weeks are documented in the script output.

Source data and licence: OpenDengue V1.3, distributed under the Creative Commons Attribution 4.0 licence (CC BY 4.0).
The panel in this repository is a derived aggregate (municipal counts summed to state-week) and is redistributed under the
same licence. If you use it, please cite OpenDengue:

- J. Clarke, A. Lim, P. Gupte, D. M. Pigott, W. G. van Panhuis, and O. J. Brady, "A global dataset of publicly available
  dengue case count data," *Sci. Data*, vol. 11, no. 1, Art. no. 296, 2024, doi: 10.1038/s41597-024-03120-7.
- OpenDengue, "OpenDengue: data from the OpenDengue database," release V1.3, Figshare, 2025,
  doi: 10.6084/m9.figshare.24259573.v4. Repository: https://github.com/OpenDengue/master-repo

## Run order (run inside this folder)
1. `python run_experiment.py`    -> predictions.csv.gz (Persistence, SeasonalNaive, Ridge-AR, RandomForest, XGBoost; regimes A, B, C)
2. `python arima_baseline.py`    -> arima_predictions.csv.gz (regimes A and B)
3. merge the two prediction files into `predictions_all.csv.gz` (done at the top of `supp.py`), then `python evaluate.py` -> results_pooled.csv, results_by_fold.csv
4. `python alerts_shap.py`       -> alert_results.csv, alert_tiers.csv, shap/permutation tables
5. `python supp.py`              -> supplementary/ (regime B by year, regime C per state, bootstrap block lengths)
6. `python supp2.py`             -> phase-wise errors, alert operating points, prevalence and threshold by year (supplementary/)

Note: `evaluate.py` reads predictions_all.csv.gz, so run `supp.py` once (or merge manually) before step 3.
Scripts write their outputs to the working folder; the copies used in the article are in `results/` and `supplementary/`.
Runtime ~30-40 min on 2 CPU cores. Random seeds are fixed.

## Leakage control
Features use data up to the forecast origin only. Each example has a forecast origin week t and a target week t+h.
A training row is used only if its target week falls strictly before the first test origin (`tdate < t0` in
`run_experiment.py`), so the latest training target is the week before the test period; no further horizon-dependent gap is added.
Alert thresholds use each state's trailing 52-week 90th percentile known at the origin.
Regimes: A rolling-origin 2019-2024; B train on targets before 2022, test 2022-2024; C leave-one-state-out plus the same shift.

## Licence
Code: MIT (see `LICENSE`). Data and derived tables: CC BY 4.0, following the OpenDengue source licence.

## Citation
Please cite the article above and the OpenDengue sources. This repository (release v1.0) is archived on Zenodo:
B. A. Glay, "Code and data for: Leakage-Controlled Tree-Ensemble Forecasting of Dengue Incidence Under Distribution Shift," v1.0, Zenodo, 2026, doi: 10.5281/zenodo.24358097.
Citation metadata are in `CITATION.cff`.
