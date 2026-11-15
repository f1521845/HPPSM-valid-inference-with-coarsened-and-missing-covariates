# HPPSM: valid inference with coarsened and missing covariates

Code for *Valid inference with coarsened and missing covariates in critical care prognostic models: a hybrid probability–possibility framework*.

## Contents

| Path | What it is |
|---|---|
| `code/hppsm_v3.py` | Core library. It contains the Royston–Parmar model with analytic derivatives and Newton refinement, logistic regression, Rubin's rules, SMC-FCS imputation, regression calibration, the error-variance bound, and the envelope search (full vertex enumeration with axis configurations, adaptive vertices, 2q + 3 configurations, dense search). |
| `code/s2_study.py` | SUPPORT2 plasmode simulation: main mechanisms M1–M4 (`run_rep` for the non-SMC methods, `smc_rep` for every SMC-FCS-based method), extended scenarios, dense-search verification and C-index. |
| `code/p12_build.py` | Builds the per-stay summary table from the raw PhysioNet 2012 files. |
| `code/p12_study.py` | PhysioNet 2012 analyses: measurement-error diagnostics, the resampling experiment, dense search, and the application with crisp and graded contours. |
| `code/me_sim.py` | Synthetic measurement-error simulation (108 scenarios). |
| `notebooks/01_SUPPORT2_HPPSM_analysis.ipynb` | Additional file 1: self-contained notebook reproducing the SUPPORT2 results. |
| `notebooks/02_PhysioNet2012_HPPSM_analysis.ipynb` | Additional file 2: self-contained notebook reproducing the PhysioNet results and the measurement-error simulation. |
| `run_all.sh` | Runs every analysis from the command line. Each step can be resumed if interrupted. |

## Data
* **SUPPORT2:** public (UCI Machine Learning Repository, doi:10.3886/ICPSR02957.v2). The code loads it through the `SurvSet` package.
* **PhysioNet/CinC Challenge 2012:** https://physionet.org/content/challenge-2012/ (ODC-BY 1.0).
  * Download `set-a`, `set-b`, `set-c` and `Outcomes-a/b/c.txt`.
  * Set `P12_RAWDIR` (command line) or `DATA_DIR` (notebook) to their folder.

## Installation
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
The versions in `requirements.txt` are those used for the paper (Python 3.13.18).

## Reproducing the results
* **Quick check (about 30 minutes):** open either notebook with `QUICK = True` and run all cells.
* **Full study:** use `QUICK = False`, or run `./run_all.sh`.
  * Total run time is about 6 hours on two cores. Most of it is the SUPPORT2 main simulation; chained-equation MI accounts for half of the time in its first 200 replicates.
  * Results are appended to CSV files after every replicate, so a run can be stopped and restarted.
  * Replicates can be split across machines with `--start`/`--reps`.
* **Seeds:** replicate *r* uses seed 10,000 + *r* (SUPPORT2) or 50,000 + *r* (PhysioNet). SMC-FCS imputation uses seed 7,000 + *r*.
* **Model fits:** fits are refined to a score norm below 1e-8, so results do not depend on optimiser tolerances.

**Verification.** Both notebooks were executed in quick mode in a clean virtual environment created from `requirements.txt`. Their replicates were compared with the full saved results and agreed to within 1e-7 (most exactly). The executed copies are in `notebooks/executed/`.

## Author
Raghad Alhuthil, Centre for Statistics in Medicine, NDORMS, University of Oxford.

## Saved results (`results_s2/`, `results_p12/`, `results_me/`)
* `results_s2/<scenario>_0.csv`: oracle, fill-in, complete-case and chained-equation MI results. These are the results used in the paper. The SMC-FCS rows in these files come from an earlier version of the sampler and are superseded.
* `results_s2/<scenario>_smc.csv`: every SMC-FCS-based method, from the final sampler.
  * The sampler uses exact truncated-normal proposals and keeps the previous value, explicitly, for any value not accepted within 1,000 proposals.
  * `HPPSM v2 (proposed)` is the primary interval (vertices + axes).
* `results_s2/<scenario>_smc_info.csv`: realised MNAR shifts and rejection-sampling diagnostics.
* `results_s2/dense_*.csv`: dense-search verification.
* `results_s2/cindex_support2.csv`: SUPPORT2 discrimination results.
* `results_p12/`: PhysioNet resampling experiment (`vertex_B_0.csv` holds the primary scenario-B interval), dense search, measurement-error diagnostics and application.
* `results_me/me_sim.csv.gz`: measurement-error simulation (gzip-compressed to stay under GitHub's 25 MB file limit; `pandas.read_csv` reads it directly).

## Licence
MIT (see `LICENSE`).
