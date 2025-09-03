# city-co2-balance
Code for the paper "Inter-city feedbacks reinforce source-sink asymmetry in regional CO₂ balance"

# Inter-city feedbacks reinforce source–sink asymmetry in regional CO₂ balance — Code & Minimal Data (RF & SEM)

> **Purpose.** This repository contains Python scripts and minimal example data to let reviewers run the Random Forest (RF) and Structural Equation Model (SEM) workflows used in the manuscript. The example files are intentionally small and anonymized so the code can be executed end‑to‑end without the full dataset.

> **Scope.** The outputs from these examples verify the pipeline and produce illustrative figures/tables, but **they will not reproduce the manuscript’s exact numerical results** because the **full dataset is not included**.

---

## 1) At a glance

* **OS / Python**: Windows 10, Python 3.11.9
* **Projects**: two independent pipelines

  * **RF**: `RF_Code.py` + `sample_data_RF_meanSHAP.xlsx` + `RF_requirements.txt`
  * **SEM**: `SEM_Code.py` + `sample_data_SEM.xlsx` + `SEM_requirements.txt`
* **How to run**: create a fresh environment (per‑project), install from `*_requirements.txt`, **manually edit the `file_path` in each `.py` to an absolute Windows path to the `.xlsx`** (this project is designed for local absolute paths), then run the script. *No other code edits are required for the sample data.*
* **License**: for peer‑review use only (update before public release).

> **Associated manuscript:** *Inter-city feedbacks reinforce source–sink asymmetry in regional CO₂ balance*.

---

## 2) Repository layout

```
.
├── RF_Code.py                 # Random Forest training + diagnostics + SHAP + PDP
├── RF_requirements.txt        # RF-specific Python dependencies
├── sample_data_RF_meanSHAP.xlsx
├── SEM_Code.py                # SEM fitting + fit statistics export
├── SEM_requirements.txt       # SEM-specific Python dependencies
└── sample_data_SEM.xlsx
```

---

## 3) Environment setup

> Create **separate** environments for RF and SEM to match their pinned dependencies.
>
> **Do not change dependency versions.** Install exactly the packages and versions listed in `*_requirements.txt`. It is OK to upgrade **`pip` itself** if needed, but do **not** run `pip install -U` on the listed packages.
>
> **Why pinned versions?** Using different package versions can lead to `ImportError`/`AttributeError`, API changes (e.g., scikit‑learn), SHAP plotting differences, or silent numerical shifts. For reproducibility, install **exact** versions with `python -m pip install -r <requirements>`.

### Recommended — `venv` (built‑in, Windows 10)

```bash
# RF
python -m venv .venv-rf
. .venv-rf/Scripts/activate
python -m pip install --upgrade pip
pip install -r RF_requirements.txt

# SEM
python -m venv .venv-sem
. .venv-sem/Scripts/activate
python -m pip install --upgrade pip
pip install -r SEM_requirements.txt
```

> Prefer `venv` for simplicity. If you personally use Conda/Mamba, you can create an environment with **Python 3.11.9** and then `pip install -r <requirements>`. Package versions must still match the files.

---

## 4) Running the RF workflow

### 4.1 Configure the input path (required)

Open `RF_Code.py` and **manually** set `file_path` to the sample data using an **absolute Windows path**. The scripts do **not** auto‑discover files and are intended for local absolute paths.

**Required — absolute Windows path**

```python
# In RF_Code.py
file_path = r"C:\path\to\sample_data_RF_meanSHAP.xlsx"
```

> **GitHub note:** Commit a neutral placeholder (e.g., `C:\path\to\sample_data_RF_meanSHAP.xlsx`). Each user who clones the repo must edit this line to their own absolute path before running.

### 4.2 Execute

From a terminal in this folder (with the RF environment activated):

```bash
python RF_Code.py
```

### 4.3 What the script does

* Loads the Excel file, creates an auxiliary feature `Fac33 = year - min(year)`; drops `FID`, `year`, and the target.
* Splits data (80/20), fits a `RandomForestRegressor` with fixed hyperparameters, and prints **R²** and **RMSE** for train/test.
* Computes **SHAP** values (tree explainer) and produces a summary plot and a lollipop‑style importance chart.
* Produces **Observed vs. Predicted** scatterplots (train/test), an **OOB error curve**, and a **panel of PDPs** for all features.

### 4.4 Expected outputs (saved alongside the data file)

* `shap_summary_plot.tif`
* `shap_bar_plot.tif`
* `model_performance_train.tif`
* `model_performance_test.tif`
* `oob_error_curve.tif`
* `pdp_plots.tif`

### 4.5 RF variables and their interpretation

* **Target (`y`)**: `CR` — *CLCER demand (XTij)*; see **Supplementary Text S2** of the manuscript for definition.
* **Predictors**: `Fac1`–`Fac32` (see **Supplementary Table S2** for variable meanings) plus the script‑generated `Fac33`.

> **Note on results:** Because the sample file is a minimal subset, the above figures are **illustrative** and not identical to the paper’s final analyses.

---

## 5) Running the SEM workflow

### 5.1 Configure the input path (required)

Open `SEM_Code.py` and **manually** set `file_path` to the sample data using an **absolute Windows path**. The scripts do **not** auto‑discover files and are intended for local absolute paths.

**Required — absolute Windows path**

```python
# In SEM_Code.py
file_path = r"C:\path\to\sample_data_SEM.xlsx"
```

> **GitHub note:** Commit a neutral placeholder (e.g., `C:\path\to\sample_data_SEM.xlsx`). Each user who clones the repo must edit this line to their own absolute path before running.

### 5.2 Execute

```bash
python SEM_Code.py
```

### 5.3 What the script does

* Reads the Excel file, drops rows with missing values, and **standardizes numeric columns** (excluding ID fields `sin_FID`, `sou_FID`).
* Fits the SEM specified in the script and exports **parameter estimates** and **goodness‑of‑fit statistics**.

### 5.4 Expected outputs (saved alongside the data file)

* `sem_model_results.xlsx`
* `sem_model_fit_stats.xlsx`

### 5.5 SEM constructs and references

* **H–L** and **H–S**: definitions and methodological details are in **Supplementary Text S6**.
* **Source/city PC1, PC3**: principal components derived from the factor loading tables (see **Supplementary Tables S7 and S8** of the manuscript). Use the same loadings that were used to score the manuscript’s dataset.

---

## 6) Data notes and limitations

* **Minimal example data (important):** The Excel files in this repository are **not the full dataset** used in the study. They are a **small, representative subset** supplied **solely to verify that the code runs** and to illustrate the pipeline. They are **insufficient to replicate the exact figures/tables of the paper**. Full‑dataset replication would require the original data described in the manuscript’s Data & Methods.
* **Columns / formats:** The sample spreadsheets contain all columns referenced by the scripts. If you substitute your own data, keep the same column names and types.

---

## 7) Reproducibility checklist

* [x] Exact Python version stated (3.11.9)
* [x] Per‑project, pinned `requirements.txt`
* [x] Deterministic RF `random_state`
* [x] All outputs written to disk with fixed filenames
* [x] Minimal data provided to run end‑to‑end

---

## 8) Troubleshooting

* **Path errors**: Make sure `file_path` is an **absolute Windows path** to the `.xlsx` (e.g., `C:\Users\<name>\...\sample_data_*.xlsx`).
* **Others can't run the scripts**: this is expected if they haven't edited the absolute path. Ask them to open the `.py` and change `file_path` to their own local absolute path. Avoid committing your personal path; use a neutral placeholder in the repo.
* **Package install issues (Windows)**: verify Python version (3.11.9) and, if needed, run `python -m pip install --upgrade pip`.
  **Do not upgrade, pin‑bump, or replace package versions listed in `*_requirements.txt`.** If dependency resolution fails, delete the environment and recreate it rather than changing versions.
* **Wrong dependency versions installed?** If you accidentally installed newer/older packages, remove the environment and reinstall exact pins:

  ```bash
  # Example: reset RF env
  deactivate 2>NUL
  rmdir /S /Q .venv-rf
  python -m venv .venv-rf & . .venv-rf/Scripts/activate
  python -m pip install --upgrade pip
  pip install -r RF_requirements.txt
  ```
* **SHAP memory/time**: On very large data, SHAP can be slow. The sample data should run quickly; reduce `n_estimators` if needed for testing.

---

## 9) License & contact

* **License:** Provided for journal peer‑review only. A final license will be set upon publication. Until then, redistribution is not permitted without authors’ consent.
* **Contact:** Prof. Qinhua Fang ([qhfang@xmu.edu.cn](mailto:qhfang@xmu.edu.cn)).

---

## 10) Change log

* v1.0 (2025‑09‑03): Initial reviewer package (RF & SEM scripts, minimal data, pinned requirements).
