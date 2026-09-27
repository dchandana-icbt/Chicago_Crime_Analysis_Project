# Chicago Crime Analysis Project

A PySpark-based analysis of the City of Chicago's public crime records (2001–present)
joined with community-area socioeconomic indicators and independently-sourced daily
weather data. Built around four tasks: a distributed ETL pipeline, exploratory data
analysis, unsupervised spatial-temporal clustering, and supervised classification
(arrest likelihood + a primary-type multiclass extension).

## Project structure

```
Chicago_Crime_Analysis_Project/
├── 00_download_chicago_datasets.ipynb   # Fetches the raw CSVs into Data/ if not already present
├── 01_data_engineering.ipynb            # Task 1 — Spark ETL: schema, cleaning, curated Parquet lake
├── 02_eda.ipynb                         # Task 2 — Temporal/spatial EDA, broadcast join, weather correlation
├── 03_clustering.ipynb                  # Task 3 — K-Means / Bisecting K-Means spatial-temporal clustering
├── 04_classification.ipynb              # Task 4 — RandomForest/GBT: binary arrest-likelihood prediction
├── 04b_classification_multiclass_extension.ipynb  # Task 4b — Primary Type multiclass extension
├── requirements.txt                     # Pinned Python dependencies
├── Data/ (a.k.a. data/)                 # Raw input CSVs (not included in this repo — see Setup below)
│   ├── Crimes_-_2001_to_Present.csv
│   ├── Census_Data_-_..._2008-2012.csv
│   ├── processed/                       # Curated Parquet lake written by 01_data_engineering
│   └── quarantine/                      # Rows rejected by Task 1's data-quality checks
├── reports/                             # CSV/JSON/Parquet outputs from every notebook
└── spark_tmp/                           # Spark's local scratch directory (safe to delete)
```

> **Note on `Data/` vs `data/`:** the notebooks are inconsistent about capitalization
> (a holdover from different authoring environments). This only works transparently on
> macOS/Windows, whose filesystems are case-insensitive — both paths resolve to the same
> folder. On Linux (e.g. a CI runner, Docker, or Colab) these are **two different
> directories**, so create/symlink one to the other, or just keep everything in a single
> `Data/` folder and export `CHICAGO_PROJECT_ROOT` accordingly (see notebooks' own path
> resolution comments in cell 1/2 of each).

## Prerequisites

- **Python 3.11** (the project was built and verified against this version)
- **Java 17** (required by PySpark; not installable via pip)
  - macOS: `brew install openjdk@17`
  - Ubuntu/Debian: `sudo apt install openjdk-17-jdk`
- **[uv](https://github.com/astral-sh/uv)** (recommended) or plain `pip` for installing dependencies
- ~2 GB free disk space for the raw crime CSV, plus room for the Parquet lake/reports
- 8GB+ RAM recommended (Spark is configured to use up to 8GB driver memory for the
  classification notebooks — see **Known gotchas** below if you have less)

## Setup

### 1. Clone/open the project and create a virtual environment

```bash
cd "Chicago_Crime_Analysis_Project"

# With uv (recommended):
uv venv .venv --python 3.11

# Or with plain venv:
python3.11 -m venv .venv
```

### 2. Install dependencies

```bash
uv pip install --python .venv/bin/python -r requirements.txt
# or, with plain pip:
source .venv/bin/activate && pip install -r requirements.txt
```

### 3. Register the venv as a Jupyter kernel

```bash
.venv/bin/python -m ipykernel install --user --name chicago-crime-prac1 \
    --display-name "Chicago Crime (PRAC1)"
```

This makes **"Chicago Crime (PRAC1)"** selectable as the kernel in Jupyter Lab, Jupyter
Notebook, or VS Code's notebook UI. Select it before running any of the project's
notebooks.

### 4. Get the raw datasets

Run **`00_download_chicago_datasets.ipynb`** top-to-bottom. It's idempotent — it checks
`data/` for the CSVs first and only downloads what's missing, in this order:
1. CSV already present → skip.
2. A local `.zip` already in `data/` → extract it.
3. Kaggle API (if you've configured a token — see the notebook's §4 for how).
4. Fallback: City of Chicago Data Portal (no login needed, works out of the box).

The crime file is large (~1.8 GB); the fallback download can take a few minutes
depending on your connection.

### 5. Run the pipeline, in order

```
00_download_chicago_datasets.ipynb   →  gets the raw CSVs
01_data_engineering.ipynb            →  builds the curated Parquet lake (run this before 02-04b)
02_eda.ipynb                         →  reads the lake, adds weather, does EDA
03_clustering.ipynb                  →  reads the lake, does spatial-temporal clustering
04_classification.ipynb              →  reads the lake, trains binary arrest-prediction models
04b_classification_multiclass_extension.ipynb  →  reads the lake, multiclass primary-type extension
```

`02`–`04b` all depend on Parquet output written by `01_data_engineering.ipynb`, so run
that one first. `02`, `03`, `04`, and `04b` are otherwise independent of each other.

You can run each notebook interactively, or headlessly from the command line:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=1800 \
    --ExecutePreprocessor.kernel_name=chicago-crime-prac1 \
    01_data_engineering.ipynb
```

(`nbconvert`, `nbclient`, and `nbformat` — included in `requirements.txt` — are only
needed for this headless-execution path; they aren't imported by any notebook's own
code.)

## What each notebook produces

| Notebook | Key outputs (in `reports/` unless noted) |
|---|---|
| `01_data_engineering` | `data/processed/*` (curated Parquet lake), `task1_summary.json`, `task1_dq_report.json`, `task1_layout_summary.json` |
| `02_eda` | `daily_counts_with_weather.parquet`, `weather_monthly_profile.csv`, `socioeconomic_correlation.csv`, `task2_weather_source_used.json` |
| `03_clustering` | `task3_cluster_centers.csv`, `task3_elbow_silhouette.csv`, `task3_cluster_vs_district.csv`, `task3_cluster_plot_sample.parquet` |
| `04_classification` | `task4_binary_results.json`, `task4_roc_points.json`, `task4_pr_points.json` |
| `04b_classification_multiclass_extension` | `task4_multiclass_summary.json`, `task4_multiclass_per_class.csv`, `task4_multiclass_model_summary.csv` |

## Known gotchas (already fixed in these notebooks, documented for reference)

- **`local[*]` + low driver memory → OutOfMemoryError.** Spark's `local[*]` master runs
  one executor thread per CPU core *inside the driver's own JVM*, so
  `spark.driver.memory` is the only heap shared across all of them. `01`–`03` use `2g`
  (fine for read-only aggregation); `04` and `04b` use `8g` because they `.cache()`
  one-hot-encoded DataFrames across millions of rows before training RandomForest/GBT.
  If you're on a machine with less RAM, lower these values in each notebook's first
  cell — but expect slower runs or renewed OOM errors on the full 2001–2022 dataset.
- **`UnresolvedAddressException` / `IllegalArgumentException` on `SparkContext` startup.**
  On macOS, if your hostname (`whatever.local`) stops resolving via mDNS/Bonjour (common
  after a network change, sleep/wake, or VPN toggle), Spark's internal Netty RPC server
  fails to bind. Every notebook's Spark session pins `spark.driver.host` /
  `spark.driver.bindAddress` to `127.0.0.1` to sidestep this entirely.
- **A crashed Spark JVM leaves a "zombie" kernel.** If a cell OOMs or the JVM otherwise
  dies mid-run, the Python kernel process usually survives — but any further
  `SparkSession.builder.getOrCreate()` call in that same kernel will fail
  (`Py4JNetworkError`/`ConnectionRefusedError`) because it's still holding a reference to
  the dead JVM's gateway. **Always fully restart the kernel** (not just re-run the cell)
  after any Spark-related crash.
- **`meteostat.Point(lat, lon)` silently returns no data** when passed to
  `meteostat.daily()` with the library's default provider — `Point` objects only resolve
  through geo-location/interpolation providers, not the bulk-file provider used here.
  `02_eda.ipynb` queries by station ID instead (`"72534"`, Chicago Midway Airport, which
  has full gap-free daily coverage for 2001–2022 — the literal nearest station to
  downtown Chicago has none).

## Data sources

- **Crimes – 2001 to Present**: [City of Chicago Data Portal](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2) (also mirrored on Kaggle as `utkarshx27/crimes-2001-to-present`)
- **Socioeconomic Indicators in Chicago, 2008–2012**: [City of Chicago Data Portal](https://data.cityofchicago.org/Health-Human-Services/Census-Data-Selected-socioeconomic-indicators-in-/kn9c-c2s2) (also mirrored on Kaggle as `umermjd11/socioeconomic-indicators-in-chicago-2008-2012`)
- **Daily weather (Chicago Midway Airport, station `72534`)**: [meteostat](https://meteostat.net/) Python package, live-fetched in `02_eda.ipynb`
