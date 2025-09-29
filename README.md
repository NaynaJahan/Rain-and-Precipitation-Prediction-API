# Weather Experiments (Sydney) using Weather API from [Open-Meteo](https://open-meteo.com/)


<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

This repository contains the experiments used to train two production models for Sydney (lat -33.8678, lon 151.2073), built from the Open-Meteo Historical Weather API:
1. `rain_or_not` – Binary classification: Whether it will rain or not in exactly +7 days from a given date.
2. `precipitation_fall` – Regression: cumulated precipitation (mm) for D+1..D+3 after a given date.

## Project Organization

```graphql
amla_at2_experiments
│
├── Makefile       
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external/       
│   ├── interim/        
│   ├── processed/      
│   └── raw/            
│
├── docs              
│
├── models/
│      ├── precipitation_fall/
│      │   └── best_precipitation_reg_pipeline.joblib
│      └── rain_or_not/
│          └── best_rain_cls_pipeline.joblib           
│
├── notebooks/          
│       ├── precipitation_fall/
|       │   ├── 36120-25SP-25238736-experiment-2.ipynb 
│       │   └── data/
│       │       │── processed/       <- The final, canonical data sets for regression modeling.
│       │       └── raw/              <- The original, immutable data dump.
│       └── rain_or_not/
|           ├── 36120-25SP-25238736-experiment-1.ipynb 
│           └── data/
│               |── processed/        <- The final, canonical data sets for classification modeling.
│               └── raw/               <- The original, immutable data dump.             
│                         
│
├── pyproject.toml     <- Project configuration file with package metadata 
│                      
│
├── references        
│
├── reports           
│   └── figures/
├── tests
│   └── test_data.py   <- Default test file; the unit tests are included in the custom python package     
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment
├── github.txt         <- Text file with Github URLs for the project, custom Python package and API
│                      
│
├── poetry.lock          
│
└── amla_at2_experiments   
    │
    ├── __init__.py             <- Makes amla_at2_experiments a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------


## Environment setup

### Requirements:

```
python==3.11.4
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.5.1
joblib==1.4.2
requests==2.32.3

```

The helper package [amla-at1](https://github.com/naynajn/amla_at1_python_pkg) from [TestPyPI](https://test.pypi.org/project/amla-at1/)  (it contains amla_at1.data.openmeteo.fetch_daily_archive and utilities):
> Package: amla-at1==2025.0.2.0 (requires joblib==1.4.2)

### Setting up with `venv` + `pip`

```bash

# From repo root
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Core scientific stack
pip install numpy==1.26.4 pandas==2.2.2 scikit-learn==1.5.1 joblib==1.4.2 python-dateutil==2.9.0.post0 requests==2.32.3

# Install helper package from TestPyPI
pip install --extra-index-url https://test.pypi.org/simple/ amla-at1==2025.0.2.0

# Notebooks
pip install jupyterlab==4.2.3 ipykernel
python -m ipykernel install --user --name amla-at2 --display-name "Python (amla-at2)"


```


--------


## Data source (Sydney-only)

Historical weather comes from [Open-Meteo (Sydney)](https://open-meteo.com/en/docs/historical-weather-api#json_return_object). The helper `fetch_daily_archive(start_date, end_date, timezone="Australia/Sydney")` downloads and returns a daily DataFrame indexed by date. Notebooks will cache CSVs under data/raw/ (auto-created) to speed up reruns.

- Latitude: -33.8678
- Longitude: 151.2073
- Timezone: Australia/Sydney

--------


## Running the experiments


### i. Classification — `rain_or_not`

#### Notebook: `notebooks/rain_or_not/36120-25SP-25238736-experiment-1.ipynb`

**What it does:**
- Loads Sydney daily history (2010–2024 by default).
- Builds supervised features using 14-day rolling windows + calendar features.
- Target: `will_rain_in_7_days` (1 if D+7 has rain, else 0).
- Tries three approaches (examples):
    - Domain + rolling (seed feature set)
    - Mutual information ranking
    - Permutation importance ranking (features merged/filtered into a final list)
- Trains and selects among:
    - AdaBoost (stumps) + Isotonic + stricter threshold
    - SGD Logistic + Isotonic calibration (better probs)
    - HistGradientBoostingClassifier + calibration
- Picks best model by validation metrics (F1 primary, with AUROC/RMSE tie-breaks).
- Saves the final model as:
```bash
models/rain_or_not/best_rain_cls_pipeline.joblib
```

**To run:**

1. Launch Jupyter:

```bash
jupyter lab
```
2. Open the classification notebook and Run All.
3. Confirm a file appears at models/rain_or_not/best_rain_cls_pipeline.joblib.
4. Attach the training columns & threshold to the saved pipeline:
```bash
setattr(best_model, "_training_columns", list(Xtr.columns))
setattr(best_model, "best_threshold_", chosen_threshold)  # e.g., from validation F1 search
```

--------


### ii. Regression — `precipitation_fall`

#### Notebook: `notebooks/precipitation_fall/36120-25SP-25238736-experiment-2.ipynb`

**What it does:**

- Loads Sydney daily history (2015–2024 by default).
- Builds features (14-day rolling + calendar), target is sum of precipitation for D+1..D+3.
- Tries three approaches (examples):
    - TweedieRegressor (log link) with grid over power, alpha
    - HistGradientBoostingRegressor with Poisson loss
    - Zero-inflated Hurdle: classifier × regressor
- Selects the lowest RMSE (tie-break: MAE then R²).
- Saves the final model as:
```bash
models/precipitation_fall/best_precipitation_reg_pipeline.joblib
```

**To run:**

1. Launch Jupyter:
```bash
jupyter lab
```
2. Open the Regression notebook and Run All.
3. Confirm a file appears at models/precipitation_fall/best_precipitation_reg_pipeline.joblib.
4. Attach the training columns to the saved pipeline:
```bash
setattr(best_model, "_training_columns", list(Xtr.columns))
```

--------


## How the FastAPI service uses these models

Deployment repo expects:
- models/rain_or_not/best_rain_cls_pipeline.joblib
- Must carry ._training_columns (list of feature names, exact order)
- Should carry .best_threshold_ (float in [0,1], default 0.5 if missing)
- models/precipitation_fall/best_precipitation_reg_pipeline.joblib
- Must carry ._training_columns

The service reconstructs features from the column names using the naming convention:

```php-template
<base>_(mean|std|sum)_<window>
```

and calendar columns:

```sql
sin_yday, cos_yday, month, dayofyear, is_summer
```

## Attribution
- [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api#json_return_object)
- [Test Python package](https://test.pypi.org/)