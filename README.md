# Tennis Predictor

An ATP tennis match outcome model using ELO ratings, surface-specific form, and
serve statistics, with live pre-match value bet recommendations sourced from the
Betfair Exchange.

Built to complement the [pl-predictor](https://github.com/SamHusbands21/sam_website)
football model and integrated into my personal website.

---

## How it works

1. **Data** — Historical ATP match results are downloaded from
   [Jeff Sackmann's tennis_atp](https://github.com/JeffSackmann/tennis_atp) repo
   (2000–present) and supplemented with
   [tennis-data.co.uk](http://www.tennis-data.co.uk) for the current season
   (updated ~daily).

2. **Features** — For each match the model uses:
   - Overall and surface-specific ELO ratings (K=32, computed pre-match)
   - Rolling win rate (last 5 and 10 matches, overall and per surface)
   - Rolling serve quality (1st serve %, points won on 1st serve, break-point save rate)
   - Fatigue signals (days since last match, duration of last match)
   - Context (rank difference, ranking points, head-to-head record, tournament
     level, best-of format)

3. **Model** — An ensemble of XGBoost (`binary:logistic`) and an
   isotonic-calibrated Random Forest. Hyperparameters are tuned with walk-forward
   `TimeSeriesSplit` cross-validation. Training window: 2005–2020.
   Hold-out test set: 2021–present.

4. **Live pipeline** — Each morning the pipeline refreshes match data, rebuilds
   ELO ratings, fetches upcoming Betfair Exchange markets, generates probability
   estimates, applies value-bet filtering (`model_prob × odds > 1.20`) and Kelly
   staking, and writes `output/recommendations.json`.

5. **Website** — Outputs are copied to `sam_website/data/` and pushed to GitHub,
   where the site reads them via static JSON.

---

## Quickstart

### Prerequisites

- Python 3.9+
- A [Betfair Exchange](https://developer.betfair.com) account with API access
  (same credentials as pl-predictor)

### Install dependencies

```powershell
cd tennis-predictor
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Configure credentials

Copy `.env.example` to `.env` and fill in your Betfair details:

```
BETFAIR_USERNAME=your_username
BETFAIR_PASSWORD=your_password
BETFAIR_APP_KEY=your_app_key
BETFAIR_CERT_PATH=certs/client-2048.crt
BETFAIR_KEY_PATH=certs/client-2048.key
```

Place your Betfair SSL certificate files in the `certs/` folder
(same files used by pl-predictor).

### Run the full pipeline (first time)

```powershell
# 1. Download historical ATP data
python -m src.collect.sackmann

# 2. Build features
python -m src.features.engineer

# 3. Train models  ← generates models/xgb_model.joblib + models/rf_model.joblib
#                    (not committed to git due to file size; must be run locally)
python -m src.models.train

# 4. Evaluate on hold-out test set
python -m src.models.evaluate

# 5. Run live pipeline (requires Betfair credentials)
python -m src.pipeline.live
```

> **Note:** Model files (`models/*.joblib`) are excluded from git because they
> exceed GitHub's 100 MB limit. You must run `python -m src.models.train` to
> generate them before running the live pipeline.

### Daily automation (Windows Task Scheduler)

Schedule `run_daily.ps1` to run each morning before markets open:

1. Open **Task Scheduler** → **Create Task**
2. **Trigger:** Daily at 08:00
3. **Action:** `powershell.exe -ExecutionPolicy Bypass -File "C:\path\to\tennis-predictor\run_daily.ps1"`
4. **Settings:** tick *Run whether user is logged on or not*

For periodic retraining (e.g. monthly) schedule `run_retrain.ps1` the same way.

---

## Project structure

```
tennis-predictor/
├── src/
│   ├── collect/
│   │   ├── sackmann.py          # Download Sackmann ATP CSVs
│   │   ├── tennis_data_co_uk.py # Supplement with tennis-data.co.uk (2025+)
│   │   └── betfair.py           # Fetch live Betfair odds
│   ├── features/
│   │   ├── elo.py               # Overall ELO system
│   │   ├── surface_elo.py       # Per-surface ELO (Hard / Clay / Grass)
│   │   └── engineer.py          # Full feature engineering pipeline
│   ├── models/
│   │   ├── train.py             # XGBoost + calibrated RF ensemble
│   │   ├── evaluate.py          # Hold-out metrics, plots, SHAP
│   │   └── calibrate.py         # Isotonic calibration helper
│   └── pipeline/
│       └── live.py              # Daily prediction pipeline
├── models/                      # Generated locally; gitignored
├── output/                      # JSON + plots consumed by the website
├── data/
│   ├── raw/                     # Gitignored; re-downloaded each run
│   └── processed/               # Gitignored; rebuilt each run
├── certs/                       # Betfair SSL certs; gitignored
├── run_daily.ps1                # Scheduled daily predictions
├── run_retrain.ps1              # Periodic full retrain
└── requirements.txt
```

---

## Methodology

Full write-up including feature importance, calibration, and limitations at
[samhusbands.com/tennis-paper.html](https://samhusbands.com/tennis-paper.html).

---

## Limitations & contributing

See the *Limitations & Next Steps* section of the methodology page for areas
most in need of improvement — surface-specific serve modelling, WTA extension,
and in-play data integration are the most impactful open items.

Pull requests welcome.
