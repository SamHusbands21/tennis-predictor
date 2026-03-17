# Tennis Betting Model - Full Retrain Script
# Run periodically (e.g. monthly) to refresh ATP data, rebuild features,
# retrain XGBoost + Random Forest models, evaluate, and push artefacts
# to the website repo.

$ErrorActionPreference = "Stop"

$PYTHON      = "C:\Users\Sam\AppData\Local\Programs\Python\Python39\python.exe"
$TENNIS_DIR  = "C:\Users\Sam\Documents\sam_website\tennis-predictor"
$SITE_DIR    = "C:\Users\Sam\Documents\sam_website"
$LOG_DIR     = Join-Path $TENNIS_DIR "logs"

New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null
$LOG_FILE = Join-Path $LOG_DIR ("retrain_" + (Get-Date -Format "yyyy-MM-dd") + ".log")

function Log($msg) {
    $line = (Get-Date -Format "HH:mm:ss") + "  " + $msg
    Write-Host $line
    Add-Content $LOG_FILE $line
}

Set-Location $TENNIS_DIR
Log "=== Full retrain started ==="

# --- Step 1: Download latest ATP data from Sackmann ---
Log "Step 1/5  Downloading latest ATP match data..."
$ErrorActionPreference = "Continue"
& $PYTHON -m src.collect.sackmann
$ec = $LASTEXITCODE
$ErrorActionPreference = "Stop"
if ($ec -ne 0) { Log "ERROR: sackmann download failed"; exit 1 }
Log "ATP data download done."

# --- Step 2: Rebuild feature matrix ---
Log "Step 2/5  Building feature matrix..."
$ErrorActionPreference = "Continue"
& $PYTHON -m src.features.engineer
$ec = $LASTEXITCODE
$ErrorActionPreference = "Stop"
if ($ec -ne 0) { Log "ERROR: engineer failed"; exit 1 }
Log "Feature engineering done."

# --- Step 3: Retrain models ---
Log "Step 3/5  Training XGBoost + Random Forest..."
$ErrorActionPreference = "Continue"
& $PYTHON -m src.models.train
$ec = $LASTEXITCODE
$ErrorActionPreference = "Stop"
if ($ec -ne 0) { Log "ERROR: train failed"; exit 1 }
Log "Training done."

# --- Step 4: Evaluate models ---
Log "Step 4/5  Evaluating models..."
$ErrorActionPreference = "Continue"
& $PYTHON -m src.models.evaluate
$ec = $LASTEXITCODE
$ErrorActionPreference = "Stop"
if ($ec -ne 0) { Log "ERROR: evaluate failed"; exit 1 }
Log "Evaluation done."

# --- Step 5: Push evaluation artefacts to website repo ---
Log "Step 5/5  Syncing evaluation artefacts to website..."
$date = Get-Date -Format "yyyy-MM-dd"

$filesToCopy = @(
    @{ Src = "output\calibration.png";         Dst = "data\tennis_calibration.png" },
    @{ Src = "output\shap_summary.png";        Dst = "data\tennis_shap_summary.png" },
    @{ Src = "output\pnl_curve.png";           Dst = "data\tennis_pnl_curve.png" },
    @{ Src = "output\threshold_sweep.png";     Dst = "data\tennis_threshold_sweep.png" },
    @{ Src = "output\website_evaluation.json"; Dst = "data\tennis_evaluation.json" }
)

foreach ($f in $filesToCopy) {
    $src = Join-Path $TENNIS_DIR $f.Src
    $dst = Join-Path $SITE_DIR   $f.Dst
    if (Test-Path $src) {
        New-Item -ItemType Directory -Force -Path (Split-Path $dst) | Out-Null
        if (Test-Path $dst) {
            icacls $dst /grant "${env:USERNAME}:(F)" /Q 2>&1 | Out-Null
            Remove-Item $dst -Force -ErrorAction SilentlyContinue
        }
        try {
            Copy-Item $src $dst -ErrorAction Stop
            Log "  Copied $($f.Src) -> $($f.Dst)"
        } catch {
            Log "  WARNING: Could not copy $($f.Src) - $($_.Exception.Message)"
        }
    } else {
        Log "  WARNING: $src not found - skipping"
    }
}

Set-Location $SITE_DIR
git add "data\tennis_calibration.png" "data\tennis_shap_summary.png" `
        "data\tennis_pnl_curve.png"    "data\tennis_threshold_sweep.png" `
        "data\tennis_evaluation.json"
$siteDiff = git diff --cached --name-only
if ($siteDiff) {
    git commit -m "chore: update tennis evaluation artefacts [$date]"
    git push
    Log "Website evaluation artefacts pushed."
} else {
    Log "No evaluation artefact changes to push."
}

Log "=== Retrain complete ==="
