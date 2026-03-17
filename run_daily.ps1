# Tennis Betting Model - Daily Pipeline Runner
# Schedule this with Windows Task Scheduler to run daily.
# Runs on your local machine so Betfair API works (UK IP).

$ErrorActionPreference = "Stop"

$PYTHON      = "C:\Users\Sam\AppData\Local\Programs\Python\Python39\python.exe"
$TENNIS_DIR  = "C:\Users\Sam\Documents\sam_website\tennis-predictor"
$SITE_DIR    = "C:\Users\Sam\Documents\sam_website"
$LOG_DIR     = Join-Path $TENNIS_DIR "logs"

New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null
$LOG_FILE = Join-Path $LOG_DIR ("pipeline_" + (Get-Date -Format "yyyy-MM-dd") + ".log")

function Log($msg) {
    $line = (Get-Date -Format "HH:mm:ss") + "  " + $msg
    Write-Host $line
    Add-Content $LOG_FILE $line
}

Log "=== Tennis pipeline started ==="

# --- Step 1: Run live pipeline ---
Log "Step 1/3  Running live pipeline..."
Set-Location $TENNIS_DIR
$ErrorActionPreference = "Continue"
& $PYTHON -m src.pipeline.live
$exitCode = $LASTEXITCODE
$ErrorActionPreference = "Stop"
if ($exitCode -ne 0) {
    Log "ERROR: Pipeline failed (exit $exitCode)"
    exit 1
}
Log "Pipeline done."

# --- Step 2: Copy to website repo ---
Log "Step 2/3  Copying recommendations.json to website..."
$src = Join-Path $TENNIS_DIR "output\recommendations.json"
$dst = Join-Path $SITE_DIR   "data\tennis_recommendations.json"
if (-not (Test-Path $src)) {
    Log "ERROR: $src not found"
    exit 1
}
New-Item -ItemType Directory -Force -Path (Split-Path $dst) | Out-Null
Copy-Item $src $dst -Force
Log "Copied to $dst"

# --- Step 3: Commit and push ---
Log "Step 3/3  Pushing to GitHub..."
Set-Location $SITE_DIR
git add "data\tennis_recommendations.json"
$diff = git diff --cached --name-only
if ($diff) {
    $date = Get-Date -Format "yyyy-MM-dd"
    git commit -m "chore: update tennis recommendations [$date]"
    git push
    Log "Pushed to GitHub."
} else {
    Log "No changes - nothing to push."
}

Log "=== Tennis pipeline complete ==="
