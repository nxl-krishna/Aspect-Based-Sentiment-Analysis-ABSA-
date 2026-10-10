# Reproduces every experiment in reports/RESULTS.md (about 45 minutes on an RTX 3050 6GB).
$ErrorActionPreference = "Continue"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = 1
$env:PYTHONWARNINGS = "ignore"
$py = ".\venv\Scripts\python.exe"
New-Item -ItemType Directory -Force logs | Out-Null

$runs = @(
    @("sent_deberta_s42",      "src.training.train_sentiment",  "--config configs/sentiment_deberta.yaml --seed 42"),
    @("extract_deberta_s42",   "src.training.train_extraction", "--config configs/extraction_deberta.yaml --seed 42"),
    @("baselines_s42",         "src.training.train_baselines",  "--seed 42"),
    @("sent_deberta_w_s42",    "src.training.train_sentiment",  "--config configs/sentiment_deberta_weighted.yaml --seed 42"),
    @("sent_distilbert_s42",   "src.training.train_sentiment",  "--config configs/sentiment_distilbert.yaml --seed 42"),
    @("sent_deberta_s1",       "src.training.train_sentiment",  "--config configs/sentiment_deberta.yaml --seed 1 --save_model false"),
    @("sent_deberta_s2",       "src.training.train_sentiment",  "--config configs/sentiment_deberta.yaml --seed 2 --save_model false"),
    @("sent_distilbert_s1",    "src.training.train_sentiment",  "--config configs/sentiment_distilbert.yaml --seed 1"),
    @("sent_distilbert_s2",    "src.training.train_sentiment",  "--config configs/sentiment_distilbert.yaml --seed 2"),
    @("extract_deberta_s1",    "src.training.train_extraction", "--config configs/extraction_deberta.yaml --seed 1 --save_model false"),
    @("extract_deberta_s2",    "src.training.train_extraction", "--config configs/extraction_deberta.yaml --seed 2 --save_model false"),
    @("baselines_s1",          "src.training.train_baselines",  "--seed 1"),
    @("baselines_s2",          "src.training.train_baselines",  "--seed 2")
)

foreach ($r in $runs) {
    $start = Get-Date
    Write-Output "=== START $($r[0]) $start"
    $cmd = "$py -m $($r[1]) $($r[2]) > logs\$($r[0]).log 2>&1"
    Invoke-Expression $cmd
    Write-Output "=== END $($r[0]) exit=$LASTEXITCODE took=$([int]((Get-Date) - $start).TotalSeconds)s"
}
