[CmdletBinding()]
param(
    [ValidateRange(1, 10000)]
    [int]$SparseTopK = 100,

    [ValidateRange(1, 1000)]
    [int]$CheckpointEvery = 10,

    [switch]$Fresh
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
$inputPath = Join-Path $projectRoot "data\public-official.json"
$outputPath = Join-Path $projectRoot "data\submission.json"

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Python environment not found: $pythonPath"
}
if (-not (Test-Path -LiteralPath $inputPath)) {
    throw "Public input not found: $inputPath"
}

$activeRuns = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match "^python" -and
    $_.CommandLine -like "*scripts\infer.py*public-official.json*submission.json*"
}
if ($activeRuns) {
    $processIds = ($activeRuns.ProcessId | Sort-Object -Unique) -join ", "
    throw "A submission inference is already running (PID: $processIds). Wait for it to finish before starting another run."
}

$inferArgs = @(
    "scripts\infer.py",
    "--config", "configs\default.yaml",
    "--input", "data\public-official.json",
    "--output", "data\submission.json",
    "--sparse-top-k", $SparseTopK,
    "--checkpoint-every", $CheckpointEvery
)
if (-not $Fresh) {
    $inferArgs += "--resume"
}

Push-Location $projectRoot
try {
    Write-Host "Creating data\submission.json (Top-K=$SparseTopK, checkpoint every $CheckpointEvery answers)..."
    & $pythonPath @inferArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Inference failed with exit code $LASTEXITCODE"
    }

    $submission = Get-Content -LiteralPath $outputPath -Raw -Encoding utf8 | ConvertFrom-Json
    $count = @($submission.PSObject.Properties).Count
    Write-Host "Done: $count predictions saved to $outputPath"
}
finally {
    Pop-Location
}
