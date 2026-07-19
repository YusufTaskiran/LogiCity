param(
    [string]$PythonExe = "python",
    [int]$Seed = 101,
    [ValidateSet("plpg", "ppo")]
    [string]$Method = "plpg",
    [Parameter(Mandatory = $true)]
    [string]$EasyCheckpoint,
    [Parameter(Mandatory = $true)]
    [string]$MediumCheckpoint,
    [Parameter(Mandatory = $true)]
    [string]$HardCheckpoint
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$figureDir = Join-Path $repoRoot "thesis\figures"
New-Item -ItemType Directory -Force -Path $figureDir | Out-Null

if ($Method -eq "plpg") {
    $configMap = @{
        "easy_5"   = "config/tasks/Nav/easy/algo/multi_shared_5x5_plpg_test_k5.yaml"
        "easy_20"  = "config/tasks/Nav/easy/algo/multi_shared_5x5_plpg_test_k20.yaml"
        "medium_5" = "config/tasks/Nav/medium/algo/multi_shared_5x5_plpg_test_k5.yaml"
        "medium_20"= "config/tasks/Nav/medium/algo/multi_shared_5x5_plpg_test_k20.yaml"
        "hard_5"   = "config/tasks/Nav/hard/algo/multi_shared_5x5_plpg_test_k5.yaml"
        "hard_20"  = "config/tasks/Nav/hard/algo/multi_shared_5x5_plpg_test_k20.yaml"
    }
} else {
    $configMap = @{
        "easy_5"   = "config/tasks/Nav/easy/algo/multi_shared_5x5_ppo_test_k5.yaml"
        "easy_20"  = "config/tasks/Nav/easy/algo/multi_shared_5x5_ppo_test_k20.yaml"
        "medium_5" = "config/tasks/Nav/medium/algo/multi_shared_5x5_ppo_test_k5.yaml"
        "medium_20"= "config/tasks/Nav/medium/algo/multi_shared_5x5_ppo_test_k20.yaml"
        "hard_5"   = "config/tasks/Nav/hard/algo/multi_shared_5x5_ppo_test_k5.yaml"
        "hard_20"  = "config/tasks/Nav/hard/algo/multi_shared_5x5_ppo_test_k20.yaml"
    }
}

$checkpointMap = @{
    "easy" = $EasyCheckpoint
    "medium" = $MediumCheckpoint
    "hard" = $HardCheckpoint
}

$runs = @(
    @{ Difficulty = "easy"; K = 5 },
    @{ Difficulty = "easy"; K = 20 },
    @{ Difficulty = "medium"; K = 5 },
    @{ Difficulty = "medium"; K = 20 },
    @{ Difficulty = "hard"; K = 5 },
    @{ Difficulty = "hard"; K = 20 }
)

foreach ($run in $runs) {
    $difficulty = $run.Difficulty
    $k = [int]$run.K
    $configKey = "{0}_{1}" -f $difficulty, $k
    $configPath = $configMap[$configKey]
    $checkpointPath = $checkpointMap[$difficulty]
    $expName = "thesis_exp3_{0}_{1}_k{2}_seed{3}" -f $Method, $difficulty, $k, $Seed

    Write-Host "Running $expName with config $configPath"
    & $PythonExe main.py --use_gym --config $configPath --exp $expName --seed $Seed --checkpoint_path $checkpointPath
    if ($LASTEXITCODE -ne 0) {
        throw "Run failed for $expName"
    }

    $sourceFrame = Join-Path $repoRoot ("rollout_vis\{0}\frame_000000.png" -f $expName)
    if (-not (Test-Path $sourceFrame)) {
        throw "Expected frame not found: $sourceFrame"
    }

    $targetFrame = Join-Path $figureDir ("exp3_{0}_k{1}_{2}.png" -f $difficulty, $k, $Method)
    Copy-Item -LiteralPath $sourceFrame -Destination $targetFrame -Force
    Write-Host "Saved thesis figure $targetFrame"
}

Write-Host "All Experiment 3 thesis frames generated in $figureDir"
