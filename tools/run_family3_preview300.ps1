param(
    [string]$PythonExe = "python",
    [string]$ResultsRoot = "results\experiment_family_3_preview300",
    [switch]$RenderGif
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$checkpointMap = @{
    "easy" = @{
        "ppo" = "checkpoints\experiment_family_1\easy_lite_ppo_seed101\best_model.zip"
        "plpg" = "checkpoints\experiment_family_1\easy_lite_plpg_seed101\best_model.zip"
        "plpg_fine" = "checkpoints\experiment_family_1\easy_lite_plpg_seed101\best_model.zip"
    }
    "medium" = @{
        "ppo" = "checkpoints\experiment_family_2\medium_lite_ppo_seed101\best_model.zip"
        "plpg" = "checkpoints\experiment_family_2\medium_lite_plpg_seed101\best_model.zip"
        "plpg_fine" = "checkpoints\experiment_family_2\medium_lite_plpg_seed101\best_model.zip"
    }
    "hard" = @{
        "ppo" = "checkpoints\experiment_family_2\hard_lite_ppo_seed101\best_model.zip"
        "plpg" = "checkpoints\experiment_family_2\hard_lite_plpg_seed101\best_model.zip"
        "plpg_fine" = "checkpoints\experiment_family_2\hard_lite_plpg_seed101\best_model.zip"
    }
}

$configMap = @{
    "ppo" = "multi_shared_5x5_ppo_test_k{0}.yaml"
    "plpg" = "multi_shared_5x5_plpg_test_k{0}.yaml"
    "plpg_fine" = "multi_shared_5x5_plpg_fine_test_k{0}.yaml"
}

$difficulties = @("easy", "medium", "hard")
$models = @("ppo", "plpg", "plpg_fine")
$kValues = @(5, 10, 15, 20)

Write-Host "Regenerating family-3 configs for 300-step visual preview..."
& $PythonExe "tools\generate_family3_continuous_configs.py" `
    --stop_mode time_budget `
    --total_steps 300 `
    --visualize `
    --frame_every 1
if ($LASTEXITCODE -ne 0) {
    throw "Config generation failed with exit code $LASTEXITCODE."
}

foreach ($difficulty in $difficulties) {
    foreach ($model in $models) {
        foreach ($k in $kValues) {
            $configName = $configMap[$model] -f $k
            $configPath = Join-Path "config\tasks\Nav\$difficulty\algo" $configName
            $checkpointPath = $checkpointMap[$difficulty][$model]
            $expName = "family3_preview300_{0}_{1}_k{2}" -f $difficulty, $model, $k
            $runDir = Join-Path $ResultsRoot (Join-Path $difficulty (Join-Path $model ("k{0}" -f $k)))
            $framesDir = Join-Path $runDir "frames"
            $rolloutPkl = Join-Path $runDir ("{0}_rollout.pkl" -f $expName)
            $gifPath = Join-Path $runDir ("{0}.gif" -f $expName)

            New-Item -ItemType Directory -Path $runDir -Force | Out-Null

            Write-Host ""
            Write-Host ("=== {0} | checkpoint={1} ===" -f $expName, $checkpointPath)
            & $PythonExe "main.py" `
                --use_gym `
                --config $configPath `
                --checkpoint_path $checkpointPath `
                --exp $expName `
                --log_dir $runDir
            if ($LASTEXITCODE -ne 0) {
                throw "Run failed for $expName with exit code $LASTEXITCODE."
            }

            if ($RenderGif) {
                New-Item -ItemType Directory -Path $framesDir -Force | Out-Null
                Write-Host ("Rendering GIF for {0}..." -f $expName)
                & $PythonExe "tools\pkl2gif.py" `
                    --pkl $rolloutPkl `
                    --ego_id 1 `
                    --output_folder $framesDir `
                    --output_gif $gifPath `
                    --duration_ms 150
                if ($LASTEXITCODE -ne 0) {
                    throw "GIF rendering failed for $expName with exit code $LASTEXITCODE."
                }
            }
        }
    }
}

Write-Host ""
Write-Host ("Finished. Results are under {0}" -f $ResultsRoot)
