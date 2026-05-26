param(
    [int]$Seed = 101,
    [string]$PythonExe = "",
    [int]$EvalFreq = 10000,
    [int]$SaveFreq = 10000,
    [int]$TotalTimesteps = 100000
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$baseConfig = Join-Path $repoRoot "config\tasks\Nav\easy\algo\plpg_ppo_eval.yaml"
$generatedDir = Join-Path $repoRoot "config\tasks\Nav\easy\algo\generated"

if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $preferredPython = "D:\Anaconda\envs\FLogicity\python.exe"
    if (Test-Path $preferredPython) {
        $PythonExe = $preferredPython
    } else {
        $PythonExe = "python"
    }
}

if (!(Test-Path $generatedDir)) {
    New-Item -ItemType Directory -Path $generatedDir | Out-Null
}

$noiseLevels = @("0.00", "0.05", "0.10", "0.20", "0.30")
$alphaLevels = @("0.01", "0.1", "0.5")
$adaptivePattern = @'
(?ms)^  adaptive_eval_schedule:\r?\n    target_tsr:\s*.*?\r?\n    consecutive_target_evals:\s*.*?\r?\n    early_eval_freq:\s*.*?\r?\n(?:    max_early_timestep:\s*.*?\r?\n)?    post_target_eval_offsets:\s*\[.*?\]\s*\r?\n    late_eval_freq:\s*.*?$
'@
$adaptiveReplacement = @'
  adaptive_eval_schedule:
    target_tsr: 1.0
    consecutive_target_evals: 2
    early_eval_freq: 2000
    max_early_timestep: 20000
    post_target_eval_offsets: [5000]
    late_eval_freq: 10000
'@

foreach ($alpha in $alphaLevels) {
    foreach ($noise in $noiseLevels) {
        $label = "bitflip" + ($noise.Replace(".", ""))
        $alphaLabel = "alpha" + ($alpha.Replace(".", ""))
        $expName = "easy_lite_plpg_${label}_${alphaLabel}_seed$Seed"
        $generatedConfig = Join-Path $generatedDir "plpg_ppo_eval_${label}_${alphaLabel}.yaml"

        $configText = Get-Content -Raw $baseConfig

        $configText = [regex]::Replace($configText, '(^\s*eval_freq:\s*)\d+(\s*$)', ('${1}' + $EvalFreq), 'Multiline')
        $configText = [regex]::Replace($configText, '(^\s*save_freq:\s*)\d+(\s*$)', ('${1}' + $SaveFreq), 'Multiline')
        $configText = [regex]::Replace($configText, '(^\s*total_timesteps:\s*)\d+(\s*$)', ('${1}' + $TotalTimesteps), 'Multiline')
        $configText = [regex]::Replace($configText, '(^\s*alpha:\s*)[0-9.]+(\s*$)', ('${1}' + $alpha), 'Multiline')
        $configText = [regex]::Replace($configText, $adaptivePattern, $adaptiveReplacement)

        $configText = $configText.Replace(
            '      safety_loss_source: "shielded"',
@'
      safety_loss_source: "shielded"
      sensor_noise:
        mode: "bit_flip"
        epsilon: __EPSILON__
'@
        )
        $configText = $configText.Replace("__EPSILON__", $noise)

        Set-Content -Path $generatedConfig -Value $configText

        Write-Host "Running $expName with bit-flip epsilon=$noise alpha=$alpha"
        & $PythonExe main.py --use_gym --config $generatedConfig --exp $expName --seed $Seed
        if ($LASTEXITCODE -ne 0) {
            throw "Run failed for $expName"
        }
    }
}

Write-Host "Completed PLPG bit-flip-noise sweep for seed $Seed and alphas $($alphaLevels -join ', ')."
