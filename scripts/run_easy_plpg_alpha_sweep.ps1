param(
    [int]$Seed = 101,
    [string]$PythonExe = "",
    [int]$EvalFreq = 2000,
    [int]$SaveFreq = 2000,
    [int]$TotalTimesteps = 40000
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$baseConfig = Join-Path $repoRoot "config\tasks\Nav\easy\algo\plpg_ppo_eval.yaml"
$generatedDir = Join-Path $repoRoot "config\tasks\Nav\easy\algo\generated"

if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
    $preferredPython = "D:\Anaconda\envs\FLogicity\python.exe"
    if (Test-Path $venvPython) {
        $PythonExe = $venvPython
    } elseif (Test-Path $preferredPython) {
        $PythonExe = $preferredPython
    } else {
        $PythonExe = "python"
    }
}

if (!(Test-Path $baseConfig)) {
    throw "Base config not found: $baseConfig"
}

if (!(Test-Path $generatedDir)) {
    New-Item -ItemType Directory -Path $generatedDir | Out-Null
}

$alphaValues = @("0.0", "0.01", "0.05", "0.1", "0.2", "0.5")

foreach ($alpha in $alphaValues) {
    $label = "alpha" + ($alpha.Replace(".", ""))
    $expName = "easy_lite_plpg_${label}_seed$Seed"
    $generatedConfig = Join-Path $generatedDir "plpg_ppo_eval_${label}.yaml"

    $configText = Get-Content -Raw $baseConfig

    $configText = [regex]::Replace($configText, '(^\s*eval_freq:\s*)\d+(\s*$)', ('${1}' + $EvalFreq), 'Multiline')
    $configText = [regex]::Replace($configText, '(^\s*save_freq:\s*)\d+(\s*$)', ('${1}' + $SaveFreq), 'Multiline')
    $configText = [regex]::Replace($configText, '(^\s*total_timesteps:\s*)\d+(\s*$)', ('${1}' + $TotalTimesteps), 'Multiline')
    $configText = [regex]::Replace($configText, '(^\s*alpha:\s*)[0-9.]+(\s*$)', ('${1}' + $alpha), 'Multiline')

    Set-Content -Path $generatedConfig -Value $configText

    Write-Host "Running $expName with alpha=$alpha"
    & $PythonExe main.py --use_gym --config $generatedConfig --exp $expName --seed $Seed
    if ($LASTEXITCODE -ne 0) {
        throw "Run failed for $expName"
    }
}

Write-Host "Completed PLPG alpha sweep for seed $Seed."
