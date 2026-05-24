param(
    [int]$Seed = 101,
    [string]$PythonExe = "",
    [int]$EvalFreq = 2500,
    [int]$SaveFreq = 2500,
    [int]$TotalTimesteps = 15000
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

foreach ($noise in $noiseLevels) {
    $epsilon = [double]::Parse($noise, [System.Globalization.CultureInfo]::InvariantCulture)
    $label = "soft" + ($noise.Replace(".", ""))
    $expName = "easy_lite_plpg_${label}_seed$Seed"
    $generatedConfig = Join-Path $generatedDir "plpg_ppo_eval_${label}.yaml"

    $configText = Get-Content -Raw $baseConfig

    $configText = [regex]::Replace($configText, '(^\s*eval_freq:\s*)\d+(\s*$)', ('${1}' + $EvalFreq), 'Multiline')
    $configText = [regex]::Replace($configText, '(^\s*save_freq:\s*)\d+(\s*$)', ('${1}' + $SaveFreq), 'Multiline')
    $configText = [regex]::Replace($configText, '(^\s*total_timesteps:\s*)\d+(\s*$)', ('${1}' + $TotalTimesteps), 'Multiline')

    $configText = $configText.Replace(
        '      safety_loss_source: "shielded"',
@'
      safety_loss_source: "shielded"
      sensor_noise:
        mode: "soft_symmetric"
        epsilon: __EPSILON__
'@
    )
    $configText = $configText.Replace("__EPSILON__", $noise)

    Set-Content -Path $generatedConfig -Value $configText

    Write-Host "Running $expName with epsilon=$noise"
    & $PythonExe main.py --use_gym --config $generatedConfig --exp $expName --seed $Seed
    if ($LASTEXITCODE -ne 0) {
        throw "Run failed for $expName"
    }
}

Write-Host "Completed PLPG soft-noise sweep for seed $Seed."
