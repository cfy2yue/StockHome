param(
    [switch]$SmokeTest
)

$ErrorActionPreference = "Stop"
Write-Host "DeepSeek key setup for stock research agent"
Write-Host "This stores the key in the current Windows user's environment variables."
Write-Host "The key will not be written to project files, reports, logs, or git."
Write-Host ""

$key = Read-Host "Paste DeepSeek API key"
$key = $key.Trim()
if ([string]::IsNullOrWhiteSpace($key)) {
    Write-Host "No key provided. Nothing changed."
    exit 1
}
if ($key -notmatch '^sk-') {
    Write-Host "Warning: the key does not start with sk-. Continuing because providers may change formats."
}

[Environment]::SetEnvironmentVariable("DEEPSEEK_API_KEY", $key, "User")
$env:DEEPSEEK_API_KEY = $key
Write-Host "DEEPSEEK_API_KEY saved for this Windows user."
Write-Host "Open a new terminal before running long workflows."

if ($SmokeTest) {
    $root = Resolve-Path (Join-Path $PSScriptRoot "..")
    $pythonCandidates = @(
        "C:\Users\lenovo\anaconda3\envs\stock-agent\python.exe",
        "python"
    )
    foreach ($candidate in $pythonCandidates) {
        try {
            & $candidate (Join-Path $root "scripts\deepseek_smoke_test.py")
            exit $LASTEXITCODE
        }
        catch {
            continue
        }
    }
    Write-Host "Could not run smoke test automatically. Please run python scripts\deepseek_smoke_test.py later."
}
