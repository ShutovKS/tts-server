# FILE: scripts/launch-windows.ps1
# VERSION: 1.1.2
# START_MODULE_CONTRACT
#   PURPOSE: Provide an interactive Windows PowerShell launcher that orchestrates profile-aware environment setup, optional model downloads, and adapter startup.
#   SCOPE: Windows-only preflight checks, service/model prompts, launcher CLI orchestration, family-env bootstrap, model artifact validation, optional Hugging Face and Piper downloads, inline-wrapper project-root fallback, and final adapter execution.
#   DEPENDS: M-LAUNCHER, M-PROFILE-RESOLVER, M-CONFIG
#   LINKS: M-WINDOWS-LAUNCHER
#   ROLE: SCRIPT
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   $SCRIPT:FAMILY_OPTIONS - Curated family menu entries mapped to runtime contours.
#   $SCRIPT:MODEL_OPTIONS - Curated model menu entries mapped to runtime family contours and local artifact folders.
#   Get-ProjectRoot - Resolve the repository root relative to the script location or an inline-wrapper fallback root.
#   ConvertTo-PlainText - Convert a secure string to plaintext for transient process-local environment use.
#   Read-TrimmedHostInput - Read one host prompt and normalize null or whitespace-padded input.
#   Resolve-HttpProbeHost - Convert bind hosts like 0.0.0.0 or :: into a client-reachable loopback probe target.
#   Select-MenuOption - Prompt the user to select one numbered option from a curated menu.
#   Select-MultipleMenuOptions - Prompt the user to select one or more numbered options from a curated menu.
#   Invoke-LauncherJson - Execute the profile-aware Python launcher and parse its JSON payload.
#   Assert-WindowsPreflight - Validate that the script runs on Windows PowerShell with Python 3.11 and ffmpeg available.
#   Test-ModelArtifacts - Validate a model folder or Hugging Face snapshot layout against required artifact groups.
#   Ensure-FamilyEnvironment - Create and verify the dedicated family environment through launcher create-env/check-env flows.
#   Ensure-ModelAvailability - Validate local model artifacts and optionally download missing assets through family-aware strategies.
#   Configure-ServiceEnvironment - Apply transient TTS_* and Telegram settings for the selected launch contour.
#   Get-RuntimeCapabilityBindings - Derive runtime capability bindings from ensured models and selected family.
#   Show-RuntimeCapabilityBindings - Print the final runtime capability binding summary before launch.
#   Wait-HttpHealthCheck - Probe the configured HTTP server until /health/live responds or timeout elapses.
#   Get-HttpServerPidFilePath - Resolve the repo-local PID metadata file for launcher-managed HTTP server instances.
#   Read-HttpServerPidFile - Load launcher-managed HTTP server PID metadata when present.
#   Clear-HttpServerPidFile - Remove stale or completed launcher-managed HTTP server PID metadata.
#   Get-TcpOwningProcessId - Resolve the PID currently listening on a target TCP port when one exists.
#   Stop-HttpServerProcess - Gracefully stop a launcher-managed HTTP server PID and clean up when needed.
#   Ensure-HttpServerLaunchTarget - Restart an existing launcher-managed HTTP server or prompt when a foreign process occupies the target port.
#   Start-SelectedService - Launch the selected adapter through the profile-aware launcher exec command.
#   Main - Run the interactive launcher flow end-to-end.
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.1.2 - Added launcher-managed HTTP server PID lifecycle so reruns restart owned processes and prompt when foreign listeners occupy the target port]
# END_CHANGE_SUMMARY

Set-StrictMode -Version 3.0
$ErrorActionPreference = 'Stop'

$SCRIPT:FAMILY_OPTIONS = @(
    [pscustomobject]@{ Key = 'qwen'; Label = 'Qwen3' },
    [pscustomobject]@{ Key = 'omnivoice'; Label = 'OmniVoice' },
    [pscustomobject]@{ Key = 'piper'; Label = 'Piper' }
)

$SCRIPT:MODEL_OPTIONS = @(
    [pscustomobject]@{ Key = 'qwen-custom-17b'; Label = 'Qwen Custom 1.7B'; Family = 'qwen'; Folder = 'Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit'; Mode = 'custom'; DownloadStrategy = 'huggingface'; RepoId = 'Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice'; ArtifactGroups = @(@('config.json'), @('model.safetensors', 'model.safetensors.index.json'), @('preprocessor_config.json'), @('tokenizer_config.json', 'vocab.json')) },
    [pscustomobject]@{ Key = 'qwen-design-17b'; Label = 'Qwen Design 1.7B'; Family = 'qwen'; Folder = 'Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit'; Mode = 'design'; DownloadStrategy = 'huggingface'; RepoId = 'Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign'; ArtifactGroups = @(@('config.json'), @('model.safetensors', 'model.safetensors.index.json'), @('preprocessor_config.json'), @('tokenizer_config.json', 'vocab.json')) },
    [pscustomobject]@{ Key = 'qwen-clone-17b'; Label = 'Qwen Clone 1.7B'; Family = 'qwen'; Folder = 'Qwen3-TTS-12Hz-1.7B-Base-8bit'; Mode = 'clone'; DownloadStrategy = 'huggingface'; RepoId = 'Qwen/Qwen3-TTS-12Hz-1.7B-Base'; ArtifactGroups = @(@('config.json'), @('model.safetensors', 'model.safetensors.index.json'), @('preprocessor_config.json'), @('tokenizer_config.json', 'vocab.json')) },
    [pscustomobject]@{ Key = 'qwen-custom-06b'; Label = 'Qwen Custom 0.6B'; Family = 'qwen'; Folder = 'Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit'; Mode = 'custom'; DownloadStrategy = 'huggingface'; RepoId = 'Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice'; ArtifactGroups = @(@('config.json'), @('model.safetensors', 'model.safetensors.index.json'), @('preprocessor_config.json'), @('tokenizer_config.json', 'vocab.json')) },
    [pscustomobject]@{ Key = 'qwen-design-06b'; Label = 'Qwen Design 0.6B'; Family = 'qwen'; Folder = 'Qwen3-TTS-12Hz-0.6B-VoiceDesign-8bit'; Mode = 'design'; DownloadStrategy = 'huggingface'; RepoId = $null; ArtifactGroups = @(@('config.json'), @('model.safetensors', 'model.safetensors.index.json'), @('preprocessor_config.json'), @('tokenizer_config.json', 'vocab.json')) },
    [pscustomobject]@{ Key = 'qwen-clone-06b'; Label = 'Qwen Clone 0.6B'; Family = 'qwen'; Folder = 'Qwen3-TTS-12Hz-0.6B-Base-8bit'; Mode = 'clone'; DownloadStrategy = 'huggingface'; RepoId = 'Qwen/Qwen3-TTS-12Hz-0.6B-Base'; ArtifactGroups = @(@('config.json'), @('model.safetensors', 'model.safetensors.index.json'), @('preprocessor_config.json'), @('tokenizer_config.json', 'vocab.json')) },
    [pscustomobject]@{ Key = 'omnivoice'; Label = 'OmniVoice'; Family = 'omnivoice'; Folder = 'OmniVoice'; Mode = 'all'; DownloadStrategy = 'huggingface'; RepoId = 'k2-fsa/OmniVoice'; ArtifactGroups = @(@('config.json'), @('model.safetensors', 'model.safetensors.index.json'), @('tokenizer_config.json', 'tokenizer.json'), @('audio_tokenizer/config.json'), @('audio_tokenizer/model.safetensors'), @('audio_tokenizer/preprocessor_config.json')) },
    [pscustomobject]@{ Key = 'piper-lessac'; Label = 'Piper en_US lessac medium'; Family = 'piper'; Folder = 'Piper-en_US-lessac-medium'; Mode = 'custom'; DownloadStrategy = 'piper'; PiperVoice = 'en_US-lessac-medium'; ArtifactGroups = @(@('model.onnx'), @('model.onnx.json')) }
)

function Get-ProjectRoot {
    if (-not [string]::IsNullOrWhiteSpace($PSScriptRoot)) {
        return (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
    }

    $fallbackRoot = $env:TTS_LAUNCH_PROJECT_ROOT
    if (-not [string]::IsNullOrWhiteSpace($fallbackRoot) -and (Test-Path $fallbackRoot)) {
        return (Resolve-Path $fallbackRoot).Path
    }

    throw 'Unable to resolve project root: neither $PSScriptRoot nor TTS_LAUNCH_PROJECT_ROOT is available.'
}

function ConvertTo-PlainText {
    param([Parameter(Mandatory = $true)][Security.SecureString]$SecureValue)
    $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($SecureValue)
    try {
        return [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
    }
    finally {
        if ($pointer -ne [IntPtr]::Zero) {
            [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
        }
    }
}

function Read-TrimmedHostInput {
    param([Parameter(Mandatory = $true)][string]$Prompt)
    $value = Read-Host $Prompt
    if ($null -eq $value) {
        return ''
    }
    return $value.Trim()
}

function Resolve-HttpProbeHost {
    param([Parameter(Mandatory = $true)][string]$BindHost)

    $normalized = $BindHost.Trim().ToLowerInvariant()
    if ([string]::IsNullOrWhiteSpace($normalized) -or $normalized -eq '0.0.0.0' -or $normalized -eq '::' -or $normalized -eq '[::]') {
        return '127.0.0.1'
    }
    return $BindHost.Trim()
}

function Select-MenuOption {
    param(
        [Parameter(Mandatory = $true)][string]$Prompt,
        [Parameter(Mandatory = $true)][object[]]$Options,
        [string]$DisplayProperty = 'Label'
    )

    Write-Host ''
    Write-Host $Prompt -ForegroundColor Cyan
    for ($index = 0; $index -lt $Options.Count; $index++) {
        Write-Host ('[{0}] {1}' -f ($index + 1), $Options[$index].$DisplayProperty)
    }
    while ($true) {
        $raw = Read-TrimmedHostInput -Prompt 'Enter option number'
        $choice = 0
        if ([int]::TryParse($raw, [ref]$choice) -and $choice -ge 1 -and $choice -le $Options.Count) {
            return $Options[$choice - 1]
        }
        Write-Warning 'Enter a number from the menu.'
    }
}

function Select-MultipleMenuOptions {
    param(
        [Parameter(Mandatory = $true)][string]$Prompt,
        [Parameter(Mandatory = $true)][object[]]$Options,
        [string]$DisplayProperty = 'Label'
    )

    if ($Options.Count -eq 0) {
        throw 'Select-MultipleMenuOptions requires at least one option.'
    }

    Write-Host ''
    Write-Host $Prompt -ForegroundColor Cyan
    for ($index = 0; $index -lt $Options.Count; $index++) {
        Write-Host ('[{0}] {1}' -f ($index + 1), $Options[$index].$DisplayProperty)
    }

    while ($true) {
        $raw = Read-TrimmedHostInput -Prompt 'Enter one or more option numbers separated by comma'
        if ([string]::IsNullOrWhiteSpace($raw)) {
            Write-Warning 'Select at least one option from the menu.'
            continue
        }

        $selectedIndexes = New-Object System.Collections.Generic.List[int]
        $isValid = $true
        foreach ($token in ($raw -split ',')) {
            $trimmed = $token.Trim()
            $choice = 0
            if (-not [int]::TryParse($trimmed, [ref]$choice) -or $choice -lt 1 -or $choice -gt $Options.Count) {
                $isValid = $false
                break
            }
            if (-not $selectedIndexes.Contains($choice - 1)) {
                $selectedIndexes.Add($choice - 1) | Out-Null
            }
        }

        if (-not $isValid -or $selectedIndexes.Count -eq 0) {
            Write-Warning 'Enter one or more valid menu numbers separated by commas.'
            continue
        }

        $selectedOptions = New-Object System.Collections.Generic.List[object]
        foreach ($selectedIndex in $selectedIndexes) {
            $selectedOptions.Add($Options[$selectedIndex]) | Out-Null
        }
        return $selectedOptions.ToArray()
    }
}

function Invoke-LauncherJson {
    param(
        [Parameter(Mandatory = $true)][string]$ProjectRoot,
        [Parameter(Mandatory = $true)][string[]]$LauncherArgs
    )

    $commandOutput = & py -3.11 -m launcher --project-root $ProjectRoot @LauncherArgs 2>&1 | Out-String
    $exitCode = $LASTEXITCODE
    if ([string]::IsNullOrWhiteSpace($commandOutput)) {
        throw "Launcher command returned no JSON output: $($LauncherArgs -join ' ')"
    }
    try {
        $payload = $commandOutput | ConvertFrom-Json
    }
    catch {
        throw "Failed to parse launcher JSON output for '$($LauncherArgs -join ' ')':`n$commandOutput"
    }
    return [pscustomobject]@{ ExitCode = $exitCode; Payload = $payload; RawOutput = $commandOutput }
}

function Assert-WindowsPreflight {
    param([Parameter(Mandatory = $true)][string]$ProjectRoot)
    $isWindowsHost = $env:OS -eq 'Windows_NT'
    if (-not $isWindowsHost) { throw "This launcher supports Windows PowerShell only." }
    if (-not (Get-Command py -ErrorAction SilentlyContinue)) { throw "The 'py' launcher was not found. Install Python 3.11+ with the Windows py launcher." }
    & py -3.11 --version | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Python 3.11 was not found through 'py -3.11'." }
    if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) { throw "ffmpeg was not found in PATH. Install ffmpeg and retry." }
    if (-not (Test-Path (Join-Path $ProjectRoot 'launcher'))) { throw "Launcher package was not found under project root: $ProjectRoot" }
}

function Get-ValidationRoots {
    param([Parameter(Mandatory = $true)][string]$ModelRoot)
    $roots = New-Object System.Collections.Generic.List[string]
    if (-not (Test-Path $ModelRoot)) { return @() }
    $roots.Add((Resolve-Path $ModelRoot).Path)
    $snapshotsPath = Join-Path $ModelRoot 'snapshots'
    if (Test-Path $snapshotsPath) {
        Get-ChildItem -Path $snapshotsPath -Directory | Sort-Object Name | ForEach-Object { $roots.Add($_.FullName) }
    }
    return $roots.ToArray()
}

function Test-ArtifactGroup {
    param([Parameter(Mandatory = $true)][string]$RootPath, [Parameter(Mandatory = $true)][string[]]$Candidates)
    foreach ($candidate in $Candidates) {
        if (Test-Path (Join-Path $RootPath $candidate)) { return $true }
    }
    return $false
}

function Test-ModelArtifacts {
    param([Parameter(Mandatory = $true)][object]$Model, [Parameter(Mandatory = $true)][string]$ModelsDir)
    $modelRoot = Join-Path $ModelsDir $Model.Folder
    $roots = Get-ValidationRoots -ModelRoot $modelRoot
    foreach ($root in $roots) {
        $allGroupsSatisfied = $true
        foreach ($group in $Model.ArtifactGroups) {
            if (-not (Test-ArtifactGroup -RootPath $root -Candidates $group)) {
                $allGroupsSatisfied = $false
                break
            }
        }
        if ($allGroupsSatisfied) {
            return [pscustomobject]@{ Available = $true; ResolvedPath = $root; ExpectedPath = $modelRoot }
        }
    }
    return [pscustomobject]@{ Available = $false; ResolvedPath = $null; ExpectedPath = $modelRoot }
}

function Ensure-FamilyEnvironment {
    param(
        [Parameter(Mandatory = $true)][string]$ProjectRoot,
        [Parameter(Mandatory = $true)][string]$Family,
        [Parameter(Mandatory = $true)][string]$Module
    )

    $createResult = Invoke-LauncherJson -ProjectRoot $ProjectRoot -LauncherArgs @('create-env', '--family', $Family, '--module', $Module, '--apply')
    if ($createResult.ExitCode -ne 0) { throw "Failed to prepare the family environment for '$Family':`n$($createResult.RawOutput)" }
    $checkResult = Invoke-LauncherJson -ProjectRoot $ProjectRoot -LauncherArgs @('check-env', '--family', $Family, '--module', $Module)
    if ($checkResult.ExitCode -ne 0) { throw "Environment check for '$Family' failed: `n$($checkResult.RawOutput)" }
    $importPayload = $checkResult.Payload.check_env.import_check
    if ($null -eq $importPayload -or $importPayload.returncode -ne 0) { throw "Runtime import check for '$Family' failed: `n$($checkResult.RawOutput)" }
    foreach ($entry in $importPayload.stdout.PSObject.Properties) {
        if (-not [bool]$entry.Value) { throw "Environment '$Family' is missing required runtime import: $($entry.Name)" }
    }
    return [string]$checkResult.Payload.check_env.expected_python_path
}

function Read-TransientToken {
    param([Parameter(Mandatory = $true)][string]$Prompt)
    return (ConvertTo-PlainText -SecureValue (Read-Host -Prompt $Prompt -AsSecureString))
}

function Invoke-HuggingFaceDownload {
    param(
        [Parameter(Mandatory = $true)][string]$PythonPath,
        [Parameter(Mandatory = $true)][string]$RepoId,
        [Parameter(Mandatory = $true)][string]$TargetDir,
        [string]$Token
    )

    $downloadCode = @'
from huggingface_hub import snapshot_download
import os
import sys

snapshot_download(
    repo_id=sys.argv[1],
    local_dir=sys.argv[2],
    token=os.environ.get("HF_TOKEN") or None,
)
'@
    $previousToken = $env:HF_TOKEN
    try {
        if ($Token) { $env:HF_TOKEN = $Token } else { Remove-Item Env:HF_TOKEN -ErrorAction SilentlyContinue }
        & $PythonPath -c $downloadCode $RepoId $TargetDir
        if ($LASTEXITCODE -ne 0) { throw "snapshot_download завершился с кодом $LASTEXITCODE" }
    }
    finally {
        if ($null -eq $previousToken) { Remove-Item Env:HF_TOKEN -ErrorAction SilentlyContinue } else { $env:HF_TOKEN = $previousToken }
    }
}

function Invoke-PiperDownload {
    param(
        [Parameter(Mandatory = $true)][string]$PythonPath,
        [Parameter(Mandatory = $true)][object]$Model,
        [Parameter(Mandatory = $true)][string]$ModelsDir
    )

    $targetDir = Join-Path $ModelsDir $Model.Folder
    New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
    & $PythonPath -m piper.download_voices $Model.PiperVoice --download-dir $targetDir
    if ($LASTEXITCODE -ne 0) { throw "Failed to download Piper voice '$($Model.PiperVoice)'." }
    $sourceModel = Join-Path $targetDir ($Model.PiperVoice + '.onnx')
    $sourceConfig = Join-Path $targetDir ($Model.PiperVoice + '.onnx.json')
    if (Test-Path $sourceModel) { Move-Item $sourceModel (Join-Path $targetDir 'model.onnx') -Force }
    if (Test-Path $sourceConfig) { Move-Item $sourceConfig (Join-Path $targetDir 'model.onnx.json') -Force }
}

function Ensure-ModelAvailability {
    param(
        [Parameter(Mandatory = $true)][string]$PythonPath,
        [Parameter(Mandatory = $true)][object]$Model,
        [Parameter(Mandatory = $true)][string]$ModelsDir
    )

    $validation = Test-ModelArtifacts -Model $Model -ModelsDir $ModelsDir
    if ($validation.Available) {
        Write-Host "Model found: $($validation.ResolvedPath)" -ForegroundColor Green
        return $validation.ResolvedPath
    }

    Write-Warning "Model '$($Model.Label)' is missing or incomplete. Expected path: $($validation.ExpectedPath)"
    $confirmation = Read-TrimmedHostInput -Prompt 'Download the model now? [y/N]'
    if ($confirmation.Trim().ToLower() -notin @('y', 'yes')) { throw 'Launch cancelled: model is not prepared locally.' }

    if ($Model.DownloadStrategy -eq 'piper') {
        Invoke-PiperDownload -PythonPath $PythonPath -Model $Model -ModelsDir $ModelsDir
    }
    elseif ($Model.DownloadStrategy -eq 'huggingface') {
        $repoId = $Model.RepoId
        if (-not [string]::IsNullOrWhiteSpace($repoId)) {
            Write-Host "Using built-in Hugging Face repo ID for $($Model.Label): $repoId" -ForegroundColor Cyan
        }
        else {
            $repoId = Read-TrimmedHostInput -Prompt 'Enter the Hugging Face repo ID for this model'
        }
        if ([string]::IsNullOrWhiteSpace($repoId)) { throw 'A Hugging Face repo ID is required for this download.' }
        $needsToken = Read-TrimmedHostInput -Prompt 'Use a temporary HF token for this download? [y/N]'
        $token = $null
        if ($needsToken.Trim().ToLower() -in @('y', 'yes')) { $token = Read-TransientToken -Prompt 'Enter HF token (it will not be saved)' }
        $targetDir = Join-Path $ModelsDir $Model.Folder
        New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
        Invoke-HuggingFaceDownload -PythonPath $PythonPath -RepoId $repoId -TargetDir $targetDir -Token $token
    }
    else {
        throw "Unknown download strategy: $($Model.DownloadStrategy)"
    }

    $afterDownload = Test-ModelArtifacts -Model $Model -ModelsDir $ModelsDir
    if (-not $afterDownload.Available) { throw "Download finished, but required artifacts for '$($Model.Label)' are still missing." }
    Write-Host "Model is ready: $($afterDownload.ResolvedPath)" -ForegroundColor Green
    return $afterDownload.ResolvedPath
}

function Get-RuntimeCapabilityBindings {
    param(
        [Parameter(Mandatory = $true)][string]$Family,
        [Parameter(Mandatory = $true)][object[]]$Models
    )

    $bindings = [ordered]@{
        family = $Family
        custom_model = $null
        design_model = $null
        clone_model = $null
    }

    foreach ($model in $Models) {
        if ($model.Mode -eq 'all') {
            $bindings.custom_model = $model.Folder
            $bindings.design_model = $model.Folder
            $bindings.clone_model = $model.Folder
            continue
        }

        if ($model.Mode -eq 'custom') { $bindings.custom_model = $model.Folder }
        elseif ($model.Mode -eq 'design') { $bindings.design_model = $model.Folder }
        elseif ($model.Mode -eq 'clone') { $bindings.clone_model = $model.Folder }
    }

    return [pscustomobject]$bindings
}

function Show-RuntimeCapabilityBindings {
    param([Parameter(Mandatory = $true)][object]$Bindings)

    Write-Host ''
    Write-Host 'Runtime capability bindings:' -ForegroundColor Cyan
    Write-Host ('  TTS_ACTIVE_FAMILY={0}' -f $Bindings.family)
    Write-Host ('  TTS_DEFAULT_CUSTOM_MODEL={0}' -f ($(if ($Bindings.custom_model) { $Bindings.custom_model } else { '<unbound>' })))
    Write-Host ('  TTS_DEFAULT_DESIGN_MODEL={0}' -f ($(if ($Bindings.design_model) { $Bindings.design_model } else { '<unbound>' })))
    Write-Host ('  TTS_DEFAULT_CLONE_MODEL={0}' -f ($(if ($Bindings.clone_model) { $Bindings.clone_model } else { '<unbound>' })))
}

function Configure-ServiceEnvironment {
    param(
        [Parameter(Mandatory = $true)][string]$ProjectRoot,
        [Parameter(Mandatory = $true)][object]$InspectPayload,
        [Parameter(Mandatory = $true)][object]$Service,
        [Parameter(Mandatory = $true)][object]$Bindings
    )

    $env:TTS_MODELS_DIR = Join-Path $ProjectRoot '.models'
    $env:TTS_OUTPUTS_DIR = Join-Path $ProjectRoot '.outputs'
    $env:TTS_VOICES_DIR = Join-Path $ProjectRoot '.voices'
    $env:TTS_UPLOAD_STAGING_DIR = Join-Path $ProjectRoot '.uploads'
    $env:TTS_ACTIVE_FAMILY = [string]$Bindings.family
    if ($Bindings.custom_model) { $env:TTS_DEFAULT_CUSTOM_MODEL = [string]$Bindings.custom_model } else { Remove-Item Env:TTS_DEFAULT_CUSTOM_MODEL -ErrorAction SilentlyContinue }
    if ($Bindings.design_model) { $env:TTS_DEFAULT_DESIGN_MODEL = [string]$Bindings.design_model } else { Remove-Item Env:TTS_DEFAULT_DESIGN_MODEL -ErrorAction SilentlyContinue }
    if ($Bindings.clone_model) { $env:TTS_DEFAULT_CLONE_MODEL = [string]$Bindings.clone_model } else { Remove-Item Env:TTS_DEFAULT_CLONE_MODEL -ErrorAction SilentlyContinue }
    if ($InspectPayload.selected_backend) {
        $env:TTS_BACKEND = [string]$InspectPayload.selected_backend
        $env:TTS_BACKEND_AUTOSELECT = 'false'
    }
    $env:TTS_REQUEST_TIMEOUT_SECONDS = '300'

    if ($Service.Key -eq 'server') {
        $bindHost = Read-TrimmedHostInput -Prompt 'Host for HTTP server [0.0.0.0]'
        $port = Read-TrimmedHostInput -Prompt 'Port for HTTP server [8000]'
        $env:TTS_HOST = if ([string]::IsNullOrWhiteSpace($bindHost)) { '0.0.0.0' } else { $bindHost }
        $env:TTS_PORT = if ([string]::IsNullOrWhiteSpace($port)) { '8000' } else { $port }
        $env:TTS_LOG_LEVEL = 'info'
    }
    elseif ($Service.Key -eq 'telegram') {
        if ([string]::IsNullOrWhiteSpace($env:TTS_TELEGRAM_BOT_TOKEN)) { $env:TTS_TELEGRAM_BOT_TOKEN = Read-TransientToken -Prompt 'Enter Telegram bot token (it will not be saved)' }
        if ([string]::IsNullOrWhiteSpace($env:TTS_TELEGRAM_BOT_TOKEN)) { throw 'TTS_TELEGRAM_BOT_TOKEN is required for Telegram launch.' }
        $allowedIds = Read-TrimmedHostInput -Prompt 'Allowed user IDs (comma-separated, optional)'
        if (-not [string]::IsNullOrWhiteSpace($allowedIds)) { $env:TTS_TELEGRAM_ALLOWED_USER_IDS = $allowedIds }
        $adminIds = Read-TrimmedHostInput -Prompt 'Admin user IDs (comma-separated, optional)'
        if (-not [string]::IsNullOrWhiteSpace($adminIds)) { $env:TTS_TELEGRAM_ADMIN_USER_IDS = $adminIds }
        $env:TTS_TELEGRAM_RATE_LIMIT_ENABLED = 'true'
        $env:TTS_TELEGRAM_RATE_LIMIT_PER_USER_PER_MINUTE = '20'
        $env:TTS_TELEGRAM_DELIVERY_STORE_PATH = Join-Path $ProjectRoot '.state/telegram_delivery_store.json'
        $env:TTS_TELEGRAM_LOG_LEVEL = 'info'
    }
    else {
        $env:TTS_AUTO_PLAY_CLI = 'true'
    }
}

function Wait-HttpHealthCheck {
    param(
        [Parameter(Mandatory = $true)][string]$BindHost,
        [Parameter(Mandatory = $true)][string]$BindPort,
        [int]$TimeoutSeconds = 30
    )

    $probeHost = Resolve-HttpProbeHost -BindHost $BindHost
    $healthUrl = "http://{0}:{1}/health/live" -f $probeHost, $BindPort
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $response = Invoke-WebRequest -UseBasicParsing -Uri $healthUrl -TimeoutSec 3
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 300) {
                Write-Host ("HTTP server is live: {0}" -f $healthUrl) -ForegroundColor Green
                return $true
            }
        }
        catch {
            Start-Sleep -Seconds 1
        }
    }

    Write-Warning ("HTTP server did not report ready at {0} within {1} seconds." -f $healthUrl, $TimeoutSeconds)
    return $false
}

function Get-HttpServerPidFilePath {
    param([Parameter(Mandatory = $true)][string]$ProjectRoot)

    return (Join-Path $ProjectRoot '.state/launcher/http-server.pid')
}

function Read-HttpServerPidFile {
    param([Parameter(Mandatory = $true)][string]$ProjectRoot)

    $pidFile = Get-HttpServerPidFilePath -ProjectRoot $ProjectRoot
    if (-not (Test-Path $pidFile)) { return $null }

    $payload = [ordered]@{}
    foreach ($line in (Get-Content -Path $pidFile -ErrorAction SilentlyContinue)) {
        if ([string]::IsNullOrWhiteSpace($line) -or $line -notmatch '=') { continue }
        $parts = $line -split '=', 2
        $payload[$parts[0]] = $parts[1]
    }
    if (-not $payload.Contains('PID')) { return $null }
    return [pscustomobject]$payload
}

function Clear-HttpServerPidFile {
    param([Parameter(Mandatory = $true)][string]$ProjectRoot)

    $pidFile = Get-HttpServerPidFilePath -ProjectRoot $ProjectRoot
    Remove-Item -Path $pidFile -ErrorAction SilentlyContinue
}

function Get-TcpOwningProcessId {
    param([Parameter(Mandatory = $true)][int]$Port)

    try {
        $connection = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction Stop | Select-Object -First 1
        if ($null -ne $connection) { return [int]$connection.OwningProcess }
    }
    catch {
        return $null
    }
    return $null
}

function Stop-HttpServerProcess {
    param(
        [Parameter(Mandatory = $true)][int]$Pid,
        [Parameter(Mandatory = $true)][string]$ProjectRoot,
        [int]$WaitSeconds = 10
    )

    $process = Get-Process -Id $Pid -ErrorAction SilentlyContinue
    if ($null -eq $process) {
        Clear-HttpServerPidFile -ProjectRoot $ProjectRoot
        return
    }

    Stop-Process -Id $Pid -ErrorAction SilentlyContinue
    $deadline = (Get-Date).AddSeconds($WaitSeconds)
    while ((Get-Date) -lt $deadline) {
        $process = Get-Process -Id $Pid -ErrorAction SilentlyContinue
        if ($null -eq $process) { break }
        Start-Sleep -Seconds 1
    }
    $process = Get-Process -Id $Pid -ErrorAction SilentlyContinue
    if ($null -ne $process) {
        Stop-Process -Id $Pid -Force -ErrorAction SilentlyContinue
    }
    Clear-HttpServerPidFile -ProjectRoot $ProjectRoot
}

function Ensure-HttpServerLaunchTarget {
    param(
        [Parameter(Mandatory = $true)][string]$ProjectRoot,
        [Parameter(Mandatory = $true)][string]$BindHost,
        [Parameter(Mandatory = $true)][string]$BindPort,
        [Parameter(Mandatory = $true)][string]$Family,
        [Parameter(Mandatory = $true)][string]$Module
    )

    $pidInfo = Read-HttpServerPidFile -ProjectRoot $ProjectRoot
    if ($null -ne $pidInfo) {
        $ownedProcess = Get-Process -Id ([int]$pidInfo.PID) -ErrorAction SilentlyContinue
        if ($null -ne $ownedProcess) {
            while ($true) {
                $managedDecision = Read-TrimmedHostInput -Prompt 'Launcher-managed HTTP server is already running. [R]estart / [K]eep existing / [C]hange port'
                switch ($managedDecision.Trim().ToLower()) {
                    'r' {
                        Write-Host ("Stopping existing launcher-managed HTTP server (PID {0}) before restart." -f $pidInfo.PID) -ForegroundColor Cyan
                        Stop-HttpServerProcess -Pid ([int]$pidInfo.PID) -ProjectRoot $ProjectRoot
                        break
                    }
                    'restart' {
                        Write-Host ("Stopping existing launcher-managed HTTP server (PID {0}) before restart." -f $pidInfo.PID) -ForegroundColor Cyan
                        Stop-HttpServerProcess -Pid ([int]$pidInfo.PID) -ProjectRoot $ProjectRoot
                        break
                    }
                    'c' {
                        $replacementPort = Read-TrimmedHostInput -Prompt 'New port for HTTP server'
                        if ([string]::IsNullOrWhiteSpace($replacementPort)) {
                            Write-Warning 'A new port is required to continue.'
                            continue
                        }
                        $env:TTS_PORT = $replacementPort
                        return
                    }
                    'change' {
                        $replacementPort = Read-TrimmedHostInput -Prompt 'New port for HTTP server'
                        if ([string]::IsNullOrWhiteSpace($replacementPort)) {
                            Write-Warning 'A new port is required to continue.'
                            continue
                        }
                        $env:TTS_PORT = $replacementPort
                        return
                    }
                    'k' { throw "Launch cancelled: existing launcher-managed HTTP server remains active on $(Resolve-HttpProbeHost -BindHost $BindHost):$BindPort." }
                    'keep' { throw "Launch cancelled: existing launcher-managed HTTP server remains active on $(Resolve-HttpProbeHost -BindHost $BindHost):$BindPort." }
                    '' { throw "Launch cancelled: existing launcher-managed HTTP server remains active on $(Resolve-HttpProbeHost -BindHost $BindHost):$BindPort." }
                    default { Write-Warning 'Enter R to restart, K to keep the existing server, or C to choose a new port.' }
                }
            }
        }
        else {
            Clear-HttpServerPidFile -ProjectRoot $ProjectRoot
        }
    }

    $resolvedPort = [int]$BindPort
    while ($true) {
        $ownerPid = Get-TcpOwningProcessId -Port $resolvedPort
        if ($null -eq $ownerPid) {
            if ($env:TTS_PORT -ne [string]$resolvedPort) { $env:TTS_PORT = [string]$resolvedPort }
            return
        }

        $decision = Read-TrimmedHostInput -Prompt 'Port is occupied by a non-launcher process. [S]top and restart / [K]eep existing / [C]hange port'
        switch ($decision.Trim().ToLower()) {
            's' {
                try {
                    Stop-Process -Id $ownerPid -ErrorAction Stop
                    Start-Sleep -Seconds 1
                }
                catch {
                    Write-Warning ("Unable to stop the process currently occupying {0}:{1}." -f (Resolve-HttpProbeHost -BindHost $BindHost), $resolvedPort)
                    continue
                }
            }
            'stop' {
                try {
                    Stop-Process -Id $ownerPid -ErrorAction Stop
                    Start-Sleep -Seconds 1
                }
                catch {
                    Write-Warning ("Unable to stop the process currently occupying {0}:{1}." -f (Resolve-HttpProbeHost -BindHost $BindHost), $resolvedPort)
                    continue
                }
            }
            'c' {
                $replacementPort = Read-TrimmedHostInput -Prompt 'New port for HTTP server'
                if ([string]::IsNullOrWhiteSpace($replacementPort)) {
                    Write-Warning 'A new port is required to continue.'
                    continue
                }
                $resolvedPort = [int]$replacementPort
            }
            'change' {
                $replacementPort = Read-TrimmedHostInput -Prompt 'New port for HTTP server'
                if ([string]::IsNullOrWhiteSpace($replacementPort)) {
                    Write-Warning 'A new port is required to continue.'
                    continue
                }
                $resolvedPort = [int]$replacementPort
            }
            'k' { throw "Launch cancelled: existing non-launcher HTTP server remains active on $(Resolve-HttpProbeHost -BindHost $BindHost):$resolvedPort." }
            'keep' { throw "Launch cancelled: existing non-launcher HTTP server remains active on $(Resolve-HttpProbeHost -BindHost $BindHost):$resolvedPort." }
            '' { throw "Launch cancelled: existing non-launcher HTTP server remains active on $(Resolve-HttpProbeHost -BindHost $BindHost):$resolvedPort." }
            default { Write-Warning 'Enter S to stop and restart, K to keep the existing server, or C to choose a new port.' }
        }
    }
}

function Start-SelectedService {
    param(
        [Parameter(Mandatory = $true)][string]$ProjectRoot,
        [Parameter(Mandatory = $true)][string]$Family,
        [Parameter(Mandatory = $true)][string]$Module,
        [Parameter(Mandatory = $true)][object]$Service
    )

    $dryRun = Invoke-LauncherJson -ProjectRoot $ProjectRoot -LauncherArgs @('exec', '--family', $Family, '--module', $Module, '--dry-run')
    Write-Host ''
    Write-Host ("Launching: {0}" -f ($dryRun.Payload.exec.command -join ' ')) -ForegroundColor Yellow

    if ($Service.Key -eq 'server') {
        Ensure-HttpServerLaunchTarget -ProjectRoot $ProjectRoot -BindHost $env:TTS_HOST -BindPort $env:TTS_PORT -Family $Family -Module $Module
        $process = Start-Process -FilePath 'py' -ArgumentList @('-3.11', '-m', 'launcher', '--project-root', $ProjectRoot, 'exec', '--family', $Family, '--module', $Module) -WorkingDirectory $ProjectRoot -PassThru
        Write-Host ("Server process started with PID {0}." -f $process.Id) -ForegroundColor Cyan
        if (-not (Wait-HttpHealthCheck -BindHost $env:TTS_HOST -BindPort $env:TTS_PORT)) {
            Stop-HttpServerProcess -Pid $process.Id -ProjectRoot $ProjectRoot
            throw 'HTTP server failed to report live after startup.'
        }
        $pidFile = Get-HttpServerPidFilePath -ProjectRoot $ProjectRoot
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $pidFile) | Out-Null
        @(
            "PID=$($process.Id)"
            "BIND_HOST=$($env:TTS_HOST)"
            "BIND_PORT=$($env:TTS_PORT)"
            "FAMILY=$Family"
            "MODULE=$Module"
        ) | Set-Content -Path $pidFile -Encoding utf8
        return
    }

    & py -3.11 -m launcher --project-root $ProjectRoot exec --family $Family --module $Module
    if ($LASTEXITCODE -ne 0) { throw "Service exited with code $LASTEXITCODE" }
}

function Main {
    $projectRoot = Get-ProjectRoot
    Assert-WindowsPreflight -ProjectRoot $projectRoot

    $serviceOptions = @(
        [pscustomobject]@{ Key = 'server'; Label = 'HTTP Server' },
        [pscustomobject]@{ Key = 'cli'; Label = 'CLI' },
        [pscustomobject]@{ Key = 'telegram'; Label = 'Telegram Bot' }
    )

    Write-Host 'Interactive Windows launcher for tts-server' -ForegroundColor Green
    $service = Select-MenuOption -Prompt 'Select service to launch' -Options $serviceOptions
    $family = Select-MenuOption -Prompt 'Select family to prepare' -Options $SCRIPT:FAMILY_OPTIONS
    $familyModels = @($SCRIPT:MODEL_OPTIONS | Where-Object { $_.Family -eq $family.Key })
    if ($familyModels.Count -eq 0) {
        throw "No model options are configured for family '$($family.Key)'."
    }

    $modelsToEnsure = @()
    if ($familyModels.Count -eq 1) {
        $modelsToEnsure = @($familyModels[0])
        Write-Host ''
        Write-Host ('Automatically preparing the only model for {0}: {1}' -f $family.Label, $familyModels[0].Label) -ForegroundColor Cyan
    }
    else {
        $modelsToEnsure = @(Select-MultipleMenuOptions -Prompt 'Select models to download if missing (multiple options can be selected, separated by comma)' -Options $familyModels)
    }

    $familyPython = Ensure-FamilyEnvironment -ProjectRoot $projectRoot -Family $family.Key -Module $service.Key
    $inspectResult = Invoke-LauncherJson -ProjectRoot $projectRoot -LauncherArgs @('inspect', '--family', $family.Key, '--module', $service.Key)
    if ($inspectResult.ExitCode -ne 0) { throw "Failed to resolve launch profile: `n$($inspectResult.RawOutput)" }
    Write-Host ''
    Write-Host ('Selected family: {0}' -f $family.Label) -ForegroundColor Cyan
    Write-Host ('Host contour: {0} / backend: {1}' -f $inspectResult.Payload.host.platform_system, $inspectResult.Payload.selected_backend) -ForegroundColor Cyan
    if (-not $inspectResult.Payload.compatible) { throw "Selected profile is incompatible with this host even after environment setup: $([string]::Join(', ', $inspectResult.Payload.reasons))" }

    $modelsDir = Join-Path $projectRoot '.models'
    New-Item -ItemType Directory -Force -Path $modelsDir | Out-Null
    foreach ($model in $modelsToEnsure) {
        Write-Host ('Ensuring model: {0}' -f $model.Label) -ForegroundColor Cyan
        Ensure-ModelAvailability -PythonPath $familyPython -Model $model -ModelsDir $modelsDir | Out-Null
    }
    $runtimeBindings = Get-RuntimeCapabilityBindings -Family $family.Key -Models $modelsToEnsure
    Show-RuntimeCapabilityBindings -Bindings $runtimeBindings
    Configure-ServiceEnvironment -ProjectRoot $projectRoot -InspectPayload $inspectResult.Payload -Service $service -Bindings $runtimeBindings
    Start-SelectedService -ProjectRoot $projectRoot -Family $family.Key -Module $service.Key -Service $service
}

try {
    Main
}
catch {
    Write-Error $_
    exit 1
}
