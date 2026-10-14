param([Parameter(Position=0)][string]$Destination)

$ErrorActionPreference = 'Stop'
$EngineFolder = Split-Path -Parent $MyInvocation.MyCommand.Path
$DownloadLock = $null

function Get-ModelSHA([string]$Path) {
    $Stream = [IO.FileStream]::new($Path, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::Read, 4MB, [IO.FileOptions]::SequentialScan)
    $SHA = [Security.Cryptography.SHA256]::Create()
    try { return [BitConverter]::ToString($SHA.ComputeHash($Stream)).Replace('-', '').ToLowerInvariant() }
    finally { $Stream.Dispose(); $SHA.Dispose() }
}

try {
    $Pins = Get-Content -LiteralPath (Join-Path $EngineFolder 'pins.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    $Model = $Pins.models.files[0]
    $FileName = [string]$Model.rfilename
    $ExpectedSize = [long]$Model.size
    $ExpectedHash = [string]$Model.lfs.sha256
    $Revision = [string]$Pins.models.revision
    $Repository = [string]$Pins.models.repo
    if ($FileName -ne [IO.Path]::GetFileName($FileName) -or $ExpectedSize -le 0 -or
        $ExpectedHash -notmatch '^[a-f0-9]{64}$' -or $Revision -notmatch '^[a-f0-9]{40}$' -or
        $Repository -notmatch '^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$') {
        throw 'Model pins are incomplete or invalid. Extract a fresh release.'
    }
    $Url = "https://huggingface.co/$Repository/resolve/$Revision/$FileName?download=true"

    if ([string]::IsNullOrWhiteSpace($Destination)) {
        Add-Type -AssemblyName System.Windows.Forms
        $Picker = New-Object System.Windows.Forms.FolderBrowserDialog
        $Picker.Description = 'Choose where to save Gyro-M (about 86 GiB). Interrupted downloads can resume.'
        $Picker.ShowNewFolderButton = $true
        try {
            if ($Picker.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) { exit 1 }
            $Destination = $Picker.SelectedPath
        }
        finally { $Picker.Dispose() }
    }
    $Destination = [IO.Path]::GetFullPath($Destination)
    if ($Destination.StartsWith('\\')) { throw 'Choose a local SSD folder rather than a network share.' }
    [IO.Directory]::CreateDirectory($Destination) | Out-Null
    $ModelPath = Join-Path $Destination $FileName
    $PartialPath = "$ModelPath.download"
    $LockPath = Join-Path $Destination '.gyro-download.lock'
    try {
        $DownloadLock = [IO.File]::Open($LockPath, [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
        [IO.File]::SetAttributes($LockPath, [IO.FileAttributes]::Hidden)
    }
    catch { throw 'Another downloader is using this folder, or the folder is not writable.' }

    if (Test-Path -LiteralPath $ModelPath) {
        $Existing = Get-Item -LiteralPath $ModelPath
        if ($Existing.Length -ne $ExpectedSize) {
            throw "Existing model has the wrong size and was left untouched: $ModelPath. Select the correct folder or preserve it under another name."
        }
        Write-Host 'Checking the existing model SHA256; this can take a few minutes...'
        if ((Get-ModelSHA $ModelPath) -ne $ExpectedHash) {
            throw "Existing model SHA256 does not match and was left untouched: $ModelPath. Preserve it under another name before downloading the pinned file."
        }
        Write-Host 'The pinned model is already present and verified.'
    }
    else {
        $Curl = Join-Path $env:SystemRoot 'System32\curl.exe'
        if (-not (Test-Path -LiteralPath $Curl)) { throw 'Windows curl.exe was not found. Install a current Windows curl build and retry.' }
        $PartialSize = 0L
        if (Test-Path -LiteralPath $PartialPath) {
            $PartialSize = (Get-Item -LiteralPath $PartialPath).Length
            if ($PartialSize -gt $ExpectedSize) {
                throw "Partial file is too large and was left untouched: $PartialPath. Preserve it under another name and retry."
            }
        }
        if ($PartialSize -lt $ExpectedSize) {
            $Drive = New-Object IO.DriveInfo([IO.Path]::GetPathRoot($Destination))
            if ($Drive.DriveFormat -eq 'FAT32') { throw 'FAT32 cannot store this model. Choose an NTFS or exFAT SSD.' }
            if ($Drive.AvailableFreeSpace -lt ($ExpectedSize - $PartialSize + 64MB)) {
                throw 'Not enough disk space for the remaining download. Free space or select another SSD.'
            }
            Write-Host "Downloading/resuming in: $Destination"
            Write-Host 'Reopen this downloader with the same folder after an interruption.'
            & $Curl --fail --location --retry 10 --retry-all-errors --continue-at - --output $PartialPath $Url
            if ($LASTEXITCODE -ne 0) { throw "curl.exe exited with code $LASTEXITCODE. The partial file was kept for retry." }
        }
        if ((Get-Item -LiteralPath $PartialPath).Length -ne $ExpectedSize) {
            throw "Download size mismatch; partial file kept at $PartialPath."
        }
        Write-Host 'Checking the downloaded model SHA256; this can take a few minutes...'
        if ((Get-ModelSHA $PartialPath) -ne $ExpectedHash) {
            throw "SHA256 mismatch; partial file kept at $PartialPath. Preserve it under another name before retrying."
        }
        # File.Move refuses to replace a model created by another program during the download.
        [IO.File]::Move($PartialPath, $ModelPath)
        Write-Host 'Download and SHA256 verification passed.'
    }
    $Utf8NoBom = New-Object Text.UTF8Encoding($false)
    [IO.File]::WriteAllText((Join-Path $EngineFolder '.model-dir'), $Destination, $Utf8NoBom)
    Write-Host 'Saved this model folder for INSTALL.cmd.'
}
catch {
    Write-Host ("Download stopped: " + $_.Exception.Message)
    exit 1
}
finally {
    if ($DownloadLock) {
        $DownloadLock.Dispose()
        Remove-Item -LiteralPath $LockPath -Force -ErrorAction SilentlyContinue
    }
}
