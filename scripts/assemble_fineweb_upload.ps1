param(
    [string]$DataDirectory = (Join-Path $PSScriptRoot '..\data\analysis_upload\fineweb_edu'),
    [string]$TrainOutput
)

$ErrorActionPreference = 'Stop'
$DataDirectory = [IO.Path]::GetFullPath($DataDirectory)
$manifestPath = Join-Path $DataDirectory 'manifest.json'
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json

foreach ($part in $manifest.train.parts) {
    $path = Join-Path $DataDirectory $part.file
    $item = Get-Item -LiteralPath $path
    if ($item.Length -ne [int64]$part.bytes) { throw "Unexpected chunk size: $($part.file)" }
    if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $part.sha256) {
        throw "Chunk SHA-256 mismatch: $($part.file)"
    }
}

$validationPath = Join-Path $DataDirectory $manifest.validation.file
$validationItem = Get-Item -LiteralPath $validationPath
if ($validationItem.Length -ne [int64]$manifest.validation.bytes) { throw 'Unexpected validation size' }
if ((Get-FileHash -LiteralPath $validationPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $manifest.validation.sha256) {
    throw 'Validation SHA-256 mismatch'
}

if (-not $TrainOutput) { $TrainOutput = Join-Path $DataDirectory $manifest.train.file }
$TrainOutput = [IO.Path]::GetFullPath($TrainOutput)
if (Test-Path -LiteralPath $TrainOutput) { throw "Refusing to overwrite existing output: $TrainOutput" }
$output = [IO.File]::Open($TrainOutput, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
try {
    foreach ($part in $manifest.train.parts) {
        $input = [IO.File]::OpenRead((Join-Path $DataDirectory $part.file))
        try { $input.CopyTo($output) } finally { $input.Dispose() }
    }
} finally { $output.Dispose() }

$result = Get-Item -LiteralPath $TrainOutput
if ($result.Length -ne [int64]$manifest.train.bytes -or
    (Get-FileHash -LiteralPath $TrainOutput -Algorithm SHA256).Hash.ToLowerInvariant() -ne $manifest.train.sha256) {
    throw "Reassembled training shard failed size or SHA-256 validation: $TrainOutput"
}
Write-Output "Reassembled and verified $TrainOutput ($($result.Length) bytes, SHA-256 $($manifest.train.sha256))"
