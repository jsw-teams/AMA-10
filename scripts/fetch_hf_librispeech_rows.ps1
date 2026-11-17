param(
    [string]$DatasetDir = 'data/raw/librispeech_hf_micro',
    [int]$TrainRows = 8,
    [int]$ValidationRows = 2,
    [int]$TrainOffset = 0,
    [int]$ValidationOffset = 0,
    [int]$DelaySeconds = 30,
    [string]$Contact = ''
)
$ErrorActionPreference = 'Stop'
if ($TrainRows -lt 1 -or $TrainRows -gt 100) { throw 'TrainRows must be between 1 and 100.' }
if ($ValidationRows -lt 1 -or $ValidationRows -gt 100) { throw 'ValidationRows must be between 1 and 100.' }
if ($TrainOffset -lt 0 -or $ValidationOffset -lt 0) { throw 'Row offsets cannot be negative.' }
if ($DelaySeconds -lt 30) { throw 'DelaySeconds must be at least 30 seconds for Hugging Face requests.' }

$repo = 'openslr/librispeech_asr'
$contactText = if ($Contact) { "contact: $Contact" } else { 'contact not provided' }
$headers = @{ 'User-Agent' = "AMA-10 LibriSpeech row collector/1.0 (local research pipeline; $contactText)" }
$token = [Environment]::GetEnvironmentVariable('HF_TOKEN')
if ($token) { $headers.Authorization = "Bearer $token" }
$lastRequest = [DateTime]::UtcNow

function Invoke-PoliteRequest([string]$Uri, [string]$OutFile = '') {
    for ($attempt = 0; $attempt -lt 5; $attempt++) {
        $waitForSpacing = [Math]::Ceiling(($lastRequest.AddSeconds($DelaySeconds) - [DateTime]::UtcNow).TotalSeconds)
        if ($waitForSpacing -gt 0) { Start-Sleep -Seconds $waitForSpacing }
        try {
            $script:lastRequest = [DateTime]::UtcNow
            if ($OutFile) {
                return Invoke-WebRequest -Uri $Uri -Headers $headers -OutFile $OutFile -PassThru -MaximumRedirection 10 -TimeoutSec 180
            }
            return Invoke-RestMethod -Uri $Uri -Headers $headers -TimeoutSec 90
        } catch {
            if ($attempt -eq 4) { throw }
            $status = 0; $retryAfter = 0; $resetSeconds = 0
            try {
                $status = [int]$_.Exception.Response.StatusCode
                $responseHeaders = $_.Exception.Response.Headers
                [void][int]::TryParse([string]$responseHeaders['Retry-After'],[ref]$retryAfter)
                $rate = [string]$responseHeaders['RateLimit']
                if ($rate -match 't=(\d+)') { $resetSeconds = [int]$Matches[1] }
            } catch { }
            if ($status -ge 400 -and $status -lt 500 -and $status -ne 429) { throw }
            $backoff = [Math]::Max($DelaySeconds,30*[Math]::Pow(2,$attempt))
            if ($status -eq 429) { $backoff = [Math]::Max(300,$backoff) }
            $backoff = [int][Math]::Min(3600,[Math]::Max($backoff,[Math]::Max($retryAfter,$resetSeconds)))
            Write-Warning "Hugging Face request failed (HTTP $status); waiting $backoff seconds before retry."
            Start-Sleep -Seconds $backoff
        }
    }
}

New-Item -ItemType Directory -Force -Path $DatasetDir | Out-Null
$audioDir = Join-Path $DatasetDir 'audio'
New-Item -ItemType Directory -Force -Path $audioDir | Out-Null
$metadataUri = "https://huggingface.co/api/datasets/$repo/revision/main"
$metadata = Invoke-PoliteRequest $metadataUri
if (-not $metadata.sha) { throw 'Hub did not return an immutable dataset revision.' }
if ([string]$metadata.cardData.license -ne 'cc-by-4.0') {
    throw "Unexpected or missing dataset-card license '$($metadata.cardData.license)'; expected cc-by-4.0."
}
Write-Host "Verified $repo license $($metadata.cardData.license), commit $($metadata.sha)."

$records = [System.Collections.Generic.List[object]]::new()
$manifestRows = [System.Collections.Generic.List[string]]::new()
$existingIds = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
$sourcesPath = Join-Path $DatasetDir 'sources.json'
$manifestPath = Join-Path $DatasetDir 'manifest.tsv'
if (Test-Path -LiteralPath $sourcesPath) {
    $existingRecords = @(Get-Content -Raw -LiteralPath $sourcesPath | ConvertFrom-Json)
    if (@($existingRecords|Where-Object source_revision -ne $metadata.sha).Count) {
        throw 'Existing rows use a different Hub revision; append into a separate dataset directory.'
    }
    foreach ($record in $existingRecords) {
        $records.Add($record)
        [void]$existingIds.Add([string]$record.source_id)
    }
    if (Test-Path -LiteralPath $manifestPath) {
        foreach ($line in Get-Content -LiteralPath $manifestPath) { $manifestRows.Add($line) }
    }
    if ($manifestRows.Count -ne $records.Count) { throw 'Existing manifest and provenance row counts differ.' }
}

$rowsBySplit = @{}
foreach ($splitSpec in @(@{name='train'; hub='train.100'; count=$TrainRows; offset=$TrainOffset},@{name='validation'; hub='validation'; count=$ValidationRows; offset=$ValidationOffset})) {
    $query = "dataset=$([Uri]::EscapeDataString($repo))&config=clean&split=$([Uri]::EscapeDataString($splitSpec.hub))&offset=$($splitSpec.offset)&length=$($splitSpec.count)"
    $response = Invoke-PoliteRequest "https://datasets-server.huggingface.co/rows?$query"
    if ($response.rows.Count -ne $splitSpec.count) { throw "Expected $($splitSpec.count) rows for $($splitSpec.hub), received $($response.rows.Count)." }
    $rowsBySplit[$splitSpec.name] = @($response.rows)
    Write-Host "Retrieved $($response.rows.Count) $($splitSpec.name) row descriptors."
}

$tab = [string][char]9
$utf8 = [Text.UTF8Encoding]::new($false)
$index = $records.Count
$fetchedThisRun = 0
foreach ($split in @('train','validation')) {
    foreach ($entry in $rowsBySplit[$split]) {
        $row = $entry.row
        if ($existingIds.Contains([string]$row.id)) {
            Write-Host "Already present: $($row.id); skipping asset download."
            continue
        }
        $audioItem = @($row.audio)[0]
        if ([string]::IsNullOrWhiteSpace([string]$row.text) -or -not $audioItem -or -not $audioItem.src) {
            throw "Row $($entry.row_idx) has no transcript or audio URL."
        }
        $assetUri = [Uri][string]$audioItem.src
        $assetPath = $assetUri.AbsolutePath
        if ($assetPath -notmatch ("/--/" + [Regex]::Escape($metadata.sha) + "/--/")) {
            throw "Audio asset for row $($entry.row_idx) is not pinned to the verified Hub commit."
        }
        if ($row.text.Contains($tab) -or $row.text.Contains([string][char]10) -or $row.text.Contains([string][char]13)) {
            throw "Row $($entry.row_idx) contains a TSV delimiter in its transcript."
        }
        $index++
        $name = 'libri_{0:D4}.wav' -f $index
        $rawPath = Join-Path $DatasetDir ($name + '.source')
        $wavPath = Join-Path $audioDir $name
        try {
            $null = Invoke-PoliteRequest ([string]$audioItem.src) $rawPath
            $sourceAudioHash = (Get-FileHash -LiteralPath $rawPath -Algorithm SHA256).Hash
            $ffmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
            if (-not $ffmpeg) { throw 'ffmpeg is required to decode the Hub audio sample to PCM WAV.' }
            & $ffmpeg.Source -nostdin -hide_banner -loglevel error -y -i $rawPath -ac 1 -ar 16000 -sample_fmt s16 -f wav $wavPath
            if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $wavPath)) { throw "ffmpeg could not decode row $($entry.row_idx)." }
        } finally {
            if (Test-Path -LiteralPath $rawPath) { Remove-Item -LiteralPath $rawPath -Force }
        }
        $record = [pscustomobject]@{
            local_audio = "audio/$name"
            transcript = [string]$row.text
            language = 'en'
            split = $split
            source_dataset = $repo
            source_revision = $metadata.sha
            source_split = if ($split -eq 'train') { 'train.clean.100' } else { 'validation.clean' }
            source_row_index = [int]$entry.row_idx
            source_id = [string]$row.id
            source_asset_path = $assetPath
            source_asset_type = [string]$audioItem.type
            source_audio_sha256 = $sourceAudioHash
            speaker_id = [string]$row.speaker_id
            chapter_id = [string]$row.chapter_id
            dataset_license = [string]$metadata.cardData.license
            dataset_url = "https://huggingface.co/datasets/$repo"
            audio_sha256 = (Get-FileHash -LiteralPath $wavPath -Algorithm SHA256).Hash
        }
        $records.Add($record)
        [void]$existingIds.Add([string]$row.id)
        $fetchedThisRun++
        $manifestRows.Add("audio/$name$tab$($record.transcript)$tab$split$tab$($record.language)")
        Write-Host "Fetched $split sample $fetchedThisRun/$($TrainRows+$ValidationRows); WAV SHA-256 $($record.audio_sha256)."
    }
}
[IO.File]::WriteAllLines($manifestPath,$manifestRows,$utf8)
[IO.File]::WriteAllText($sourcesPath,($records|ConvertTo-Json -Depth 8),$utf8)
Write-Host "Saved $($records.Count) paired examples with train/validation split provenance in $DatasetDir."
