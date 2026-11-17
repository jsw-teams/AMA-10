param(
    [Parameter(Mandatory = $true)][string]$RepoId,
    [Parameter(Mandatory = $true)][string]$FilePath,
    [Parameter(Mandatory = $true)][string]$Revision,
    [Parameter(Mandatory = $true)][string]$ExpectedLicense,
    [Parameter(Mandatory = $true)][string]$OutputPath,
    [string]$Contact = '',
    [int]$RequestDelaySeconds = 30
)
$ErrorActionPreference = 'Stop'

# Fetch one explicitly chosen file. Resolve a branch/tag to a commit first,
# require the caller's license review, and download using that immutable SHA.
if ($RequestDelaySeconds -lt 30) { throw 'RequestDelaySeconds must be at least 30 seconds for Hugging Face requests.' }
$headers = @{ 'User-Agent' = "AMA-10 Hugging Face dataset fetcher/1.0 (local research pipeline; contact: $(if ($Contact) {$Contact} else {'not provided'}))" }
$token = [Environment]::GetEnvironmentVariable('HF_TOKEN')
if ($token) { $headers.Authorization = "Bearer $token" }
$repoPath = (($RepoId -split '/') | ForEach-Object { [Uri]::EscapeDataString($_) }) -join '/'
$revisionEscaped = [Uri]::EscapeDataString($Revision)
$metadataUri = "https://huggingface.co/api/datasets/$repoPath/revision/$revisionEscaped"

function Invoke-PoliteRequest([string]$Uri, [string]$OutFile = '') {
    for ($attempt = 0; $attempt -lt 5; $attempt++) {
        try {
            if ($OutFile) {
                return Invoke-WebRequest -Uri $Uri -Headers $headers -OutFile $OutFile -PassThru -MaximumRedirection 10 -TimeoutSec 180
            }
            return Invoke-RestMethod -Uri $Uri -Headers $headers -TimeoutSec 60
        } catch {
            $status = 0; $retryAfter = 0; $resetSeconds = 0
            try {
                $status = [int]$_.Exception.Response.StatusCode
                $responseHeaders = $_.Exception.Response.Headers
                [void][int]::TryParse([string]$responseHeaders['Retry-After'], [ref]$retryAfter)
                $rate = [string]$responseHeaders['RateLimit']
                if ($rate -match 't=(\d+)') { $resetSeconds = [int]$Matches[1] }
            } catch { }
            if ($status -ge 400 -and $status -lt 500 -and $status -ne 408 -and $status -ne 429) {
                throw "Non-retryable Hugging Face response HTTP $status for $Uri."
            }
            if ($attempt -eq 4) { throw }
            $wait = [Math]::Max($RequestDelaySeconds, 30 * [Math]::Pow(2, $attempt))
            if ($status -eq 429) { $wait = [Math]::Max($wait, 300) }
            $wait = [Math]::Max($wait, [Math]::Max($retryAfter, $resetSeconds))
            $wait = [int][Math]::Min(3600, $wait)
            Write-Warning "Hugging Face request failed (HTTP $status); waiting $wait seconds before retry."
            Start-Sleep -Seconds $wait
        }
    }
}

$metadata = Invoke-PoliteRequest $metadataUri
if (-not $metadata.sha) { throw "Hub did not return a pinned commit SHA for $RepoId@$Revision." }
$license = [string]$metadata.cardData.license
if (-not $license) { throw 'Dataset card has no machine-readable license; refusing to download.' }
if ($license -ne $ExpectedLicense) { throw "License mismatch: dataset says '$license', expected '$ExpectedLicense'." }
Write-Host "Verified dataset card license '$license'; pinned revision $($metadata.sha)."
Start-Sleep -Seconds $RequestDelaySeconds

$fileEscaped = (($FilePath -split '/') | ForEach-Object { [Uri]::EscapeDataString($_) }) -join '/'
$downloadUri = "https://huggingface.co/datasets/$repoPath/resolve/$($metadata.sha)/$fileEscaped"
$parent = Split-Path -Parent $OutputPath
if ($parent) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
$tempPath = "$OutputPath.partial"
try {
    $null = Invoke-PoliteRequest $downloadUri $tempPath
    if (-not (Test-Path -LiteralPath $tempPath) -or (Get-Item -LiteralPath $tempPath).Length -eq 0) {
        throw 'Download produced an empty or missing file.'
    }
    Move-Item -LiteralPath $tempPath -Destination $OutputPath -Force
} finally {
    if (Test-Path -LiteralPath $tempPath) { Remove-Item -LiteralPath $tempPath -Force }
}
$sha256 = (Get-FileHash -LiteralPath $OutputPath -Algorithm SHA256).Hash
$manifest = [pscustomobject]@{
    repo_id = $RepoId
    requested_revision = $Revision
    resolved_commit = $metadata.sha
    file_path = $FilePath
    local_path = $OutputPath
    dataset_card_license = $license
    dataset_card_url = "https://huggingface.co/datasets/$RepoId"
    file_url = $downloadUri
    retrieval_utc = (Get-Date).ToUniversalTime().ToString('o')
    sha256 = $sha256
    note = 'Dataset-card license verified; independently review licenses of all underlying media before training or redistribution.'
}
$manifestPath = "$OutputPath.sources.json"
$manifest | ConvertTo-Json -Depth 5 | Set-Content -Encoding utf8 $manifestPath
Write-Host "Saved $OutputPath ($sha256) and provenance $manifestPath"
