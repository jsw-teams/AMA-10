param(
    [string]$TrainPath = 'data/raw/wikipedia_train.txt',
    [string]$ValidationPath = 'data/raw/wikipedia_validation.txt',
    [string]$ManifestPath = 'data/raw/wikipedia_sources.json',
    [int]$RequestDelaySeconds = 30,
    [string]$Contact = ''
)
$ErrorActionPreference = 'Stop'
if ($RequestDelaySeconds -lt 30) { throw 'RequestDelaySeconds must be at least 30 seconds for Wikimedia requests.' }
$titles = @(
    'Recurrent neural network',
    'Language model',
    'Natural language processing',
    'UTF-8',
    'C (programming language)',
    'OpenCL',
    'Gradient descent',
    'Machine learning'
)
$records = [System.Collections.Generic.List[object]]::new()
$trainText = [System.Collections.Generic.List[string]]::new()
$validationText = [System.Collections.Generic.List[string]]::new()
$contactPart = if ($Contact) { "; contact: $Contact" } else { '' }
$userAgent = "AMA-10 corpus fetcher/1.0 (local research pipeline)$contactPart"
foreach ($index in 0..($titles.Count - 1)) {
    $encodedTitle = [Uri]::EscapeDataString($titles[$index])
    $uri = "https://en.wikipedia.org/w/api.php?action=query&format=json&prop=extracts%7Cinfo%7Crevisions&explaintext=1&inprop=url&rvprop=ids&maxlag=5&titles=$encodedTitle"
    $response = $null
    for ($attempt = 0; $attempt -lt 5 -and -not $response; $attempt++) {
        $rateLimited = $false
        try {
            $response = Invoke-RestMethod -Uri $uri -UserAgent $userAgent -TimeoutSec 60
            if ($response.error) {
                $rateLimited = $response.error.code -match 'ratelimit|too-many-requests|lag'
                $message = $response.error.info
                $response = $null
                throw $message
            }
        } catch {
            if ($attempt -eq 4) { throw "Wikipedia fetch failed for $($titles[$index]): $_" }
            $response = $null
            $waitSeconds = [int][Math]::Min(600, 30 * [Math]::Pow(2, $attempt))
            $statusCode = 0
            $retryAfter = $null
            try {
                $statusCode = [int]$_.Exception.Response.StatusCode
                $retryAfter = $_.Exception.Response.Headers['Retry-After']
            } catch { }
            if ($rateLimited -or $statusCode -eq 429) { $waitSeconds = [Math]::Max(60, $waitSeconds) }
            $retrySeconds = 0
            if ([int]::TryParse([string]$retryAfter, [ref]$retrySeconds)) {
                $waitSeconds = [Math]::Max($waitSeconds, [Math]::Min(600, $retrySeconds))
            }
            Write-Warning "Request throttled or failed for $($titles[$index]); waiting $waitSeconds seconds before retry."
            Start-Sleep -Seconds $waitSeconds
        }
    }
    $page = $response.query.pages.PSObject.Properties.Value | Select-Object -First 1
    if ([string]::IsNullOrWhiteSpace($page.extract)) { throw "No extract returned for $($titles[$index])" }
    $document = "$($page.title)`n$($page.extract)`n"
    if ($index -lt 6) { $trainText.Add($document) } else { $validationText.Add($document) }
    $records.Add([pscustomobject]@{
        title = $page.title
        pageid = $page.pageid
        revision = $page.revisions[0].revid
        url = $page.fullurl
        retrieval_utc = (Get-Date).ToUniversalTime().ToString('o')
        license = 'CC BY-SA 4.0'
        license_url = 'https://creativecommons.org/licenses/by-sa/4.0/'
        split = $(if ($index -lt 6) { 'train' } else { 'validation' })
    })
    Write-Host "$($index + 1)/$($titles.Count) $($page.title): $($page.extract.Length) chars"
    if ($index -lt ($titles.Count - 1)) { Start-Sleep -Seconds $RequestDelaySeconds }
}
foreach ($path in @($TrainPath, $ValidationPath, $ManifestPath)) {
    $parent = Split-Path -Parent $path
    if ($parent) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
}
$utf8 = [Text.UTF8Encoding]::new($false)
[IO.File]::WriteAllText($TrainPath, ($trainText -join "`n"), $utf8)
[IO.File]::WriteAllText($ValidationPath, ($validationText -join "`n"), $utf8)
$records | ConvertTo-Json -Depth 5 | Set-Content -Encoding utf8 $ManifestPath
