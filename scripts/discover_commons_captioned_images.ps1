param(
    [string]$OutputCsv = 'data/raw/commons_captioned_expanded_1080.csv',
    [int]$ImagesPerTopic = 8,
    [int]$RequestDelaySeconds = 30
)
$ErrorActionPreference = 'Stop'
if ($ImagesPerTopic -lt 2 -or $ImagesPerTopic -gt 20) { throw 'ImagesPerTopic must be in [2, 20].' }
if ($RequestDelaySeconds -lt 30) { throw 'RequestDelaySeconds must be at least 30 seconds for Wikimedia requests.' }
$headers = @{ 'User-Agent' = 'AMA-10 Commons caption corpus discovery/1.0 (local research pipeline; contact not provided)' }
$api = 'https://commons.wikimedia.org/w/api.php'
$topics = @('bird','fox','landscape','building','car','flower','apple','portrait','waterfall','skyline')
$cacheDir = 'build/commons-caption-discovery'
$null = New-Item -ItemType Directory -Force -Path $cacheDir
$searchCachePath = Join-Path $cacheDir 'search-cache.json'
$eligibilityCachePath = Join-Path $cacheDir 'eligible-cache.json'
$searchCache = @{}
if (Test-Path -LiteralPath $searchCachePath) {
    foreach ($entry in @(Get-Content -LiteralPath $searchCachePath -Raw | ConvertFrom-Json)) { $searchCache[[string]$entry.topic] = @($entry.results) }
}
$existing = @{}
$existingManifest = 'data/raw/commons_captioned_diverse_1080/sources.json'
if (Test-Path -LiteralPath $existingManifest) {
    foreach ($record in @(Get-Content -LiteralPath $existingManifest -Raw | ConvertFrom-Json)) {
        $existing[[string]$record.file_title.ToLowerInvariant()] = $true
    }
}
$candidates = [Collections.Generic.List[object]]::new()
foreach ($topic in $topics) {
    if (-not $searchCache.ContainsKey($topic)) {
        $encoded = [Uri]::EscapeDataString($topic)
        $uri = "${api}?action=query&format=json&list=search&srnamespace=6&srlimit=20&srsearch=$encoded"
        $result = Invoke-RestMethod -Uri $uri -Headers $headers -TimeoutSec 60
        $searchCache[$topic] = @($result.query.search | ForEach-Object { [pscustomobject]@{title=[string]$_.title;pageid=[int64]$_.pageid} })
        @($searchCache.GetEnumerator() | ForEach-Object { [pscustomobject]@{topic=$_.Key;results=$_.Value} }) |
            ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $searchCachePath -Encoding utf8
        Start-Sleep -Seconds $RequestDelaySeconds
    }
    foreach ($hit in $searchCache[$topic]) {
        $key = ([string]$hit.title).ToLowerInvariant()
        if (-not $existing.ContainsKey($key) -and -not @($candidates | Where-Object { $_.title.ToLowerInvariant() -eq $key }).Count) {
            $candidates.Add([pscustomobject]@{ topic=$topic; title=[string]$hit.title; pageid=[int64]$hit.pageid })
        }
    }
}
$eligible = [Collections.Generic.List[object]]::new()
if (Test-Path -LiteralPath $eligibilityCachePath) {
    foreach ($entry in @(Get-Content -LiteralPath $eligibilityCachePath -Raw | ConvertFrom-Json)) { $eligible.Add($entry) }
}
$allowedLicenses = @('CC0','CC0 1.0','CC BY 2.0','CC BY 2.5','CC BY 3.0','CC BY 4.0',
                     'CC BY-SA 2.0','CC BY-SA 2.5','CC BY-SA 3.0','CC BY-SA 4.0','Public domain')
if (-not $eligible.Count) { for ($offset=0; $offset -lt $candidates.Count; $offset+=50) {
    $batch = @($candidates | Select-Object -Skip $offset -First 50)
    $encodedTitles = @($batch | ForEach-Object { [Uri]::EscapeDataString($_.title) }) -join '%7C'
    $infoUri = "${api}?action=query&format=json&redirects=1&prop=info%7Cimageinfo&inprop=url&iiprop=url%7Cextmetadata&iiextmetadatalanguage=en&iiurlwidth=1080&titles=$encodedTitles"
    $info = Invoke-RestMethod -Uri $infoUri -Headers $headers -TimeoutSec 90
    Start-Sleep -Seconds $RequestDelaySeconds
    $pages = @{}
    foreach ($page in $info.query.pages.PSObject.Properties.Value) { $pages[$page.title.ToLowerInvariant()] = $page }
    $valid = @($batch | Where-Object {
        $page=$pages[$_.title.ToLowerInvariant()]
        $page -and $page.imageinfo -and $page.imageinfo[0].thumburl -and
        ([string]$page.imageinfo[0].extmetadata.LicenseShortName.value) -in $allowedLicenses
    })
    if (-not $valid.Count) { continue }
    $ids = @($valid | ForEach-Object { "M$($pages[$_.title.ToLowerInvariant()].pageid)" }) -join '%7C'
    $labelsUri = "https://commons.wikimedia.org/w/api.php?action=wbgetentities&format=json&props=labels&languages=en&ids=$ids"
    $labels = Invoke-RestMethod -Uri $labelsUri -Headers $headers -TimeoutSec 60
    foreach ($candidate in $valid) {
        $page = $pages[$candidate.title.ToLowerInvariant()]
        $id = "M$($page.pageid)"
        $caption = [string]$labels.entities.$id.labels.en.value
        if (-not [string]::IsNullOrWhiteSpace($caption) -and -not $caption.Contains("`t") -and
            -not $caption.Contains("`r") -and -not $caption.Contains("`n")) {
            $eligible.Add([pscustomobject]@{ topic=$candidate.topic; title=$page.title; caption=$caption })
        }
    }
    Start-Sleep -Seconds $RequestDelaySeconds
} }
if ($eligible.Count) { $eligible | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $eligibilityCachePath -Encoding utf8 }
$specs = [Collections.Generic.List[object]]::new()
foreach ($topic in $topics) {
    $group = @($eligible | Where-Object topic -eq $topic | Select-Object -First $ImagesPerTopic)
    if ($group.Count -lt $ImagesPerTopic) { throw "Only $($group.Count) licensed captioned files found for topic '$topic'; refusing an imbalanced incomplete set." }
    for ($i=0; $i -lt $group.Count; $i++) {
        $split = if ($i -ge $ImagesPerTopic-2) { 'validation' } else { 'train' }
        $specs.Add([pscustomobject]@{ title=$group[$i].title; split=$split })
    }
}
$full = [IO.Path]::GetFullPath($OutputCsv)
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $full) | Out-Null
$specs | Export-Csv -LiteralPath $full -NoTypeInformation -Encoding utf8
Write-Host "topics=$($topics.Count) images=$($specs.Count) train=$( @($specs|Where-Object split -eq 'train').Count ) validation=$( @($specs|Where-Object split -eq 'validation').Count ) output=$full"
