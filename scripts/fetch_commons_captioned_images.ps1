param(
    [string]$InputCsv = 'data/raw/commons_captioned_seed.csv',
    [string]$OutputDir = 'data/raw/commons_captioned_seed',
    [int]$RequestDelaySeconds = 30,
    [string]$Contact = ''
)
$ErrorActionPreference = 'Stop'
if ($RequestDelaySeconds -lt 30) { throw 'RequestDelaySeconds must be at least 30 seconds for Wikimedia requests.' }
$ffmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
if (-not $ffmpeg) { throw 'ffmpeg is required to convert downloaded thumbnails to P6 PPM.' }
$specs = @(Import-Csv -LiteralPath $InputCsv)
if (-not $specs.Count -or @($specs | Where-Object split -eq 'train').Count -eq 0 -or
    @($specs | Where-Object split -eq 'validation').Count -eq 0) {
    throw 'Input CSV needs at least one train and one validation row.'
}
foreach ($spec in $specs) {
    if ($spec.title -notmatch '^File:' -or $spec.split -notin @('train','validation')) {
        throw "Invalid source row: $($spec | ConvertTo-Json -Compress)"
    }
}
$contactPart = if ($Contact) { "; contact: $Contact" } else { '; contact not provided' }
$headers = @{ 'User-Agent' = "AMA-10 Commons caption corpus fetcher/1.0 (local research pipeline$contactPart)" }
$api = 'https://commons.wikimedia.org/w/api.php'

function Invoke-WikimediaRequest([string]$Uri, [switch]$Download, [string]$OutFile) {
    for ($attempt = 0; $attempt -lt 5; $attempt++) {
        try {
            if ($Download) {
                return Invoke-WebRequest -Uri $Uri -Headers $headers -OutFile $OutFile -PassThru -MaximumRedirection 10 -TimeoutSec 180
            }
            return Invoke-RestMethod -Uri $Uri -Headers $headers -TimeoutSec 60
        } catch {
            if ($attempt -eq 4) { throw }
            $status = 0; $retryAfter = 0
            try {
                $status = [int]$_.Exception.Response.StatusCode
                [void][int]::TryParse([string]$_.Exception.Response.Headers['Retry-After'],[ref]$retryAfter)
            } catch { }
            $wait = [Math]::Max($RequestDelaySeconds,30 * [Math]::Pow(2,$attempt))
            if ($status -eq 429) { $wait = [Math]::Max(120,$wait) }
            $wait = [int][Math]::Min(900,[Math]::Max($wait,$retryAfter))
            Write-Warning "Commons request failed for '$Uri' (HTTP ${status}: $($_.Exception.Message)); waiting $wait seconds before retry."
            Start-Sleep -Seconds $wait
        }
    }
}

$encodedTitles = @($specs | ForEach-Object { [Uri]::EscapeDataString($_.title) }) -join '%7C'
$queryUri = "${api}?action=query&format=json&redirects=1&maxlag=5&prop=info%7Cimageinfo&inprop=url&iiprop=url%7Cextmetadata&iiextmetadatalanguage=en&iiurlwidth=64&titles=$encodedTitles"
$query = Invoke-WikimediaRequest $queryUri
Start-Sleep -Seconds $RequestDelaySeconds
$pages = @{}
foreach ($page in $query.query.pages.PSObject.Properties.Value) {
    $pages[$page.title.ToLowerInvariant()] = $page
}
$entityIds = @($pages.Values | ForEach-Object { "M$($_.pageid)" }) -join '%7C'
if (-not $entityIds) { throw 'Commons returned no pages for the requested titles.' }
$captionUri = "${api}?action=wbgetentities&format=json&props=labels&languages=en&ids=$entityIds"
$captions = Invoke-WikimediaRequest $captionUri
$allowedLicenses = @('CC0','CC0 1.0','CC BY 2.0','CC BY 2.5','CC BY 3.0','CC BY 4.0',
                     'CC BY-SA 2.0','CC BY-SA 2.5','CC BY-SA 3.0','CC BY-SA 4.0','Public domain')
$null = New-Item -ItemType Directory -Force -Path $OutputDir
$manifest = [System.Collections.Generic.List[object]]::new()
$tsvRows = [System.Collections.Generic.List[string]]::new()
$utf8 = [Text.UTF8Encoding]::new($false)
$index = 0
foreach ($spec in $specs) {
    $page = $pages[$spec.title.ToLowerInvariant()]
    if (-not $page -or -not $page.imageinfo -or $page.missing) { throw "Commons file not found: $($spec.title)" }
    $info = $page.imageinfo[0]
    $license = [string]$info.extmetadata.LicenseShortName.value
    if ($license -notin $allowedLicenses) { throw "Refusing unsupported or unclear license '$license' on $($page.title)." }
    $entityId = "M$($page.pageid)"
    $caption = [string]$captions.entities.$entityId.labels.en.value
    if ([string]::IsNullOrWhiteSpace($caption) -or $caption.Contains("`t") -or $caption.Contains("`n")) {
        throw "No usable English structured caption for $($page.title)."
    }
    if (-not $info.thumburl) { throw "Commons did not return a thumbnail URL for $($page.title)." }
    $index++
    $stem = '{0:D4}' -f $index
    $downloadPath = Join-Path $OutputDir "$stem.download"
    $ppmPath = Join-Path $OutputDir "$stem.ppm"
    $null = Invoke-WikimediaRequest $info.thumburl -Download -OutFile $downloadPath
    if (-not (Test-Path -LiteralPath $downloadPath) -or (Get-Item -LiteralPath $downloadPath).Length -eq 0) {
        throw "Empty image download for $($page.title)."
    }
    & $ffmpeg.Source -y -loglevel error -i $downloadPath -vf 'scale=64:64:force_original_aspect_ratio=decrease,pad=64:64:(ow-iw)/2:(oh-ih)/2' -frames:v 1 -f image2 -vcodec ppm $ppmPath
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $ppmPath)) { throw "PPM conversion failed for $($page.title)." }
    Remove-Item -LiteralPath $downloadPath -Force
    $hash = (Get-FileHash -LiteralPath $ppmPath -Algorithm SHA256).Hash
    $tsvRows.Add("$stem.ppm`t$caption`t$($spec.split)")
    $manifest.Add([pscustomobject]@{
        local_image = $ppmPath
        image_sha256 = $hash
        split = $spec.split
        caption = $caption
        caption_language = 'en'
        caption_license = 'CC0 (Commons structured file captions)'
        media_info_id = $entityId
        file_title = $page.title
        file_page_url = $page.fullurl
        thumbnail_url = $info.thumburl
        original_url = $info.url
        image_license = $license
        image_license_url = $info.extmetadata.LicenseUrl.value
        artist_html = $info.extmetadata.Artist.value
        retrieval_utc = (Get-Date).ToUniversalTime().ToString('o')
        preprocessing = '64x64 RGB thumbnail, aspect-preserving resize with black padding, P6 PPM'
    })
    Write-Host "$($spec.split): $($page.title) [$license] $($ppmPath)"
    if ($index -lt $specs.Count) { Start-Sleep -Seconds $RequestDelaySeconds }
}
$tsvPath = Join-Path $OutputDir 'manifest.tsv'
$jsonPath = Join-Path $OutputDir 'sources.json'
[IO.File]::WriteAllLines($tsvPath,$tsvRows,$utf8)
$manifest | ConvertTo-Json -Depth 6 | Set-Content -Encoding utf8 $jsonPath
Write-Host "Saved $($specs.Count) pairs to $tsvPath and provenance to $jsonPath"
