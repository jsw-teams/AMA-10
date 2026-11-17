param(
    [string]$TrainPath = 'data/raw/multilingual_train.txt',
    [string]$ValidationPath = 'data/raw/multilingual_validation.txt',
    [string]$ManifestPath = 'data/raw/multilingual_sources.json',
    [int]$RequestDelaySeconds = 30,
    [string]$Contact = ''
)
$ErrorActionPreference = 'Stop'
if ($RequestDelaySeconds -lt 30) { throw 'RequestDelaySeconds must be at least 30 seconds for Wikimedia requests.' }

$pages = @(
    @{ lang = 'en'; title = 'Natural language processing'; split = 'train' },
    @{ lang = 'en'; title = 'Language model'; split = 'validation' },
    @{ lang = 'zh'; title = '自然语言处理'; split = 'train' },
    @{ lang = 'zh'; title = '机器学习'; split = 'validation' },
    @{ lang = 'ja'; title = '自然言語処理'; split = 'train' },
    @{ lang = 'ja'; title = '機械学習'; split = 'validation' },
    @{ lang = 'es'; title = 'Procesamiento de lenguajes naturales'; split = 'train' },
    @{ lang = 'es'; title = 'Aprendizaje automático'; split = 'validation' },
    @{ lang = 'fr'; title = 'Traitement automatique des langues'; split = 'train' },
    @{ lang = 'fr'; title = 'Apprentissage automatique'; split = 'validation' }
)

$records = [System.Collections.Generic.List[object]]::new()
$trainText = [System.Collections.Generic.List[string]]::new()
$validationText = [System.Collections.Generic.List[string]]::new()
$contactPart = if ($Contact) { "; contact: $Contact" } else { '' }
$userAgent = "AMA-10 multilingual corpus fetcher/1.0 (local research pipeline)$contactPart"

foreach ($group in ($pages | Group-Object lang)) {
    $language = $group.Name
    $titles = @($group.Group | ForEach-Object { $_.title })
    $joinedTitles = [Uri]::EscapeDataString([string]::Join('|', $titles))
    $uri = "https://$language.wikipedia.org/w/api.php?action=query&format=json&prop=extracts%7Cinfo%7Crevisions&explaintext=1&exintro=1&exlimit=max&inprop=url&rvprop=ids&maxlag=5&titles=$joinedTitles"
    $response = $null
    for ($attempt = 0; $attempt -lt 4 -and -not $response; $attempt++) {
        $rateLimited = $false
        try {
            $response = Invoke-RestMethod -Uri $uri -UserAgent $userAgent -TimeoutSec 60
            if ($response.error) {
                $rateLimited = $response.error.code -match 'ratelimit|too-many-requests'
                $message = $response.error.info
                $response = $null
                throw $message
            }
        } catch {
            if ($attempt -eq 3) { throw "Wikipedia fetch failed for $language`: $_" }
            $response = $null
            $waitSeconds = [int][Math]::Min(600, 30 * [Math]::Pow(2, $attempt))
            $statusCode = 0
            $retryAfter = $null
            try {
                $statusCode = [int]$_.Exception.Response.StatusCode
                $retryAfter = $_.Exception.Response.Headers['Retry-After']
            } catch { }
            if ($rateLimited -or $statusCode -eq 429) {
                $waitSeconds = [Math]::Max(60, $waitSeconds)
            }
            $retrySeconds = 0
            if ([int]::TryParse([string]$retryAfter,[ref]$retrySeconds)) {
                $waitSeconds = [Math]::Max($waitSeconds,[Math]::Min(600,$retrySeconds))
            }
            Write-Warning "Request throttled or failed for $language; waiting $waitSeconds seconds before retry."
            Start-Sleep -Seconds $waitSeconds
        }
    }
    $byTitle = @{}
    foreach ($page in $response.query.pages.PSObject.Properties.Value) {
        if (-not $page.missing -and -not [string]::IsNullOrWhiteSpace($page.extract)) {
            $byTitle[$page.title.ToLowerInvariant()] = $page
        }
    }
    foreach ($spec in $group.Group) {
        $key = $spec.title.ToLowerInvariant()
        $page = $byTitle[$key]
        if (-not $page) {
            $normalized = $spec.title.Replace('_', ' ').ToLowerInvariant()
            $page = $byTitle[$normalized]
        }
        if (-not $page) { throw "No extract found for $language`: $($spec.title)" }
        $document = "$($page.title)`n$($page.extract)`n"
        if ($spec.split -eq 'train') { $trainText.Add($document) }
        else { $validationText.Add($document) }
        $records.Add([pscustomobject]@{
            language = $language
            title = $page.title
            pageid = $page.pageid
            revision = $page.revisions[0].revid
            url = $page.fullurl
            api_url = $uri
            extract_scope = 'lead introduction'
            retrieval_utc = (Get-Date).ToUniversalTime().ToString('o')
            license = 'CC BY-SA 4.0'
            license_url = 'https://creativecommons.org/licenses/by-sa/4.0/'
            split = $spec.split
        })
        Write-Host "$language`: $($page.title) chars=$($page.extract.Length) split=$($spec.split)"
    }
    Start-Sleep -Seconds $RequestDelaySeconds
}

foreach ($path in @($TrainPath, $ValidationPath, $ManifestPath)) {
    $parent = Split-Path -Parent $path
    if ($parent) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
}
$utf8 = [Text.UTF8Encoding]::new($false)
[IO.File]::WriteAllText($TrainPath, ($trainText -join "`n"), $utf8)
[IO.File]::WriteAllText($ValidationPath, ($validationText -join "`n"), $utf8)
$records | ConvertTo-Json -Depth 5 | Set-Content -Encoding utf8 $ManifestPath
