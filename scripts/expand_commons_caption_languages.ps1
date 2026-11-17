param(
    [string]$DatasetDir = 'data/raw/commons_captioned_diverse',
    [string[]]$Languages = @('en','fr','es','ja','zh'),
    [int]$RetryBaseSeconds = 30,
    [int]$RequestDelaySeconds = 30,
    [string]$Contact = ''
)
$ErrorActionPreference = 'Stop'
if ($RetryBaseSeconds -lt 30) { throw 'RetryBaseSeconds must be at least 30 seconds.' }
if ($RequestDelaySeconds -lt 30) { throw 'RequestDelaySeconds must be at least 30 seconds for Wikimedia requests.' }
$sourcesPath = Join-Path $DatasetDir 'sources.json'
$manifestPath = Join-Path $DatasetDir 'manifest.tsv'
$sourceRows = @(Get-Content -Raw -LiteralPath $sourcesPath | ConvertFrom-Json)
$unique = @($sourceRows | Group-Object media_info_id | ForEach-Object { $_.Group[0] })
if (-not $unique.Count) { throw 'No Commons MediaInfo source records found.' }
$ids = @($unique | ForEach-Object { $_.media_info_id }) -join '%7C'
$languageList = ($Languages | Select-Object -Unique) -join '|'
$languageQuery = [Uri]::EscapeDataString($languageList)
$uri = "https://commons.wikimedia.org/w/api.php?action=wbgetentities&format=json&maxlag=5&props=labels&languages=$languageQuery&ids=$ids"
$contactPart = if ($Contact) { "; contact: $Contact" } else { '; contact not provided' }
$userAgent = "AMA-10 Commons multilingual caption metadata fetcher/1.0 (local research pipeline$contactPart)"
$response = $null
for ($attempt=0; $attempt -lt 5 -and -not $response; $attempt++) {
    try {
        $response = Invoke-RestMethod -Uri $uri -UserAgent $userAgent -TimeoutSec 60
        if ($response.error) { $message=$response.error.info; $response=$null; throw $message }
    } catch {
        if ($attempt -eq 4) { throw }
        $status=0;$retryAfter=0
        try {
            $status=[int]$_.Exception.Response.StatusCode
            [void][int]::TryParse([string]$_.Exception.Response.Headers['Retry-After'],[ref]$retryAfter)
        } catch { }
        $wait=[Math]::Max($RetryBaseSeconds,30*[Math]::Pow(2,$attempt))
        if ($status -eq 429) { $wait=[Math]::Max(120,$wait) }
        $wait=[int][Math]::Min(900,[Math]::Max($wait,$retryAfter))
        Write-Warning "Commons caption request failed (HTTP $($status): $($_.Exception.Message)); waiting $wait seconds before retry."
        Start-Sleep -Seconds $wait
    }
}
Start-Sleep -Seconds $RequestDelaySeconds
$records=[System.Collections.Generic.List[object]]::new()
$rows=[System.Collections.Generic.List[string]]::new()
$tab=[string][char]9
$langCounts=@{}
foreach($source in $unique){
    $imagePath=Join-Path $DatasetDir $source.local_image
    if(-not(Test-Path -LiteralPath $imagePath)){throw "Missing image file: $imagePath"}
    $hash=(Get-FileHash -LiteralPath $imagePath -Algorithm SHA256).Hash
    if($hash -ne $source.image_sha256){throw "Image hash mismatch: $imagePath"}
    $entity=$response.entities.($source.media_info_id)
    if(-not $entity){throw "No caption entity returned for $($source.media_info_id)"}
    foreach($language in ($Languages|Select-Object -Unique)){
        $caption=[string]$entity.labels.$language.value
        if([string]::IsNullOrWhiteSpace($caption)){continue}
        if($caption.Contains($tab) -or $caption.Contains([string][char]10) -or $caption.Contains([string][char]13)){
            throw "Caption contains a TSV delimiter for $($source.media_info_id) [$language]."
        }
        $record=ConvertFrom-Json (ConvertTo-Json $source -Depth 8)
        $record.caption=$caption
        $record.caption_language=$language
        $record.caption_license='CC0 (Commons structured file captions)'
        $records.Add($record)
        $rows.Add("$($source.local_image)$tab$caption$tab$($source.split)$tab$language")
        if(-not $langCounts.ContainsKey($language)){$langCounts[$language]=0}
        $langCounts[$language]++
    }
}
if(-not $records.Count){throw 'No captions found in the requested languages.'}
$utf8=[Text.UTF8Encoding]::new($false)
[IO.File]::WriteAllLines($manifestPath,$rows,$utf8)
$records|ConvertTo-Json -Depth 8|Set-Content -Encoding utf8 $sourcesPath
$summary=($langCounts.GetEnumerator()|Sort-Object Name|ForEach-Object{"$($_.Name)=$($_.Value)"})-join ', '
Write-Host "Wrote $($records.Count) aligned caption rows for $($unique.Count) images ($summary); image train/validation splits are preserved."
