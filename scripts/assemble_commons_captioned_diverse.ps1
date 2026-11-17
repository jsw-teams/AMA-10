param(
    [string]$InputCsv = 'data/raw/commons_captioned_diverse.csv',
    [string]$OutputDir = 'data/raw/commons_captioned_diverse',
    [string]$ExistingSeedDir = 'data/raw/commons_captioned_seed',
    [int]$RequestDelaySeconds = 30,
    [string]$Contact = ''
)
$ErrorActionPreference = 'Stop'
$fetcher = Join-Path $PSScriptRoot 'fetch_commons_captioned_images.ps1'
& $fetcher -InputCsv $InputCsv -OutputDir $OutputDir -RequestDelaySeconds $RequestDelaySeconds -Contact $Contact

$sourcePath = Join-Path $OutputDir 'sources.json'
$newRecords = @(Get-Content -Raw -LiteralPath $sourcePath | ConvertFrom-Json)
$records = [System.Collections.Generic.List[object]]::new()
$rows = [System.Collections.Generic.List[string]]::new()
$tab = [string][char]9
foreach ($record in $newRecords) {
    $name = [IO.Path]::GetFileName($record.local_image)
    $record.local_image = $name
    $records.Add($record)
    $rows.Add("$name$tab$($record.caption)$tab$($record.split)$tab$($record.caption_language)")
}

$existingPath = Join-Path $ExistingSeedDir 'sources.json'
$oldRecords = @(Get-Content -Raw -LiteralPath $existingPath | ConvertFrom-Json)
$index = 0
foreach ($record in $oldRecords) {
    $index++
    $oldName = [IO.Path]::GetFileName($record.local_image)
    $newName = 'kingfisher_{0:D2}.ppm' -f $index
    Copy-Item -LiteralPath (Join-Path $ExistingSeedDir $oldName) -Destination (Join-Path $OutputDir $newName) -Force
    $record.local_image = $newName
    $records.Add($record)
    $rows.Add("$newName$tab$($record.caption)$tab$($record.split)$tab$($record.caption_language)")
}
$utf8 = [Text.UTF8Encoding]::new($false)
[IO.File]::WriteAllLines((Join-Path $OutputDir 'manifest.tsv'),$rows,$utf8)
$records | ConvertTo-Json -Depth 8 | Set-Content -Encoding utf8 (Join-Path $OutputDir 'sources.json')
Write-Host "Assembled $($rows.Count) licensed image-caption pairs in $OutputDir"
