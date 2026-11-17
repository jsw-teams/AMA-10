param(
    [string]$InputPath = "data/raw/multilingual_validation.txt",
    [string]$ManifestPath = "data/raw/multilingual_sources.json",
    [string]$OutputDirectory = "data/processed/multilingual_validation",
    [ValidateSet("train","validation")][string]$Split = "validation"
)

$ErrorActionPreference = "Stop"
$utf8 = [Text.UTF8Encoding]::new($false)
$resolvedInput = (Resolve-Path -LiteralPath $InputPath).Path
$resolvedManifest = (Resolve-Path -LiteralPath $ManifestPath).Path
$text = [IO.File]::ReadAllText($resolvedInput, $utf8)
$records = [IO.File]::ReadAllText($resolvedManifest, $utf8) | ConvertFrom-Json
$selected = @($records | Where-Object { $_.split -eq $Split })
if ($selected.Count -eq 0) { throw "Manifest has no $Split pages." }

$matches = @()
$lastStart = -1
foreach ($record in $selected) {
    $pattern = "(?m)^" + [regex]::Escape([string]$record.title) + "\r?$"
    $found = [regex]::Matches($text, $pattern)
    $next = $null
    foreach ($candidate in $found) {
        if ($candidate.Index -gt $lastStart) { $next = $candidate; break }
    }
    if (-not $next) {
        throw "No source title line remains in manifest order for $($record.language): $($record.title); found $($found.Count)."
    }
    $lastStart = $next.Index
    $matches += [pscustomobject]@{ Record = $record; Start = $next.Index }
}
$matches = @($matches | Sort-Object Start)
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$documentsByLanguage = @{}
for ($i = 0; $i -lt $matches.Count; $i++) {
    $start = $matches[$i].Start
    $end = if ($i + 1 -lt $matches.Count) { $matches[$i + 1].Start } else { $text.Length }
    $document = $text.Substring($start, $end - $start).Trim()
    if ($document.Length -eq 0) { throw "Empty validation extract for $($matches[$i].Record.language)." }
    $language = [string]$matches[$i].Record.language
    if (-not $documentsByLanguage.ContainsKey($language)) {
        $documentsByLanguage[$language] = [System.Collections.Generic.List[string]]::new()
    }
    $documentsByLanguage[$language].Add($document)
}
foreach ($language in ($documentsByLanguage.Keys | Sort-Object)) {
    $path = Join-Path $OutputDirectory ("{0}.txt" -f $language)
    $combined = $documentsByLanguage[$language] -join "`n"
    [IO.File]::WriteAllText((Join-Path (Get-Location) $path),$combined,$utf8)
    Write-Output ("language={0} pages={1} chars={2} file={3}" -f `
        $language, $documentsByLanguage[$language].Count, $combined.Length, $path)
}
