param(
    [Parameter(Mandatory=$true)][string]$Language,
    [string]$CacheDirectory='data/cache/wikipedia_expansion',
    [string]$CorpusPath='',
    [string]$OutputPath='build/wikipedia-validation-pages.txt'
)
$ErrorActionPreference='Stop'
if (-not $CorpusPath) { $CorpusPath="data/processed/multilingual_expansion_validation_144/$Language.txt" }
if ($Language -notmatch '^[a-z]{2,3}$') { throw 'Language must be a two- or three-letter lowercase code.' }
$cachePath=Join-Path $CacheDirectory "$Language.json"
if (-not (Test-Path -LiteralPath $cachePath)) { throw "Missing source cache: $cachePath" }
$text=[IO.File]::ReadAllText((Resolve-Path -LiteralPath $CorpusPath),[Text.UTF8Encoding]::new($false))
$records=@(Get-Content -LiteralPath $cachePath -Raw | ConvertFrom-Json | Where-Object {$_.metadata.split -eq 'validation'})
if (-not $records.Count) { throw "No validation records in $cachePath" }
$matches=[Collections.Generic.List[object]]::new()
$searchFrom=0
foreach($record in $records){
    $title=[string]$record.metadata.title
    $pattern='(?m)^'+[regex]::Escape($title)+'\r?$'
    $found=[regex]::Match($text.Substring($searchFrom),$pattern)
    if(-not $found.Success){throw "Could not locate page title in corpus order: $title"}
    $start=$searchFrom+$found.Index
    $matches.Add([pscustomobject]@{Start=$start;Title=$title})
    $searchFrom=$start+$found.Length
}
if($matches[0].Start -ne 0){throw 'Corpus has an unassigned prefix before the first source page.'}
$documents=[Collections.Generic.List[string]]::new()
for($i=0;$i -lt $matches.Count;$i++){
    $start=$matches[$i].Start
    $end=if($i+1 -lt $matches.Count){$matches[$i+1].Start}else{$text.Length}
    $document=$text.Substring($start,$end-$start).Trim([char[]]@("`r","`n"))
    if([string]::IsNullOrWhiteSpace($document)){throw "Empty page extract: $($matches[$i].Title)"}
    $documents.Add($document)
}
$parent=Split-Path -Parent $OutputPath
if($parent){New-Item -ItemType Directory -Force -Path $parent|Out-Null}
[IO.File]::WriteAllText($OutputPath,($documents -join [char]12),[Text.UTF8Encoding]::new($false))
Write-Output "language=$Language pages=$($documents.Count) output=$OutputPath bytes=$((Get-Item -LiteralPath $OutputPath).Length)"
