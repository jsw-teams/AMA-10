param(
    [string]$CacheDirectory='data/cache/wikipedia_expansion',
    [string]$ReferenceTrainPath='data/raw/multilingual_expansion_20lang_train.txt',
    [string]$ValidationPath='data/raw/multilingual_expansion_20lang_validation.txt',
    [string]$OutputTrainPath='data/processed/multilingual_20lang_interleaved_train.txt',
    [string]$OutputManifestPath='data/processed/multilingual_20lang_interleaved_sources.json'
)
$ErrorActionPreference='Stop'
$cacheFiles=@(Get-ChildItem -LiteralPath $CacheDirectory -Filter '*.json' -File | Sort-Object BaseName)
if(-not $cacheFiles.Count){throw "No language caches found under $CacheDirectory"}
$queues=[ordered]@{}
$manifest=[Collections.Generic.List[object]]::new()
$seen=[Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
$trainByteCount=0L
foreach($file in $cacheFiles){
    $language=$file.BaseName
    $records=@(Get-Content -LiteralPath $file.FullName -Raw | ConvertFrom-Json)
    $trainRecords=[Collections.Generic.List[object]]::new()
    foreach($record in $records){
        $meta=$record.metadata
        if($meta.language -ne $language -or $meta.license -ne 'CC BY-SA 4.0' -or
           $meta.split -notin @('train','validation') -or -not $meta.pageid -or
           [string]::IsNullOrWhiteSpace([string]$meta.title) -or
           [string]::IsNullOrWhiteSpace([string]$record.extract)){
            throw "Invalid cached source metadata in $($file.Name)"
        }
        $key="$language`:$($meta.pageid)"
        if(-not $seen.Add($key)){throw "Duplicate cached page ID $key"}
        if($meta.split -eq 'train'){$trainRecords.Add($record)}
    }
    if(-not $trainRecords.Count){throw "No training pages in $($file.Name)"}
    $queues[$language]=$trainRecords
    foreach($record in $trainRecords){
        $doc="$($record.metadata.title)`n$($record.extract)`n"
        $trainByteCount += [Text.Encoding]::UTF8.GetByteCount($doc)
    }
}
$maxRounds=0
foreach($language in $queues.Keys){if($queues[$language].Count -gt $maxRounds){$maxRounds=$queues[$language].Count}}
$documents=[Collections.Generic.List[string]]::new()
$rank=0
for($round=0;$round -lt $maxRounds;$round++){
    foreach($language in $queues.Keys){
        if($round -ge $queues[$language].Count){continue}
        $record=$queues[$language][$round]
        $documents.Add("$($record.metadata.title)`n$($record.extract)`n")
        $rank++
        $manifest.Add([pscustomobject]@{
            order=$rank; round=$round+1; language=$language
            title=[string]$record.metadata.title; pageid=[int64]$record.metadata.pageid
            revision=[int64]$record.metadata.revision; url=[string]$record.metadata.url
            license=[string]$record.metadata.license; split='train'
        })
    }
}
$trainText=$documents -join "`n"
$actualBytes=[Text.Encoding]::UTF8.GetByteCount($trainText)
$referenceBytes=(Get-Item -LiteralPath $ReferenceTrainPath).Length
if($actualBytes -ne $referenceBytes){throw "Interleaved stream has $actualBytes bytes; reference cache set has $referenceBytes. Refusing to publish a partial corpus."}
$parent=Split-Path -Parent $OutputTrainPath
if($parent){New-Item -ItemType Directory -Force -Path $parent|Out-Null}
$manifestParent=Split-Path -Parent $OutputManifestPath
if($manifestParent){New-Item -ItemType Directory -Force -Path $manifestParent|Out-Null}
[IO.File]::WriteAllText($OutputTrainPath,$trainText,[Text.UTF8Encoding]::new($false))
$manifest|ConvertTo-Json -Depth 5|Set-Content -LiteralPath $OutputManifestPath -Encoding utf8
$trainName=[IO.Path]::GetFileNameWithoutExtension($OutputTrainPath)
if(-not $trainName.EndsWith('_train',[StringComparison]::Ordinal)){throw 'OutputTrainPath must end in _train.txt so its validation path is unambiguous.'}
$validationOutput=Join-Path (Split-Path -Parent $OutputTrainPath) (($trainName.Substring(0,$trainName.Length-6))+'_validation.txt')
Copy-Item -LiteralPath $ValidationPath -Destination $validationOutput -Force
$distribution=$queues.Keys|ForEach-Object {[pscustomobject]@{language=$_;train_pages=$queues[$_].Count}}|ConvertTo-Json -Compress
Write-Output "train_bytes=$actualBytes reference_bytes=$referenceBytes pages=$rank rounds=$maxRounds train_sha256=$((Get-FileHash -LiteralPath $OutputTrainPath -Algorithm SHA256).Hash) validation_sha256=$((Get-FileHash -LiteralPath $validationOutput -Algorithm SHA256).Hash) manifest=$OutputManifestPath"
Write-Output "train_pages_by_language=$distribution"

