param(
    [string]$CacheDirectory='data/cache/wikipedia_expansion',
    [string]$ReferenceTrainPath='data/raw/multilingual_expansion_20lang_train.txt',
    [string]$ValidationPath='data/raw/multilingual_expansion_20lang_validation.txt',
    [string]$OutputTrainPath='data/processed/multilingual_20lang_epoch_shuffle_train.txt',
    [string]$OutputManifestPath='data/processed/multilingual_20lang_epoch_shuffle_sources.json',
    [ValidateRange(1,100)][int]$Epochs=5,
    [int]$Seed=10
)
$ErrorActionPreference='Stop'
$files=@(Get-ChildItem -LiteralPath $CacheDirectory -Filter '*.json' -File|Sort-Object BaseName)
if(-not $files.Count){throw "No language source caches found in $CacheDirectory"}
$pages=[Collections.Generic.List[object]]::new()
$seen=[Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
foreach($file in $files){
    $language=$file.BaseName
    foreach($record in @(Get-Content -LiteralPath $file.FullName -Raw|ConvertFrom-Json)){
        $meta=$record.metadata
        if($meta.language -ne $language -or $meta.license -ne 'CC BY-SA 4.0' -or
           $meta.split -notin @('train','validation') -or -not $meta.pageid -or
           [string]::IsNullOrWhiteSpace([string]$meta.title) -or
           [string]::IsNullOrWhiteSpace([string]$record.extract)){
            throw "Invalid source record in $($file.Name)"
        }
        if($meta.split -ne 'train'){continue}
        $key="$language`:$($meta.pageid)"
        if(-not $seen.Add($key)){throw "Duplicate training page ID $key"}
        $pages.Add([pscustomobject]@{
            language=$language; title=[string]$meta.title; pageid=[int64]$meta.pageid
            revision=[int64]$meta.revision; url=[string]$meta.url; license=[string]$meta.license
            extract=[string]$record.extract
        })
    }
}
if(-not $pages.Count){throw 'No training pages were found.'}
$referenceBytes=(Get-Item -LiteralPath $ReferenceTrainPath).Length
$epochBytes=0L
foreach($page in $pages){$epochBytes += [Text.Encoding]::UTF8.GetByteCount("$($page.title)`n$($page.extract)`n")}
$epochBytes += [Math]::Max(0,$pages.Count-1)
if($epochBytes -ne $referenceBytes){throw "Page records reconstruct $epochBytes bytes, expected $referenceBytes."}
$blocks=[Collections.Generic.List[string]]::new()
$manifest=[Collections.Generic.List[object]]::new()
for($epoch=0;$epoch -lt $Epochs;$epoch++){
    $ordered=[Collections.Generic.List[object]]::new()
    foreach($page in $pages){$ordered.Add($page)}
    $rng=[Random]::new([int]($Seed+$epoch))
    for($i=$ordered.Count-1;$i -gt 0;$i--){$j=$rng.Next($i+1);$tmp=$ordered[$i];$ordered[$i]=$ordered[$j];$ordered[$j]=$tmp}
    $docs=[Collections.Generic.List[string]]::new()
    for($i=0;$i -lt $ordered.Count;$i++){
        $page=$ordered[$i]
        $docs.Add("$($page.title)`n$($page.extract)`n")
        $manifest.Add([pscustomobject]@{
            epoch=$epoch+1; order=$i+1; language=$page.language; title=$page.title
            pageid=$page.pageid; revision=$page.revision; url=$page.url; license=$page.license
        })
    }
    $blocks.Add($docs -join "`n")
}
$trainText=$blocks -join "`n"
$expectedBytes=[long]$epochBytes*$Epochs+($Epochs-1)
$actualBytes=[Text.Encoding]::UTF8.GetByteCount($trainText)
if($actualBytes -ne $expectedBytes){throw "Shuffled stream has $actualBytes bytes; expected $expectedBytes."}
$trainParent=Split-Path -Parent $OutputTrainPath
$manifestParent=Split-Path -Parent $OutputManifestPath
if($trainParent){New-Item -ItemType Directory -Force -Path $trainParent|Out-Null}
if($manifestParent){New-Item -ItemType Directory -Force -Path $manifestParent|Out-Null}
[IO.File]::WriteAllText($OutputTrainPath,$trainText,[Text.UTF8Encoding]::new($false))
$manifest|ConvertTo-Json -Depth 5|Set-Content -LiteralPath $OutputManifestPath -Encoding utf8
$trainName=[IO.Path]::GetFileNameWithoutExtension($OutputTrainPath)
if(-not $trainName.EndsWith('_train',[StringComparison]::Ordinal)){throw 'OutputTrainPath must end in _train.txt so the validation file can be named consistently.'}
$validationName=$trainName.Substring(0,$trainName.Length-6)+'_validation.txt'
$validationOutput=Join-Path (Split-Path -Parent $OutputTrainPath) $validationName
Copy-Item -LiteralPath $ValidationPath -Destination $validationOutput -Force
Write-Output "epochs=$Epochs seed=$Seed pages_per_epoch=$($pages.Count) train_bytes_per_epoch=$epochBytes total_train_bytes=$actualBytes train_sha256=$((Get-FileHash -LiteralPath $OutputTrainPath -Algorithm SHA256).Hash) validation_sha256=$((Get-FileHash -LiteralPath $validationOutput -Algorithm SHA256).Hash) manifest=$OutputManifestPath"
