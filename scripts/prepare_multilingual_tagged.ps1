param(
    [string]$CacheDirectory='data/cache/wikipedia_expansion',
    [string]$ReferenceTrainPath='data/raw/multilingual_expansion_20lang_train.txt',
    [string]$ReferenceValidationPath='data/raw/multilingual_expansion_20lang_validation.txt',
    [string]$OutputDirectory='data/processed/multilingual_20lang_tagged'
)
$ErrorActionPreference='Stop'
$files=@(Get-ChildItem -LiteralPath $CacheDirectory -Filter '*.json' -File|Sort-Object BaseName)
if(-not $files.Count){throw "No language source caches found in $CacheDirectory"}
$trainPages=[Collections.Generic.List[object]]::new()
$validationPages=[Collections.Generic.List[object]]::new()
$seen=[Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
$languages=[Collections.Generic.List[string]]::new()
foreach($file in $files){
    $language=$file.BaseName
    if($language -notmatch '^[a-z]{2,3}$'){throw "Invalid language code in $($file.Name)"}
    $languages.Add($language)
    foreach($record in @(Get-Content -LiteralPath $file.FullName -Raw|ConvertFrom-Json)){
        $meta=$record.metadata
        if($meta.language -ne $language -or $meta.license -ne 'CC BY-SA 4.0' -or
           $meta.split -notin @('train','validation') -or -not $meta.pageid -or
           -not $meta.revision -or -not $meta.url -or
           [string]::IsNullOrWhiteSpace([string]$meta.title) -or
           [string]::IsNullOrWhiteSpace([string]$record.extract)){
            throw "Invalid or unlicensed source record in $($file.Name)"
        }
        $key="$language`:$($meta.pageid)"
        if(-not $seen.Add($key)){throw "Duplicate source page ID $key"}
        $page=[pscustomobject]@{
            language=$language; split=[string]$meta.split; title=[string]$meta.title
            extract=[string]$record.extract; pageid=[int64]$meta.pageid
            revision=[int64]$meta.revision; url=[string]$meta.url; license=[string]$meta.license
        }
        if($meta.split -eq 'train'){$trainPages.Add($page)}else{$validationPages.Add($page)}
    }
}
function Join-PageDocuments([object[]]$Pages,[bool]$Tagged){
    $documents=[Collections.Generic.List[string]]::new()
    foreach($page in $Pages){
        $prefix=if($Tagged){"<lang:$($page.language)>`n"}else{''}
        $documents.Add("$prefix$($page.title)`n$($page.extract)`n")
    }
    return ($documents -join "`n")
}
$baseTrain=Join-PageDocuments $trainPages.ToArray() $false
$baseValidation=Join-PageDocuments $validationPages.ToArray() $false
if([Text.Encoding]::UTF8.GetByteCount($baseTrain) -ne (Get-Item -LiteralPath $ReferenceTrainPath).Length -or
   [Text.Encoding]::UTF8.GetByteCount($baseValidation) -ne (Get-Item -LiteralPath $ReferenceValidationPath).Length){
    throw 'Cached pages do not reconstruct the pinned raw train and validation files; refusing to publish.'
}
function Get-Utf8Sha256([string]$Text){
    $sha=[Security.Cryptography.SHA256]::Create()
    try{$bytes=$sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($Text));return ([BitConverter]::ToString($bytes)).Replace('-','')}
    finally{$sha.Dispose()}
}
if((Get-Utf8Sha256 $baseTrain) -ne (Get-FileHash -LiteralPath $ReferenceTrainPath -Algorithm SHA256).Hash -or
   (Get-Utf8Sha256 $baseValidation) -ne (Get-FileHash -LiteralPath $ReferenceValidationPath -Algorithm SHA256).Hash){
    throw 'Cached page text or order differs from the pinned raw files; refusing to publish.'
}
$trainText=Join-PageDocuments $trainPages.ToArray() $true
$validationText=Join-PageDocuments $validationPages.ToArray() $true
$output=[IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Force -Path $output|Out-Null
[IO.File]::WriteAllText((Join-Path $output 'train.txt'),$trainText,[Text.UTF8Encoding]::new($false))
[IO.File]::WriteAllText((Join-Path $output 'validation.txt'),$validationText,[Text.UTF8Encoding]::new($false))
$byLanguage=[Collections.Generic.List[object]]::new()
foreach($language in $languages){
    $languagePages=@($validationPages|Where-Object language -eq $language)
    $languageText=Join-PageDocuments $languagePages $true
    $path=Join-Path $output "validation-$language.txt"
    [IO.File]::WriteAllText($path,$languageText,[Text.UTF8Encoding]::new($false))
    $byLanguage.Add([pscustomobject]@{
        language=$language; validation_pages=$languagePages.Count
        validation_sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
    })
}
$manifest=[pscustomobject]@{
    dataset='Wikimedia revision-attributed 20-language lead corpus'
    license='CC BY-SA 4.0'
    language_prefix_format='<lang:xx> on a separate line before every article title'
    train_pages=$trainPages.Count; validation_pages=$validationPages.Count
    train_source_sha256=(Get-FileHash -LiteralPath $ReferenceTrainPath -Algorithm SHA256).Hash
    validation_source_sha256=(Get-FileHash -LiteralPath $ReferenceValidationPath -Algorithm SHA256).Hash
    tagged_train_sha256=(Get-FileHash -LiteralPath (Join-Path $output 'train.txt') -Algorithm SHA256).Hash
    tagged_validation_sha256=(Get-FileHash -LiteralPath (Join-Path $output 'validation.txt') -Algorithm SHA256).Hash
    train_sources=@($trainPages|ForEach-Object{[pscustomobject]@{language=$_.language;title=$_.title;pageid=$_.pageid;revision=$_.revision;url=$_.url;license=$_.license}})
    validation_sources=@($validationPages|ForEach-Object{[pscustomobject]@{language=$_.language;title=$_.title;pageid=$_.pageid;revision=$_.revision;url=$_.url;license=$_.license}})
    validation_by_language=$byLanguage
}
$manifest|ConvertTo-Json -Depth 8|Set-Content -LiteralPath (Join-Path $output 'sources.json') -Encoding utf8
Write-Output "languages=$($languages.Count) train_pages=$($trainPages.Count) validation_pages=$($validationPages.Count) train_bytes=$([Text.Encoding]::UTF8.GetByteCount($trainText)) validation_bytes=$([Text.Encoding]::UTF8.GetByteCount($validationText)) train_sha256=$((Get-FileHash -LiteralPath (Join-Path $output 'train.txt') -Algorithm SHA256).Hash) validation_sha256=$((Get-FileHash -LiteralPath (Join-Path $output 'validation.txt') -Algorithm SHA256).Hash) output=$output"
