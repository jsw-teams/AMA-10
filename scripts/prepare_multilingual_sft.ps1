param(
    [string]$CacheDirectory = 'data/cache/wikipedia_expansion',
    [string]$OutputDirectory = 'data/processed/multilingual_sft',
    [int]$MaxResponseCharacters = 2000,
    [string[]]$Languages = @('en','es','fr','ja','zh','de','pt','ru','ar')
)
$ErrorActionPreference = 'Stop'
if ($MaxResponseCharacters -lt 100) { throw 'MaxResponseCharacters must be at least 100.' }
if (-not $Languages.Count -or @($Languages | Select-Object -Unique).Count -ne $Languages.Count) { throw 'Languages must be a non-empty list without duplicates.' }
$languages = $Languages
$train = [Collections.Generic.List[object]]::new()
$validation = [Collections.Generic.List[object]]::new()
$provenance = [Collections.Generic.List[object]]::new()
$pairSources = [Collections.Generic.List[object]]::new()
$seenPageIds = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)

function New-LanguagePrompt([string]$Language,[string]$Title) {
    switch ($Language) {
        'en' { return "Answer in English: What is $Title?`nAssistant:" }
        'es' { return "Responde en español: ¿Qué es $Title?`nAsistente:" }
        'fr' { return "Réponds en français : Qu’est-ce que $Title ?`nAssistant :" }
        'ja' { return "日本語で答えてください。$Titleとは何ですか？`n回答:" }
        'zh' { return "请用中文回答：什么是$Title？`n回答：" }
        'de' { return "Antworte auf Deutsch: Was ist $Title?`nAssistent:" }
        'pt' { return "Responda em português: O que é $Title?`nAssistente:" }
        'ru' { return "Ответь по-русски: Что такое $Title?`nАссистент:" }
        'ar' { return "أجب باللغة العربية: ما هو $Title؟`nالمساعد:" }
        'bn' { return "বাংলায় উত্তর দিন: $Title কী?`nসহকারী:" }
        'fa' { return "به فارسی پاسخ بده: $Title چیست؟`nدستیار:" }
        'hi' { return "हिंदी में उत्तर दें: $Title क्या है?`nसहायक:" }
        'id' { return "Jawab dalam bahasa Indonesia: Apa itu $Title?`nAsisten:" }
        'it' { return "Rispondi in italiano: Che cos'è $Title?`nAssistente:" }
        'ko' { return "한국어로 답하세요: $Title이란 무엇인가요?`n도우미:" }
        'nl' { return "Antwoord in het Nederlands: Wat is $Title?`nAssistent:" }
        'pl' { return "Odpowiedz po polsku: Czym jest $Title?`nAsystent:" }
        'sv' { return "Svara på svenska: Vad är $Title?`nAssistent:" }
        'tr' { return "Türkçe cevap ver: $Title nedir?`nAsistan:" }
        'vi' { return "Hãy trả lời bằng tiếng Việt: $Title là gì?`nTrợ lý:" }
        default { throw "Unsupported language: $Language" }
    }
}
function ConvertTo-TsvField([string]$Value) {
    return $Value.Replace('\','\\').Replace("`t",'\t').Replace("`r",'\r').Replace("`n",'\n')
}

foreach ($language in $languages) {
    $cachePath = Join-Path $CacheDirectory "$language.json"
    if (-not (Test-Path -LiteralPath $cachePath)) { throw "Missing cached source file: $cachePath" }
    $cacheHash = (Get-FileHash -LiteralPath $cachePath -Algorithm SHA256).Hash
    $records = @(Get-Content -LiteralPath $cachePath -Raw | ConvertFrom-Json)
    $languageTrain = 0
    $languageValidation = 0
    foreach ($record in $records) {
        $meta = $record.metadata
        if ($meta.language -ne $language -or $meta.license -ne 'CC BY-SA 4.0' -or
            -not $meta.pageid -or -not $meta.revision -or -not $meta.url -or
            $meta.split -notin @('train','validation') -or
            [string]::IsNullOrWhiteSpace([string]$meta.title) -or
            [string]::IsNullOrWhiteSpace([string]$record.extract)) {
            throw "Invalid or unlicensed cached source record in $cachePath"
        }
        $sourceKey = "$language`:$($meta.pageid)"
        if (-not $seenPageIds.Add($sourceKey)) { throw "Duplicate Wikimedia page ID $sourceKey" }
        $title = ([string]$meta.title).Trim()
        $response = ([string]$record.extract).Trim()
        if ($response.Length -gt $MaxResponseCharacters) {
            $cut = $response.LastIndexOfAny([char[]]@('.','!','?','。','！','？'),$MaxResponseCharacters)
            if ($cut -ge [int]($MaxResponseCharacters * 0.6)) { $response = $response.Substring(0,$cut+1) }
            else { $response = $response.Substring(0,$MaxResponseCharacters) }
        }
        if ([string]::IsNullOrWhiteSpace($response)) { throw "Empty response for $($meta.url)" }
        $pair = [pscustomobject]@{
            language = $language
            title = $title
            pageid = [string]$meta.pageid
            revision = [string]$meta.revision
            url = [string]$meta.url
            license = [string]$meta.license
            prompt = New-LanguagePrompt $language $title
            response = $response
        }
        if ($meta.split -eq 'train') {
            $train.Add($pair); $languageTrain++
            $rowIndex = $train.Count - 1
        } else {
            $validation.Add($pair); $languageValidation++
            $rowIndex = $validation.Count - 1
        }
        $pairSources.Add([pscustomobject]@{
            split = [string]$meta.split
            row_index = $rowIndex
            language = $language
            title = $title
            pageid = [string]$meta.pageid
            revision = [string]$meta.revision
            url = [string]$meta.url
            license = [string]$meta.license
        })
    }
    if ($languageTrain -lt 1 -or $languageValidation -lt 1) {
        throw "Language '$language' must have training and validation pages."
    }
    $provenance.Add([pscustomobject]@{
        language = $language
        source_cache = $cachePath
        source_cache_sha256 = $cacheHash
        train_pairs = $languageTrain
        validation_pairs = $languageValidation
        license = 'CC BY-SA 4.0'
    })
}

$output = [IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Force -Path $output | Out-Null
function Write-PairFile([string]$Path,[object[]]$Pairs) {
    $lines = [Collections.Generic.List[string]]::new()
    foreach ($pair in $Pairs) {
        $lines.Add((ConvertTo-TsvField $pair.prompt) + "`t" + (ConvertTo-TsvField $pair.response))
    }
    $temp = "$Path.partial"
    [IO.File]::WriteAllLines($temp,$lines,[Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath $temp -Destination $Path -Force
}
$trainPath = Join-Path $output 'train.tsv'
$validationPath = Join-Path $output 'validation.tsv'
Write-PairFile $trainPath $train.ToArray()
Write-PairFile $validationPath $validation.ToArray()
foreach ($language in $languages) {
    Write-PairFile (Join-Path $output "train-$language.tsv") @($train | Where-Object language -eq $language)
    Write-PairFile (Join-Path $output "validation-$language.tsv") @($validation | Where-Object language -eq $language)
}
$manifest = [pscustomobject]@{
    dataset = 'Wikimedia multilingual lead extracts'
    languages = $languages
    license = 'CC BY-SA 4.0'
    split = 'Preserved per-page train/validation split from the revision-pinned cache; aligned topic pages use the same split across languages.'
    prompt_format = 'Language-specific question asking for an answer in the named language; response is the source article lead extract.'
    max_response_characters = $MaxResponseCharacters
    train_pairs = $train.Count
    validation_pairs = $validation.Count
    sources = $provenance
    pair_sources = $pairSources
    train_sha256 = (Get-FileHash -LiteralPath $trainPath -Algorithm SHA256).Hash
    validation_sha256 = (Get-FileHash -LiteralPath $validationPath -Algorithm SHA256).Hash
    created_utc = [DateTime]::UtcNow.ToString('o')
}
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $output 'sources.json') -Encoding utf8
Write-Host "Prepared $($train.Count) multilingual training and $($validation.Count) validation pairs from $($languages.Count) languages; output=$output"
