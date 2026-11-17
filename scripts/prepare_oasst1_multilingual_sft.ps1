param(
    [string]$InputPath = 'data/raw/oasst1/ready.messages.jsonl.gz',
    [string]$OutputDirectory = 'data/processed/oasst1_multilingual_sft',
    [string[]]$Languages = @('en','es','fr','ja','zh','de','pt-BR','ru','ar'),
    [int]$MaxTrainRowsPerLanguage = 500,
    [int]$MaxValidationRowsPerLanguage = 100,
    [int]$MaxPromptCharacters = 2000,
    [int]$MaxResponseCharacters = 4000
)
$ErrorActionPreference = 'Stop'
if ($MaxTrainRowsPerLanguage -lt 1 -or $MaxValidationRowsPerLanguage -lt 1) { throw 'Per-language row caps must be positive.' }
if ($MaxPromptCharacters -lt 1 -or $MaxResponseCharacters -lt 1) { throw 'Character caps must be positive.' }
if (-not $Languages.Count -or @($Languages | Select-Object -Unique).Count -ne $Languages.Count) { throw 'Languages must be a non-empty list without duplicates.' }

$source = (Resolve-Path -LiteralPath $InputPath).Path
$provenancePath = "$source.sources.json"
if (-not (Test-Path -LiteralPath $provenancePath)) { throw 'The HF source provenance sidecar is required.' }
$provenance = Get-Content -LiteralPath $provenancePath -Raw | ConvertFrom-Json
if ($provenance.repo_id -ne 'OpenAssistant/oasst1' -or
    $provenance.dataset_card_license -ne 'apache-2.0' -or
    $provenance.file_path -ne '2023-04-12_oasst_ready.messages.jsonl.gz') {
    throw 'Unexpected source repository, file, or license in provenance.'
}
$sourceHash = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash
if ($sourceHash -ne $provenance.sha256) { throw 'Compressed source hash does not match the HF provenance record.' }

$messages = [Collections.Generic.Dictionary[string,object]]::new()
$stream = [IO.File]::OpenRead($source)
$gzip = [IO.Compression.GZipStream]::new($stream,[IO.Compression.CompressionMode]::Decompress)
$reader = [IO.StreamReader]::new($gzip,[Text.Encoding]::UTF8,$true)
$auditedMessages = 0
try {
    while (($line = $reader.ReadLine()) -ne $null) {
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        $row = ConvertFrom-Json -InputObject $line
        if (-not $row.message_id -or -not $row.message_tree_id -or -not $row.text -or $row.lang -notin $Languages) { continue }
        $labels = $row.labels
        $valid = $row.review_result -eq $true -and $row.deleted -eq $false -and
                 $row.tree_state -eq 'ready_for_export' -and
                 $labels.spam.value -eq 0 -and $labels.lang_mismatch.value -eq 0 -and
                 $labels.pii.value -eq 0 -and $labels.not_appropriate.value -eq 0
        if (-not $valid) { continue }
        $messages[[string]$row.message_id] = [pscustomobject]@{
            id = [string]$row.message_id; parent = [string]$row.parent_id
            tree = [string]$row.message_tree_id; role = [string]$row.role
            text = [string]$row.text; lang = [string]$row.lang
        }
        $auditedMessages++
    }
} finally {
    $reader.Dispose(); $gzip.Dispose(); $stream.Dispose()
}

$byLanguage = @{}
foreach ($language in $Languages) {
    $byLanguage[$language] = [pscustomobject]@{
        train = [Collections.Generic.List[object]]::new()
        validation = [Collections.Generic.List[object]]::new()
    }
}
foreach ($message in $messages.Values) {
    if ($message.role -ne 'assistant' -or -not $message.parent -or -not $messages.ContainsKey($message.parent)) { continue }
    $prompt = $messages[$message.parent]
    if ($prompt.role -ne 'prompter' -or $prompt.tree -ne $message.tree -or $prompt.lang -ne $message.lang) { continue }
    if ($prompt.text.Length -gt $MaxPromptCharacters -or $message.text.Length -gt $MaxResponseCharacters) { continue }
    $bucket = [Convert]::ToInt32($message.tree.Substring(0,2),16) % 10
    $split = if ($bucket -eq 0) { 'validation' } else { 'train' }
    $formattedPrompt = "User: $($prompt.text)`nAssistant:"
    $byLanguage[$message.lang].$split.Add([pscustomobject]@{
        id = $message.id; parent_id = $prompt.id; tree_id = $message.tree
        language = $message.lang; prompt = $formattedPrompt; response = $message.text
    })
}

function ConvertTo-TsvField([string]$Value) {
    return $Value.Replace('\','\\').Replace("`t",'\t').Replace("`r",'\r').Replace("`n",'\n')
}
function Write-PairFile([string]$Path,[object[]]$Pairs) {
    $lines = [Collections.Generic.List[string]]::new()
    foreach ($pair in $Pairs) { $lines.Add((ConvertTo-TsvField $pair.prompt) + "`t" + (ConvertTo-TsvField $pair.response)) }
    $temp = "$Path.partial"
    [IO.File]::WriteAllLines($temp,$lines,[Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath $temp -Destination $Path -Force
}

$train = [Collections.Generic.List[object]]::new()
$validation = [Collections.Generic.List[object]]::new()
$sourceRows = [Collections.Generic.List[object]]::new()
$counts = [Collections.Generic.List[object]]::new()
$output = [IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Force -Path $output | Out-Null
foreach ($language in $Languages) {
    foreach ($split in @('train','validation')) {
        $cap = if ($split -eq 'train') { $MaxTrainRowsPerLanguage } else { $MaxValidationRowsPerLanguage }
        $chosen = @($byLanguage[$language].$split | Sort-Object id | Select-Object -First $cap)
        if ($split -eq 'train') { $train.AddRange([object[]]$chosen) } else { $validation.AddRange([object[]]$chosen) }
        $languagePath = Join-Path $output "$split-$language.tsv"
        Write-PairFile $languagePath $chosen
        if ($split -eq 'validation') {
            $validationLines = [Collections.Generic.List[string]]::new()
            foreach ($pair in $chosen) { $validationLines.Add($pair.prompt); $validationLines.Add($pair.response) }
            $validationTextPath = Join-Path $output "$split-$language.txt"
            [IO.File]::WriteAllText($validationTextPath,($validationLines -join "`n") + "`n",[Text.UTF8Encoding]::new($false))
        }
        foreach ($pair in $chosen) {
            $sourceRows.Add([pscustomobject]@{
                language = $language; split = $split; message_id = $pair.id
                parent_message_id = $pair.parent_id; message_tree_id = $pair.tree_id
            })
        }
        $counts.Add([pscustomobject]@{
            language = $language; split = $split; available_pairs = $byLanguage[$language].$split.Count
            selected_pairs = $chosen.Count; distinct_trees = @($chosen | Select-Object -ExpandProperty tree_id -Unique).Count
        })
    }
}

$train = @($train | Sort-Object language,id)
$validation = @($validation | Sort-Object language,id)
$trainPath = Join-Path $output 'train.tsv'
$validationPath = Join-Path $output 'validation.tsv'
Write-PairFile $trainPath $train
Write-PairFile $validationPath $validation
$validationLines = [Collections.Generic.List[string]]::new()
foreach ($pair in $validation) { $validationLines.Add($pair.prompt); $validationLines.Add($pair.response) }
$validationTextPath = Join-Path $output 'validation.txt'
[IO.File]::WriteAllText($validationTextPath,($validationLines -join "`n") + "`n",[Text.UTF8Encoding]::new($false))
$manifest = [pscustomobject]@{
    dataset = 'OpenAssistant/oasst1'; file = $provenance.file_path
    revision = $provenance.resolved_commit; license = $provenance.dataset_card_license
    source_sha256 = $sourceHash; audited_messages = $auditedMessages
    split = 'whole message_tree_id; first byte modulo 10 assigns validation bucket 0'
    filters = 'ready_for_export; selected languages; matching prompt/assistant language; review_result=true; deleted=false; spam/lang_mismatch/PII/not_appropriate labels all zero'
    prompt_format = 'User: <prompt> newline Assistant:'
    max_prompt_characters = $MaxPromptCharacters; max_response_characters = $MaxResponseCharacters
    languages = $Languages; max_train_rows_per_language = $MaxTrainRowsPerLanguage
    max_validation_rows_per_language = $MaxValidationRowsPerLanguage
    counts = $counts.ToArray(); rows = $sourceRows.ToArray()
    train_sha256 = (Get-FileHash -LiteralPath $trainPath -Algorithm SHA256).Hash
    validation_sha256 = (Get-FileHash -LiteralPath $validationPath -Algorithm SHA256).Hash
    validation_text_sha256 = (Get-FileHash -LiteralPath $validationTextPath -Algorithm SHA256).Hash
    created_utc = [DateTime]::UtcNow.ToString('o')
    note = 'The Hub card declares Apache-2.0; this does not establish separate rights to every user-submitted message.'
}
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $output 'sources.json') -Encoding utf8
Write-Host "Prepared $($train.Count) train and $($validation.Count) validation multilingual pairs from $($Languages.Count) languages; audited selected-language messages=$auditedMessages; output=$output"
