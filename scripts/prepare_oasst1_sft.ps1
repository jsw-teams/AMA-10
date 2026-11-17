param(
    [string]$InputPath = 'data/raw/oasst1/ready.messages.jsonl.gz',
    [string]$OutputDirectory = 'data/processed/oasst1_sft',
    [int]$TrainRows = 200,
    [int]$ValidationRows = 50,
    [int]$MaxPromptCharacters = 2000,
    [int]$MaxResponseCharacters = 4000
)
$ErrorActionPreference = 'Stop'
if ($TrainRows -lt 1 -or $ValidationRows -lt 1) { throw 'TrainRows and ValidationRows must be positive.' }
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
$rowsRead = 0
try {
    while (($line = $reader.ReadLine()) -ne $null) {
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        $row = ConvertFrom-Json -InputObject $line
        if (-not $row.message_id -or -not $row.message_tree_id -or -not $row.text) { continue }
        $labels = $row.labels
        $valid = $row.review_result -eq $true -and $row.deleted -eq $false -and
                 $row.lang -eq 'en' -and $row.tree_state -eq 'ready_for_export' -and
                 $labels.spam.value -eq 0 -and $labels.lang_mismatch.value -eq 0 -and
                 $labels.pii.value -eq 0 -and $labels.not_appropriate.value -eq 0
        if (-not $valid) { continue }
        $messages[$row.message_id] = [pscustomobject]@{
            id = [string]$row.message_id; parent = [string]$row.parent_id
            tree = [string]$row.message_tree_id; role = [string]$row.role
            text = [string]$row.text; lang = [string]$row.lang
        }
        $rowsRead++
    }
} finally {
    $reader.Dispose();$gzip.Dispose();$stream.Dispose()
}

$train = [Collections.Generic.List[object]]::new()
$validation = [Collections.Generic.List[object]]::new()
foreach ($message in $messages.Values) {
    if ($message.role -ne 'assistant' -or -not $message.parent) { continue }
    if (-not $messages.ContainsKey($message.parent)) { continue }
    $prompt = $messages[$message.parent]
    if ($prompt.role -ne 'prompter' -or $prompt.tree -ne $message.tree -or
        $prompt.text.Length -gt $MaxPromptCharacters -or $message.text.Length -gt $MaxResponseCharacters) { continue }
    $bucket = [Convert]::ToInt32($message.tree.Substring(0,2),16) % 10
    $formattedPrompt = "User: $($prompt.text)`nAssistant:"
    $pair = [pscustomobject]@{ id=$message.id; tree=$message.tree; prompt=$formattedPrompt; response=$message.text }
    if ($bucket -eq 0) { $validation.Add($pair) } else { $train.Add($pair) }
}
$train = @($train | Sort-Object id | Select-Object -First $TrainRows)
$validation = @($validation | Sort-Object id | Select-Object -First $ValidationRows)
if ($train.Count -lt $TrainRows -or $validation.Count -lt $ValidationRows) {
    throw "Not enough audited pairs after filtering (train=$($train.Count), validation=$($validation.Count))."
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
$output = [IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Force -Path $output | Out-Null
$trainPath = Join-Path $output 'train.tsv'
$validPath = Join-Path $output 'validation.tsv'
Write-PairFile $trainPath $train
Write-PairFile $validPath $validation
$manifest = [pscustomobject]@{
    dataset = 'OpenAssistant/oasst1'; file = $provenance.file_path
    revision = $provenance.resolved_commit; license = $provenance.dataset_card_license
    source_sha256 = $sourceHash; train_rows = $train.Count; validation_rows = $validation.Count
    filtered_english_messages = $rowsRead; split = 'message_tree_id first-byte modulo 10; bucket 0 validation'
    filters = 'ready_for_export; English; review_result=true; deleted=false; spam/lang_mismatch/PII/not_appropriate labels all zero'
    prompt_format = 'User: <prompt> newline Assistant:'
    max_prompt_characters = $MaxPromptCharacters; max_response_characters = $MaxResponseCharacters
    train_sha256 = (Get-FileHash -LiteralPath $trainPath -Algorithm SHA256).Hash
    validation_sha256 = (Get-FileHash -LiteralPath $validPath -Algorithm SHA256).Hash
    created_utc = [DateTime]::UtcNow.ToString('o')
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $output 'sources.json') -Encoding utf8
Write-Host "Prepared $($train.Count) train and $($validation.Count) validation English pairs; audited messages=$rowsRead; output=$output"
