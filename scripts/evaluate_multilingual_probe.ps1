param(
    [Parameter(Mandatory=$true)][string]$ModelPath,
    [Parameter(Mandatory=$true)][string]$TokenizerPath,
    [string]$AdapterPath = '',
    [string]$ExecutablePath = 'build/ama10.exe',
    [string]$OutputDirectory = 'build/multilingual_probe',
    [int]$NewTokens = 48,
    [double]$FrequencyPenalty = 0.5,
    [double]$MinimumExpectedScriptFraction = 0.5,
    [double]$MaximumRepeatedFourgramRatio = 0.75,
    [switch]$FailOnThreshold
)
$ErrorActionPreference = 'Stop'
if ($NewTokens -lt 1 -or $NewTokens -gt 512) { throw 'NewTokens must be in [1, 512].' }
if ($FrequencyPenalty -lt 0 -or $FrequencyPenalty -gt 10) { throw 'FrequencyPenalty must be in [0, 10].' }
if ($MinimumExpectedScriptFraction -lt 0 -or $MinimumExpectedScriptFraction -gt 1) { throw 'MinimumExpectedScriptFraction must be in [0, 1].' }
if ($MaximumRepeatedFourgramRatio -lt 0 -or $MaximumRepeatedFourgramRatio -gt 1) { throw 'MaximumRepeatedFourgramRatio must be in [0, 1].' }
foreach ($path in @($ModelPath,$TokenizerPath,$ExecutablePath)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Required file not found: $path" }
}
if ($AdapterPath -and -not (Test-Path -LiteralPath $AdapterPath -PathType Leaf)) { throw "Adapter file not found: $AdapterPath" }
$output = [IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Force -Path $output | Out-Null
$probes = @(
    [pscustomobject]@{ language='en'; expected='latin'; prompt='Respond in English: My favorite season is' },
    [pscustomobject]@{ language='es'; expected='latin'; prompt='Responde en español: Mi estación favorita es' },
    [pscustomobject]@{ language='fr'; expected='latin'; prompt='Réponds en français : Ma saison préférée est' },
    [pscustomobject]@{ language='ja'; expected='kana_or_cjk'; prompt='日本語で答えてください。好きな季節は' },
    [pscustomobject]@{ language='zh'; expected='cjk'; prompt='请用中文回答：我最喜欢的季节是' },
    [pscustomobject]@{ language='de'; expected='latin'; prompt='Antworte auf Deutsch: Meine Lieblingsjahreszeit ist' },
    [pscustomobject]@{ language='pt'; expected='latin'; prompt='Responda em português: Minha estação favorita é' },
    [pscustomobject]@{ language='ru'; expected='cyrillic'; prompt='Ответь по-русски: Моё любимое время года — это' },
    [pscustomobject]@{ language='ar'; expected='arabic'; prompt='أجب باللغة العربية: فصلي المفضل هو' }
)
$results = [Collections.Generic.List[object]]::new()
foreach ($probe in $probes) {
    $promptPath = Join-Path $output "$($probe.language)-prompt.txt"
    $generatedPath = Join-Path $output "$($probe.language)-generated.txt"
    [IO.File]::WriteAllText($promptPath,$probe.prompt,[Text.UTF8Encoding]::new($false))
    $generateArgs = @('generate',$ModelPath,$TokenizerPath,$promptPath,$generatedPath,$NewTokens,'cpu',$FrequencyPenalty)
    if ($AdapterPath) { $generateArgs += $AdapterPath }
    $commandOutput = @(& $ExecutablePath @generateArgs 2>&1)
    if ($LASTEXITCODE -ne 0) { throw "Generation failed for $($probe.language): $($commandOutput -join ' ')" }
    $metricsLine = [string]($commandOutput | Where-Object { $_ -match '^generation_distinct_token_ratio=' } | Select-Object -First 1)
    if (-not $metricsLine) { throw "Generation metrics missing for $($probe.language)." }
    $metrics = @{}
    foreach ($match in [regex]::Matches($metricsLine,'([a-z0-9_]+)=([0-9.]+)')) { $metrics[$match.Groups[1].Value] = [double]$match.Groups[2].Value }
    $codepoints = $metrics.codepoints
    $expectedCount = switch ($probe.expected) {
        'latin' { $metrics.scripts_latin }
        'kana_or_cjk' { $metrics.kana + $metrics.cjk }
        'cjk' { $metrics.cjk }
        'cyrillic' { $metrics.cyrillic }
        'arabic' { $metrics.arabic }
    }
    $fraction = if ($codepoints -gt 0) { $expectedCount / $codepoints } else { 0.0 }
    $passed = $metrics.invalid_utf8_bytes -eq 0 -and
              $fraction -ge $MinimumExpectedScriptFraction -and
              $metrics.repeated_4gram_ratio -le $MaximumRepeatedFourgramRatio
    $results.Add([pscustomobject]@{
        language = $probe.language
        expected_script = $probe.expected
        expected_script_count = $expectedCount
        codepoints = $codepoints
        expected_script_fraction = [Math]::Round($fraction,6)
        invalid_utf8_bytes = $metrics.invalid_utf8_bytes
        distinct_token_ratio = $metrics.generation_distinct_token_ratio
        repeated_4gram_ratio = $metrics.repeated_4gram_ratio
        passed = $passed
        prompt = $probe.prompt
        generated_path = $generatedPath
    })
}
$failed = @($results | Where-Object { -not $_.passed })
$report = [pscustomobject]@{
    model_path = [IO.Path]::GetFullPath($ModelPath)
    model_sha256 = (Get-FileHash -LiteralPath $ModelPath -Algorithm SHA256).Hash
    tokenizer_path = [IO.Path]::GetFullPath($TokenizerPath)
    tokenizer_sha256 = (Get-FileHash -LiteralPath $TokenizerPath -Algorithm SHA256).Hash
    adapter_path = if ($AdapterPath) { [IO.Path]::GetFullPath($AdapterPath) } else { $null }
    adapter_sha256 = if ($AdapterPath) { (Get-FileHash -LiteralPath $AdapterPath -Algorithm SHA256).Hash } else { $null }
    executable_path = [IO.Path]::GetFullPath($ExecutablePath)
    new_tokens = $NewTokens
    frequency_penalty = $FrequencyPenalty
    minimum_expected_script_fraction = $MinimumExpectedScriptFraction
    maximum_repeated_fourgram_ratio = $MaximumRepeatedFourgramRatio
    overall_pass = ($failed.Count -eq 0)
    probes = $results
    limitation = 'Script-block adherence is a shallow diagnostic; it does not score semantic quality, grammar, or answer correctness.'
    created_utc = [DateTime]::UtcNow.ToString('o')
}
$reportPath = Join-Path $output 'report.json'
$report | ConvertTo-Json -Depth 7 | Set-Content -LiteralPath $reportPath -Encoding utf8
foreach ($result in $results) {
    Write-Host "$($result.language): expected_fraction=$($result.expected_script_fraction) repeated4=$($result.repeated_4gram_ratio) passed=$($result.passed)"
}
Write-Host "overall_pass=$($report.overall_pass) report=$reportPath"
if ($FailOnThreshold -and $failed.Count) { exit 1 }
