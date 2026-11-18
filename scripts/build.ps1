param([switch]$Test,[switch]$OpenCL,[switch]$Parquet)
$ErrorActionPreference = 'Stop'

$compiler = Get-Command gcc -ErrorAction SilentlyContinue
if ($compiler) {
    $gcc = $compiler.Source
} else {
    $roots = Get-ChildItem (Join-Path $env:LOCALAPPDATA 'Microsoft\WinGet\Packages') -Directory -Filter 'BrechtSanders.WinLibs*' -ErrorAction SilentlyContinue
    $gcc = $null
    foreach ($root in $roots) {
        $candidate = Join-Path $root.FullName 'mingw64\bin\gcc.exe'
        if (Test-Path -LiteralPath $candidate) { $gcc = $candidate; break }
    }
}
if (-not $gcc) { throw 'GCC was not found. Install a C11 GCC toolchain and add it to PATH.' }
$gccBin = Split-Path -Parent $gcc
if (($env:PATH -split [IO.Path]::PathSeparator) -notcontains $gccBin) {
    $env:PATH = "$gccBin$([IO.Path]::PathSeparator)$env:PATH"
}

$rootDir = Split-Path -Parent $PSScriptRoot
$buildDir = Join-Path $rootDir 'build'
if ($Parquet) { & (Join-Path $PSScriptRoot 'fetch_duckdb.ps1') }
$cloudflareTeacherFlag = [Environment]::GetEnvironmentVariable('AMA10_USE_CLOUDFLARE_TEACHER','Process')
$cloudflareAccountId = [Environment]::GetEnvironmentVariable('CLOUDFLARE_ACCOUNT_ID','Process')
New-Item -ItemType Directory -Force -Path $buildDir | Out-Null
Push-Location $rootDir
try {
    $common = @('-std=c11','-O3','-Wall','-Wextra','-Wpedantic','-Werror','-I','include')
    $core = @('src/model.c','src/affine_model.c','src/encoder.c','src/image.c','src/image_gen.c','src/audio.c','src/tokenizer.c','src/checkpoint.c','src/multimodal_checkpoint.c','src/dataset.c','src/eval.c','src/lora.c','src/tools.c','src/q8.c')
    $openclArgs = @()
    if ($OpenCL) {
        $openclRoot = $env:CUDA_PATH
        if (-not $openclRoot -or -not (Test-Path -LiteralPath (Join-Path $openclRoot 'include\CL\cl.h'))) {
            throw 'OpenCL headers were not found under CUDA_PATH.'
        }
        $openclArgs = @('-DAMA10_ENABLE_OPENCL=1','-I',(Join-Path $openclRoot 'include'),'src/opencl_backend.c','src/affine_scan_opencl.c','src/image_gen_opencl.c','-L',(Join-Path $openclRoot 'lib\x64'),'-lOpenCL')
    }
    & $gcc @common @core @openclArgs src/main.c -lm -o (Join-Path $buildDir 'ama10.exe')
    if ($LASTEXITCODE -ne 0) { throw "AMA-10 build failed with exit code $LASTEXITCODE" }
    & $gcc @common @core src/image_caption_train.c -lm -o (Join-Path $buildDir 'ama10_image_caption_train.exe')
    if ($LASTEXITCODE -ne 0) { throw "Image-caption trainer build failed with exit code $LASTEXITCODE" }
    & $gcc @common @core src/image_caption_generate.c -lm -o (Join-Path $buildDir 'ama10_image_caption_generate.exe')
    if ($LASTEXITCODE -ne 0) { throw "Image-caption generator build failed with exit code $LASTEXITCODE" }
    & $gcc @common @core src/image_gen_train.c -lm -o (Join-Path $buildDir 'ama10_image_gen_train.exe')
    if ($LASTEXITCODE -ne 0) { throw "Image generation trainer build failed with exit code $LASTEXITCODE" }
    if ($OpenCL) {
        & $gcc @common '-I' (Join-Path $env:CUDA_PATH 'include') '-DAMA10_ENABLE_OPENCL=1' src/opencl_backend.c src/affine_scan_opencl.c src/image_gen_opencl.c @core src/image_gen_train.c '-L' (Join-Path $env:CUDA_PATH 'lib\x64') '-lOpenCL' -lm -o (Join-Path $buildDir 'ama10_image_gen_train_opencl.exe')
        if ($LASTEXITCODE -ne 0) { throw "OpenCL image generation trainer build failed with exit code $LASTEXITCODE" }
    }
    & $gcc @common @core src/image_gen_generate.c -lm -o (Join-Path $buildDir 'ama10_image_gen_generate.exe')
    if ($LASTEXITCODE -ne 0) { throw "Image generator build failed with exit code $LASTEXITCODE" }
    & $gcc @common @core src/image_score.c -lm -o (Join-Path $buildDir 'ama10_image_score.exe')
    if ($LASTEXITCODE -ne 0) { throw "Image scorer build failed with exit code $LASTEXITCODE" }
    & $gcc @common @core src/q8_cli.c -lm -o (Join-Path $buildDir 'ama10_q8.exe')
    if ($LASTEXITCODE -ne 0) { throw "Q8 inference CLI build failed with exit code $LASTEXITCODE" }
    & $gcc @common @core src/audio_transcribe_train.c -lm -o (Join-Path $buildDir 'ama10_audio_transcribe_train.exe')
    if ($LASTEXITCODE -ne 0) { throw "Audio-transcription trainer build failed with exit code $LASTEXITCODE" }
    & $gcc @common @core src/audio_transcribe_generate.c -lm -o (Join-Path $buildDir 'ama10_audio_transcribe_generate.exe')
    if ($LASTEXITCODE -ne 0) { throw "Audio-transcription generator build failed with exit code $LASTEXITCODE" }
    & $gcc @common @core src/tts.c src/tts_train.c -lm -o (Join-Path $buildDir 'ama10_tts_train.exe')
    if ($LASTEXITCODE -ne 0) { throw "TTS trainer build failed with exit code $LASTEXITCODE" }
    & $gcc @common @core src/tts.c src/tts_generate.c -lm -o (Join-Path $buildDir 'ama10_tts_generate.exe')
    if ($LASTEXITCODE -ne 0) { throw "TTS generator build failed with exit code $LASTEXITCODE" }
    & $gcc @common @core src/sft_train.c -lm -o (Join-Path $buildDir 'ama10_sft_train.exe')
    if ($LASTEXITCODE -ne 0) { throw "SFT trainer build failed with exit code $LASTEXITCODE" }
    & $gcc @common @core src/kd_train.c -lm -o (Join-Path $buildDir 'ama10_kd_train.exe')
    if ($LASTEXITCODE -ne 0) { throw "KD trainer build failed with exit code $LASTEXITCODE" }
    & $gcc @common @core src/lora_train.c -lm -o (Join-Path $buildDir 'ama10_lora_train.exe')
    if ($LASTEXITCODE -ne 0) { throw "LoRA trainer build failed with exit code $LASTEXITCODE" }
    & $gcc @common @core src/supervisor.c src/judge_json.c src/cloudflare_ai.c -lm -lwinhttp -o (Join-Path $buildDir 'ama10_supervisor.exe')
    if ($LASTEXITCODE -ne 0) { throw "Supervisor build failed with exit code $LASTEXITCODE" }
    & $gcc @common @core src/tool_runtime.c -lm -o (Join-Path $buildDir 'ama10_tool_runtime.exe')
    if ($LASTEXITCODE -ne 0) { throw "Tool runtime build failed with exit code $LASTEXITCODE" }
    & $gcc @common @core src/tool_chat.c -lm -o (Join-Path $buildDir 'ama10_tool_chat.exe')
    if ($LASTEXITCODE -ne 0) { throw "Tool chat build failed with exit code $LASTEXITCODE" }
    & $gcc @common src/judge_json.c src/cloudflare_ai.c src/teacher_judge.c -lwinhttp -o (Join-Path $buildDir 'ama10_teacher_judge.exe')
    if ($LASTEXITCODE -ne 0) { throw "Cloudflare teacher build failed with exit code $LASTEXITCODE" }
    & $gcc @common src/swanlab_metrics.c src/swanlab_cli.c -lwinhttp -o (Join-Path $buildDir 'ama10_swanlab.exe')
    if ($LASTEXITCODE -ne 0) { throw "SwanLab C metric uploader build failed with exit code $LASTEXITCODE" }
    & $gcc @common src/pipeline_ledger.c src/pipeline_ledger_cli.c -o (Join-Path $buildDir 'ama10_pipeline_ledger.exe')
    if ($LASTEXITCODE -ne 0) { throw "Pipeline ledger build failed with exit code $LASTEXITCODE" }
    & $gcc @common '-DAMA10_TEXT_SPLIT_CLI=1' src/text_split.c -o (Join-Path $buildDir 'ama10_text_split.exe')
    if ($LASTEXITCODE -ne 0) { throw "Document split utility build failed with exit code $LASTEXITCODE" }
    & $gcc @common src/pipeline_ledger.c src/shard_fetch.c src/shard_fetch_cli.c -lwinhttp -lbcrypt -o (Join-Path $buildDir 'ama10_fetch_shard.exe')
    if ($LASTEXITCODE -ne 0) { throw "Official shard fetcher build failed with exit code $LASTEXITCODE" }
    if ($Parquet) {
        $duckdbDir = Join-Path $buildDir 'deps\duckdb-2.0.0'
        & $gcc @common '-I' $duckdbDir src/pipeline_ledger.c src/parquet_ingest.c src/parquet_ingest_cli.c -lbcrypt -o (Join-Path $buildDir 'ama10_parquet_ingest.exe')
        if ($LASTEXITCODE -ne 0) { throw "Parquet shard processor build failed with exit code $LASTEXITCODE" }
    }
    & $gcc @common src/tokenizer.c src/eval.c src/text_score.c -lm -o (Join-Path $buildDir 'ama10_score_text.exe')
    if ($LASTEXITCODE -ne 0) { throw "Text scorer build failed with exit code $LASTEXITCODE" }
    if ($OpenCL) {
        & $gcc @common '-I' (Join-Path $env:CUDA_PATH 'include') '-DAMA10_ENABLE_OPENCL=1' src/opencl_backend.c src/affine_scan_opencl.c src/image_gen_opencl.c @core benchmarks/opencl_inference.c '-L' (Join-Path $env:CUDA_PATH 'lib\x64') '-lOpenCL' -lm -o (Join-Path $buildDir 'ama10_opencl_bench.exe')
        if ($LASTEXITCODE -ne 0) { throw "OpenCL benchmark build failed with exit code $LASTEXITCODE" }
    }
    if ($Test) {
        & $gcc @common src/pipeline_ledger.c tests/pipeline_ledger_test.c -o (Join-Path $buildDir 'ama10_pipeline_ledger_test.exe')
        if ($LASTEXITCODE -ne 0) { throw "Pipeline ledger test build failed with exit code $LASTEXITCODE" }
        & (Join-Path $buildDir 'ama10_pipeline_ledger_test.exe')
        if ($LASTEXITCODE -ne 0) { throw "Pipeline ledger test failed with exit code $LASTEXITCODE" }
        [Environment]::SetEnvironmentVariable('AMA10_USE_CLOUDFLARE_TEACHER','0','Process')
        & $gcc @common src/judge_json.c src/cloudflare_ai.c tests/cloudflare_ai_test.c -lwinhttp -o (Join-Path $buildDir 'ama10_cloudflare_ai_test.exe')
        if ($LASTEXITCODE -ne 0) { throw "Cloudflare parser test build failed with exit code $LASTEXITCODE" }
        & (Join-Path $buildDir 'ama10_cloudflare_ai_test.exe')
        if ($LASTEXITCODE -ne 0) { throw "Cloudflare parser test failed with exit code $LASTEXITCODE" }
        & $gcc @common src/swanlab_metrics.c tests/swanlab_test.c -lwinhttp -o (Join-Path $buildDir 'ama10_swanlab_test.exe')
        if ($LASTEXITCODE -ne 0) { throw "SwanLab uploader test build failed with exit code $LASTEXITCODE" }
        & (Join-Path $buildDir 'ama10_swanlab_test.exe')
        if ($LASTEXITCODE -ne 0) { throw "SwanLab uploader validation test failed with exit code $LASTEXITCODE" }
        & $gcc -std=c11 -O3 -Wall -Wextra -Wpedantic -Werror -I include src/model.c src/affine_model.c src/tokenizer.c tests/core_test.c -lm -o (Join-Path $buildDir 'ama10_core_test.exe')
        if ($LASTEXITCODE -ne 0) { throw "Test build failed with exit code $LASTEXITCODE" }
        & (Join-Path $buildDir 'ama10_core_test.exe')
        if ($LASTEXITCODE -ne 0) { throw "Core test failed with exit code $LASTEXITCODE" }
        & $gcc @common src/model.c src/affine_model.c src/encoder.c tests/encoder_test.c -lm -o (Join-Path $buildDir 'ama10_encoder_test.exe')
        if ($LASTEXITCODE -ne 0) { throw "Encoder test build failed with exit code $LASTEXITCODE" }
        & (Join-Path $buildDir 'ama10_encoder_test.exe')
        if ($LASTEXITCODE -ne 0) { throw "Encoder test failed with exit code $LASTEXITCODE" }
        & $gcc @common src/model.c src/affine_model.c src/encoder.c src/image.c tests/image_test.c -lm -o (Join-Path $buildDir 'ama10_image_test.exe')
        if ($LASTEXITCODE -ne 0) { throw "Image test build failed with exit code $LASTEXITCODE" }
        & (Join-Path $buildDir 'ama10_image_test.exe')
        if ($LASTEXITCODE -ne 0) { throw "Image test failed with exit code $LASTEXITCODE" }
        & $gcc @common src/image.c src/image_gen.c tests/image_gen_test.c -lm -o (Join-Path $buildDir 'ama10_image_gen_test.exe')
        if ($LASTEXITCODE -ne 0) { throw "Image generation test build failed with exit code $LASTEXITCODE" }
        & (Join-Path $buildDir 'ama10_image_gen_test.exe') $buildDir
        if ($LASTEXITCODE -ne 0) { throw "Image generation test failed with exit code $LASTEXITCODE" }
        & $gcc @common src/audio.c tests/audio_test.c -lm -o (Join-Path $buildDir 'ama10_audio_test.exe')
        if ($LASTEXITCODE -ne 0) { throw "Audio test build failed with exit code $LASTEXITCODE" }
        & (Join-Path $buildDir 'ama10_audio_test.exe')
        if ($LASTEXITCODE -ne 0) { throw "Audio test failed with exit code $LASTEXITCODE" }
        & $gcc @common src/audio.c src/tts.c tests/tts_test.c -lm -o (Join-Path $buildDir 'ama10_tts_test.exe')
        if ($LASTEXITCODE -ne 0) { throw "TTS test build failed with exit code $LASTEXITCODE" }
        & (Join-Path $buildDir 'ama10_tts_test.exe') $buildDir
        if ($LASTEXITCODE -ne 0) { throw "TTS test failed with exit code $LASTEXITCODE" }
        $audioFixtureDir = Join-Path $buildDir 'audio_asr_fixture'
        New-Item -ItemType Directory -Force -Path $audioFixtureDir | Out-Null
        Copy-Item -LiteralPath (Join-Path $buildDir 'audio-fixture.wav') -Destination (Join-Path $audioFixtureDir 'sample.wav') -Force
        $tab = [string][char]9
        $manifest = Join-Path $audioFixtureDir 'manifest.tsv'
        [IO.File]::WriteAllLines($manifest,@("sample.wav${tab}a${tab}train${tab}en","sample.wav${tab}a${tab}validation${tab}en"),[Text.UTF8Encoding]::new($false))
        $audioFixturePrefix = Join-Path $audioFixtureDir 'fixture'
        & (Join-Path $buildDir 'ama10_audio_transcribe_train.exe') $manifest 2 $audioFixturePrefix
        if ($LASTEXITCODE -ne 0) { throw "Audio transcription training integration failed with exit code $LASTEXITCODE" }
        & (Join-Path $buildDir 'ama10_audio_transcribe_generate.exe') "$audioFixturePrefix.amm" (Join-Path $audioFixtureDir 'sample.wav') (Join-Path $audioFixtureDir 'transcript.txt') 8 en
        if ($LASTEXITCODE -ne 0) { throw "Audio transcription generation integration failed with exit code $LASTEXITCODE" }
        & (Join-Path $buildDir 'ama10_audio_transcribe_generate.exe') "$audioFixturePrefix.amm" (Join-Path $audioFixtureDir 'sample.wav') (Join-Path $audioFixtureDir 'transcript-lm.txt') 8 en 0 $manifest 0.1 0
        if ($LASTEXITCODE -ne 0) { throw "Audio CTC language-model integration failed with exit code $LASTEXITCODE" }
        $ttsFixturePrefix = Join-Path $audioFixtureDir 'tts-fixture.ama'
        & (Join-Path $buildDir 'ama10_tts_train.exe') $manifest 2 $ttsFixturePrefix
        if ($LASTEXITCODE -ne 0) { throw "TTS training integration failed with exit code $LASTEXITCODE" }
        & (Join-Path $buildDir 'ama10_tts_train.exe') $manifest 4 $ttsFixturePrefix --resume
        if ($LASTEXITCODE -ne 0) { throw "TTS resume integration failed with exit code $LASTEXITCODE" }
        $ttsPrompt = Join-Path $audioFixtureDir 'tts-prompt.txt'
        [IO.File]::WriteAllText($ttsPrompt,'a',[Text.UTF8Encoding]::new($false))
        & (Join-Path $buildDir 'ama10_tts_generate.exe') $ttsFixturePrefix $ttsPrompt (Join-Path $audioFixtureDir 'tts-output.wav') en 50
        if ($LASTEXITCODE -ne 0) { throw "TTS generation integration failed with exit code $LASTEXITCODE" }
        if (-not (Test-Path -LiteralPath (Join-Path $audioFixtureDir 'tts-output.wav'))) { throw 'TTS WAV output was not created' }
        $sftTrain = Join-Path $buildDir 'sft-train.tsv'
        $sftValid = Join-Path $buildDir 'sft-valid.tsv'
        [IO.File]::WriteAllLines($sftTrain,@("2+2?${tab}4","hi${tab}hello"),[Text.UTF8Encoding]::new($false))
        [IO.File]::WriteAllLines($sftValid,@("2+2?${tab}4"),[Text.UTF8Encoding]::new($false))
        $metricTrain = Join-Path $buildDir 'metric-train.txt'
        $metricValid = Join-Path $buildDir 'metric-validation.txt'
        $metricCheckpoint = Join-Path $buildDir 'metric-fixture.ama'
        $metricCsv = Join-Path $buildDir 'metric-fixture.csv'
        [IO.File]::WriteAllText($metricTrain,(('small recurrent model training fixture. ' * 8) + "`n"),[Text.UTF8Encoding]::new($false))
        [IO.File]::WriteAllText($metricValid,(('held out validation text. ' * 4) + "`n"),[Text.UTF8Encoding]::new($false))
        & (Join-Path $buildDir 'ama10.exe') train $metricTrain $metricValid 2 0.0001 256 $metricCheckpoint --width 8 --metrics-csv $metricCsv
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $metricCsv)) { throw 'Training metric CSV integration failed' }
        if ((Get-Content -LiteralPath $metricCsv).Count -lt 7) { throw 'Training metric CSV did not contain expected scalar samples' }
        $sftPrompt = Join-Path $buildDir 'sft-generation-prompt.txt'
        [IO.File]::WriteAllText($sftPrompt,"User: 2+2?`nAssistant:",[Text.UTF8Encoding]::new($false))
        & (Join-Path $buildDir 'ama10_sft_train.exe') $sftTrain $sftValid 4 (Join-Path $buildDir 'sft-fixture.ama')
        if ($LASTEXITCODE -ne 0) { throw "SFT training integration failed with exit code $LASTEXITCODE" }
        if (-not (Test-Path -LiteralPath (Join-Path $buildDir 'sft-fixture.ama'))) { throw 'SFT checkpoint was not created' }
        $preservedSft = Join-Path $buildDir 'sft-preserved-fixture.ama'
        & (Join-Path $buildDir 'ama10_sft_train.exe') $sftTrain $sftValid 2 $preservedSft 0.00003 --init (Join-Path $buildDir 'sft-fixture.ama') --preserve-from (Join-Path $buildDir 'sft-fixture.ama') --preserve-weight 0.1
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $preservedSft)) { throw 'Teacher-preserved SFT integration failed' }
        & (Join-Path $buildDir 'ama10.exe') export-safetensors (Join-Path $buildDir 'sft-fixture.ama') (Join-Path $buildDir 'sft-fixture.safetensors')
        if ($LASTEXITCODE -ne 0) { throw 'Checkpoint SafeTensors export failed' }
        & (Join-Path $buildDir 'ama10.exe') generate (Join-Path $buildDir 'sft-fixture.safetensors') (Join-Path $buildDir 'sft-fixture.ama.tok') $sftPrompt (Join-Path $buildDir 'safetensors-generation.txt') 8 cpu 0
        if ($LASTEXITCODE -ne 0) { throw 'SafeTensors inference integration failed' }
        $kdTrain = Join-Path $buildDir 'kd-train.txt'
        $kdValid = Join-Path $buildDir 'kd-valid.txt'
        [IO.File]::WriteAllText($kdTrain,(('2+2? 4 hi hello ' * 40) + "`n"),[Text.UTF8Encoding]::new($false))
        [IO.File]::WriteAllText($kdValid,(('2+2? 4 hi hello ' * 8) + "`n"),[Text.UTF8Encoding]::new($false))
        $widthFixture = Join-Path $buildDir 'width-fixture.ama'
        & (Join-Path $buildDir 'ama10.exe') train $kdTrain $kdValid 2 0.0001 256 $widthFixture --width 64
        if ($LASTEXITCODE -ne 0) { throw 'Configurable-width training integration failed' }
        & (Join-Path $buildDir 'ama10.exe') train $kdTrain $kdValid 4 0.0001 256 $widthFixture --resume --width 64
        if ($LASTEXITCODE -ne 0) { throw 'Configurable-width resume integration failed' }
        & (Join-Path $buildDir 'ama10.exe') train $kdTrain $kdValid 4 0.0001 256 $widthFixture --resume --width 32 *> $null
        if ($LASTEXITCODE -eq 0) { throw 'Width-mismatched checkpoint resume was accepted' }
        & (Join-Path $buildDir 'ama10_kd_train.exe') (Join-Path $buildDir 'sft-fixture.ama') $kdTrain $kdValid 2 (Join-Path $buildDir 'kd-fixture.ama') 8
        if ($LASTEXITCODE -ne 0) { throw "KD training integration failed with exit code $LASTEXITCODE" }
        if (-not (Test-Path -LiteralPath (Join-Path $buildDir 'kd-fixture.ama'))) { throw 'KD checkpoint was not created' }
        & (Join-Path $buildDir 'ama10_lora_train.exe') (Join-Path $buildDir 'sft-fixture.ama') $sftTrain $sftValid 2 (Join-Path $buildDir 'lora-fixture.lora') 4
        if ($LASTEXITCODE -ne 0) { throw "LoRA training integration failed with exit code $LASTEXITCODE" }
        & (Join-Path $buildDir 'ama10.exe') generate (Join-Path $buildDir 'sft-fixture.ama') (Join-Path $buildDir 'sft-fixture.ama.tok') $sftPrompt (Join-Path $buildDir 'lora-generation-output.txt') 8 cpu 0 (Join-Path $buildDir 'lora-fixture.lora')
        if ($LASTEXITCODE -ne 0) { throw "LoRA inference integration failed with exit code $LASTEXITCODE" }
        $supervisorReport = Join-Path $buildDir 'supervisor-fixture.json'
        & (Join-Path $buildDir 'ama10_supervisor.exe') (Join-Path $buildDir 'sft-fixture.ama') (Join-Path $buildDir 'sft-fixture.ama') $kdValid $supervisorReport (Join-Path $buildDir 'supervisor-fixture.ama')
        if ($LASTEXITCODE -ne 0) { throw "Supervisor integration failed with exit code $LASTEXITCODE" }
        $supervisorJson = Get-Content -LiteralPath $supervisorReport -Raw | ConvertFrom-Json
        if ($supervisorJson.decision -ne 'accept') { throw 'Supervisor did not accept an identical reference/candidate checkpoint' }
        $teacherStateReport = Join-Path $buildDir 'supervisor-teacher-state-fixture.json'
        [Environment]::SetEnvironmentVariable('AMA10_USE_CLOUDFLARE_TEACHER','1','Process')
        [Environment]::SetEnvironmentVariable('CLOUDFLARE_ACCOUNT_ID',$null,'Process')
        & (Join-Path $buildDir 'ama10_supervisor.exe') (Join-Path $buildDir 'sft-fixture.ama') (Join-Path $buildDir 'sft-fixture.ama') $kdValid $teacherStateReport (Join-Path $buildDir 'supervisor-teacher-state-fixture.ama')
        if ($LASTEXITCODE -ne 0) { throw "Supervisor Cloudflare teacher configuration integration failed with exit code $LASTEXITCODE" }
        $teacherStateJson = Get-Content -LiteralPath $teacherStateReport -Raw | ConvertFrom-Json
        if ($teacherStateJson.decision -ne 'rollback' -or $teacherStateJson.reason -ne 'teacher_judgment_unavailable' -or $teacherStateJson.teacher.status -ne 'missing_account_or_api_key' -or -not $teacherStateJson.teacher.enabled) {
            throw 'Supervisor did not fail closed when Cloudflare teacher account configuration was missing'
        }
        [Environment]::SetEnvironmentVariable('AMA10_USE_CLOUDFLARE_TEACHER','0','Process')
        $supervisorLanguageReport = Join-Path $buildDir 'supervisor-language-fixture.json'
        $supervisorLanguageFiles = "en=$kdValid;de=$kdValid"
        & (Join-Path $buildDir 'ama10_supervisor.exe') (Join-Path $buildDir 'sft-fixture.ama') (Join-Path $buildDir 'sft-fixture.ama') $kdValid $supervisorLanguageReport (Join-Path $buildDir 'supervisor-language-fixture.ama') 0.01 0.75 $supervisorLanguageFiles 0.1 0.0
        if ($LASTEXITCODE -ne 0) { throw "Supervisor language-stratified integration failed with exit code $LASTEXITCODE" }
        $supervisorLanguageJson = Get-Content -LiteralPath $supervisorLanguageReport -Raw | ConvertFrom-Json
        if ($supervisorLanguageJson.decision -ne 'accept' -or -not $supervisorLanguageJson.language_gate_enabled -or -not $supervisorLanguageJson.script_gate_enabled -or $supervisorLanguageJson.minimum_expected_script_fraction -ne 0.0 -or $supervisorLanguageJson.language_validation.Count -ne 2 -or @($supervisorLanguageJson.language_validation | Where-Object { -not $_.passed -or $_.expected_script -ne 'latin' }).Count -ne 0) {
            throw 'Supervisor did not report passing per-language validation gates for an identical checkpoint'
        }
        $supervisorScriptReport = Join-Path $buildDir 'supervisor-script-fixture.json'
        & (Join-Path $buildDir 'ama10_supervisor.exe') (Join-Path $buildDir 'sft-fixture.ama') (Join-Path $buildDir 'sft-fixture.ama') $kdValid $supervisorScriptReport (Join-Path $buildDir 'supervisor-script-fixture.ama') 0.01 0.75 $supervisorLanguageFiles 0.1 0.6
        if ($LASTEXITCODE -ne 0) { throw "Supervisor script-adherence integration failed with exit code $LASTEXITCODE" }
        $supervisorScriptJson = Get-Content -LiteralPath $supervisorScriptReport -Raw | ConvertFrom-Json
        if ($supervisorScriptJson.decision -ne 'rollback' -or $supervisorScriptJson.reason -ne 'candidate_expected_script_fraction_too_low' -or @($supervisorScriptJson.language_validation | Where-Object { $_.script_passed }).Count -ne 0) {
            throw 'Supervisor did not reject a candidate below the expected-script threshold'
        }
        $supervisorLimitReport = Join-Path $buildDir 'supervisor-limit-fixture.json'
        & (Join-Path $buildDir 'ama10_supervisor.exe') (Join-Path $buildDir 'sft-fixture.ama') (Join-Path $buildDir 'sft-fixture.ama') $kdValid $supervisorLimitReport (Join-Path $buildDir 'supervisor-limit-fixture.ama') 0.01 0.1
        if ($LASTEXITCODE -ne 0) { throw "Supervisor repetition-limit integration failed with exit code $LASTEXITCODE" }
        $supervisorLimitJson = Get-Content -LiteralPath $supervisorLimitReport -Raw | ConvertFrom-Json
        if ($supervisorLimitJson.decision -ne 'rollback' -or $supervisorLimitJson.reason -ne 'candidate_exceeds_absolute_repetition_limit' -or $supervisorLimitJson.max_repeated_fourgram_ratio -ne 0.1) {
            throw 'Supervisor did not reject a candidate above the absolute repetition limit'
        }
        & $gcc @common src/tools.c tests/tools_test.c -lm -o (Join-Path $buildDir 'ama10_tools_test.exe')
        if ($LASTEXITCODE -ne 0) { throw "Tools test build failed with exit code $LASTEXITCODE" }
        & (Join-Path $buildDir 'ama10_tools_test.exe')
        if ($LASTEXITCODE -ne 0) { throw "Tools test failed with exit code $LASTEXITCODE" }
        $toolRequest = Join-Path $buildDir 'tool-request.txt'
        $toolResponse = Join-Path $buildDir 'tool-response.txt'
        [IO.File]::WriteAllText($toolRequest,"<|tool_call|>calculator${tab}6*7<|end_tool_call|>",[Text.UTF8Encoding]::new($false))
        & (Join-Path $buildDir 'ama10_tool_runtime.exe') $toolRequest $toolResponse
        if ($LASTEXITCODE -ne 0 -or -not ((Get-Content -LiteralPath $toolResponse -Raw).Contains("42"))) { throw 'Tool runtime did not execute the calculator call' }
        & (Join-Path $buildDir 'ama10_tool_chat.exe') (Join-Path $buildDir 'sft-fixture.ama') (Join-Path $buildDir 'sft-fixture.ama.tok') $sftPrompt (Join-Path $buildDir 'tool-chat-fixture.txt') 1 8
        if ($LASTEXITCODE -ne 0) { throw "Tool chat inference integration failed with exit code $LASTEXITCODE" }
        & (Join-Path $buildDir 'ama10.exe') generate (Join-Path $buildDir 'sft-fixture.ama') (Join-Path $buildDir 'sft-fixture.ama.tok') $sftPrompt (Join-Path $buildDir 'sft-generation-output.txt') 8 cpu 0.5
        if ($LASTEXITCODE -ne 0) { throw "Text generation with repetition penalty integration failed with exit code $LASTEXITCODE" }
        $q8Fixture = Join-Path $buildDir 'sft-fixture.q8'
        & (Join-Path $buildDir 'ama10_q8.exe') quantize (Join-Path $buildDir 'sft-fixture.ama') $q8Fixture
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $q8Fixture)) { throw 'Q8 checkpoint conversion integration failed' }
        & (Join-Path $buildDir 'ama10_q8.exe') generate $q8Fixture "$q8Fixture.tok" $sftPrompt (Join-Path $buildDir 'q8-generation-output.txt') 8
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath (Join-Path $buildDir 'q8-generation-output.txt'))) { throw 'Q8 generation integration failed' }
        & (Join-Path $buildDir 'ama10_q8.exe') evaluate $q8Fixture "$q8Fixture.tok" $kdValid
        if ($LASTEXITCODE -ne 0) { throw 'Q8 evaluation integration failed' }
        & $gcc -std=c11 -O3 -Wall -Wextra -Wpedantic -Werror -I include src/tokenizer.c tests/tokenizer_test.c -o (Join-Path $buildDir 'ama10_tokenizer_test.exe')
        if ($LASTEXITCODE -ne 0) { throw "Tokenizer test build failed with exit code $LASTEXITCODE" }
        & (Join-Path $buildDir 'ama10_tokenizer_test.exe')
        if ($LASTEXITCODE -ne 0) { throw "Tokenizer test failed with exit code $LASTEXITCODE" }
        & $gcc -std=c11 -O3 -Wall -Wextra -Wpedantic -Werror -I include src/model.c src/affine_model.c src/tokenizer.c src/checkpoint.c tests/checkpoint_test.c -lm -o (Join-Path $buildDir 'ama10_checkpoint_test.exe')
        if ($LASTEXITCODE -ne 0) { throw "Checkpoint test build failed with exit code $LASTEXITCODE" }
        & (Join-Path $buildDir 'ama10_checkpoint_test.exe')
        if ($LASTEXITCODE -ne 0) { throw "Checkpoint test failed with exit code $LASTEXITCODE" }
        & $gcc @common src/model.c src/affine_model.c tests/safetensors_test.c -lm -o (Join-Path $buildDir 'ama10_safetensors_test.exe')
        if ($LASTEXITCODE -ne 0) { throw "SafeTensors test build failed with exit code $LASTEXITCODE" }
        & (Join-Path $buildDir 'ama10_safetensors_test.exe')
        if ($LASTEXITCODE -ne 0) { throw "SafeTensors test failed with exit code $LASTEXITCODE" }
        & $gcc @common src/model.c src/affine_model.c src/q8.c tests/q8_test.c -lm -o (Join-Path $buildDir 'ama10_q8_test.exe')
        if ($LASTEXITCODE -ne 0) { throw "Q8 test build failed with exit code $LASTEXITCODE" }
        & (Join-Path $buildDir 'ama10_q8_test.exe') $buildDir
        if ($LASTEXITCODE -ne 0) { throw "Q8 test failed with exit code $LASTEXITCODE" }
        & $gcc @common src/model.c src/affine_model.c src/encoder.c src/tokenizer.c src/checkpoint.c src/multimodal_checkpoint.c tests/multimodal_checkpoint_test.c -lm -o (Join-Path $buildDir 'ama10_multimodal_checkpoint_test.exe')
        if ($LASTEXITCODE -ne 0) { throw "Multimodal checkpoint test build failed with exit code $LASTEXITCODE" }
        & (Join-Path $buildDir 'ama10_multimodal_checkpoint_test.exe')
        if ($LASTEXITCODE -ne 0) { throw "Multimodal checkpoint test failed with exit code $LASTEXITCODE" }
        & $gcc @common src/tokenizer.c src/dataset.c tests/dataset_test.c -o (Join-Path $buildDir 'ama10_dataset_test.exe')
        if ($LASTEXITCODE -ne 0) { throw "Dataset test build failed with exit code $LASTEXITCODE" }
        & (Join-Path $buildDir 'ama10_dataset_test.exe')
        if ($LASTEXITCODE -ne 0) { throw "Dataset test failed with exit code $LASTEXITCODE" }
        & $gcc @common src/tokenizer.c src/eval.c tests/eval_test.c -o (Join-Path $buildDir 'ama10_eval_test.exe')
        if ($LASTEXITCODE -ne 0) { throw "Evaluation test build failed with exit code $LASTEXITCODE" }
        & (Join-Path $buildDir 'ama10_eval_test.exe')
        if ($LASTEXITCODE -ne 0) { throw "Evaluation test failed with exit code $LASTEXITCODE" }
        & $gcc @common @core tests/lora_test.c -lm -o (Join-Path $buildDir 'ama10_lora_test.exe')
        if ($LASTEXITCODE -ne 0) { throw "LoRA test build failed with exit code $LASTEXITCODE" }
        & (Join-Path $buildDir 'ama10_lora_test.exe')
        if ($LASTEXITCODE -ne 0) { throw "LoRA test failed with exit code $LASTEXITCODE" }
        if ($OpenCL) {
            & $gcc @common '-DAMA10_ENABLE_OPENCL=1' '-I' (Join-Path $env:CUDA_PATH 'include') src/image.c src/image_gen.c src/image_gen_opencl.c tests/image_gen_opencl_test.c '-L' (Join-Path $env:CUDA_PATH 'lib\x64') '-lOpenCL' -lm -o (Join-Path $buildDir 'ama10_image_gen_opencl_test.exe')
            if ($LASTEXITCODE -ne 0) { throw "OpenCL spatial codec test build failed with exit code $LASTEXITCODE" }
            & (Join-Path $buildDir 'ama10_image_gen_opencl_test.exe') $buildDir (Join-Path $rootDir 'kernels\ama10_recurrent.cl')
            if ($LASTEXITCODE -ne 0) { throw "OpenCL spatial codec test failed with exit code $LASTEXITCODE" }
            & $gcc @common '-I' (Join-Path $env:CUDA_PATH 'include') src/affine_scan_opencl.c tests/affine_scan_test.c '-L' (Join-Path $env:CUDA_PATH 'lib\x64') '-lOpenCL' -lm -o (Join-Path $buildDir 'ama10_affine_scan_test.exe')
            if ($LASTEXITCODE -ne 0) { throw "Affine scan test build failed with exit code $LASTEXITCODE" }
            & (Join-Path $buildDir 'ama10_affine_scan_test.exe') (Join-Path $rootDir 'kernels\ama10_affine_scan.cl')
            if ($LASTEXITCODE -ne 0) { throw "Affine scan test failed with exit code $LASTEXITCODE" }
        & $gcc @common '-DAMA10_ENABLE_OPENCL=1' '-I' (Join-Path $env:CUDA_PATH 'include') src/opencl_backend.c src/affine_scan_opencl.c src/image_gen_opencl.c @core tests/opencl_test.c '-L' (Join-Path $env:CUDA_PATH 'lib\x64') '-lOpenCL' -lm -o (Join-Path $buildDir 'ama10_opencl_test.exe')
            if ($LASTEXITCODE -ne 0) { throw "OpenCL test build failed with exit code $LASTEXITCODE" }
            & (Join-Path $buildDir 'ama10_opencl_test.exe') (Join-Path $rootDir 'kernels\ama10_recurrent.cl')
            if ($LASTEXITCODE -ne 0) { throw "OpenCL test failed with exit code $LASTEXITCODE" }
            $hybridCheckpoint = Join-Path $buildDir 'opencl-adamw-integration.ama'
            & (Join-Path $buildDir 'ama10.exe') train $kdTrain $kdValid 2 0.0001 256 $hybridCheckpoint --opencl-adamw
            if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $hybridCheckpoint)) { throw 'OpenCL AdamW training integration failed' }
            & (Join-Path $buildDir 'ama10.exe') train $kdTrain $kdValid 4 0.0001 256 $hybridCheckpoint --resume --opencl-adamw
            if ($LASTEXITCODE -ne 0) { throw 'OpenCL AdamW resume integration failed' }
        }
    }
} finally {
    [Environment]::SetEnvironmentVariable('AMA10_USE_CLOUDFLARE_TEACHER',$cloudflareTeacherFlag,'Process')
    [Environment]::SetEnvironmentVariable('CLOUDFLARE_ACCOUNT_ID',$cloudflareAccountId,'Process')
    Pop-Location
}

