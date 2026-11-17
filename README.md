# AMA-10

AMA-10 is a new C11 research project for an adaptive recurrent multimodal model. The current milestone includes a CPU language model, trainable dense modality encoder, small image-caption and licensed audio-transcription pilots, and an experimental text-to-speech path. The recurrent core uses manual BPTT, AdamW and hybrid Muon updates, and a trainable byte-pair tokenizer. The multimodal pipelines use the shared core, but their tiny datasets produce poor outputs; AMA-10 is not yet a capable assistant.

## Build and tests

Requires a C11 GCC toolchain. The included PowerShell build script discovers GCC on `PATH` or in the common user-scoped WinLibs location:

```powershell
.\scripts\build.ps1 -Test
```

Tests cover forward/reset behavior, tiny-sequence overfit, numerical gradient agreement for token and dense modality inputs, trainable encoder updates, image/audio decoding and framing, conditioned image/audio-to-text update and generation paths, normalized WER/CER, AdamW accumulation parity, Muon hybrid updates, checkpoint round trips and exact resume parity, tokenizer determinism/round trips, and NVIDIA RTX 4050 OpenCL inference parity when `-OpenCL` is enabled.

## Train

The checked-in text sample contains six Wikipedia articles for training and two separate held-out articles under CC BY-SA 4.0. Preserve revision attribution and share-alike terms; see [data attribution](data/README.md). Training starts from random model weights and learns BPE merges only from the training split, using up to an 8 MiB sample by default. Files larger than the budget are sampled at evenly spaced offsets; `--tokenizer-sample-bytes N` selects a different bounded budget. The encoded train and validation data are kept in fingerprinted, disk-backed token caches; training and evaluation read only the needed windows rather than retaining both full token arrays in RAM:

```powershell
New-Item -ItemType Directory -Force .\checkpoints | Out-Null
.\build\ama10.exe train .\data\raw\wikipedia_train.txt .\data\raw\wikipedia_validation.txt 10000 0.0001 512 .\checkpoints\ama10-bpe-v0.3.ama
```

The tokenizer has IDs 0–255 for byte fallback plus up to 256 learned merges. The CLI baseline uses batch size one, fixed depth two, 24-token TBPTT chunks, and width 32. Set `--width N` (8–4096) to train a larger or smaller recurrent core; resume requires the requested width to match the checkpoint. Pass `--accumulation N` after the checkpoint path to aggregate gradients across N chunks before each optimizer step; recurrent state continues across chunks and resets at corpus wrap. `--lr-decay D` enables an inverse-square-root schedule `lr(step)=base_lr/sqrt(1+D*(step-1))`; zero (the default) keeps a constant rate, and the absolute optimizer step is preserved across resumes. For a fresh random-weight experiment with a fixed existing tokenizer, pass `--tokenizer-file path.tok`; its vocabulary must match the requested size, and it cannot be combined with `--resume`. This lets architecture ablations reuse the same token cache and keeps tokenizer differences out of model comparisons. The C training API also exposes activity-threshold adaptive depth, with BPTT through executed steps; its stopping rule is a heuristic and has not yet been benchmarked for quality or compute savings. Use `--optimizer muon` for Newton-Schulz Muon on hidden matrices and AdamW on embeddings, output, and biases; its auxiliary AdamW rate is 0.015 of the hidden-matrix rate. Training metrics are token NLL for the learned BPE vocabulary. Checkpoint v6 stores model weights, tokenizer merges, optimizer kind and moments, recurrent state, update count, token cursor, accumulation and schedule settings, and fingerprints of both data files. It encodes the input-conditioned update-gate equation; older checkpoints are rejected because their weights were trained under different recurrence dynamics. A `.tok` companion is written for standalone generation. Snapshots are written every 100 optimizer updates and at the requested total.

To continue from that snapshot, pass the same data files, learning rate, and vocabulary, and set the desired total update count:

```powershell
.\build\ama10.exe train .\data\raw\wikipedia_train.txt .\data\raw\wikipedia_validation.txt 20000 0.0001 512 .\checkpoints\ama10-bpe-v0.3.ama --resume
```

Resume checks that data fingerprints and training settings match. The update argument is the final total, so this example advances a 10,000-update snapshot to 20,000 updates. Use the same `--accumulation N`, `--lr-decay D`, and `--tokenizer-sample-bytes N` on both commands.

An experimental hybrid training mode runs recurrent sequence forward propagation, the output-head projection, full-sequence BPTT gradients, and clipped AdamW updates through OpenCL; the CPU handles loss orchestration and data loading. Build with `-OpenCL`, then add `--opencl-adamw` to a fresh or resumed AdamW training command. Muon is not supported by this path. Optimizer moments stay on the device across updates and synchronize to the checkpoint before saving. The mode remains experimental: it was slower than CPU at widths 64, 256, and 512 in current end-to-end trials, though the gap narrowed at wider dimensions. OpenCL tests compare full-sequence gradients against CPU BPTT and compare consecutive parameter and optimizer-moment updates against CPU AdamW.

A fresh Muon-hybrid run can use a separately chosen hidden-matrix learning rate:

```powershell
.\build\ama10.exe train .\data\raw\multilingual_train.txt .\data\raw\multilingual_validation.txt 3000 0.002 512 .\checkpoints\ama10-muon.ama --accumulation 2 --lr-decay 0.1 --optimizer muon
```

To grow a v6 checkpoint's BPE vocabulary, append merges from the same fingerprinted training file and resize its embedding/output matrices and optimizer state. First train the 512-token source checkpoint, then grow and resume it:

```powershell
.\build\ama10.exe train .\data\raw\multilingual_expansion_train.txt .\data\raw\multilingual_expansion_validation.txt 10000 0.0001 512 .\checkpoints\ama10-multilingual-expansion-gated-512-v0.1.ama --accumulation 2 --lr-decay 0.1
.\build\ama10.exe grow-vocabulary .\checkpoints\ama10-multilingual-expansion-gated-512-v0.1.ama .\checkpoints\ama10-multilingual-expansion-gated-512-v0.1.ama.tok .\data\raw\multilingual_expansion_train.txt 1024 .\checkpoints\ama10-multilingual-expansion-gated-grown-v0.1.ama
.\build\ama10.exe train .\data\raw\multilingual_expansion_train.txt .\data\raw\multilingual_expansion_validation.txt 15000 0.0001 1024 .\checkpoints\ama10-multilingual-expansion-gated-grown-v0.1.ama --accumulation 2 --lr-decay 0.1 --resume
```

Growth preserves existing token IDs, parameters, optimizer moments, optimizer step, and checkpoint provenance. It maps the saved cursor through source-byte offsets into the new token cache and resets recurrent state because old token boundaries no longer apply. The printed `actual` vocabulary size is authoritative when training resumes.

The expanded multilingual baseline can be reproduced from the pinned CC BY-SA page leads. Supply an operator contact in the User-Agent with `-Contact 'name or email'` when available; the fetcher makes requests serially, pauses between language groups, caches successful groups, and honors `Retry-After` with exponential backoff:

```powershell
.\scripts\fetch_multilingual_expansion.ps1
.\build\ama10.exe train .\data\raw\multilingual_expansion_train.txt .\data\raw\multilingual_expansion_validation.txt 7200 0.0001 1024 .\checkpoints\ama10-multilingual-expansion-gated-v0.1.ama --accumulation 2 --lr-decay 0.1
.\scripts\split_multilingual_validation.ps1 -InputPath data/raw/multilingual_expansion_validation.txt -ManifestPath data/raw/multilingual_expansion_sources.json -OutputDirectory data/processed/multilingual_expansion_validation
foreach ($langFile in Get-ChildItem data/processed/multilingual_expansion_validation/*.txt) { .\build\ama10.exe evaluate .\checkpoints\ama10-multilingual-expansion-gated-v0.1.ama .\checkpoints\ama10-multilingual-expansion-gated-v0.1.ama.tok $langFile.FullName }
```

The politely fetched 20-language extension uses a separate output set so the nine-language corpus remains reproducible. It has 225 train and 58 validation pages (seven missing train titles are listed in the manifest sidecar); the same per-language 30-second request spacing and 429 backoff apply. The fresh random-init run and held-out scores are recorded in [the experiment log](docs/experiments.md). `scripts/prepare_multilingual_interleaved.ps1` constructs a deterministic round-robin order, and `scripts/prepare_multilingual_tagged.ps1` adds an explicit `<lang:xx>` prefix to each article from the same verified local caches; both matched 20k-update experiments are documented as rejected ablations. To reproduce the grouped-order model run after fetching:

```powershell
.\scripts\fetch_multilingual_expansion.ps1 -TrainPath data/raw/multilingual_expansion_20lang_train.txt -ValidationPath data/raw/multilingual_expansion_20lang_validation.txt -ManifestPath data/raw/multilingual_expansion_20lang_sources.json
.\build\ama10.exe train data/raw/multilingual_expansion_20lang_train.txt data/raw/multilingual_expansion_20lang_validation.txt 20000 0.0001 1024 checkpoints/ama10-20lang-fresh-v0.1.ama --width 128 --accumulation 2 --lr-decay 0.1 --metrics-csv build/20lang-fresh-metrics.csv
```

## SwanLab metrics

The C trainer can write scalar metrics to a `step,key,value` CSV, and the standalone C uploader sends those values to a SwanLab project without the Python SDK. Create a project in SwanLab first, then set `SWANLAB_API_KEY` in the environment:

```powershell
$env:SWANLAB_API_KEY = '<your API key>'
.\build\ama10.exe train .\data\raw\wikipedia_train.txt .\data\raw\wikipedia_validation.txt 1000 0.0001 512 .\checkpoints\metrics-run.ama --metrics-csv .\build\training-metrics.csv
.\build\ama10_swanlab.exe your-workspace your-project 'AMA-10 baseline' .\build\training-metrics.csv
```

The uploader creates an experiment in the existing project, defines scalar series, submits the CSV values, and marks the experiment finished. Set `SWANLAB_API_HOST` to a self-hosted server origin when needed, or pass the origin as the last CLI argument. The uploader targets SwanLab's current SDK HTTP ingestion routes; those routes are not part of the public OpenAPI contract and may change between SDK releases. Network upload requires a valid key and is not exercised by the offline test suite.

## Generate

Pass a UTF-8 prompt file. Model and tokenizer vocabularies must match:

```powershell
.\build\ama10.exe generate .\checkpoints\ama10-bpe-v0.3.ama .\checkpoints\ama10-bpe-v0.3.ama.tok .\data\prompt.txt .\build\answer.bin 128
```

Generation is greedy and writes decoded bytes. Prompts must be valid UTF-8; candidate tokens are filtered so generated output remains valid UTF-8, including a complete final code point. This is appropriate for text generation and intentionally disallows arbitrary binary outputs. It also reports distinct-token ratio, repeated 4-gram ratio, UTF-8 validity, and Unicode script counts to help quantify repetition and language following. The current prototype still produces incoherent, repetitive text; see [experiment results](docs/experiments.md).

Inference accepts native model files and AMA-10 F32 SafeTensors weights. Export a native model or training checkpoint, then pass the SafeTensors file and its matching tokenizer to `generate`:

```powershell
.\build\ama10.exe export-safetensors .\checkpoints\ama10-bpe-v0.3.ama .\build\ama10-bpe-v0.3.safetensors
.\build\ama10.exe generate .\build\ama10-bpe-v0.3.safetensors .\checkpoints\ama10-bpe-v0.3.ama.tok .\data\prompt.txt .\build\answer.bin 128
```

CPU inference can also use a checksummed per-row symmetric INT8 weight format. Converting a training checkpoint writes a `.q8` model and its matching `.tok` tokenizer sidecar; an F32-only model can be converted by supplying its tokenizer path:

```powershell
.\build\ama10_q8.exe quantize .\checkpoints\ama10-oasst1-multilingual-wiki-selected-v0.1.ama .\build\model.q8
.\build\ama10_q8.exe generate .\build\model.q8 .\build\model.q8.tok .\data\prompt.txt .\build\q8-answer.txt 128 0.2
.\build\ama10_q8.exe evaluate .\build\model.q8 .\build\model.q8.tok .\data\processed\multilingual_expansion_validation_144\en.txt
```

On the width-128, vocabulary-1024 selected checkpoint, Q8 reduced resident parameter storage from 1,311,744 to 335,360 bytes (74.4%). On 4,131 English validation predictions, token NLL changed from 5.112789 to 5.116218. Three end-to-end 256-token generation runs averaged 37.8 ms in Q8 and 55.4 ms in F32 on an Intel Core i5-13500H; process startup and checkpoint loading are included, so treat the timing as a local measurement rather than a general throughput claim. See [the experiment log](docs/experiments.md).

Use `evaluate` to score every next-token target in a UTF-8 text file, carrying recurrent state across 512-token reads. It reports token NLL and approximate bits per source byte so different tokenizer vocabularies can be compared. This is distinct from the trainer's fast 512-token prefix diagnostic:

```powershell
.\build\ama10.exe evaluate .\checkpoints\ama10-multilingual-expansion-gated-v0.1.ama .\checkpoints\ama10-multilingual-expansion-gated-v0.1.ama.tok .\data\processed\multilingual_expansion_validation\ja.txt
```

To compare adaptive inference with the fixed two-step baseline on the same full stream, append `adaptive-max-depth halt-threshold`; the command reports fixed two-step and fixed maximum-depth baselines alongside adaptive token NLL, mean recurrent steps, and CPU evaluation time. Append `state-reset-period` (or supply it as the only optional value) to compare continuous state with periodic resets. For example, `.\build\ama10.exe evaluate model.ama model.ama.tok validation.txt 4 0.8 24`.

Use `ama10 evaluate-documents <checkpoint> <tokenizer> <form-feed-pages-file>` to compare state carry across documents with a reset at each page while scoring the same within-page predictions. `scripts/prepare_wikipedia_eval_pages.ps1 -Language en` builds an exact-boundary input from the cached Wikipedia source titles. The included [split script](scripts/split_multilingual_validation.ps1) reconstructs per-language held-out files from the retrieval manifest and combined validation text.

The C audio frontend reads 16 kHz PCM16 mono/stereo RIFF WAV, folds stereo to mono, and produces 40 causal log-mel features from 25 ms Hamming-windowed frames every 10 ms. `ama10_audio_transcribe_train` trains on every acoustic frame with a CTC sequence objective and a 40-to-32 audio encoder; inference uses an 8-prefix CTC beam decoder with optional blank-score penalty and training-manifest bigram fusion. `ama10_score_text` reports normalized WER/CER. The CTC pilot remains poor: more data and byte-level targets both lowered CTC loss without yielding useful held-out transcripts.

The C audio library provides log-domain CTC forward-backward; `ama10_train_soft_multimodal_sequence` backpropagates its normalized framewise class posteriors through the shared recurrent core and dense input encoder. One-hot soft-target updates match the existing hard-target path, and finite differences validate the CTC posteriors against the sequence-loss gradient. The original 16/4 LibriSpeech pilot reached validation NLL 5.33 per target, but beam decoding was unusable. The polite, revision-pinned HF expansion now has 32 train/eight validation clips with speaker-disjoint splits. On this expanded set, both sequential and deterministic per-epoch shuffled 2,500-update CTC runs reduced validation NLL from 67.84 to 5.32/5.30 per target. The shuffled checkpoint's fixed penalty-4 decoding scored pooled WER 0.939 and CER 0.832 on all eight clips; default decoding remained nearly entirely blank (WER 1.000/CER 0.998). These tiny-set results are mixed and still far from usable ASR. See [the experiment log](docs/experiments.md) and [audio data provenance](data/README.md).

Supervised fine-tuning is available as `ama10_sft_train`: it reads escaped UTF-8 prompt/response TSV, masks loss on prompt tokens, can initialize from an existing text checkpoint, and saves a checkpoint plus tokenizer sidecar. Optional `--preserve-from teacher.ama --preserve-weight 0.25` blends teacher next-token distributions into SFT: response targets mix teacher probabilities with the gold token, and prompt positions use teacher distributions. Example: `ama10_sft_train.exe train.tsv validation.tsv 1000 sft.ama 0.00003 --init pretrained.ama --preserve-from pretrained.ama --preserve-weight 0.25`. The OpenAssistant preparation script filters reviewed English user/assistant pairs and holds out whole conversation trees. Its 200/50-pair pilot lowered response-token NLL, but free generation remained incoherent; at a higher learning rate it collapsed to repeated NUL bytes. This is training-pipeline evidence, not successful assistant behavior.

For a language-conditioned continuation, `.\scripts\prepare_multilingual_sft.ps1` turns the revision-attributed Wikipedia lead cache into 144 train and 36 held-out pairs, balanced across English, Spanish, French, Japanese, Chinese, German, Portuguese, Russian, and Arabic (16/4 per language). The source-page split is preserved and the manifest maps each TSV row to its page ID, revision, URL, and CC BY-SA 4.0 attribution. Compatible pretraining checkpoint widths from 8 to 4096 are supported. Example: `.\build\ama10_sft_train.exe .\data\processed\multilingual_sft\train.tsv .\data\processed\multilingual_sft\validation.tsv 500 .\checkpoints\ama10-multilingual-9lang-144p-sft-v0.1.ama 0.00003 --init .\checkpoints\ama10-multilingual-9lang-144p-from-scratch-v0.2.ama`. On this expanded dataset, a fresh 500-update pretrain at learning rate 1e-4 lowered held-out token NLL from 7.016 to 4.746. A 500-update SFT pass lowered held-out response NLL from 8.173 to 6.574 and improved full-stream bits-per-byte in eight of nine languages; Arabic regressed. The script-selection probe passes six languages and fails Japanese, Chinese, and Russian. The supervisor rejects this SFT checkpoint because Arabic NLL regressed and generation remains repetitive. See [the experiment log](docs/experiments.md) for measurements and checkpoint hashes.

Teacher-to-student distillation is available as `ama10_kd_train.exe teacher.ama train.txt validation.txt updates student.ama [student-width=16] [learning-rate=0.0001]`. It reuses the teacher tokenizer, builds fingerprinted token caches, forms full-vocabulary teacher softmax targets in C, and trains a narrower C student through soft-target BPTT. A real multilingual expansion run (32-wide teacher → 16-wide student, 1,000 updates) reduced held-out teacher/student KL 0.053928→0.024569 and student gold-token NLL 6.958796→6.936258. This establishes a working distillation path, not parity in generation quality. The checkpoint and complete run notes are in [the experiment log](docs/experiments.md).

Low-rank output-head adaptation is available as `ama10_lora_train.exe base.ama train.tsv validation.tsv updates adapter.lora [rank=8] [learning-rate=0.0003]`. It freezes the base model, trains rank-r factors in C with Adam, and saves a separately checksummed adapter bound to the base checkpoint and tokenizer fingerprints. Apply it during CPU generation by adding the adapter path after the repetition penalty: `ama10.exe generate base.ama base.ama.tok prompt.txt output.txt 128 cpu 0.5 adapter.lora`. OpenCL adapter inference is not implemented yet. The first 1,000-update rank-8 OASST run reduced response-token NLL only 6.941806→6.935594; held-out generation remained incoherent, so this is pipeline validation rather than an assistant-quality result.

The C checkpoint supervisor compares compatible checkpoints on a streamed validation file, then generates a fixed UTF-8-safe continuation from each. It accepts a candidate only if aggregate NLL stays within tolerance, repetition does not rise by more than 0.05, generated bytes are valid UTF-8, and repeated 4-grams stay at or below 0.75 by default. Optionally pass `language=validation.txt;...` and a per-language NLL regression tolerance to enforce named language gates. For recognized language codes, it also gates the candidate continuation's expected Unicode script fraction (0.50 by default, configurable) and records it in JSON; this is a script-selection diagnostic, not a measure of meaning or grammar. The full CLI is `ama10_supervisor.exe reference.ama candidate.ama validation.txt report.json selected.ama [nll-tolerance] [max-repeat4] [en=en.txt;de=de.txt;...] [max-language-nll-regression=0.1] [min-expected-script-fraction=0.5]`.

An optional Cloudflare Workers AI teacher adds a pairwise quality gate after the deterministic checks. Put the API token alone in the root `api.key` file (the path is Git-ignored), set `CLOUDFLARE_ACCOUNT_ID`, and explicitly enable requests with `AMA10_USE_CLOUDFLARE_TEACHER=1`. The only allowed teacher model is OpenAI `@cf/openai/gpt-oss-20b`, hosted through Workers AI; the C client rejects other model IDs so mainland-China providers cannot be selected through an environment variable or CLI argument. The reference and candidate continuations are sent with the shared prompt, and only a valid JSON judgment preferring B accepts the candidate. Missing configuration, HTTP errors, or malformed judgments fail closed. The build integration tests force the teacher off and make no external requests. `ama10_teacher_judge` uses the same fixed model for manual pairwise judgments.

The C tool runtime accepts `<|tool_call|>name<TAB>arguments<|end_tool_call|>` blocks and currently allowlists a safe arithmetic calculator and bounded local text search; it never runs shell commands. `ama10_tool_chat.exe checkpoint.ama tokenizer.tok prompt.txt transcript.txt [max-turns=3] [tokens-per-turn=128] [repetition-penalty=0.5]` feeds tool results back into the model context. `scripts/prepare_calculator_tool_sft.ps1` builds deterministic synthetic arithmetic traces for response-only SFT. The 256/64-example run reduced validation response NLL 6.915634→5.493999 over 6,000 updates, but free generation still did not emit a tool call. The runtime is exercised with a known calculator call; learned autonomous tool selection remains unproven.

Text generation accepts an optional per-token frequency penalty after the backend argument: `ama10.exe generate model.ama model.ama.tok prompt.txt output.txt 128 cpu 0.5`. This can reduce repeated-token collapse; it does not make incoherent outputs meaningful.

An optional OpenCL C backend runs recurrent inference on a GPU. Build it with the NVIDIA OpenCL headers/import library available under `CUDA_PATH`, then append `opencl` after the token count:

```powershell
.\scripts\build.ps1 -OpenCL -Test
.\build\ama10.exe generate .\checkpoints\ama10-bpe-v0.3.ama .\checkpoints\ama10-bpe-v0.3.ama.tok .\build\prompt.txt .\build\answer.bin 128 opencl
```

The tests compare GPU logits, recurrent state, adaptive early halting, and depth counters against CPU results, including persistent-stream parity. Greedy decoding keeps recurrent state on the device and transfers only next-token logits per step; call `ama10_opencl_reset` before using the stream API. Run `.\build\ama10_opencl_bench.exe 256 4096 500` for a warm-state speed comparison. At width 32 / vocabulary 1,024, the RTX 4050 inference path remains slower than CPU; speedups at larger inference shapes have not been established. Training can opt into the experimental `--opencl-adamw` hybrid path described above; the path routes recurrent forward propagation and BPTT through OpenCL but remains slower than CPU in measured end-to-end runs.

## Image-caption seed experiment

The rights-traceable Commons seed contains three training photos and one held-out photo. Fetch it politely (requires `ffmpeg`) and train a fresh shared-core image-caption model with:

```powershell
.\scripts\fetch_commons_captioned_images.ps1
.\scripts\build.ps1
.\build\ama10_image_caption_train.exe data\raw\commons_captioned_seed\manifest.tsv 1000 checkpoints\ama10-commons-seed
```

The trainer uses 16 RGB image patches, a trainable dense encoder, and causal caption loss. It reports held-out caption token NLL before and after training and reload-checks the single .amm bundle containing model weights, tokenizer, optimizer moments, recurrent state, training progress, and image encoder. The four kingfisher examples are a plumbing and tiny-sample overfit check, not evidence of general image understanding.

Resume to a larger total update count with the same dataset and settings:

```powershell
./build/ama10_image_caption_train.exe data/raw/commons_captioned_seed/manifest.tsv 1200 checkpoints/ama10-commons-seed --resume checkpoints/ama10-commons-seed.amm
```

Resume checks the image/caption split hashes and all recorded training settings before continuing.

The experimental inverse text-to-image path quantizes non-overlapping 16×16 RGB patches into a 64-entry train-only codebook, then trains the shared AMA core to predict a variable-length patch-code sequence from a language-tagged caption. The generator accepts requested output dimensions and emits P6 PPM. The codebook is still low capacity and produces patch mosaics. Train and generate with:

```powershell
.\build\ama10_image_gen_train.exe data\raw\commons_captioned_diverse\manifest.tsv 1000 checkpoints\ama10-commons-text-to-image-vq64-v0.3.ama
.\build\ama10_image_gen_train.exe data\raw\commons_captioned_diverse\manifest.tsv 5000 checkpoints\ama10-commons-text-to-image-vq64-v0.3.ama --resume
Set-Content -NoNewline build\imggen-prompt.txt 'A RED FOX IN SWEDEN'
.\build\ama10_image_gen_generate.exe checkpoints\ama10-commons-text-to-image-vq64-v0.3.ama build\imggen-prompt.txt build\generated-image.ppm en 0.8
```

The variable-resolution generator accepts optional width and height after temperature, for example `... en 0.8 1080 640`. A fresh patch-16, 64-center checkpoint trained for 1,000 updates and resumed to 5,000 on 14 local train rows and 6 validation rows; held-out image-code NLL fell 5.604289→5.196232. Its 1080×640 output is an unrecognizable patch mosaic. A 128-center comparison raised matched-size PSNR to 4.867 dB, but worsened code NLL to 5.451 and still produced a mosaic. These experiments do not establish semantic image generation.

This 11-image Commons set is only a pipeline experiment. After 5,000 updates, validation image-code NLL fell 5.583538→3.842064. A sampled held-out fox prompt scored PSNR 4.331 dB and native 8×8-window SSIM 0.066 against its reference; outputs remain noisy and do not demonstrate semantic image generation. See [the experiment log](docs/experiments.md) for the reset-state correction, checkpoint hashes, and the earlier greedy mode-collapse result.

Compare two P6 images with the native C metric utility: `.\build\ama10_image_score.exe reference.ppm generated-image.ppm`. It reports global RGB PSNR and mean 8×8-window RGB SSIM.

Generate a caption from a P6 PPM image using the saved encoder and shared recurrent state:

```powershell
./build/ama10_image_caption_generate.exe checkpoints/ama10-commons-seed.amm data/raw/commons_captioned_seed/0004.ppm build/caption.txt 29
```

The output is filtered to valid UTF-8. This pilot currently produces weak captions; the command verifies the inference path, not caption quality.

For the 11-image set, run the polite fetch-and-assembly script, add the available English, French, Spanish, and Japanese captions in one batched metadata request, then train from scratch:

```powershell
./scripts/assemble_commons_captioned_diverse.ps1
./scripts/expand_commons_caption_languages.ps1
./build/ama10_image_caption_train.exe data/raw/commons_captioned_diverse/manifest.tsv 1000 checkpoints/ama10-commons-multiclass
./build/ama10_image_caption_generate.exe checkpoints/ama10-commons-multiclass.amm data/raw/commons_captioned_diverse/0006.ppm build/fox-caption.txt 29 en
```

The three held-out images cover a fox, Vancouver skyline, and a kingfisher; available captions for each image stay within its split. The current generated strings are UTF-8-valid but incoherent; see the per-run scores and samples in [the experiment log](docs/experiments.md).

The multilingual manifest has four tab-separated fields: image path, caption, image split, and caption language. Captions are conditioned with a `<lang:xx>` prefix; pass `en`, `fr`, `es`, or `ja` as the optional final generator argument. The small pilot has only English/French/Spanish validation captions, and Spanish has one held-out caption.

## Audio-transcription pilot

The Hugging Face collector verifies the dataset card license, resolves a commit, checks each audio asset URL against that commit, uses the dataset viewer for selected rows, waits at least 30 seconds between every Hub/API/audio request, honors `Retry-After` and rate-limit reset hints, and records speaker/chapter/split IDs plus source and converted-WAV SHA-256 hashes. It appends offset batches without re-downloading existing row IDs. The current set has 48 `train.clean.100` clips from five speakers and 12 separate `validation.clean` clips from four held-out speakers (about 22.3 MB total):

```powershell
.\scripts\fetch_hf_librispeech_rows.ps1 -TrainRows 8 -ValidationRows 2
.\scripts\fetch_hf_librispeech_rows.ps1 -TrainRows 8 -ValidationRows 2 -TrainOffset 10000 -ValidationOffset 200
.\scripts\fetch_hf_librispeech_rows.ps1 -TrainRows 16 -ValidationRows 4 -TrainOffset 20000 -ValidationOffset 1000
.\scripts\fetch_hf_librispeech_rows.ps1 -TrainRows 16 -ValidationRows 4 -TrainOffset 24000 -ValidationOffset 2000
.\scripts\build.ps1 -Test
.\build\ama10_audio_transcribe_train.exe data\raw\librispeech_hf_micro\manifest.tsv 2000 checkpoints\ama10-librispeech-ctc-byte256-48x12-v1.1
.\build\ama10_audio_transcribe_generate.exe checkpoints\ama10-librispeech-ctc-byte256-48x12-v1.1.amm data\raw\librispeech_hf_micro\audio\libri_0057.wav build\libri-hypothesis.txt 512 en 2.5
# Optional transcript bigram fusion; currently experimental and off by default.
.\build\ama10_audio_transcribe_generate.exe checkpoints\ama10-librispeech-ctc-byte256-48x12-v1.1.amm data\raw\librispeech_hf_micro\audio\libri_0057.wav build\libri-hypothesis-lm.txt 512 en 2 data\raw\librispeech_hf_micro\manifest.tsv 0.1 0.0
```

This remains a small pipeline experiment, not a representative ASR benchmark. On the 48/12 split, the 3,800-update BPE model reached held-out CTC NLL 5.176271 per target, but fixed penalty-4 decoding scored WER/CER 0.9674/0.8552. A 2,000-update byte-level run lowered per-character CTC NLL 27.85→2.93; the blank-penalty sweep still found no useful WER setting (the lowest CER was 0.872 with WER 1.32). The optional bigram prior failed grouped validation. Normalization lowercases ASCII and removes punctuation before scoring. Full results and checkpoint hashes are in [the experiment log](docs/experiments.md).

## Experimental text-to-speech

This prototype fits a 64-centroid codebook over train-only log-mel frames, then trains the shared AMA core to predict audio-code IDs after a text prompt. At inference, predicted mel centroids are rendered to a mono 16 kHz WAV using an approximate Griffin–Lim phase reconstruction. Train and generate with:

```powershell
.\build\ama10_tts_train.exe data\raw\librispeech_hf_micro\manifest.tsv 1101 checkpoints\ama10-librispeech-tts-vq64-48x12-v0.1.ama
Set-Content -NoNewline build\tts-prompt.txt 'THE BLUE BIRD IS SINGING OUTSIDE'
.\build\ama10_tts_generate.exe checkpoints\ama10-librispeech-tts-vq64-48x12-v0.1.ama build\tts-prompt.txt build\speech.wav en
```

The trainer writes the model, tokenizer, and codebook sidecars (`.tok` and `.vq`); resume with `--resume` and the existing model path. On the 48/12 LibriSpeech slice, 1,101 updates reduced validation loss from 5.735589 to 4.634959. The model did not learn to emit EOS, so generation uses a training-derived duration estimate. It produced structurally valid PCM16 files at 16 kHz for both a held-out transcript (about 10.96 s) and the prompt above (about 2.18 s). These checks establish the training and WAV-generation path only; intelligibility and naturalness have not been evaluated, and the codebook/Griffin–Lim reconstruction is highly approximate.

Run `.\scripts\evaluate_multilingual_probe.ps1 -ModelPath model.ama -TokenizerPath model.ama.tok -FailOnThreshold` for nine fixed language prompts; add `-AdapterPath adapter.lora` to evaluate a LoRA adapter. The JSON report records valid UTF-8, repetition, and expected script-block fraction (Latin for English/Spanish/French/German/Portuguese, kana-or-CJK for Japanese, CJK for Chinese, Cyrillic for Russian, Arabic for Arabic); each output must meet the 0.5 expected-script fraction by default. This is a shallow language-selection diagnostic, not a grammar or semantic-quality score.

## Scope still missing

Broader paired multimodal data and batching, larger multilingual speech training, high-quality TTS and image generation, broad streaming ingestion and corpus deduplication, standardized language/task evaluation, faster OpenCL training and adapter inference, reliable model-initiated tool use, RLAIF, automated training scheduling, broader SwanLab live training integration, quantized training and on-demand modality loading, and long-running capability experiments remain future work. The C image path decodes PPM/JPEG/PNG/BMP/TGA, supports high-resolution overlapping tiles for captioning, and has a variable-resolution patch-code generator. Its outputs remain mosaics, and the available Commons image-caption corpus is tiny. SafeTensors support currently covers AMA-10 F32 model weights only; it does not include tokenizer, optimizer, training state, or arbitrary model architectures. See [the algorithm note](docs/algorithm.md), [experiment log](docs/experiments.md), and [research log](docs/research.md).

A longer continuation of the 144-page multilingual pretraining run to 20,000 updates (about five passes over the stream) improved full-stream held-out bits per byte in all nine languages; the fixed language-generation probe passed seven languages with the configured frequency penalty, while raw greedy generation remained repetitive. A 500-update balanced SFT from this checkpoint lowered response NLL and passed eight of nine probe languages, but the supervisor rolled it back because raw repeated 4-grams rose from 0.424 to 0.640. A lower-weight teacher-preserved SFT also improved per-language losses but raised repetition further (0.720), and was rejected. See [the experiment log](docs/experiments.md) for per-language values and checkpoint hashes; those trials retained the 20,000-update pretraining checkpoint at that point.

The current SFT trainer deterministically shuffles the training rows once per epoch using a seed derived from the train-file fingerprint, so language-grouped input files do not create language-grouped update blocks. `scripts/prepare_oasst1_multilingual_sft.ps1` builds a tree-disjoint multilingual conversation set from the pinned OASST1 export and emits per-language validation streams. The initial OASST-only pass improved Russian generation but failed the Arabic NLL and raw-repetition gates; the supervisor retained the pretraining checkpoint at that stage. See the [experiment log](docs/experiments.md) and [data notes](data/README.md) for the exact split, counts, and hashes.

The accepted multilingual continuation is [ama10-oasst1-multilingual-wiki-selected-v0.1.ama](checkpoints/ama10-oasst1-multilingual-wiki-selected-v0.1.ama), selected by the C supervisor on combined OASST1 and Wikipedia validation. It improves the OASST candidate's aggregate NLL and raw repetition while passing all nine named validation-loss gates. The fixed script probe passes eight languages but still fails Arabic script adherence; see [the supervisor report](build/oasst1-wiki-adapt-supervisor.json) and [experiment log](docs/experiments.md).








