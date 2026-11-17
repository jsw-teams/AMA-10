# AMA-10 data acquisition and experiment plan

## Source management

`data/pipeline/datasets.json` is the canonical registry of all 26 requested sources. It records the official landing point, known native shard form, acquisition route, version uncertainty, and whether access or source identity is unresolved. An unresolved or pending record remains incomplete. Before a shard is fetched, record its immutable revision or release identifier, official URL, expected size/checksum when published, and local path in `data/pipeline/status.jsonl`.

The status file is a local append-only JSON Lines index, not a network database. It lives at `data/pipeline/status.jsonl`; no database host, port, username, or connection string is configured. A 503 from Hugging Face is a temporary HTTP response and has no database connection associated with it.

Each shard lifecycle should be represented with events: `planned`, `downloading`, `downloaded`, `validated`, `processed`, `tokenized`, `consumed`, `failed_retryable`, or `failed_terminal`. The C11 `ama10_pipeline_ledger` utility appends validated JSONL events with a dataset ID, shard ID, UTC time, source revision/URL, local path, byte counters, SHA-256, document count, valid token count, and consumed-token count. Windows appends are exclusively locked and flushed. It rejects unknown dataset IDs/states, malformed SHA-256 strings, validation bytes above acquired bytes, and consumed tokens above valid tokens. Example syntax:

```powershell
.\build\ama10_pipeline_ledger.exe data/pipeline/status.jsonl fineweb_edu sample/100BT/000_00004.parquet planned v1.4.0 https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/<commit>/sample/100BT/000_00004.parquet data/cache/fineweb_edu/000_00004.parquet - 0 0 0 0 0
```

Use an immutable commit in place of `<commit>` before scheduling the shard. A partial file is not counted as downloaded/validated or consumed. A recovered download must validate the final source hash before processing.

On Windows, the C11 `ama10_fetch_shard` executable downloads one HTTPS shard serially. It requires at least 30 seconds between requests, honors `Retry-After` and Hugging Face `RateLimit` reset values, accepts only HTTPS, checks Hub `resolve` URLs against the supplied immutable commit, resumes only when an ETag/Last-Modified validator is available and a 206 Content-Range starts at the saved byte offset, enforces a caller-provided per-shard byte ceiling, computes SHA-256, optionally enforces the official expected SHA-256, atomically commits the final file, and appends `downloaded` only after commit. Usage is:

```powershell
.\build\ama10_fetch_shard.exe data/pipeline/status.jsonl fineweb_edu data/CC-MAIN-2016-07/train-00001-of-00013.parquet 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9 https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9/data/CC-MAIN-2016-07/train-00001-of-00013.parquet data/cache/fineweb_edu/train-00001-of-00013.parquet 940520f9f41073006e4ec95473761cbe66a84f31316c106c3d2d3eda9eb56b89 3000000000 30
```

## Initial text mixture

The first sampler target is 38% English, 29% Chinese, 21% multilingual, and 12% code/math, measured in valid training tokens. These are sampling targets, not acquisition completion claims. The sampler must report actual consumed tokens per dataset and language; later mixture changes require recorded evaluation evidence. Image and speech sources are tracked by modality counts and training steps separately rather than silently folded into text-token percentages.

## Processing path

1. Resolve the official release/Hub revision and inspect the official shard listing; do not crawl rendered dataset pages for bulk data.
2. Download one shard at a time, serially, through the existing polite request policy. Preserve the current 30-second minimum spacing for Hugging Face calls and honor `Retry-After` and rate-limit reset headers. Resume only when the server supports byte ranges and the partial file's validator matches; otherwise restart that shard after recording the failed partial.
3. Verify published checksums, then record local SHA-256 and source byte count. Keep raw and derived files under the 100 GB working cache budget; evict only fully validated raw shards after their derived products and provenance are durable.
4. Stream native records into UTF-8 validation, document normalization, language tagging/identification, configured quality filters, and cross-source deduplication. Record each rejection reason and pre/post-filter counts. Tokenize incrementally with the exact training tokenizer and produce deterministic bounded shards.
5. Expose only fully committed processed shards to the trainer. The sampler reads the ledger and cannot count a shard twice after restart. Each checkpoint records shard IDs and token offsets it consumed.
6. Keep image payloads, OCR annotations, video references, and audio streams linked by stable sample IDs. Validate media headers, decode success, dimensions/duration, and annotation references before exposing records to multimodal training.

The current C11 dataset module caches tokenized local text files; it does not yet implement a general native Parquet/JSONL/WebDataset acquisition and decode pipeline. Therefore no entry in the requested catalog is represented as production-ready until that functionality and its end-to-end shard evidence exist. Official Parquet processing needs a C-compatible reader dependency or an explicitly documented C ABI; compressed JSONL needs a C-compatible Zstandard dependency; neither is silently emulated by downloading a small sample.

## Experiment sequence and evidence

- Text pretraining: causal next-token objective, document-level train/validation separation, per-language NLL and bits-per-byte, UTF-8 checks, script probes, and repetition diagnostics.
- Chinese: report Simplified and Traditional Chinese separately; do not infer Traditional coverage from Simplified-only records.
- Mathematics/code: source-separated evaluations and source ablations, with token-level and task-level measures.
- Image: contrastive/alignment and caption/VQA/OCR objectives using committed image-text records; measure image counts, decode failures, and task metrics.
- Speech: FLEURS ASR/speech-text alignment, reporting hours, sample rate, CER/WER per language.
- Joint training: compare single-modality and mixed-modality steps, shared-weight gradient norms, throughput, hardware usage, and held-out capability changes.
- Post-training: SFT/distillation must identify the exact consumed source shards and teacher model. The configured teacher allowlist excludes mainland-China models.

Training reports must distinguish actual downloads, validated bytes, processed records, valid tokens, sampler consumption, parameter-update steps, and evaluation results. No source is complete based on its registration or on a small smoke test.

## Current status snapshot (2026-11-17)

All 26 designated sources are represented in the registry. At the snapshot time, FineWeb-Edu shard 1 had 2,358,493,358 raw bytes acquired and SHA-256 validated, then 759,857 unique documents were processed from 759,858 rows. Its deterministic 5% split contains 721,641 train and 38,216 validation documents. Tokenization/training was still running; valid and consumed token counts remain zero until the trainer commits token caches and updates. All other designated corpora remain pending, gated, or unresolved. The pre-existing LibriSpeech collector completed its already-running polite request and the local corpus now contains 80 paired examples (64 train / 16 validation) in `data/raw/librispeech_hf_micro/`; this is an existing speech smoke corpus, not FLEURS and not a substitute for the designated speech source. No designated dataset had supplied a training update at the time the earlier LibriSpeech status note was written; the FineWeb-Edu training job described below is now in progress.


## 2026-11-17 — first designated shard training started

The official FineWeb-Edu CC-MAIN-2016-07/train-00001-of-00013.parquet shard was downloaded at its pinned Hub revision and validated against the published SHA-256. The C11 DuckDB C-ABI processor streamed the full Parquet shard, normalized UTF-8 text, removed one exact duplicate within the shard, and emitted an intact processed corpus plus deterministic document-level train/validation files. The source score column was null for all rows, so no score threshold was applied. Train/validation byte counts and SHA-256 values are in data/processed/fineweb_edu/sources.json.

An actual ama10 train job has started on the complete train and validation text files, with a 1,024-token tokenizer vocabulary, width 128, AdamW, accumulation 2, and a 10,000-update target. The trainer first encodes the full files to persistent token caches, then performs genuine next-token gradient updates and evaluates held-out NLL. At this report point the process is still building its training token cache; no valid-token count, update count, or loss result is yet claimed. This is a small current-architecture experiment, not the requested 1.45B model or full multi-source production run.


### Automotive frame manifest

Automotive data entries are additive candidates. Frame-level records should follow `data/pipeline/vision-frame.schema.json`, retaining dataset revision, sequence ID, frame order, nanosecond timestamp, camera ID, RGB image path and dimensions, calibration, annotation, pose references, and train/validation/test split. Video remains in source frame order; do not collapse a sequence into an unordered image list. Group weights for general imagery, automotive vision, document/OCR, multimodal instruction, and speech remain unset until independent capability measurements justify a mix.
