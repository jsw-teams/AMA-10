# AMA-10 experiment log

## 2026-11-17 — multilingual held-out and language-following baseline

**Checkpoint:** `checkpoints/ama10-multilingual-expansion-gated-v0.1.ama`, 7,200 AdamW updates, width 32, vocabulary 1,024. Used the existing disjoint per-language validation files under `data/processed/multilingual_expansion_validation/`; `ama10.exe evaluate` streamed each complete file. Bits per byte are derived from token NLL normalized by source bytes, so compare them across languages only as tokenizer/model diagnostics, not as a cross-language quality ranking.

| Language | Validation bytes | Tokens | Token NLL | Bits/byte |
|---|---:|---:|---:|---:|
| English | 4,446 | 1,957 | 6.922720 | 4.393905 |
| Spanish | 4,982 | 1,972 | 6.916238 | 3.947548 |
| French | 3,385 | 1,443 | 6.916669 | 4.250872 |
| Japanese | 1,659 | 751 | 6.920818 | 4.513847 |
| Chinese | 1,486 | 803 | 6.953112 | 5.413888 |

**Language-following probe:** greedy CPU generation with frequency penalty 0.5, 48 new tokens, using these fixed prefixes: `Respond in English: My favorite season is`; `Responde en español: Mi estación favorita es`; `Réponds en français : Ma saison préférée est`; `日本語で答えてください。好きな季節は`; `请用中文回答：我最喜欢的季节是`. All five outputs were valid UTF-8, with distinct-token ratio 1.0 and repeated-4-gram ratio 0.0, but script counts were mixed in every continuation: English 40 Latin / 19 CJK / 9 kana; Spanish 52 Latin / 18 CJK / 7 kana / 1 Arabic; French 58 Latin / 18 CJK / 5 kana / 1 Cyrillic; Japanese 54 Latin / 12 CJK / 5 kana; Chinese 45 Latin / 27 CJK / 4 kana. Manual inspection confirmed nonsensical mixed-script fragments, so the model fails this language-following probe. These results are a small diagnostic baseline, not a standardized language benchmark; the current validation corpus is limited to Wikipedia lead extracts.

## 2026-11-17 — matched multilingual width scaling

Trained a width-128 model from random initialization for 7,200 AdamW updates on the same 63,381-byte training file and 15,976-byte validation file as the width-32 baseline. Both use seed 10, vocabulary 1,024, depth 2, sequence length 24, accumulation 2, learning rate `1e-4`, inverse-square-root decay 0.1, and the same tokenizer sidecar (SHA-256 `2B259B529C597831B5E0FD6AD0AFAA5020480296121325DBC364554ACEB56363`). The width-128 run was checkpointed periodically and resumed from update 4,900 to 7,200 after interruption; it did not load width-32 model weights. Trainer's 512-token prefix diagnostics changed from train/validation NLL 6.587181/6.552926 before training to 6.449827/6.400399 after training.

Reproduce from fresh random weights with `.\build\ama10.exe train .\data\raw\multilingual_expansion_train.txt .\data\raw\multilingual_expansion_validation.txt 7200 0.0001 1024 .\checkpoints\ama10-multilingual-expansion-width128-v0.1.ama --width 128 --accumulation 2 --lr-decay 0.1`. The CLI defaults to seed 10 and an 8 MiB tokenizer sample. Evaluate each language with `ama10.exe evaluate <checkpoint> <checkpoint>.tok data/processed/multilingual_expansion_validation/<lang>.txt`; generation prompts and their probe results are recorded above.

Full-stream evaluation uses the same held-out files and checkpoint weights as the preceding width-32 baseline:

| Language | Width-32 bits/byte | Width-128 bits/byte | Change |
|---|---:|---:|---:|
| English | 4.393905 | 4.018802 | -0.375103 |
| Spanish | 3.947548 | 3.630832 | -0.316716 |
| French | 4.250872 | 3.956562 | -0.294310 |
| Japanese | 4.513847 | 4.347376 | -0.166471 |
| Chinese | 5.413888 | 5.257069 | -0.156819 |

Greedy 48-token continuations with the same five language prompts, CPU backend, and frequency penalty 0.5 remained nonsensical. Outputs were valid UTF-8 and had no repeated 4-grams, but their script counts were dominated by Latin in every case (English 94 Latin / 1 kana; Spanish 91 / 1; French 87 / 1; Japanese 93 / 1; Chinese 96 Latin / 0 CJK / 1 kana). This controlled width increase improves next-token compression on all five small validation files but does not produce language-following behavior. The checkpoint SHA-256 is `84BC5547EF115ED84F75C153C5555D17A72669CDB27160D860945A6FDBD70A3C`; the checkpoint is 3,948,404 bytes.

## 2026-11-17 — width-128 OASST SFT and absolute repetition gate

The SFT trainer previously rejected initialized models unless width and input width were exactly 32. It now accepts compatible model widths 8–4096 while requiring `input_width == width` and exact tokenizer vocabulary agreement, so the post-training stage can use the width-128 pretrain checkpoint. `scripts/build.ps1 -Test` passed after the change. A real 100-additional-update continuation on the reviewed English OASST1 split used the width-128 checkpoint at update 7,200, learning rate `3e-5`, 1,000 training pairs, 200 tree-held-out validation pairs, and the inherited tokenizer and optimizer state. Validation response-token NLL improved 6.401765→6.156682 over 86,713 response tokens.

Full-stream multilingual evaluation after SFT showed small English/Spanish/French improvements but Japanese/Chinese regressions:

| Language | Pre-SFT bits/byte | Post-SFT bits/byte | Change |
|---|---:|---:|---:|
| English | 4.018802 | 3.889408 | -0.129394 |
| Spanish | 3.630832 | 3.569531 | -0.061301 |
| French | 3.956562 | 3.887259 | -0.069303 |
| Japanese | 4.347376 | 4.402139 | +0.054763 |
| Chinese | 5.257069 | 5.280694 | +0.023625 |

The same 48-token language prompts still produced garbled outputs; Japanese and Chinese continuations were almost entirely Latin-script. On the combined multilingual validation file, the supervisor measured reference/candidate NLL 6.425050/6.342568 and repeated-4-gram ratios 0.992/0.992. Its previous relative-only rule accepted this equally collapsed candidate. That failure motivated an absolute candidate repetition ceiling (default 0.75), in addition to the existing +0.05 relative tolerance and NLL gate. The integration suite now verifies both identical-checkpoint acceptance under a passing ceiling and rollback when an identical fixture exceeds a deliberately strict ceiling. Re-running the real OASST candidate with the default ceiling selected the pre-SFT reference and reported reason `candidate_exceeds_absolute_repetition_limit`. The candidate checkpoint remains an experiment artifact, not the selected model.

## 2026-11-17 — language-conditioned Wikipedia SFT probe

Added `scripts/prepare_multilingual_sft.ps1`, which builds a 40/10 train/validation TSV from the revision-pinned per-language Wikipedia cache. It preserves the topic-aligned page split across English, Spanish, French, Japanese, and Chinese (8 training pages and 2 disjoint held-out pages per language), caps response excerpts at 2,000 characters, emits per-language TSVs, and records row-to-page ID/revision/URL/license provenance. The generated combined files hash to train `08E5B8119CA3B23D8BAF5E6C6EB01391B65D06E904C16F17858F700998640008` and validation `B8B25AAA10ADC154562A67B4214BCB56B4C576E2810C3822A31E3DF2C3854A33`.

Fine-tuned the width-128 pretraining checkpoint from update 7,200 with translated language-specific questions and the matched article lead as the answer. The 100-update pass lowered mean held-out response-token NLL 6.501919→6.409848. A separate 500-update pass from the same pretraining checkpoint lowered it to 6.315261 across 5,378 held-out response tokens; it did not continue from the 100-update candidate. The 500-update checkpoint reports optimizer update 7,700 and has SHA-256 `63D7AD5D04ECDC50C5F1BA3F0CC3B3F3833C9D673772A90AE76C43EB3DC779A2`.

Full-stream held-out text metrics moved as follows after the 500-update pass:

| Language | Pre-SFT bits/byte | Post-SFT bits/byte | Change |
|---|---:|---:|---:|
| English | 4.018802 | 3.888250 | -0.130552 |
| Spanish | 3.630832 | 3.535046 | -0.095786 |
| French | 3.956562 | 3.845846 | -0.110716 |
| Japanese | 4.347376 | 4.314554 | -0.032822 |
| Chinese | 5.257069 | 5.167921 | -0.089148 |

Despite the consistent held-out compression gains, the five fixed 48-token generations remained garbled and predominantly Latin-script: Latin codepoints were 58/79 English, 57/78 Spanish, 60/79 French, 60/80 Japanese, and 58/79 Chinese. All were valid UTF-8 with repeated-4-gram ratio zero. This targeted SFT improves the measured losses, but does not demonstrate language following or semantic answering; the five-generation probe remains a failure.

Added `scripts/evaluate_multilingual_probe.ps1` to reproduce these five prompts and report per-language expected-script fraction. With the default 0.5 threshold, the 500-update model passes English, Spanish, and French script selection but fails Japanese and Chinese at 0.0 expected-script fraction. The machine-readable report is `build/width128-multilingual-sft500-probe/report.json`. The metric explicitly does not measure grammar or semantics.

## 2026-11-16 — CPU byte model, first full-corpus run

**Initialization and data:** deterministic random initialization, seed 10; 32 hidden/input width; 256 byte IDs; 20,544 trainable parameters; six full Wikipedia extracts for training (189,305 bytes) and two separate article extracts for validation (80,852 bytes). Source titles, revision IDs, and URLs are in `data/raw/wikipedia_sources.json`. Data license is CC BY-SA 4.0.

**Training:** C11 CPU executable built by GCC 16.2.0; 10,000 updates; AdamW learning rate `1e-4`, weight decay `0.01`; gradient norm clip 1; batch size 1; depth 2; 24-byte TBPTT chunks with recurrent state carried between chunks and reset at corpus wrap. This processes 240,000 targets (about 1.27 passes through the training bytes). Wall time was 11.371 s including evaluation, checkpoint write, and generation, or about 21.1k target bytes/s by this coarse measurement.

| Measurement | Before | After |
|---|---:|---:|
| Training prefix next-byte NLL (first 512 bytes) | 5.766956 | 2.667820 |
| Held-out page prefix next-byte NLL (first 512 bytes) | 5.758281 | 2.625086 |

These are narrow checks on an eight-article sample, not a standard benchmark or evidence of general language ability. The checkpoint is `checkpoints/ama10-v0.1.ama` (82,208 bytes; SHA-256 `B60402765D22290932CB2BCC468651D1DC9BE4C770C9AD7DD8A7AC0E72DBFD8D`). It stores weights and dimensions only; optimizer state and exact training cursor are not saved.

**Generation result:** greedy decoding produced 128 bytes beginning `re the the the the ...`. The continuation repeats a short phrase and is a failure for usable free generation. The model has not passed the requested continuity/repetition evaluation.

## 2026-11-16 — learned BPE and loadable inference checkpoint

The C tokenizer learned 256 merges from training articles only, yielding a 512-ID vocabulary with 256 byte fallback IDs. Training text encoded to 89,839 tokens and validation to 39,858 tokens. The model therefore had 36,928 parameters. It was freshly initialized with seed 10 and trained for 10,000 updates at depth 2, width 32, sequence chunks of 24 tokens, AdamW learning rate `1e-4`, weight decay `0.01`, and gradient norm clip 1.

| Measurement | Before | After |
|---|---:|---:|
| Train token NLL (first 512 tokens) | 6.320324 | 5.273163 |
| Held-out article token NLL (first 512 tokens) | 6.335848 | 5.296598 |

Training plus BPE learning/encoding, evaluation, checkpoint writing, and generation took 28.096 s (about 8.5k update-target tokens/s for the complete command). Token NLL is a different unit from the byte-model NLL above. The new model file is 147,732 bytes (SHA-256 `5A2D130789275784D2E55FBC86DB005A25F639FBEA89BEA82184656B10159610`); its 3,892-byte tokenizer companion has SHA-256 `5C3AB1EC7CDD1FA28E380FF42BFA7541C7AC1C35AC47AA75C371F861F7EA4324`. A separate `generate` command loaded both files and emitted 64 tokens from a UTF-8 prompt.

**Generation result:** decoded output still repeats fragments such as `ine` and `reine`; the model again fails free-generation quality despite lower held-out NLL. Model/tokenizer files load for inference. Training checkpoint recovery was added afterward; this recorded experiment predates that format.

**Correctness work:** the first numerical-gradient test found that time steps were backpropagated in forward order; reverse-time BPTT fixed this. A subsequent evaluator/trainer parity check found that the trainer read state-vector columns before all entries were staged; separating state staging from matrix computation fixed it. The final test suite passes deterministic forward, reset, tiny-sequence overfit, train/eval parity, and finite-difference magnitude checks on one high-magnitude scalar from each of eight parameter groups. This is an initial gradient gate, not exhaustive gradient coverage.

## Known next improvements

Increase and diversify the licensed corpus, use larger evaluation spans and standard byte-level baselines, improve checkpoint/resume, investigate adaptive depth and state carry, add stochastic sampling only after measurement, and eliminate repetition through model/data/training changes validated on held-out text. Do not treat sampling tricks as a substitute for fixing the model's greedy collapse.

## 2026-11-16 — polite multilingual Wikipedia lead sample

**Acquisition:** `scripts/fetch_multilingual_wikipedia.ps1` retrieved two revision-pinned article leads per language in English, Spanish, French, Japanese, and Chinese. Each language was requested serially as one grouped API query, with `maxlag=5`, a 15-second pause between languages, and exponential retry handling that honors `Retry-After` and waits at least 60 seconds for 429/rate-limited responses. The script accepts an operator contact through `-Contact`; none was supplied for this run, so its User-Agent identifies the local research pipeline. The run completed without further 429 responses. `data/raw/multilingual_sources.json` records titles, revision IDs, retrieval time, page URLs, lead-only scope, and CC BY-SA 4.0 attribution. This is a tiny lead-introduction sample, not a broad multilingual corpus. Train and validation pages are disjoint within each language.

**Data:** train 5,364 bytes (SHA-256 `836A88E6B7D4C6D6B2133375EBFEAD5C28F99DB69AD08A981EBCADCEF0D50224`); validation 6,575 bytes (SHA-256 `E61C83CABD423A57A1C0FC40C7303D6F38D1D8614A2579E7563B653473DB06C2`). BPE budget 8 MiB, vocabulary 512, 2,393 train tokens, 3,980 validation tokens. Fresh seed-10 model, width 32, depth 2, 3,000 updates, sequence 24, accumulation 2, AdamW learning rate `1e-4`, weight decay `0.01`, inverse-square-root decay `0.1`; wall time 10.7 s.

| Measurement | Before | After |
|---|---:|---:|
| Train token NLL | 6.321599 | 6.211518 |
| Held-out token NLL | 6.354003 | 6.194930 |

Checkpoint `checkpoints/ama10-multilingual-v0.1.ama` SHA-256 `1E54601BDCFD69D08468411E8CB6C849EDA735847F473366BF6965D92D9D535B`; tokenizer companion SHA-256 `A4B6D4F41F5D9433965DE2F1FD0ABD3A70C5902BDB0EFAC0780F33CE55BF66F6`. Greedy 128-token sample had distinct-token ratio 0.6875 and repeated-4gram ratio 0.0000; it contained 42 invalid UTF-8 bytes. Before the decoding constraint described below, five 96-token language prompts had 17–37 invalid bytes and mixed scripts. The tiny multilingual sample still does not establish multilingual competence.

## 2026-11-16 — UTF-8-constrained decoding follow-up

Those invalid bytes motivated a text-decoding constraint. The C inference path now validates the prompt as UTF-8 and only considers token byte strings that continue a valid UTF-8 stream; on the final generation step it requires each code point to be complete. This applies to BPE tokens and byte-fallback IDs alike. Tokenizer encoding/decoding remains byte-exact; this is a text-generation constraint.

On the same checkpoint, 96-token greedy continuations from five language prompts all produced valid UTF-8 (zero invalid bytes each), verified by strict decoding. The outputs remain poor: repeated 4-gram ratios were English 0.1398, Spanish 0.7097, French 0.1828, Japanese 0.0538, and Chinese 0.6559. They visibly mix scripts and unrelated fragments. The constraint guarantees valid UTF-8 only; it does not improve meaning or language adherence.

**Verification:** `scripts/build.ps1 -OpenCL -Test` passed the CPU core/gradient/overfit checks, tokenizer UTF-8/emoji/NUL round trip, checkpoint resume parity, streamed dataset boundary check, generation-metrics tests, and OpenCL CPU/GPU parity on the RTX 4050 Laptop GPU. The existing suite has no dedicated constrained-decoder unit test; end-to-end output was checked on the five prompts with a strict UTF-8 decoder.

## 2026-11-16 — AdamW versus Muon hybrid pilot

Implemented an optional hybrid optimizer: Muon updates the four hidden matrices (`wx`, `wh`, `wm`, `wu`); AdamW updates embedding/output matrices and biases. The update uses 0.95 Nesterov momentum, five quintic Newton-Schulz steps, shape scaling, global gradient clipping, and decoupled weight decay. Auxiliary AdamW uses 0.015 of the Muon hidden-matrix learning rate. Optimizer selection is exposed as `--optimizer adamw|muon` and is stored in checkpoint v5; v4 checkpoints still load as AdamW. Both direct and accumulated training paths use the same optimizer implementation.

**Pilot protocol:** same seed 10, train/validation bytes and tokenization, vocabulary 512, width 32, depth 2, 24-token chunks, accumulation 2, 1,000 updates, inverse-sqrt decay 0.1. Each run started with train/validation NLL 6.321599 / 6.354003. A Muon hidden LR of `0.002` (auxiliary AdamW LR `3e-5`) gave NLL 6.184488 / 6.177555. AdamW at `3e-5` gave 6.286791 / 6.332454; AdamW at `1e-4` gave 6.239813 / 6.222978. The paired 3e-5 AdamW and Muon runs took 7.4 s total including build, training, evaluation, generation, and checkpoint writes. The Muon run improved this tiny held-out lead sample most among these settings, but this is not a sound optimizer ranking: it is one seed and corpus, only three learning-rate configurations, and the hidden/auxiliary parameter groups have different rates. No benchmark-level claim is made.

**Correctness:** CPU/OpenCL regression suite passed with a Muon finite-value/update test. A separate checkpoint test saved a one-step Muon state, reloaded it, and matched the next model weights, both moment arrays, and recurrent state bit-for-bit against uninterrupted training. A CLI resume also advanced the Muon pilot from update 1,000 to 1,100, and the earlier version-4 AdamW multilingual checkpoint still loaded for generation.

## 2026-11-16 — full-stream, per-language evaluation

The trainer's pre/post NLL diagnostics intentionally cap scoring at the first 512 tokens. A standalone `evaluate` command now scores every next-token target, carries recurrent state across 512-token data reads, uses the checkpoint's bundled tokenizer (and verifies its `.tok` companion), and reports the corpus token count and optimizer metadata. `scripts/split_multilingual_validation.ps1` uses the source manifest's exact page titles to reproduce each page-disjoint language file.

The 3,000-update AdamW multilingual checkpoint scored as follows over the complete held-out page leads:

| Language | Tokens | Full-stream token NLL |
|---|---:|---:|
| English | 480 | 6.19413 |
| Spanish | 855 | 6.24134 |
| French | 1,485 | 6.23794 |
| Japanese | 338 | 6.24557 |
| Chinese | 815 | 6.23228 |

This avoids the earlier prefix-only comparison and shows that the aggregate score hides language variation. The validation sample is still only one short lead page per language (338–1,485 tokens), and NLL near `ln(512) ≈ 6.238` is weak prediction, not evidence of understanding or instruction following. The command was exercised on all five page splits; the reusable CPU/OpenCL suite passed after its addition.

## 2026-11-16 — expanded multilingual lead corpus

To get beyond the 10-page pipeline smoke sample, `scripts/fetch_multilingual_expansion.ps1` politely retrieved 40 training and 10 page-disjoint validation leads (8/2 per language) in five grouped, serial API calls. It pauses 20 seconds between languages, sets `maxlag=5`, uses exponential retries with a 120-second minimum after 429 and `Retry-After` support, and writes reusable per-language caches so successful groups are not downloaded again after a later group fails. The fetch completed without a 429. The data remains CC BY-SA 4.0; each exact revision and canonical URL is in `data/raw/multilingual_expansion_sources.json`.

Training text is 63,381 bytes (SHA-256 `0243BFE845810F189A5A191496BBBD563C9FABB529E830D8107173C66FF99280`); validation is 15,976 bytes (SHA-256 `BB5C3A6D7C880D5E4F43620BC0D6AA4CEE3EB6DF6F3928986143AAB627D9CEE`). Fresh random weights, seed 10, BPE vocabulary 512 fit only on train, width 32, depth 2, 24-token chunks, 10,000 AdamW updates, accumulation 2, LR `1e-4`, inverse-square-root decay 0.1, weight decay 0.01. The run took 38.8 seconds and tokenized to 34,575 training / 8,696 validation tokens.

The trainer's 512-token prefix diagnostic moved train/validation NLL from 6.322843 / 6.338564 to 6.126660 / 6.106869. Full-stream, per-language evaluation on the held-out pages is more informative:

| Language | Validation tokens | Full-stream token NLL |
|---|---:|---:|
| English | 2,379 | 6.09832 |
| Spanish | 2,512 | 6.11393 |
| French | 1,774 | 6.09843 |
| Japanese | 979 | 6.22525 |
| Chinese | 1,039 | 6.27145 |

This points to a substantial script/language gap: the Japanese and Chinese losses are materially worse than the Latin-script languages, despite all languages appearing in training. The validation set is still just two topic-related lead introductions per language, so this is a diagnostic rather than a generalization benchmark. Five 96-token constrained greedy generations all remained valid UTF-8, but their outputs contained unrelated cross-script text; repeated 4-gram rates ranged from 0.0215 to 0.1613. Unicode validity is not language adherence.

Using the trained tokenizer, the 8-page-per-language training split encoded to 6,134 English, 7,621 Spanish, 6,557 French, 5,510 Japanese, and 8,699 Chinese tokens (17.8%, 22.1%, 19.0%, 15.9%, 25.2%). Japanese is underrepresented by token exposure, but Chinese has the largest share and the worst held-out NLL. A simple equal-byte mix is therefore not an evidence-based fix; next work should inspect tokenizer merge allocation and compare matched script-aware BPE or larger CJK-specific training coverage.

## 2026-11-16 — vocabulary-size ablation at matched data passes

The evaluation CLI now reports byte-normalized cross entropy (`bits_per_byte = total token NLL / raw UTF-8 bytes / ln(2)`), allowing a cautious comparison across different tokenizations. A 1,024-token BPE vocabulary encoded the same train data as 25,000 tokens versus 34,575 tokens at vocabulary 512. To control for repeated data exposure, runs used 10,000 updates at vocab 512 and 7,200 updates at vocab 1,024; both process about 480k and 345.6k target tokens, respectively, or about 13.9 passes over their tokenized training split. Same seed, width 32, depth 2, sequence 24, accumulation 2, data, LR `1e-4`, and decay 0.1. Parameter counts are 36,928 versus 69,696, so compute and model capacity are not matched; the 1,024-token run also took roughly 70 seconds versus about 39 seconds.

| Language | 512 vocab bits/byte | 1,024 vocab bits/byte | Change |
|---|---:|---:|---:|
| English | 4.7057 | 4.4027 | -0.3030 |
| Spanish | 4.4457 | 3.9549 | -0.4908 |
| French | 4.6083 | 4.2619 | -0.3465 |
| Japanese | 5.2945 | 4.5169 | -0.7776 |
| Chinese | 6.3201 | 5.4130 | -0.9071 |

At this controlled exposure, the larger tokenizer lowered byte-normalized loss for all five languages and reduced the Japanese/Chinese gap. This supports testing a larger vocabulary further, but cannot isolate tokenizer quality from the nearly doubled model parameters and longer runtime. Five 96-token generations from the matched-pass 1,024 checkpoint were all valid UTF-8, with repeated 4-gram ratios from 0.0000 to 0.0323; text still mixed scripts and was not coherent. Lower NLL and lower repetition scores do not establish language following or general competence.

## 2026-11-17 — grow vocabulary on a trained checkpoint

The new `grow-vocabulary` command extended the 512-ID multilingual expansion checkpoint to 1,024 IDs using the same fingerprinted training file used to create it. Existing BPE IDs and merges stay fixed; new embedding/output rows are initialized, old model parameters and optimizer moments are retained, new optimizer moments start at zero, and the optimizer step is preserved. The saved token cursor was translated through its raw byte offset into the new token cache, and recurrent state was reset because token boundaries changed. The grown checkpoint resumed successfully at update 10,000 and trained to update 15,000 with accumulation 2 and inverse-square-root decay 0.1.

Full-stream validation at update 15,000:

| Language | Bytes | Tokens | Token NLL | Bits/byte |
|---|---:|---:|---:|---:|
| English | 4,446 | 1,957 | 6.7905 | 4.3100 |
| Spanish | 4,982 | 1,972 | 6.8208 | 3.8931 |
| French | 3,385 | 1,443 | 6.8180 | 4.1902 |
| Japanese | 1,659 | 751 | 6.9278 | 4.5184 |
| Chinese | 1,486 | 803 | 6.9687 | 5.4260 |

Against the scratch 1,024-vocabulary run at 7,200 updates and roughly matched token exposure, this continuation is lower on English, Spanish, and French and slightly higher on Japanese and Chinese. It is not a controlled causal comparison: the grown model spent its first 10,000 updates at vocabulary 512, has a different schedule trajectory, and the validation split is small. It does show that the format and training path can grow a vocabulary without discarding learned rows or optimizer progress. This does not establish improved generation quality; the prototype's coherence and language-following limits remain.

A 96-token greedy continuation from the grown checkpoint after the English prompt `Artificial intelligence is` remained valid UTF-8 but mixed Latin, CJK, kana, and other symbols. Its repeated 4-gram ratio was 0.1613 and distinct-token ratio 0.5208. The higher repetition than the scratch 1,024-vocabulary run's five prompt results reinforces that this growth run is an operational experiment, not evidence of a generation-quality win.

## 2026-11-17 — input-conditioned recurrent write gate

The previous cell set `g = sigmoid(Wu h + bg)`, so current-token content could change the candidate but not directly choose its fast-state write strength. Following the selective-update principle in Mamba, the recurrence now uses `g = sigmoid(Wx x + Wu h + bg)`. It ties the input gate to the existing candidate input projection `Wx`, adding no parameters. Backpropagation sums candidate-path and gate-path derivatives into `Wx` and the token embedding. Checkpoint v6 and standalone weights format v3 identify this changed equation; older checkpoints and model files are rejected so runs cannot silently resume under different dynamics.

The change was evaluated from random seed 10 against the prior recurrent gate at matched vocabulary (1,024), training text, 7,200 updates, 345,600 target tokens, depth 2, width 32, sequence length 24, accumulation 2, AdamW, base rate `1e-4`, and inverse-square-root decay 0.1. Model parameter count is unchanged at 69,696. Training prefix NLL moved from 7.000867 to 6.914567 and validation prefix NLL from 7.047327 to 6.927058. Full-stream held-out bits/byte were:

| Language | Recurrent-only gate | Input-conditioned gate | Change |
|---|---:|---:|---:|
| English | 4.4027 | 4.3939 | -0.0088 |
| Spanish | 3.9549 | 3.9475 | -0.0074 |
| French | 4.2619 | 4.2509 | -0.0110 |
| Japanese | 4.5169 | 4.5138 | -0.0031 |
| Chinese | 5.4130 | 5.4139 | +0.0009 |

The validation changes are tiny and mixed in direction; this run does not establish a quality improvement. Five 96-token constrained greedy continuations remained valid UTF-8, with repeated 4-gram ratios 0.0000, 0.0430, 0.0000, 0.0323, and 0.0323 and distinct-token ratios 0.8333, 0.8021, 0.8021, 0.7708, and 0.7812. All still mixed scripts instead of following the prompt language. After making the trainer's own diagnostic continuation use the same UTF-8 candidate filter, a same-total resume at update 7,200 emitted 128 valid UTF-8 tokens, repeated 4-gram ratio 0.0080, distinct-token ratio 0.6953. The experiment supports numerical stability and continued study of input-dependent writes, but a separate gate projection, multiple seeds, and larger language-following evaluation are still needed before retaining the change as a quality win.

## 2026-11-17 — dense modality input and trainable encoder foundation

The recurrent API now accepts finite dense feature vectors directly, returns `dL/dfeatures` from manual BPTT, and can jointly optimize a trainable `tanh(Ax+b)` C encoder with AdamW. Encoder parameters, moments, and step count have a separate versioned `.enc` file format. Numerical checks show the feature-input model path matches the token-embedding path when supplied identical vectors, the returned feature gradient agrees with finite differences, and the encoder's own input gradient agrees with finite differences. A masked mixed-sequence integration test trains text targets after a two-frame visual feature prefix, verifies gradients flow back through the unsupervised prefix, updates both encoder and recurrent core, and leaves text-step feature gradients at zero. The complete OpenCL build/test suite passed afterward. This remains infrastructure evidence only: no image/audio file decoder, real paired multimodal dataset, task-quality score, or atomic joint checkpoint has been added yet.

## 2026-11-17 — polite source fetch paths

The single-language Wikipedia fetcher now uses `maxlag=5`, a descriptive User-Agent with optional contact, ten-second serial spacing, exponential retries, and a longer wait on 429 / MediaWiki lag responses while honoring `Retry-After`. This brings it in line with the conservative behavior already used by the multilingual collectors. A new `scripts/fetch_hf_dataset_file.ps1` supports one explicitly selected Hugging Face dataset file: it checks machine-readable card license against a caller-supplied expected value, resolves the chosen ref to an immutable commit before downloading, waits at least 30 seconds between metadata and resolver requests, honors `Retry-After` and the Hub `RateLimit` reset time, and writes SHA-256 provenance. This script has not been used to download a corpus; no HF dataset was selected because repository-card licensing alone cannot establish the rights for every underlying media item. PowerShell parser checks passed for all four fetch scripts. No network fetch was run in this step.

## 2026-11-17 — causal image-to-caption update

Added `ama10_train_image_caption`: image patches are encoded as a masked feature prefix; the last patch state predicts the first caption token, then each caption token predicts the next. This makes the image condition causally observable to the first caption target and propagates later caption losses backward across the caption and image-prefix steps before updating the patch encoder. The helper resets recurrent state at each image to avoid cross-example contamination. Its first integration run exposed that mixed-sequence gradients are sized for the full image-plus-text length; the buffer allocation was corrected to that full size. Verification: `scripts/build.ps1 -OpenCL -Test` passed the core, modality encoder finite-difference, image PPM decode/patchify and image-conditioned caption update, tokenizer, checkpoint, cache, eval, and RTX 4050 OpenCL parity checks. The image-caption test is synthetic, so this validates alignment and gradient flow plumbing only, not caption quality or real-data training. Real paired data and unified model-plus-encoder checkpointing remain outstanding.

## 2026-11-17 — first real image-caption training pilot

Added a deliberately polite Commons fetcher using the documented MediaWiki APIs: one batched file-metadata request, one structured-caption request after a 15-second pause, then serial thumbnail downloads spaced 15 seconds apart. It validates each file's individual image license and requires an English structured caption. Commons captions are CC0; the photos keep their separate file licenses, recorded alongside artist and source URL in `data/raw/commons_captioned_seed/sources.json`. The dataset has three training kingfisher images and one held-out image, converted to 64x64 RGB P6 PPM; `manifest.tsv` and the input CSV reproduce the set.

The new `ama10_image_caption_train` C executable tokenizes only training captions, initializes a fresh width-32 recurrent model and 768-to-32 dense patch encoder, then trains on the three image-caption pairs for 1,000 AdamW updates. It cycles through examples, uses 16 16x16 patches per image, depth two, base LR 0.001 with inverse-square-root coefficient 0.01, and weight decay 0.01. Training captions total 180 bytes / 63 BPE tokens; validation is 29 tokens. Held-out caption token NLL fell from 5.786707 to 4.726536; the last logged training chunk loss was 0.51204. Reloading the three saved files reproduced validation NLL exactly to 1e-6. This is a one-image, same-species validation set and only demonstrates that real paired samples can drive a trainable update; it is not a meaningful generalization or caption-quality score. The model, tokenizer, and encoder are still separate files, and the pilot has no optimizer-resume path.

Reproduce the fetch with `scripts/fetch_commons_captioned_images.ps1`, then build and run `scripts/build.ps1` and `build/ama10_image_caption_train.exe data/raw/commons_captioned_seed/manifest.tsv 1000 checkpoints/ama10-commons-kingfisher-v0.1`. Output SHA-256 values: `.ama` `1435675FBDB5F3D73ABE314D3445FD3F18CDF594EA8C689419AFFAC6850F7828`; `.enc` `AA6A452E778BCBD32CD77857BB97B65837066B7E935055D907BC49835ADB68F7`; `.tok` `03103661A96ACDC90EE14A901FE1A9C4D5095339F4A8BC50A0B47BF31CCCD36E`.

## 2026-11-17 — atomic multimodal checkpoint bundle

Added `.amm` version 1, which packages the existing versioned core checkpoint and trainable encoder checkpoint into one checksummed file and uses staged writes plus backup recovery for replacement. Load verifies the container lengths and both component hashes, loads into temporary objects, checks model/encoder input-width compatibility, and only replaces caller state after every component validates. A new test confirms bit-exact model weights, AdamW moments and step, recurrent state, tokenizer, encoder weights/moments and step, and progress after round-trip; it corrupts a payload byte and verifies rejection leaves already-loaded destination state intact.

The image-caption trainer now saves and reload-checks one `.amm` bundle rather than separate model/tokenizer/encoder files. A fresh 1,000-update run reproduced the previous pilot exactly: validation NLL 5.786707 → 4.726536 over 29 caption tokens. Bundle SHA-256: `860CE1F22F91FFCBBC5F81E0638E7EAAA07E5ADA255E64E073C0DB5FC8464BA6`. The new multimodal-checkpoint test and the full `scripts/build.ps1 -OpenCL -Test` suite passed, including RTX 4050 CPU/GPU parity.

## 2026-11-17 — image-caption training resume parity

`ama10_image_caption_train` now accepts `--resume <checkpoint.amm>` and a requested total update count. It rebuilds the dataset and tokenizer inputs, verifies exact train/validation hashes, tokenizer budget, architecture, optimizer, schedule, sequence length, and cursor metadata, then continues with saved optimizer moments and update count. A 1,000→1,200 continuation and an independent fresh 1,200-update run produced byte-identical `.amm` files (SHA-256 `40A03CADDD29CF582172F6FE832ED42EBF02F29D7AF6A92880BEB8DDD1C81A5A`), verifying exact resume parity for this deterministic batch-one experiment. Validation NLL rose from 4.726536 at update 1,000 to 4.867761 at update 1,200, while the last train chunk loss fell to 0.69064; this small run shows overfitting pressure and argues against treating additional updates as automatically beneficial. README documents the resume invocation.

## 2026-11-17 — image-conditioned caption inference

Added `ama10_image_caption_generate`, which loads the single `.amm` bundle, decodes and patchifies a P6 PPM image, runs its patch sequence through the saved encoder and recurrent core, then greedily generates text from the final image-conditioned state. Candidate tokens are filtered so the output remains valid UTF-8 and ends on a complete code point. On the held-out image after update 1,000 it wrote 29 valid UTF-8 tokens / 63 bytes. The text began `Close up ofemale common kingfisher (Alcedo atthis)` and ended with unrelated fragments; it failed to identify the dragonfly larva in the reference caption. This verifies the image → encoder → shared core → text output path, while confirming that this pilot has poor caption quality. Reproduction command is in README.

## 2026-11-17 — multi-category Commons caption pilot

Expanded the seed to 11 licensed, rights-traceable image-caption pairs using seven new Commons files plus the four already fetched kingfishers. The new training samples cover Ama Dablam, an Amsterdam bicycle, a red apple, a fox catching a vole, and an apple tree; held-out samples are a different fox photo and a Vancouver skyline. Train/validation is 8/3 images and 211/127 BPE target tokens. `sources.json` has a verified SHA-256 for all 11 PPMs; all hashes matched after assembly. The new captions' text comes from Commons structured labels and each image's exact license / attribution stays in the manifest. The assembly script avoids re-downloading the earlier four photos.

Fresh random initialization, width 32, depth 2, BPE vocabulary 328, 1,000 updates, AdamW, base LR 0.001 with inverse-square-root coefficient 0.01, weight decay 0.01. Held-out token NLL was 5.840592 before and 4.768022 after training across all 127 targets. The last training chunk loss was 2.60245. Greedy 29-token outputs for the held-out fox, city, and kingfisher photos all passed UTF-8 filtering but remained incoherent (for example, the fox output was `A fee cle  ound.  oond.   lee common .`). The result is lower validation NLL on a more diverse but still tiny English set, not successful captioning. Checkpoint SHA-256: `F6DDB3EEE97F90709516EE898D5C5A0EC4423E1133537965446AB191F3F69CCD`. Reproduce with `scripts/assemble_commons_captioned_diverse.ps1`, then `scripts/build.ps1` and `build/ama10_image_caption_train.exe data/raw/commons_captioned_diverse/manifest.tsv 1000 checkpoints/ama10-commons-multiclass`.

The 1,000→1,200 update resume matched a fresh 1,200-update run byte-for-byte (SHA-256 `ABA2B9C9BDB2CA3B91232EC42AC444C9E6955207A743180BB26FD39AC9E40BE4`). Validation NLL rose from 4.768022 to 4.832275 while the final training chunk loss fell to 2.24554, again showing the small dataset's overfitting tendency.

## 2026-11-17 — multilingual language-conditioned caption pilot

Fetched Commons structured labels in one batched metadata request with the existing polite client behavior and enriched the 11-image set without downloading images again. The resulting four-column manifest contains 20 aligned caption rows: 14 training rows (8 English, 4 French, 2 Japanese) and 6 validation rows (3 English, 2 French, 1 Spanish). The language metadata is retained in `sources.json`; file-level image licensing and SHA-256 records remain attached to each source image.

Added a `<lang:xx>` text prefix to each caption during tokenizer training and causal training. The image patches are followed by the language prefix, and the final prefix state predicts the first caption token. Inference accepts an optional language code and applies the same prefix. Resume hashes and sequence-length metadata include language and prefix tokens, so unconditioned checkpoints cannot resume as conditioned runs. The image test covers both conditioned and unconditioned update paths.

Fresh width-32/depth-2 model, vocabulary 400, 1,000 AdamW updates. Validation token NLL changed from 6.045773 to 5.148887 across 360 targets. Per-language values: English 6.049620→5.044702 (127 tokens), French 6.036434→5.220713 (174), Spanish 6.065030→5.161328 (59). Japanese has no held-out image, and Spanish has only one. Despite lower NLL, 30-token greedy outputs for the held-out fox image were identical across `en`, `fr`, and `es`: `AAma Dablam. NNNNNNNNNNNNNNNNNNNNNNNNN`. They are valid UTF-8 but neither language-following nor image-grounded. This is evidence of conditional training plumbing and loss reduction on a tiny set, not useful multilingual caption quality. Reloaded validation NLL was 5.148887. Checkpoint SHA-256: `580745D380D3AFBDF9FD1ADA8B8E902C665C03B28141396C0121AA2C102D629C` (`checkpoints/ama10-commons-conditioned-v0.3.amm`).

Reproduce the data enrichment with `scripts/expand_commons_caption_languages.ps1` (one batched Commons API request, no image re-download), then build with `scripts/build.ps1 -OpenCL -Test` and train with `build/ama10_image_caption_train.exe data/raw/commons_captioned_diverse/manifest.tsv 1000 checkpoints/ama10-commons-conditioned-v0.3`.

## 2026-11-17 — PCM speech input foundation

Added a C11 RIFF/WAVE reader for PCM16 16 kHz mono/stereo. It averages stereo samples to mono and extracts 400-sample Hamming-windowed frames at a 160-sample hop (25 ms / 10 ms). Unsupported sample format, rate, channel count, empty/misaligned payloads, and inconsistent block/byte rates are rejected. The synthetic fixture verifies signed sample conversion, stereo averaging, edge windowing, and zero padding. `scripts/build.ps1 -OpenCL -Test` passes including this fixture and the existing RTX 4050 parity checks.

This entry originally covered input plumbing only; the following experiment records the paired trainer and real data added afterward. OpenSLR lists LibriSpeech train-clean-100 as 6.3 GB under CC BY 4.0, while the 337 MB dev-clean split is for held-out evaluation.

## 2026-11-17 — C audio-to-text training pilot

Added `ama10_train_feature_conditioned_text`, a general dense-feature-sequence-to-text causal update that trains the modality encoder and shared recurrent core. The new C trainer consumes a four-column TSV (`relative WAV path`, `transcript`, `train|validation`, `language`), reads 16 kHz PCM16 WAV, applies a train-only BPE tokenizer sample with `<lang:en>` prefix, evaluates transcript token NLL, saves/reloads one `.amm` bundle, and resumes only when audio/transcript hashes and settings match. The C generator conditions on audio frames plus a language prefix. `ama10_score_text` computes punctuation-stripped, ASCII-lowercased WER and Unicode-codepoint CER. This reuses the shared AMA recurrent core rather than an external ASR model.

The polite Hugging Face collector verifies the machine-readable `cc-by-4.0` dataset card, pins repository commit `71cacbfb7e2354c4226d01e70d77d5fca3d04ba1`, requests eight `clean/train.100` examples and two `clean/validation` examples, then serially downloads and converts them to 16 kHz mono PCM WAV with at least 30 seconds between every request. No 429 occurred. The resulting 4.02 MB set has eight contiguous train utterances from speaker 374 / chapter 180298 and two validation utterances from speaker 2277 / chapter 149896. `data/raw/librispeech_hf_micro/sources.json` records row IDs, split, transcript, source revision, speaker/chapter and local WAV hashes. This is deliberately a tiny slice, not representative ASR training data.

From random weights, width 32 / depth 2 / vocabulary 453, 500 AdamW updates over eight train clips: held-out transcript-token NLL fell 6.428474→5.207644 over 111 target tokens; the final logged train chunk was 4.32078. Greedy 40-token outputs on the two held-out WAVs scored WER 1.020833 (49 edits / 48 reference words) and CER 0.728033 (174 edits / 239 normalized Unicode characters). Both outputs repeated `MME`-like fragments and were poor transcriptions. The lower token NLL is therefore not an ASR quality win. Checkpoint SHA-256: `4C11F2B3BA660872B266AA8220181CDD11F9CC46C2441B2C02E39669C5404B73` (`checkpoints/ama10-librispeech-hf-micro-v0.2.amm`). An independent fresh 8-update bundle exactly matched a 4→8 resumed run, SHA-256 `413C21B02EE7CCA60528D5910978F0BB034E3D985E78AC4B84097FD488ED2E1F`.

The synthetic WAV integration fixture also trains, saves/reloads, and generates through the new audio path. `scripts/build.ps1 -OpenCL -Test` passes with this fixture, WER/CER unit tests, and GPU inference parity. Reproduce collection with `scripts/fetch_hf_librispeech_rows.ps1 -TrainRows 8 -ValidationRows 2`, then build and train with `ama10_audio_transcribe_train.exe data/raw/librispeech_hf_micro/manifest.tsv 500 checkpoints/ama10-librispeech-hf-micro-v0.2`.

## 2026-11-17 — log-mel front end and expanded ASR slice

The first audio model receives a complete utterance through raw overlapping waveform windows, then has to recover all transcript tokens from the final recurrent state. This is a severe information bottleneck. As an initial feature-side improvement, added a C radix-2 512-point FFT and 40 triangular mel filters over 25 ms Hamming windows at a 10 ms hop. Each filter outputs fixed-scaled log power, with no utterance-wide normalization, so an individual frame uses no future samples. Trainer and generator both consume these 40 features. A front-end version marker is included in train/validation hashes; the old 400-wide raw-window checkpoints cannot resume in this 40-wide feature model. A synthetic 1 kHz tone fixture confirms spectral concentration in the expected mel region.

On the original matched 8-train / 2-validation slice, both runs started from their fixed random seeds and trained 500 updates with identical model width, schedule, token data, and evaluation limit. Raw windows gave validation NLL 6.428474→5.207644, WER 1.020833 (49/48), CER 0.728033 (174/239). Log-mel gave NLL 6.262311→5.137072, WER 1.083333 (52/48), CER 0.690376 (165/239). This is mixed evidence: NLL and CER improved slightly while WER worsened, and both are unusably high. It does not establish a front-end quality win.

Expanded the same pinned `71cacbfb7e2354c4226d01e70d77d5fca3d04ba1` LibriSpeech revision with eight rows at train offset 10,000 and two validation rows at offset 200, appending without re-downloading existing IDs. The final data has 16 train clips from two speakers/chapters and four held-out clips from two disjoint speakers/chapters; all 20 converted-WAV hashes were independently verified. Source-blob hashes are present for the appended batch; the original ten rows retain their source IDs/revision and converted-WAV hash. Total WAV size is 7,922,840 bytes. No 429 occurred and the collector kept 30-second spacing.

Fresh log-mel training, width 32 / depth 2 / vocabulary 512, 500 AdamW updates: held-out token NLL 6.319282→5.166077 over 189 targets. Forty-token greedy decoding scored aggregate WER 1.063291 (84/79) and CER 0.770531 (319/414) across four utterances. The model still emits repeated fragments, so this remains a failed ASR-quality result. Checkpoint SHA-256: `8B247F61318EB9068EA975993B0AF1CA4A7368ACF7FB5194799BF528094C2116` (`checkpoints/ama10-librispeech-hf-mel-16x4-v0.4.amm`). A fresh 8-update run matched a 4→8 resumed run exactly, SHA-256 `BD0BB5C8976169B7D45F9ABC001506B6B490E17FE3DF4C4F4C12DEEDB7157855`. `scripts/build.ps1 -Test` passed after adding the log-mel transform, tone check, WER/CER tests, and synthetic audio train/generate integration.

## 2026-11-17 — monotonic chunk-aligned speech decoding experiment

The prefix model's audio bottleneck motivated a causal aligned update: average-pool contiguous log-mel frames into one context per training BPE token, feed the `<lang:en>` condition, then interleave each context (which predicts its target token) and the teacher-forced previous token. `ama10_train_aligned_feature_text` backpropagates through this full interleaved sequence and the shared audio encoder. Inference partitions the frames into the user-requested number of output steps and follows the same feature/token alternation. This preserves the shared AMA recurrent core and gives each output step local acoustic context. The alignment itself is not learned: training divides frames uniformly over known target tokens, while inference divides by the requested maximum token count. That difference is a material limitation.

Fresh 16/4 LibriSpeech log-mel run, width 32 / depth 2 / vocabulary 512, 500 updates: validation NLL 6.418193→5.208201; greedy 40-token output WER 1.493671 (118/79 words), CER 0.818841 (339/414 chars). Extending this checkpoint to 5,000 updates overfit: NLL rose 5.208201→5.664704; WER/CER were still 1.113924 (88/79) and 0.797101 (330/414). The prefix model at 500 updates had WER/CER 1.063291/0.770531, so the new alignment path did not improve the measured speech result. The 500-update bundle hash is `9A6C16D676DBE7CCA51751A0B5543ECF68FC0FC8BB6105455285EE65ABCBA7D8`; the 5,000-update bundle hash is `DC382EB522A6D255AD2ABB1ABAD947549351892BD1773A12A42913BF16FCAA8A`. A fresh 8-update bundle matched a 4→8 resume byte-for-byte, SHA-256 `125DD284398530B5245360113F8216C6249A6DF75B3F944B5E32FF5F7DD637`.

The resume metadata includes a new audio alignment hash marker; a previous log-mel prefix checkpoint was rejected. The synthetic integration test now exercises log-mel extraction, segment pooling, aligned training, checkpoint reload, and aligned generation. The full OpenCL suite is rerun after these changes.

## 2026-11-17 — CTC sequence criterion primitives

Added a log-domain CTC forward-backward routine over frame-major logits. It expands the transcript with blank states, computes the sequence negative log-likelihood, and returns framewise class posteriors. The shared core now accepts normalized soft target distributions in its masked multimodal BPTT path, allowing CTC posteriors to drive gradients through dense audio features and the recurrent model. A regression check confirms one-hot soft targets exactly match the hard-target update; CTC tests verify each posterior row sums to one, finite-difference agreement between NLL derivatives and `softmax - posterior`, and rejection of a repeated-token target with too few frames. `scripts/build.ps1 -Test` passes. The existing ASR trainer has not yet been switched to CTC, and no new ASR quality result is claimed.

## 2026-11-17 — CTC framewise LibriSpeech pilot

Switched the trainer from uniform transcript-time pooling to every causal log-mel frame. Each update computes logits after the language prompt, runs CTC forward-backward with byte-token ID 0 reserved as blank, then trains the shared model and audio encoder from framewise posterior targets. Checkpoint hashes include a new CTC/blank marker, sequence length records the longest acoustic-frame sequence, and generation greedily collapses repeated labels and blanks. The synthetic C test uses a one-byte transcript so even its single-frame WAV is a feasible CTC target; end-to-end train, save/reload, and generation pass.

Fresh 16/4 LibriSpeech run, width 32/depth 2/vocabulary 512, 500 updates: validation CTC NLL per target fell from 65.521667 to 19.666545. Greedy CTC decoding across the four held-out utterances had macro WER 1.000000 and CER 0.979361; each hypothesis collapsed to a single token. A resume to 1,500 updates lowered validation CTC NLL further to 5.329343, but default greedy decoding collapsed to blank for all four utterances (macro WER/CER 1.000/1.000). Prefix beam search (width 8, top 16 nonblank classes) with blank penalties 0–4 was then measured on the same validation split. Penalty 4 produced macro WER 1.002885 and CER 0.724666, versus 1.000/1.000 at penalty 0, but outputs repeated `THE` and `E`; selecting a penalty on this validation set is diagnostic only. This is still a failed ASR-quality result. `scripts/build.ps1 -Test` passed after the trainer and decoder changes.

## 2026-11-17 — C11 response-only SFT path and OASST1 pilot

Added `ama10_sft_train`, a C trainer over escaped UTF-8 `prompt<TAB>response` rows. It teacher-forces concatenated prompt and response tokens but masks loss until the first response target, evaluates response-token NLL only, supports optional initialization from an existing checkpoint (including optimizer moments), and verifies the saved checkpoint/tokenizer by reload. It saves a `.tok` sidecar and records source fingerprints/progress. The source-preparation script filters OpenAssistant's ready-for-export messages to English messages with positive review, not deleted, and zero spam, language-mismatch, PII, and inappropriate-content labels. It reconstructs direct prompter→assistant pairs and splits by message-tree ID to prevent conversation leakage; escaped TSV preserves embedded tabs/newlines. The compressed source file is pinned to commit `5c38bf14fe3c7bb31bb8e13eb4f87b2c6fd78d93`, SHA-256 `286A6E9A5A413B3272AE9C0B5A20D327983DEA1C24342AE28CB244A6DA65185C`.

With 200 train pairs, 50 validation pairs and the 7,200-update multilingual checkpoint, 200 SFT updates at `3e-4` lowered held-out response NLL 6.9325→5.7907 but caused free generation to collapse to repeated NULs (distinct-token ratio 0.0125, repeated 4-gram ratio 0.9870). Repeating from the same base with `3e-5` lowered NLL only to 6.9042 and produced incoherent text without the NUL collapse (distinct-token ratio 0.65, repeated 4-gram ratio 0.039). The SFT command exposes learning rate and defaults to `3e-5`; neither pilot demonstrates capable assistant behavior. `scripts/build.ps1 -Test` passes with SFT fixture training, response-only validation loss, checkpoint save/reload, and the rest of the C suite.

Expanded to 1,000 train / 200 validation conversation trees and 1,000 updates at `1e-4` from the same pretrained checkpoint. Response NLL fell 6.931560→5.816568; English Wikipedia validation token NLL also fell 6.916070→6.086886. Despite those token-level improvements, free generation on three held-out OASST prompts emitted repetitive `i` sequences (distinct-token ratio 0.025; repeated 4-gram ratio 0.86–0.91). Added an optional output-frequency penalty to the C text decoder, which subtracts `penalty * prior_count(token)` from each candidate logit without changing training weights. On one fixed validation prompt, penalty 0.0 had diversity/repeated-4gram metrics 0.025/0.909, penalty 0.5 had 0.650/0.000, and penalty 2.0 had 1.000/0.000; all outputs remained semantically incoherent. This improves repetition metrics but does not solve assistant quality. The CLI accepts `generate model tokenizer prompt output tokens cpu repetition-penalty` with penalty in [0,10].

## 2026-11-17 — C11 teacher-to-student distillation

Added `ama10_kd_train`, which loads a teacher checkpoint and its tokenizer, streams train/validation text through fingerprinted token caches, computes full-vocabulary teacher distributions in C, then updates a narrower student with `ama10_train_soft_multimodal_sequence`. The CLI accepts teacher, train text, validation text, update count, output checkpoint, optional student width, and optional learning rate. It reports validation teacher→student KL and student gold-token NLL before/after; the saved model and tokenizer sidecar are reload-checked. A two-update fixture runs in `scripts/build.ps1 -Test`.

Real run used the multilingual expansion checkpoint (width 32, vocabulary 1,024) as teacher and the multilingual expansion train/validation files, with width 16, sequence length 64, depth 2, learning rate `1e-4`, and 1,000 updates. On 6,927 validation next-token positions, teacher/student KL fell 0.053928→0.024569 and student gold-token NLL fell 6.958796→6.936258. Teacher-soft training CE declined from 6.958522 at update 100 to 6.938134 at update 1,000. Output: `checkpoints/ama10-multilingual-kd-width16-v0.1.ama`, SHA-256 `47C6454365CA55C61D9090482DDBDCD1281272F90F8F5911D9D28907D05C1A67`. This measures distribution imitation on the same narrow corpus; it does not establish student generation quality or broad capability. `scripts/build.ps1 -Test` passed with the KD fixture and existing C suites.

## 2026-11-17 — C11 LoRA output-head adaptation

Added a C LoRA trainer for prompt/response TSV. It freezes the loaded model, updates only rank-r A/B output-head factors with Adam, evaluates response-only NLL, and stores a separate checksummed `.lora` adapter tied to exact base-checkpoint and tokenizer fingerprints. CPU text generation accepts the adapter as an optional final argument and rejects mismatched checkpoints. A unit test checks the adapter's analytical factor derivative against a finite difference, round-trip serialization, and base binding; the build integration also trains, reloads, and runs generation through a rank-4 fixture.

Real run used the distilled width-16 checkpoint, rank 8, 1,000 updates at `1e-4`, and the OASST1 1,000/200 tree-held-out TSV split. Validation response NLL decreased 6.941806→6.935594. Adapter: `checkpoints/ama10-kd16-oasst-lora-r8-v0.1.lora`, SHA-256 `5B935A22A7E8D7FB018CE1FD1D30EB5160A71CCDFB77256E6F44FDE1D452D170`. A 128-token held-out generation remained incoherent, mixing scripts and words with frequent fragments despite a repeated 4-gram ratio of 0.152; no instruction-following improvement is claimed. Adapter inference currently runs on CPU only. `scripts/build.ps1 -Test` passed with adapter finite-difference/serialization checks, training fixture and generation.

## 2026-11-17 — C checkpoint supervision and regression recovery

Added `ama10_supervisor`, which loads reference and candidate training checkpoints, verifies model/tokenizer compatibility, reuses a fingerprinted validation-token cache, and compares full-stream causal NLL. It also generates a deterministic UTF-8-constrained 128-token continuation from the same validation prefix and compares repeated 4-gram ratios. The candidate is selected only if its NLL stays within a configurable absolute tolerance and repetition is no more than 0.05 worse; otherwise the reference checkpoint is atomically copied into the selected path. The report is JSON and the selected tokenizer sidecar is written next to the checkpoint. Identical reference/candidate control passed and selected the candidate.

Regression trial compared the 7,200-update multilingual reference to the 1,000-pair OASST SFT checkpoint on the multilingual expansion validation corpus. Candidate NLL improved 6.921277→6.477446, but repeated 4-gram ratio degraded 0.192→0.888. The supervisor rejected it and restored the reference; SHA-256 of reference and recovered checkpoint both equal `4C3EBCB51FE128EF496683E69922BC2C8EC87004B28D92A2AB3DC4AF7468348D`. This confirms the gate can catch a concrete loss/repetition tradeoff. It does not evaluate semantic correctness, call an external teacher, launch or resume training, schedule data, or upload SwanLab metrics. The C suite passes with an identical-checkpoint acceptance integration case.

## 2026-11-17 — C tool dispatcher and synthetic call traces

Added a C allowlist runtime and model conversation loop. Calls use `<|tool_call|>name<TAB>arguments<|end_tool_call|>`; supported tools are a bounded recursive-descent arithmetic calculator (finite arithmetic, parentheses, powers, division-by-zero checks) and case-insensitive local text search returning at most three bounded snippets. No shell or arbitrary process tool exists. `ama10_tool_chat` generates an assistant turn, executes one valid call, adds a framed untrusted result to the transcript/context, and generates a follow-up. The tool parser rejects malformed blocks and unknown names. Tests cover arithmetic precedence, malformed expressions, divide-by-zero, bounded search, protocol dispatch, calculator runtime output, and CPU chat inference.

`scripts/prepare_calculator_tool_sft.ps1` creates 256 train and 64 validation arithmetic traces with seed 20261117. SFT from the 7,200-update multilingual checkpoint for 1,000 updates at `3e-5` reduced validation response NLL 6.915634→6.181994. A further 5,000 updates at `1e-5` reduced it to 5.493999 (6,000 additional updates total). However, free generation on `Use the calculator tool to compute 23 * 19.` emitted repeated fragments rather than a call block, even with a 1.0 repetition penalty; the C dispatcher was verified on known call text, but learned tool selection did not pass. Checkpoints: `ama10-multilingual-calculator-tools-v0.1.ama` (8,200 total updates) and `ama10-multilingual-calculator-tools-v0.2.ama` (13,200 total updates). This is a failed autonomous-tool-use experiment despite lower teacher-forced loss. `scripts/build.ps1 -Test` passed after adding the runtime and chat-loop fixtures.

## 2026-11-17 — Persistent-state OpenCL decoding benchmark

The RTX 4050 parity test already passed for the state-visible OpenCL API, but its per-token transfers made the small reference configuration slower than CPU. Added a persistent stream API that leaves fast/slow recurrent state in device buffers and transfers only vocabulary logits each step. CPU/OpenCL logits matched within `3e-4` for both fixed depth and early halting; the OpenCL full suite passed after the change. Greedy `opencl` generation was exercised on the SFT fixture checkpoint and emitted 32 valid UTF-8 bytes.

On this machine, width 32 / vocabulary 1,024 / depth 2 measured CPU 106,690 tokens/s, state-visible OpenCL 7,269 tokens/s, and persistent OpenCL 15,184 tokens/s (still 7.0× slower than CPU). A sweep of freshly initialized benchmark models measured persistent OpenCL speedups of 0.28× at width 64 / vocabulary 1,024, 0.90× at width 128 / vocabulary 2,048, 1.76× at width 256 / vocabulary 4,096, and 4.79× at width 512 / vocabulary 8,192; at the last size, CPU was 355 tokens/s and OpenCL 1,700 tokens/s. These are brief single runs on an RTX 4050 Laptop GPU, not a broad performance claim or a trained-model result. They show a CPU/GPU crossover near width 256 for these vocabulary ratios. For the current small research checkpoints CPU remains faster. The test also confirms stream decode is rejected until an explicit device-state reset.

## 2026-11-17 — Hybrid OpenCL AdamW update

Added an optional C training path that keeps forward/backward propagation on CPU but runs the clipped AdamW update over a flattened set of all eight parameter groups with an OpenCL kernel. Device moments persist across optimizer updates; host model weights are refreshed after each update and moments synchronize before checkpoint writes. The update stages results before committing to host model memory, and clears accumulated gradients only on success. The RTX 4050 test compares CPU and OpenCL weights plus first/second moments over two consecutive updates, including a gradient norm large enough to exercise clipping; all passed. A from-scratch width-32, vocabulary-512, 8-update Wikipedia run produced the same printed train/validation NLL as CPU to six decimals (train 6.329172→6.319828; validation 6.340924→6.345048). A 100-update timing comparison took 0.416 s on CPU and 1.105 s in hybrid mode. A separate 2→4-update resume restored optimizer step 2 and completed successfully. The OpenCL optimizer overhead still outweighs any benefit at this small parameter count; this is an experimental hybrid path, not a GPU-accelerated full training implementation. BPTT remains CPU-only, and Muon is unsupported.

## 2026-11-17 — Configurable main-trainer width and scaling trial

Added `--width N` (8–4096) to fresh and resumed main-trainer runs. Resume now rejects a width that differs from the checkpoint model. The default remains width 32. CPU and hybrid OpenCL runs from random seed 10, vocabulary 512, and Wikipedia train/validation each ran 20 updates at width 256: wall times were 1.493 s and 2.226 s respectively. At width 512, ten fresh updates took 6.096 s CPU and 12.796 s hybrid. Resuming the width-512 OpenCL checkpoint from update 10 to 12 succeeded; over the trainer's 512-token evaluation window, train NLL moved 6.102170→6.033870 and validation NLL 6.219945→6.201154. Its 128-token free continuation still had repeated 4-grams at 0.960. A width-256 request against the width-512 checkpoint was rejected. The OpenCL build and integration suite passed with fresh/resumed width-64 training, a width mismatch rejection, and the AdamW hybrid. Larger dimensions increased CPU BPTT cost, but this partial optimizer offload still did not make training faster; a GPU BPTT implementation remains necessary.

## 2026-11-17 — OpenCL output-head training slice

The initial output-head-only implementation was extended to recurrent BPTT. Reverse-time fast/slow deltas now persist on the GPU, and the sequence cache, token IDs, and output-head hidden deltas upload once per microbatch; kernels accumulate all seven recurrent/input gradient groups without per-cell host transfers. Output-head gradients form the eighth group. The OpenCL test compares all eight groups against the CPU reference for a seven-token, depth-three sequence. `scripts/build.ps1 -OpenCL -Test` passed on the RTX 4050, including fresh/resumed training fixtures. End-to-end timing on the multilingual English expansion files (11,873 train bytes, 4,446 validation bytes, vocabulary 256) still favors CPU: at width 64, ten updates took 0.091 s CPU vs 0.792 s hybrid; width 256, five updates took 0.600 s vs 2.285 s; width 512, three updates took 2.097 s vs 6.096 s. Runs used seed 10, depth two, AdamW, and the same learning rate. The hybrid now computes full BPTT on-device but does not accelerate training at these shapes; per-cell kernel launch overhead and CPU recurrent forward propagation remain targets for optimization.

## 2026-11-17 — OpenCL recurrent forward and cache reuse

Added a training-specific OpenCL forward kernel. The GPU now computes all recurrent layers and tokens, writes the eight-field trajectory cache, and retains that cache for GPU BPTT; only the final hidden vector per token and final slow state return to the host for loss/state handling. CPU/OpenCL loss, final recurrent state, and all eight parameter-gradient groups matched in the seven-token, depth-three parity fixture. scripts/build.ps1 -OpenCL -Test passed on the RTX 4050, including fresh/resumed training. On the same multilingual English expansion files (11,873 train bytes, 4,446 validation bytes, vocabulary 256), CPU vs hybrid total runtime was 0.089 vs 1.810 s for ten updates at width 64, 0.560 vs 1.225 s for three updates at width 256, and 2.112 vs 2.673 s for three updates at width 512. Runs used seed 10, depth two, AdamW, and learning rate 1e-4. GPU forward narrows the gap substantially at widths 256–512, but the hybrid remains slower; CPU loss calculation and sequential kernel launch overhead are still costs.

## 2026-11-17 — polite nine-language Wikipedia coverage expansion

Expanded the revision-pinned lead corpus from five to nine Wikipedia editions: English, Spanish, French, Japanese, Chinese, German, Portuguese, Russian, and Arabic. Each language has eight training leads and two disjoint validation leads. The fetcher makes one grouped MediaWiki API request per language, waits at least 30 seconds between requests, sets `maxlag=5`, sends a descriptive User-Agent, honors `Retry-After`, and backs off at least 120 seconds after 429. Completed language batches are cached so an interrupted run can resume without refetching them. A few unavailable localized titles were corrected after the API returned missing pages; no title-discovery preflight was used. No 429 occurred during the successful collection.

The resulting files contain 72 training and 18 validation pages. SHA-256: training `A7EC1522FD5D0FBE23B7FA4082993E9D1D67D7FA3D23A2C74B18AFA43A028D76`, validation `09DF51DE9054ACC697986C4D7C512BDABB68E6462F0A6F0DF04B6F82F2979CF7`, source manifest `B2CE21C61AD46979AAE1A0500A431EF3EAE1B380D64CE2B05879E58209637F89`. `scripts/prepare_multilingual_sft.ps1` now emits balanced language-specific prompts for all nine languages (72/18 pairs); the generation probe covers Latin, Kana/CJK, Cyrillic, and Arabic script blocks. Hugging Face fetchers also retain a 30-second minimum delay per request/download and honor rate-limit response headers. These small lead extracts support pipeline experiments, not broad language-quality claims.
## 2026-11-17 — fresh nine-language pretraining and SFT

Using the expanded revision-pinned 72/18 page split, trained a new width-128, depth-2, 1,024-token model from random initialization (seed 10), AdamW, constant `3e-4`, for 500 updates. This run did not initialize from any older checkpoint. Training used 126,758 bytes / 52,811 tokens and validation used 30,620 bytes / 12,958 tokens. Training token NLL moved 6.992378→7.492115 while held-out NLL moved 7.015560→6.868658, so this short run shows a train/evaluation divergence rather than uniform optimization. Checkpoint SHA-256 `95339AB320C8AE9445E283B6B1579A9F779A9CCFD2D306895F46750360C211EF`; tokenizer SHA-256 `E21212854092AABF53C8DA8EFB5E8B4CBA599F8056D49E2D5D816DEC93564083`.

Continued with 500 additional masked-response SFT updates on the 72/18 balanced language-specific article prompt pairs (`3e-5`); the checkpoint stores global update 1,000. Mean held-out response-token NLL fell 7.082464→6.376673 over 10,576 response tokens. Full-stream held-out bits/byte by language also fell in every language:

| Language | Pre-SFT bits/byte | Post-SFT bits/byte | Change |
|---|---:|---:|---:|
| Arabic | 3.800580 | 3.714468 | -0.086112 |
| German | 5.906635 | 4.892847 | -1.013788 |
| English | 4.036023 | 3.764178 | -0.271845 |
| Spanish | 3.839522 | 3.669521 | -0.170001 |
| French | 4.263563 | 4.050598 | -0.212965 |
| Japanese | 5.509983 | 5.055272 | -0.454711 |
| Portuguese | 4.159630 | 3.837335 | -0.322295 |
| Russian | 4.175758 | 4.132190 | -0.043568 |
| Chinese | 6.887750 | 7.347398 | +0.459648 |

The fixed nine-prompt generation probe improved script fractions on all four failing languages relative to the fresh pre-SFT model, but still failed its 0.5 threshold for Japanese (0.232), Chinese (0.230), Russian (0.328), and Arabic (0.317). English, Spanish, French, German, and Portuguese passed. All outputs were valid UTF-8 and had zero repeated 4-grams; neither metric checks meaning or grammatical quality. The Chinese full-stream bits/byte regression is a clear counterexample to broad improvement despite the mean SFT loss decrease. SFT model SHA-256 `3863285198FA30612111819FD6B0BC8DDE9E1FFBD710B1769949749F474C8A5C`; it uses the unchanged tokenizer hash above.

Updated `scripts/split_multilingual_validation.ps1` to resolve repeated source titles by matching remaining title occurrences in manifest order; English and Russian both contain the title “Data mining”. The splitter produced exactly two held-out pages per language, and the C evaluator reported per-language token NLL/bits-per-byte from each resulting file. The overall probe report is `build/multilingual-9lang-sft-probe/report.json`. This remains a small lead-extract experiment and is not broad multilingual capability evidence.
A second 500-update continuation at `1e-5` lowered response-token NLL further, 6.376673→6.309510, but its per-language full-stream check showed mixed movement relative to the 1,000-update checkpoint:

| Language | 1,000 updates bits/byte | 1,500 updates bits/byte | Change |
|---|---:|---:|---:|
| Arabic | 3.714468 | 3.735494 | +0.021026 |
| German | 4.892847 | 7.692955 | +2.800108 |
| English | 3.764178 | 3.741603 | -0.022575 |
| Spanish | 3.669521 | 3.636107 | -0.033414 |
| French | 4.050598 | 4.000466 | -0.050132 |
| Japanese | 5.055272 | 4.975774 | -0.079498 |
| Portuguese | 3.837335 | 3.797028 | -0.040307 |
| Russian | 4.132190 | 3.303694 | -0.828496 |
| Chinese | 7.347398 | 7.529541 | +0.182143 |

The script probe fractions for Japanese / Chinese / Russian / Arabic moved from 0.232 / 0.230 / 0.328 / 0.317 to 0.288 / 0.269 / 0.447 / 0.450, still below the 0.5 gate. This shows why aggregate response NLL is insufficient for checkpoint selection: later SFT has conflicting language-level effects, including large regressions in German and Chinese. The 1,500-update checkpoint SHA-256 is `8998508B89D53D76BD1419F399B4559892104D719E299F6C56AC07D5D26D0B21`; retain the 1,000-update checkpoint as a separate candidate rather than claiming the later checkpoint is globally better.
## 2026-11-17 — language-stratified checkpoint supervision

The aggregate supervisor gate could accept a candidate while concealing per-language loss regressions. Added an optional semicolon-separated `name=validation.txt` list to `ama10_supervisor`; it evaluates each stream using the reference and candidate tokenizer, reports per-language hashes, token counts, NLL deltas, and pass/fail, and rolls back if any language exceeds a configurable NLL regression tolerance (default 0.10 nats/token). Existing invocations without the list retain the previous aggregate, UTF-8, and repetition behavior. Duplicate/malformed language entries are rejected; up to 32 language streams are supported.

A real comparison of the nine-language 1,000-update and 1,500-update SFT checkpoints used an aggregate tolerance of 0.05 and language tolerance of 0.10. Aggregate NLL changed 7.982314→8.005641, so it passed that tolerance. The new gates caught German NLL worsening 5.749203→9.039391 and Chinese 8.085409→8.285847; seven other language streams passed. The supervisor returned `candidate_language_nll_regression` and selected the 1,000-update reference. Its generated repeat-4 ratio changed 0.608→0.632, which remained within the configured ceiling and relative allowance. Full report: `build/9lang-sft-supervisor.json`.

Added a two-language identical-checkpoint integration case for report serialization and acceptance. `scripts/build.ps1 -Test` passed all C unit and integration suites, including the new per-language supervisor fixture. The real loss comparisons come from two held-out pages per language and are noisy; this gate is a configurable regression guard, not a semantic evaluator.
## 2026-11-17 — expanded nine-language corpus and fresh model run

Expanded each language from 8/2 to 16 training / 4 held-out Wikipedia leads, adding eight aligned training topics and two held-out topics per language. The updated fetcher reuses cached pages by requested title, fetches only missing titles in grouped serial API calls, retains the 30-second minimum delay and `maxlag=5`/429 backoff, and now rejects duplicate page IDs before writing corpus files. One localized title was redirected to an already included German page; it was replaced with a distinct `Datenkompression` page. No 429 occurred. Each language now has 20 manifest rows and exactly 16/4 train/validation; all train and validation page IDs are disjoint within language.

The corpus files contain 225,721 training bytes and 66,847 validation bytes (the extracted text payloads). SHA-256: train `8E4F401DB577931A94C4ACDE3E609499FC06E1397B83575E8DC75737620D7D36`, validation `1F58987FBEEAC8E64C701CA2E472E7B60ACE921E1EE60C2F34BF2058521FD567`, source manifest `7D30FC76A2349EF9D3A4BEA10115220CAF5998E5E1F9A99EF4D2EEB7F5C0CC1F`. The balanced prompt files contain 144/36 pairs: train SHA-256 `21B97D63E19F84D991B648E22005FBFB8428B5CE9B65060F4F146FBAF956B6C0`, validation `8503C67F7B050D95301A22A8B673BD25CFB845C6ED08C7E95675C86901AC7DF4`.

A fresh width-128, depth-2, vocabulary-1,024 run from random weights at `3e-4` for 1,000 updates was a failed optimization: train NLL 7.015696→7.814888, validation NLL 7.015133→7.780798, and repetition ratio 0.992. A separate fresh run at `1e-4` for 500 updates reduced train NLL 7.015696→5.079560 and validation NLL 7.015133→4.745945. Its checkpoint SHA-256 is `FF0019BEA948CD192237FA0867D6B171CAE44701FDE25C6E365A8D92150C141F` (tokenizer hash is shared with the 1,024-vocabulary SFT runs). Greedy output remained highly repetitive (repeat-4 0.920); the fixed probe had only Arabic above the expected-script threshold before SFT.

A 500-update response-masked SFT pass at `3e-5` lowered held-out response NLL 8.173270→6.573810 over 24,380 response tokens. Per-language held-out bits-per-byte changed as follows:

| Language | Pre-SFT bits/byte | Post-SFT bits/byte | Change |
|---|---:|---:|---:|
| Arabic | 2.391524 | 2.478463 | +0.086939 |
| German | 6.926212 | 5.440817 | -1.485395 |
| English | 5.915590 | 4.549066 | -1.366524 |
| Spanish | 5.329231 | 4.024999 | -1.304232 |
| French | 5.765012 | 4.384148 | -1.380864 |
| Japanese | 6.372306 | 5.218834 | -1.153472 |
| Portuguese | 5.582680 | 4.197449 | -1.385231 |
| Russian | 4.763850 | 3.467684 | -1.296166 |
| Chinese | 7.396644 | 6.297770 | -1.098874 |

The script probe passed English, Spanish, French, German, Portuguese, and Arabic; it failed Japanese (0.026), Chinese (0.012), and Russian (0.130). The multilingual supervisor compared the pre-SFT and SFT models with a 0.10 aggregate and per-language NLL tolerance. Aggregate NLL improved 8.314975→7.946265, but Arabic NLL regressed 4.748570→4.921195, beyond tolerance, and free generation also exceeded the 0.75 repeat-4 ceiling (0.856). It selected the pre-SFT model and reported `candidate_language_nll_regression`; report: `build/9lang-144p-supervisor.json`. Checkpoint hashes: pre-SFT `FF0019BEA948CD192237FA0867D6B171CAE44701FDE25C6E365A8D92150C141F`, SFT `2E57E6F4B2A92A388A0D6C37E9FD142BF2BDC76BB1C3A932D80C847BF9D30F30`. Both checkpoints remain available for follow-up. These losses come from four held-out pages per language and do not establish language understanding or semantic generation quality.
## 2026-11-17 — multilingual LoRA adaptation trials

Extended `scripts/evaluate_multilingual_probe.ps1` with optional `-AdapterPath`, preserving its base-checkpoint mode and recording adapter hash/path in the JSON report. Applied the existing C rank-8 output-head LoRA trainer to the fresh 144-page pre-SFT checkpoint and balanced 144/36 prompt pairs.

At rank 8, learning rate `3e-4`, and 1,000 updates, held-out response NLL decreased 8.173270→6.427828. Generation with the same 0.5 frequency penalty used by the probe collapsed: all nine prompts had repeated-4-gram ratio 0.9778; only the Latin prompts matched their expected script. Adapter SHA-256 `6F403E7903E03A261C3F48B0C35A26289026236308B0C65E1C9CF8D562E5777A`.

A second fresh rank-8 adapter at `1e-4` for 500 updates decreased response NLL only to 7.929609. All nine prompts still failed the probe; repeated-4-gram ratios were 0.8889 (Russian 0.8667). Adapter SHA-256 `EB83CBB042C6B3DBF3CFC746144D413CF9CFBBD56E6BB03B417C1DF944DBDE8D`. Both adapters are retained as failed experiments. The lower teacher-forced loss did not preserve free-generation diversity, so LoRA is not a selected multilingual capability checkpoint.
## 2026-11-17 — teacher-preserved SFT experiments

Extended `ama10_sft_train` with optional `--preserve-from teacher.ama --preserve-weight λ`. It loads a frozen, tokenizer-compatible teacher, computes next-token distributions in C, and trains through the existing soft-target BPTT path. At response positions the target is `(1−λ)·one_hot(gold) + λ·teacher_distribution`; the prompt-preserving variant trains prompt positions directly against the teacher distribution. `scripts/build.ps1 -Test` passed after adding an initialized-model SFT fixture with teacher preservation and checkpoint reload.

Two 500-update, width-128 runs started from the same 144-page pretraining checkpoint, with learning rate `3e-5` and λ=0.25. The first applied the blend only to response positions: response NLL 8.173270→6.805463, raw supervisor repeat-4 0.896→0.152, but Arabic validation NLL regressed 4.748570→5.130015 (beyond the 0.10 gate). The fixed script probe passed only Arabic; supervisor rolled back the candidate with `candidate_language_nll_regression`. Checkpoint SHA-256 `47A576A4B63F759522B35F64B727DD22E153739ADAD0BFC1C0889A8C07861742`.

The second also trained prompt positions against pure teacher distributions: response NLL 8.173270→6.837819 and raw repeat-4 0.896→0.640. Arabic NLL regressed 4.748570→5.135392; the script probe again passed only Arabic and the supervisor rolled back. Checkpoint SHA-256 `1B93CBE8077E83A4834C4F254F38801D77AE038C3098D4CC1B691BB607E23803`. These controlled runs show the teacher term reduces raw repetition but does not preserve Arabic evaluation or yield multilingual generation. Neither candidate is selected; the frozen-target SFT option remains experimental.
## 2026-11-17 — expanded multilingual pretraining, 5,000 updates

Trained a fresh width-128, depth-2 model with a 1,024-token BPE vocabulary on the current 144-page multilingual stream at learning rate `1e-4`. The 5,000 updates processed approximately 120,000 tokens against 96,896 training tokens. Aggregate training token NLL decreased 7.015696→6.891361 and held-out NLL decreased 7.015133→6.869580. Per-language held-out bits per byte were Arabic 3.525, German 5.480, English 4.610, Spanish 4.172, French 4.503, Japanese 5.063, Portuguese 4.386, Russian 3.644, and Chinese 6.169. Checkpoint SHA-256 `F30430B405F379ED3BB0E2082808C9295026881E914136CB807B9E6062E9B947`; tokenizer SHA-256 `0B3110A2ED3408D32DED44ADF7A4C5442A0D6277043264BD3C4462244DA20F5E`.

The nine-language generation probe had zero repeated four-grams in every output, but expected-script fractions ranged from 0.014 (Japanese) to 0.382 (French); all nine failed the 0.5 threshold. This is a failed language-generation checkpoint despite the reduced repetition and modest held-out loss improvement. The probe measures script adherence only, not semantic quality.

Continued the exact checkpoint with matching fingerprints and optimizer state to 20,000 total updates (about 4.95 passes over the 96,896-token stream). On the trainer's 512-target diagnostics, train NLL decreased 6.891361→5.919872 and validation NLL 6.869580→5.912891. Full held-out streams improved in all nine languages versus the 5,000-update checkpoint: Arabic 3.525→3.028, German 5.480→4.675, English 4.610→3.802, Spanish 4.172→3.398, French 4.503→3.733, Japanese 5.063→3.790, Portuguese 4.386→3.398, Russian 3.644→2.929, and Chinese 6.169→3.887 bits per byte. Checkpoint SHA-256 `5564EA7DD7ADEBECBA097C19DF4E29726B0860A8143C84AE68DEBEB1E12F6688`; tokenizer SHA-256 remains `0B3110A2ED3408D32DED44ADF7A4C5442A0D6277043264BD3C4462244DA20F5E`.

The fixed multilingual script probe with frequency penalty 0.5 passed seven languages (English, Spanish, French, German, Japanese, Portuguese, Chinese) and failed Russian and Arabic; every probe had zero repeated four-grams and valid UTF-8. Default unpenalized training diagnostic generation still collapsed (repeat-4 ratio 0.88, entirely CJK), so decoding stability remains unresolved. The per-language validation improvements demonstrate a real benefit from longer training but do not establish comprehension or fluent generation. Probe report: `build/multilingual-9lang-144p-pretrain20000-probe/report.json`.

A 500-update language-conditioned SFT continuation from the 20,000-update base lowered held-out response-token NLL 5.472781→4.883738. Full-stream held-out NLL improved for eight languages and regressed only slightly for Chinese (5.1497→4.5079 bits/byte-equivalent token NLL 4.4996→4.5079); the fixed probe passed eight languages (including Arabic) and failed Russian. The supervisor compared all nine language streams: aggregate NLL improved 5.566250→4.908417 and every named stream stayed within the 0.10 regression tolerance, but raw repeat-4 rose from 0.424 to 0.640, exceeding its allowed +0.05 regression. It rolled back to the 20,000-update base. SFT candidate SHA-256 `BF66BF6D258DA5DBD1F3C664583EFD07B183D418513DDDB48925D0C26FB749E3`; report `build/multilingual-9lang-144p-sft20k-supervisor.json`. This is a stronger teacher-forced fit but not an accepted generation checkpoint.

The follow-up SFT used the same 20,000-update base, 250 updates, learning rate `3e-5`, and teacher-preservation weight 0.05. Held-out response NLL improved 5.472781→4.998282. All nine language validation NLLs improved or stayed within 0.10, and the fixed probe passed seven languages (English, Spanish, French, German, Japanese, Portuguese, Chinese), failing Russian and Arabic. The supervisor rejected this checkpoint because raw repeat-4 increased 0.424→0.720, beyond the +0.05 relative tolerance (though below its absolute 0.75 ceiling). Candidate SHA-256 `F8DF88BFD1C6E248A7C9DC87EAAAC04CC4D9BC89811B1A7C882B8BC43F8B84CE`; report `build/multilingual-9lang-144p-sft20k-preserve005-supervisor.json`. The 0.05 preservation term retained more language validation performance than the 0.25 trials from the earlier 5,000-update base but did not fix generation diversity.

## 2026-11-17 — shuffled multilingual SFT and OASST1 conversation data

Changed `src/sft_train.c` to visit each training pair once per epoch in a deterministic Fisher–Yates order seeded from the train-file fingerprint, then reshuffle for the next epoch. This replaces the prior contiguous file-order update schedule, which grouped multilingual pairs by language. `scripts/build.ps1 -Test` passed with the change, including normal and teacher-preserved SFT checkpoint fixtures.

A controlled 500-update pass over the existing 144-page Wikipedia SFT set started from the 20,000-update pretrain checkpoint. Response NLL fell 5.472781→4.889979. The fixed generation probe passed eight of nine languages (Arabic now passed; Russian still failed), but the supervisor rejected the candidate: aggregate NLL improved 5.566250→4.920223 and every language NLL passed, while raw repeat-4 increased 0.424→0.664. Candidate SHA-256 `AD972AA39404E927C15EB5705667CED51EBD70705BD9818167D51B9708EA678D`; report `build/multilingual-9lang-144p-sft20k-shuffle-supervisor.json`. A sequential 500-update run from the same base achieved marginally lower NLL 4.908417 and lower repetition 0.640, but also failed the repetition regression gate. Shuffling is kept as the repeatable language-mixing schedule, not claimed as a quality win.

Added `scripts/prepare_oasst1_multilingual_sft.ps1` to derive reviewed prompt/response pairs from the already downloaded, commit-pinned OASST1 export. It checks the stored source hash and declared `apache-2.0` card license, applies the existing review/deletion/PII/language-mismatch/spam filters, requires prompt/assistant language agreement, splits entire message trees, writes per-language TSV and validation text, and records message/tree IDs and hashes. It prepared 3,670 train and 680 validation pairs across English, Spanish, French, Japanese, Chinese, German, Brazilian Portuguese, Russian, and Arabic. Selected Arabic has only 26 train and 3 validation pairs; Japanese has 144/23. An audit over all 4,350 selected rows found zero message-tree overlap between train and validation. Source SHA-256 `286A6E9A5A413B3272AE9C0B5A20D327983DEA1C24342AE28CB244A6DA65185C`; train TSV SHA-256 `7B9E5D62C95A0A27643696FDDD0A4115FB23AB15E0D1BFD191984EEA4F07C7C8`; validation TSV SHA-256 `968E7C3A2D0BE05A10E18EF884E6CB811A68A43EDC851C0BA86D7684BA815C33`.

One deterministic shuffled pass over those OASST pairs lowered held-out response NLL 5.517582→4.992050. Its script probe passed eight languages, including Russian, and failed Arabic. The nine-stream supervisor showed aggregate NLL 5.644088→5.026746; eight language streams improved, while Arabic regressed 6.039468→6.200924 (beyond tolerance). Raw repeat-4 fell 0.792→0.752 but still exceeded the 0.75 absolute ceiling. It selected the 20,000-update pretraining reference. Candidate SHA-256 `C4BA1BEC8C1E131311F6FE55ED1CB7646B3C48217ABD07876D66C93F7AFEB340`; report `build/oasst1-multilingual-sft-supervisor.json`. The lower Arabic result is consistent with its small 26/3 pair split, but this is an inference, not proof of the cause. OASST's repository card declaration does not establish independent rights to all user-submitted text.

A final 144-update continuation at learning rate `1e-5` used the balanced 144-pair Wikipedia language SFT set to adapt the OASST candidate. Wikipedia response NLL fell 5.225627→5.128323. The fixed probe still failed Arabic expected-script adherence but passed the other eight languages, including Russian. The supervisor compared the candidate against the OASST SFT parent on combined OASST + Wikipedia validation text and accepted it: aggregate NLL 5.042394→5.026950, repeated-4-gram ratio 0.752→0.680, valid UTF-8, and all nine named language NLL gates passed. Arabic improved 7.125591→6.572047 on the combined stream; the other languages also passed. Report: `build/oasst1-wiki-adapt-supervisor.json`. Accepted checkpoint SHA-256 `4DFBA111603AF7328C7F6075836874A00C8EC878599B97F3DEE1F305CD171956`; tokenizer SHA-256 `0B3110A2ED3408D32DED44ADF7A4C5442A0D6277043264BD3C4462244DA20F5E`. The durable selected copy is `checkpoints/ama10-oasst1-multilingual-wiki-selected-v0.1.ama`. This is a gate-passing multilingual checkpoint with an unresolved Arabic script-generation failure; the script probe does not establish semantic quality.

## 2026-11-17 — polite LibriSpeech expansion and shuffled CTC training

Extended the pinned `openslr/librispeech_asr` CC BY 4.0 sample using `scripts/fetch_hf_librispeech_rows.ps1` at train offset 20,000 and validation offset 1,000, adding 16/4 rows. The fetcher made serial requests with at least 30 seconds between metadata, row, and audio requests, and retried a transient HTTP 0 with backoff; no 429 occurred. The resulting 32/8 set has no duplicate source IDs or speaker overlap across splits. Train rows span speakers/chapters 374/180298 (8), 909/131045 (8), and 730/358 (16); validation rows span 2086/149220 (2), 2277/149896 (2), and 5338/24640 (4). It contains 14,412,240 bytes of converted WAV. The revision remains `71cacbfb7e2354c4226d01e70d77d5fca3d04ba1`. `sources.json` SHA-256: `23E2E6B54CEE186286C7F5DA9C1BFF695B12D3E9DF21DCBA397EAC14DFD0B1D6`.

Changed `src/audio_transcribe_train.c` to sample every train row once per epoch in deterministic Fisher–Yates order. The epoch permutation is derived from the training fingerprint and epoch number, so mid-epoch resume reconstructs the same order. The frontend fingerprint was bumped to `AMA10_LOGMEL40_FFT512_HAMMING_CTC_BLANK0_SHUFFLE_V2`, preventing accidental resume from a checkpoint trained with sequential ordering. `scripts/build.ps1 -Test` passed, including CTC checkpoint save/reload and generation integration.

A fresh random-init shuffled run for 2,500 updates reduced held-out CTC NLL per target from `67.840485` to `5.304610` over 463 reference tokens. This is nearly unchanged from the sequential-order 2,500-update run (`5.323833`). The shuffled checkpoint SHA-256 is `40484BB87544234D35661F2FE0B9778B36AC4B6EDD590A1589230E30D3DB54EA` at `checkpoints/ama10-librispeech-ctc-32x8-expanded-shuffle-v0.9.amm`.

Fixed decoding with max 40 tokens and blank penalty 0 produced pooled WER `1.000` (181/181) and CER `0.997978` (987/989). Penalty 4 produced WER `0.939227` (170/181) and CER `0.832154` (823/989). Against the sequential checkpoint's same-eight-clip penalty-4 result (WER `1.187845`, CER `0.697791`), shuffling reduced word insertions but raised character errors; hypotheses remain poor for every speaker group. Keep this checkpoint as a measured experiment, not a selected recognizer. Per-clip references and hypotheses are in `build/asr-ctc-32x8-expanded-shuffle-v0.9-final/`.

## 2026-11-17 — exploratory CTC transcript bigram fusion

Added an optional C11 bigram prior to `src/audio_transcribe_generate.c`. It reads only matching-language `train` rows from the supplied TSV manifest, tokenizes each transcript with the checkpoint tokenizer, counts BPE unigram/bigram frequencies, and adds a smoothed conditional-to-unigram log-ratio only when the decoder extends a collapsed CTC prefix. The insertion bonus is a separate parameter; both fusion controls default to zero when omitted. No audio checkpoint or training data was changed. `scripts/build.ps1 -Test` passed, including the new fixture invocation that loads a train-manifest LM during CTC generation.

Evaluated a grid of six LM weights (0–0.8) and four insertion bonuses (-0.1–0.2) on the 8 held-out clips, grouped by speaker. The no-LM penalty-4 baseline was WER 0.9392 (170/181), CER 0.8322 (823/989). Choosing settings on two speaker groups and scoring the third gave WER/CER of 0.9375/0.8536 on group A, 0.9677/0.8743 on B, and 0.9314/0.8487 on C. Grouped CV totals were WER 0.9392 (170/181), CER 0.8544 (845/989): no word-error improvement and worse character error. On the full validation set, the lowest-WER grid point (LM weight 0, insertion bonus -0.1) scored WER 0.9337/CER 0.8392; the lowest-CER point (weight 0.4, bonus 0.2) scored WER 0.9613/CER 0.7735. These are validation-tuned figures and not generalization evidence. Keep fusion optional and disabled by default; the acoustic recognizer remains unusable. Sweep outputs are in `build/ctc-lm-grid-v1/`.

## 2026-11-17 — 48/12 speech expansion and byte-target CTC trial

A request at train offset 30,000 returned zero rows, so the collector stopped before downloading any assets. Repeated with train offset 24,000 and validation offset 2,000. The same polite collector fetched 16/4 new clips at a minimum 30-second interval, pinned to revision `71cacbfb7e2354c4226d01e70d77d5fca3d04ba1` and card license CC BY 4.0; no 429 occurred. The resulting set has 60 clips (48/12), 22,309,160 converted WAV bytes, zero duplicate source IDs, zero train/validation speaker overlap, zero missing audio files, and zero WAV hash mismatches. Train speaker/chapter counts: 374/180298 (8), 909/131045 (8), 730/358 (16), 27/123349 (15), 4406/16882 (1). Validation counts: 2086/149220 (2), 2277/149896 (2), 5338/24640 (4), 3000/15664 (4). Sources manifest SHA-256 `7FE3DCB4C4B7AA0F1558602F1EAE0A250C8853BE3272A35C8A0B1689C2623233`.

A fresh shuffled BPE model trained 3,800 updates on 48/12, roughly matching the earlier update exposures per training clip. Validation CTC NLL fell 68.226909→5.176271 per BPE target over 872 tokens. On the same 12 clips at blank penalty 4, the prior 32/8 checkpoint scored WER/CER 0.9555/0.8492, while the 48/12 checkpoint scored 0.9674/0.8552. On the four new validation clips, WER was unchanged at 0.9744; CER improved 0.8690→0.8491. This small held-out group signal did not translate into an overall win. Checkpoint SHA-256 `26B1B2BE724A26ACA45B88F20214AE8B3724DC230A9964B4BDA3982A71103CDB`.

## 2026-11-17 — experimental shared-core text-to-speech

Added a C11 TTS trainer and generator on the shared AMA recurrent core. The trainer fits 64 train-only k-means centroids over 40D log-mel frames pooled to 50 frames/s, then predicts quantized acoustic codes after a language-tagged transcript. It writes a normal `.ama` model plus tokenizer and VQ codebook sidecars; VQ v2 records the training-derived frames-per-transcript-byte duration ratio and can still load v1 codebooks. Generation maps predicted codes to approximate mel magnitudes, estimates phase with deterministic random-phase Griffin–Lim, and writes mono PCM16 WAV. The unit test checks codebook fitting, checksum persistence, quantization, synthesis, and WAV reload; the build integration test trains, resumes, and generates a fixture WAV.

On the revision-pinned LibriSpeech 48/12 sample, a fresh 1,000-update run reduced held-out token NLL 5.735589→4.672703. Resuming to 1,100 updates reached 4.635425; after VQ v1→v2 migration, one more update reached 4.634959. The 1,101-update validation loss remains high. The generator failed to emit EOS, so it uses the train-derived duration estimate: a held-out transcript produced a 10.96 s WAV, while `THE BLUE BIRD IS SINGING OUTSIDE` produced 2.18 s. `ffprobe` confirmed both files are mono, 16 kHz WAVs (10.975 s and 2.195 s, accounting for frame rounding). This verifies the data/training/checkpoint/audio-write path, not intelligibility or naturalness. Model SHA-256 `F78F0A332285E904D9A32E92F78A29E1636621A0B2CF5C5AD8861881EA5C481A`; tokenizer SHA-256 `BF2B90AC2B408ECDFB3AE2364720DFB44E47DAF610C1EE69AF71699E4919FE65`; VQ v2 SHA-256 `945D3D60B90CD1BC4C2D826940A16DE66052D951CC285DE5F557134C3B7FECF4`.

## 2026-11-17 — shared-core text-to-image patch generator

Added a C11 image-codebook, trainer, and text-conditioned generator. It fits 64 k-means centroids from training-set 8×8 RGB patches only (192 floats per patch), encodes each 64×64 source image as 64 visual codes, and trains the shared recurrent core to predict the code sequence after a byte-tokenized `<lang:xx> caption newline NUL` prefix. The `.vq` file has a checksum and binds to the train split hash; model and byte tokenizer are saved as `.ama` and `.tok`. Generation samples 64 patch codes at a configurable temperature (default 0.8) and writes a P6 PPM. The codebook API test covers fit, save/load, quantization, reconstruction and PPM reload. The trainer resets recurrent state for each independent image-caption pair and shuffles train rows deterministically by epoch; adding the reset bumped the data fingerprint to `AMA10_TEXT_TO_IMAGE_VQ64_PATCH8_RESETSTATE_V2` and the corrected trial started from random weights.

On the 11-image Commons sample (14 train caption rows, six validation caption rows; eight training and three held-out images), a fresh 1,000-update run reduced all-token validation NLL 5.561940→4.714318 and image-code-only NLL 5.583538→4.476537. Resuming the same checkpoint to 5,000 updates lowered the respective losses to 4.069213 and 3.842064. One sampled output using the held-out fox caption scored 4.331 dB PSNR and 0.066140 8×8-window SSIM with the new native C scorer; FFmpeg's differently windowed SSIM filter reported 0.151372 on that image. The output is a noisy patch mosaic, not semantically recognizable image generation. Greedy decoding on the earlier state-contaminated experiment collapsed to a solid black image (2.50 dB PSNR); sampling was added to expose code diversity. On the corrected run, the sampled 1,000-update image scored 4.18 dB PSNR, showing that lower code NLL has not yet yielded meaningful fidelity. Source manifest SHA-256 `2E92DB6D6D98C3EA688A23C27FAEAF64BBE5352ABC9C9453BF29E952D4B4EC7B`. 5,000-update checkpoint SHA-256 `7996581A73862F6480184D327AE2D7EE94D697B04A93B5BD047FD2A40BC20613`; tokenizer SHA-256 `BF2B90AC2B408ECDFB3AE2364720DFB44E47DAF610C1EE69AF71699E4919FE65`; codebook SHA-256 `5B47C9CC25F91C9EE1526288AC8C306B97D2BE0B29D6FA2F41942E21E14E083D`.

## 2026-11-17 — per-row INT8 CPU inference

Added a separate C11 `.q8` inference format and CLI. Conversion is per-output-row symmetric int8: each matrix row stores signed byte weights plus one float32 scale; biases, recurrent states, activations, and logits remain float32. The payload is dimensioned and checksummed, written with staged replacement, and load is transactional. The Q8 runner supports UTF-8-checked generation, repetition penalties, and validation text NLL. It reads the `.tok` sidecar on demand and does not load optimizer moments or training state. `tests/q8_test.c` checks repeated atomic save, roundtrip, checksum corruption rejection, memory accounting, and nine-token recurrent logits against F32. The fixture's maximum logit difference was 0.006832998 and RMS difference 0.00209112.

Quantized the accepted 23,814-update multilingual checkpoint (`checkpoints/ama10-oasst1-multilingual-wiki-selected-v0.1.ama`, width/input-width 128, vocabulary 1,024). F32 parameters occupy 1,311,744 bytes; Q8 weights, scales, and biases occupy 335,360 resident bytes (74.4% less). The `.q8` file is 335,392 bytes. On the 4,131 next-token predictions in `data/processed/multilingual_expansion_validation_144/en.txt`, F32 token NLL was 5.11278874 and Q8 was 5.11621787 (delta +0.00342913). Q8 generation completed 64 tokens with valid UTF-8. On this Intel Core i5-13500H, three end-to-end 256-token generations averaged 37.8 ms for Q8 and 55.4 ms for F32; each timing includes process startup, model loading and output writing, so it is an indicative local result. The exact sampled continuations diverged after shared tokens (byte agreement 50.6%), as expected from recurrent error accumulation; the low teacher-forced NLL change is stronger evidence than exact greedy-output matching. Q8 SHA-256 `5D2B372A4EFD67146DD91F87872D7D7FAEAB219B49DFE2EEBC7CF39ACFB3AD10`; tokenizer sidecar SHA-256 `0B3110A2ED3408D32DED44ADF7A4C5442A0D6277043264BD3C4462244DA20F5E`.

Then changed the fresh CTC tokenizer budget to 256, making transcript labels byte/character-level and changing the data-pipeline fingerprint to `AMA10_LOGMEL40_FFT512_HAMMING_CTC_BYTE256_BLANK0_SHUFFLE_V3`; this forced random initialization. A 2,000-update shuffled run reduced per-byte held-out CTC NLL 27.851816→2.932684 over 1,847 target bytes. Despite that, default decoding was blank (WER/CER 1.000/1.000); penalty 4 caused repeated `E` insertions (WER 6.614/CER 2.139). A 10-point blank-penalty sweep's lowest CER was 0.872 at penalty 2.5, but WER was 1.321. This target change did not fix ASR and is not selected. Byte-level checkpoint SHA-256 `FBFE8530A29D67B88D15CC81DFE465874574615A085939ADA66C07F68BC7FD3E`. Per-clip output and pooled score reports are under `build/asr-ctc-48x12-comparison-v1.0.json` and `build/asr-ctc-byte256-48x12-v1.1/`.

## 2026-11-17 — C SwanLab scalar uploader

Added optional per-update CSV output to the C training CLI (`--metrics-csv`) and a standalone C/WinHTTP uploader (`ama10_swanlab.exe`). The uploader exchanges `SWANLAB_API_KEY` for a SwanLab session, looks up an existing project, creates an experiment, declares scalar series, uploads scalar batches, and closes the run. It uses no Python SDK. Its write endpoints follow the current SwanLab SDK implementation, not the stable public OpenAPI; network upload was not exercised because no API key or workspace was configured. `tests/swanlab_test.c` verifies the client refuses to make a request when credentials are absent. See the [SwanLab REST API overview](https://docs.swanlab.cn/en/api/rest/) and the current [SDK scalar upload implementation](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/sdk/internal/core_python/api/upload.py).
## 2026-11-17 — 20-language polite Wikipedia expansion

Added eleven language editions (Bengali, Persian, Hindi, Indonesian, Italian, Korean, Dutch, Polish, Swedish, Turkish, Vietnamese) to the existing nine-language fetch set. The collector waits at least 30 seconds between language requests, sends `maxlag=5` and a descriptive User-Agent, honors `Retry-After`, and backs off at least 120 seconds on HTTP 429. The fetch completed without a 429. Seven unavailable train titles are listed in `data/raw/multilingual_expansion_20lang_sources.missing.json`; all 58 requested validation leads were retrieved. The resulting corpus has 225 train / 58 validation article leads, 392,273 / 100,539 bytes, and keeps page-level train/validation separation. The source manifest records revision IDs, URLs, retrieval times, and CC BY-SA 4.0 terms. SHA-256: train `644F327E5CBBEF0BA540250AA4325064CE345B8873E134D51672B4DE7545B56E`; validation `9B9CBE9340F58A9BB22DF48E65DBC234FBE66E3CC62B11058CE317C0BE4993A4`; source manifest `796DA03CD50D220AD0AD1852491DFE8E412CEBDAAE4576A65BF9C8BAD9671F9F`.

Prepared 225/58 language-conditioned SFT pairs under `data/processed/multilingual_sft_20lang/`, carrying row-level source provenance. Train TSV SHA-256 `0F8DFF4F9492C0A8DF5E014AF4BCF9D43F1FC3298610DD3A24B49FE95737A767`; validation TSV `E7ACEB485CD8D385AB1E196CBF364781AF9440E8DAAD3BADD483D1F6F70AB93B`.

Fresh pretraining used seed 10, width 128, vocabulary 1,024, accumulation 2, learning rate `1e-4`, inverse-square-root decay 0.1, and 20,000 updates. It started from random initialization. Stream train/validation token NLL changed 7.039829→6.649181 and 6.964666→6.639658. Checkpoint SHA-256 `D6551C6CF1D0862D0351A50547B0C03CD407BB38A95EF6E462B112A4296DF595`; tokenizer sidecar `3DA0EFE31A638A12F552363605ACE805D53B97ABE89368C67B9F41113CCF282B`.

Per-language bits/byte, comparing the earlier 23,814-update nine-language checkpoint to this fresh model on the same held-out article leads:

| Language | Earlier | Fresh |
| --- | ---: | ---: |
| Arabic | 3.344 | 3.685 |
| Bengali | 10.581 | 3.594 |
| German | 4.201 | 4.991 |
| English | 3.440 | 4.168 |
| Spanish | 3.130 | 3.895 |
| Persian | 5.748 | 3.551 |
| French | 3.334 | 4.199 |
| Hindi | 10.640 | 3.465 |
| Indonesian | 4.254 | 4.586 |
| Italian | 3.794 | 4.194 |
| Japanese | 3.530 | 5.436 |
| Korean | 10.642 | 6.421 |
| Dutch | 3.851 | 4.421 |
| Polish | 5.448 | 5.286 |
| Portuguese | 3.208 | 4.041 |
| Russian | 2.215 | 4.059 |
| Swedish | 4.276 | 4.704 |
| Turkish | 5.629 | 4.795 |
| Vietnamese | 9.010 | 5.086 |
| Chinese | 3.864 | 6.608 |

This demonstrates markedly better byte efficiency for several newly added writing systems, while regression in several original languages shows substantial capacity dilution at this small model/data scale. These results do not establish useful text generation: training diagnostics still reported a 0.936 repeated-4-gram ratio. Evaluation outputs are preserved in `build/20lang-baseline.txt` and `build/20lang-fresh-eval.txt`, with update metrics in `build/20lang-fresh-metrics.csv`.

A 500-update language-conditioned SFT continuation at `5e-5` reduced held-out response-token NLL from 6.473479 to 6.450538 on 58 prompt/response pairs. It did not pass the existing nine-language generation probe: English, Spanish, French, German, and Portuguese passed the expected-script threshold, while Japanese, Chinese, Russian, and Arabic generated no expected-script characters. Some per-language Wikipedia held-out losses also worsened after SFT. This is recorded as a diagnostic checkpoint, not a selected model. SFT checkpoint SHA-256 `1CB7126661EEAC9E223C10A7A84FF55264371409B8551D78ECD3D72BEECAC51C`; tokenizer SHA-256 `3DA0EFE31A638A12F552363605ACE805D53B97ABE89368C67B9F41113CCF282B`. Per-language results are in `build/20lang-sft500-eval.txt`; the standard probe report is `build/multilingual_probe/report.json`.

## 2026-11-17 — supervisor expected-script gate

Extended `ama10_supervisor` so named language validation streams also run a deterministic candidate continuation and report its expected Unicode-script fraction. Supported tags include Latin-script languages, CJK, Japanese kana/CJK, Korean Hangul, Cyrillic, Arabic/Persian, Hindi Devanagari, and Bengali. The default minimum is 0.50, matching `evaluate_multilingual_probe.ps1`; callers can set the optional final CLI argument from 0 to 1. Candidates below the threshold roll back with `candidate_expected_script_fraction_too_low`. Unknown tags retain their per-language NLL gate and are marked as script-unknown. Expanded C evaluation metrics with a Bengali-script counter. Integration coverage checks the passing threshold and a stricter threshold that rejects the identical candidate; Unicode-block tests include Bengali. This is shallow language-selection evidence and cannot judge semantic correctness.

Applied the gate to the 20-language SFT candidate versus its fresh-pretrain parent. With aggregate/per-language NLL tolerance 10 and repetition maximum 1 (isolating script adherence), the supervisor rolled back: only 12/20 language continuations met the 0.50 threshold. Arabic, Bengali, Persian, Hindi, Japanese, Korean, Russian, and Chinese each generated 0% of the expected script. The real report is `build/20lang-sft500-supervisor.json`; the selected output is the pretraining parent at `build/20lang-supervisor-selected.ama`. This agrees with the standard nine-language probe's rejection and prevents a modest aggregate response-NLL improvement from masking language-selection failures.



## 2026-11-17 — adaptive-depth held-out evaluation

Added an adaptive inference evaluator to the public C API and `ama10 evaluate`. When passed a maximum depth and halt threshold, the CLI now reports fixed two-step NLL, fixed maximum-depth NLL, adaptive NLL, and mean executed depth on the same uninterrupted token stream. The C regression suite checks that threshold 2.0 exactly matches fixed-depth evaluation and threshold 0.0 stops at the required minimum of two steps.

On the accepted 23,814-update checkpoint `ama10-oasst1-multilingual-wiki-selected-v0.1.ama` (SHA-256 `4DFBA111603AF7328C7F6075836874A00C8EC878599B97F3DEE1F305CD171956`), the 4,131-prediction English validation stream measured NLL 5.11278874 at fixed depth two and 5.12003127 at fixed depth four. With maximum depth four, threshold 0.88 yielded NLL 5.10330548 at mean depth 2.570806; threshold 0.90 yielded 5.10546903 at 2.909223; threshold 0.92 yielded 5.10963289 at 3.347131. Threshold 0.80 used mean depth 2.008715 with NLL 5.11024141. The activity heuristic therefore changed both compute and loss on this stream, but this single-language, single-checkpoint result does not show a robust speedup or general quality improvement. Runtime was not benchmarked, and the threshold needs selection on separate validation data before use as a default.

The adaptive-depth sweep was extended to the nine available language validation streams at threshold 0.88, max depth 4. Each of the three evaluation paths was timed inside the evaluator and run three times; values below are median CPU seconds. Fixed-depth NLL is deterministic; mean adaptive depth is in microsteps per token.

| Language | Fixed-2 NLL | Fixed-4 NLL | Adaptive NLL | Mean depth | Fixed-2 s | Fixed-4 s | Adaptive s |
|---|---:|---:|---:|---:|---:|---:|---:|
| English | 5.112789 | 5.120031 | 5.103305 | 2.571 | 0.335 | 0.492 | 0.389 |
| Spanish | 5.207763 | 5.247341 | 5.220085 | 2.649 | 0.282 | 0.425 | 0.329 |
| French | 5.118071 | 5.148891 | 5.140856 | 2.624 | 0.260 | 0.387 | 0.303 |
| German | 5.279997 | 5.315633 | 5.298011 | 2.479 | 0.138 | 0.205 | 0.154 |
| Portuguese | 5.087416 | 5.152094 | 5.122556 | 2.725 | 0.255 | 0.383 | 0.299 |
| Japanese | 4.871448 | 4.955788 | 4.955886 | 2.762 | 0.240 | 0.367 | 0.291 |
| Chinese | 4.472983 | 4.732962 | 4.709768 | 2.896 | 0.163 | 0.246 | 0.198 |
| Russian | 4.228197 | 4.223145 | 4.226616 | 2.261 | 0.338 | 0.504 | 0.364 |
| Arabic | 6.640503 | 6.559606 | 6.721964 | 2.293 | 0.320 | 0.474 | 0.340 |

At this threshold, adaptive inference was 6–22% slower than fixed depth two and 19–28% faster than fixed depth four on these runs. It lowered NLL versus fixed depth two in English and Russian; the other seven languages regressed. Compared with fixed depth four, it was faster across all nine streams and had lower NLL in six. These measurements argue against enabling this heuristic by default: the current max-depth-four procedure is a compute/quality tradeoff that needs threshold selection or a learned halting rule, not a general improvement. The timings exclude process startup and model loading and were measured on one Windows laptop/CPU; they are not portable throughput claims.


## 2026-11-17 — recurrent-state carry ablation

Extended `ama10 evaluate` with an optional state-reset period, allowing one model and token stream to be scored with its state carried continuously or zeroed at a fixed token interval. The fixed two-step path is unchanged; a reset run reports its own token NLL and timing. On the same 23,814-update selected checkpoint, resetting every 24 tokens was worse in seven languages but improved Japanese by 0.146 nats/token and Arabic by 1.242. Exact NLL (continuous → reset-every-24; delta): English 5.112789→5.257868 (+0.145079), Spanish 5.207763→5.402561 (+0.194798), French 5.118071→5.406768 (+0.288697), German 5.279997→5.308129 (+0.028132), Portuguese 5.087416→5.295396 (+0.207980), Japanese 4.871448→4.725483 (−0.145965), Chinese 4.472983→4.737171 (+0.264188), Russian 4.228197→4.599130 (+0.370932), Arabic 6.640503→5.398388 (−1.242114). English resetting every 128 tokens scored 5.152858 versus 5.112789 carried continuously.

This is a fixed-interval ablation, not a document-boundary experiment: the reset points can cut through an article, and the per-language outcomes conflict. It confirms state carry materially changes held-out loss, but it does not yet establish the right state lifetime or whether reset gains persist with document-aligned segmentation. Training still carries recurrent state across TBPTT chunks and resets only at corpus wrap.

## 2026-11-17 — page-boundary state carry evaluation

The fixed-interval reset ablation can cut through a document, so I added `scripts/prepare_wikipedia_eval_pages.ps1` and the C `evaluate-documents` command. The helper uses cached, revision-attributed validation page titles to split each existing per-language stream on its actual four page starts and substitutes form-feed separators. Replacing those separators with LF reconstructs each original validation file byte-for-byte for all nine languages. The C evaluator tokenizes each page independently, excludes the cross-page target from both conditions, carries state through all page tokens in one condition, and resets it at each page start in the other. Each row below scores the same within-page targets (total predictions = total tokens minus four pages).

| Language | Predictions | Carry NLL | Reset at page NLL | Reset minus carry |
|---|---:|---:|---:|---:|
| English | 4,128 | 5.113251 | 5.125178 | +0.011927 |
| Spanish | 3,518 | 5.208953 | 5.206967 | −0.001986 |
| French | 3,221 | 5.119711 | 5.138623 | +0.018912 |
| German | 1,692 | 5.279727 | 5.267976 | −0.011751 |
| Portuguese | 3,135 | 5.088475 | 5.070903 | −0.017572 |
| Japanese | 2,970 | 4.871189 | 4.854158 | −0.017031 |
| Chinese | 1,997 | 4.470687 | 4.488462 | +0.017775 |
| Russian | 4,168 | 4.231631 | 4.245725 | +0.014094 |
| Arabic | 3,948 | 6.634167 | 6.464862 | −0.169305 |

Resetting at article boundaries improved five languages and worsened four. The effect is small in eight streams and substantial for Arabic, so this checkpoint does not support one global carry/reset policy. Page alignment removes the arbitrary-boundary confound, but four validation pages per language are too few to establish a general memory rule. The evaluator and page preparer are diagnostics; training still carries state across TBPTT chunks and resets at corpus wrap.

## 2026-11-17 — rejected 25,000-update 20-language continuation

Resumed the v0.1 fresh-from-random checkpoint from update 20,000 to 25,000 on the same pinned 20-language corpus, width 128, vocabulary 1,024, accumulation 2, AdamW, base learning rate `1e-4`, inverse-square-root decay `0.1`, sequence length 24, and seed 10. The original v0.1 checkpoint remains unchanged. The trainer's 512-token diagnostics worsened: train NLL 6.649181→6.657655 and validation NLL 6.639658→6.660856. Final training mean chunk loss was 6.33624; this is not directly comparable to full-stream NLL.

The C supervisor compared v0.2 against v0.1 on the complete 47,014-token 20-language validation stream with deterministic language probes. It rolled back v0.2: full-stream NLL increased 6.412491→6.436085, repeated-4-gram ratio rose 0.904→0.912, and only 9/20 continuation probes met the 0.50 expected-script gate. UTF-8 remained valid. The supervisor-selected file's hash exactly matches the untouched v0.1 parent. The rejected v0.2 checkpoint SHA-256 is `F1172FE734CACB251F624638F3247F2E23882FEC781386D769E991650C25C1E2`; tokenizer SHA-256 remains `3DA0EFE31A638A12F552363605ACE805D53B97ABE89368C67B9F41113CCF282B`. Checkpoint and report are retained for analysis at `checkpoints/ama10-20lang-continued-v0.2.ama` and `build/20lang-continued-supervisor.json`.

## 2026-11-17 — matched 20-language interleaved-order experiment

Added `scripts/prepare_multilingual_interleaved.ps1`, which reads the cached, revision-attributed page records and writes a deterministic round-robin document stream across 20 languages. It retains all 225 training pages and the exact 392,273 training bytes; the validation file is copied byte-for-byte (SHA-256 `9B9CBE9340F58A9BB22DF48E65DBC234FBE66E3CC62B11058CE317C0BE4993A4`). The interleaved train SHA-256 is `65A3B507E20142986403DD74318B26E74688AD20DDE5D76E377F19604EEF1F19`. Its BPE tokenizer hash matches the grouped-order baseline exactly (`3DA0EFE31A638A12F552363605ACE805D53B97ABE89368C67B9F41113CCF282B`), so this experiment changes document order while retaining the same vocabulary.

Trained a new model from random seed 10 for 20,000 updates, width 128, vocabulary 1,024, accumulation 2, AdamW, learning rate `1e-4`, and inverse-square-root decay 0.1. The run resumed from its preserved 10,000-update checkpoint without resetting optimizer moments. Its trainer validation diagnostic at 512 tokens ended at 6.703896 versus 6.639658 for the grouped baseline. On the same full per-language held-out files, bits per byte improved in 9/20 languages and regressed in 11; deltas (interleaved minus grouped) ranged from −0.188 in Russian to +0.195 in Indonesian.

The deterministic supervisor probe is decisive: the grouped 20,000-update checkpoint passed 12/20 expected-script checks, while the interleaved checkpoint passed 0/20 at the 0.50 threshold. Its self-evaluated full-stream NLL was slightly lower (6.394839 vs 6.412491), but repeated-4-gram ratio worsened from 0.904 to 0.928 and all language probes failed. Therefore this round-robin document order is rejected despite a small aggregate NLL gain; balancing page counts alone is not an adequate multilingual sampler. Interleaved checkpoint SHA-256 `0C9B8B226331167D0C121FF2185FF2363FD1E4945E69B6B02FDDF000BED61210`. Checkpoints are retained at `checkpoints/ama10-20lang-interleaved-v0.1.ama` (10k) and `checkpoints/ama10-20lang-interleaved-v0.2.ama` (20k); the report is `build/20lang-interleaved-20k-probe.json`.

## 2026-11-17 — rejected explicit language-tag pretraining

Added `scripts/prepare_multilingual_tagged.ps1` to prepend `<lang:xx>` before every source article. Before writing the derivative data, it verifies SHA-256 equality between cached page reconstruction and the original grouped training and validation files. It retains all 225/58 pages and their order, writes 20 tagged language-specific validation streams, and preserves page IDs, revisions, URLs, and CC BY-SA 4.0 attribution in `data/processed/multilingual_20lang_tagged/sources.json`. Tagged train and validation SHA-256 are `FB4225AAF39BF3E48FD291CD02786C3BA317EE9798E9211D06E862F9065F5F61` and `CD01730F684E4C6B1A2C9B51852D84E87EE88FEBA48A706B3EA1C63C1028C869`.

Trained a fresh seed-10 width-128 model for 20,000 updates with the grouped article order and the same optimizer schedule as the untagged baseline. On the exact same tagged full validation stream, candidate NLL/bits-per-byte were 6.420840/4.330038 versus 6.421192/4.347315 for the untagged baseline. Per-language bits/byte improved in 16/20 languages (largest gains: Chinese −0.196, Swedish −0.115, Polish −0.110, Arabic −0.085) and regressed in four (largest: Spanish +0.061, French +0.048, English +0.043). Despite this small held-out likelihood gain, language selection became worse: using the same tagged prompts and the 0.50 script threshold, the untagged baseline passed 12/20 probes and the tagged model passed 0/20. Its repeated-4-gram ratio also rose from 0.848 to 0.888. Therefore explicit article tags in this pretraining format are rejected; the original grouped checkpoint remains the selected baseline. Teacher requests were disabled. Candidate checkpoint SHA-256 `73BAC97E1BF24002E333C85E2D34327C55EEEE85ED5BFE791F8D8C05772EFBFC`; tokenizer SHA-256 `209CD6C01CD0550485379A0F00E2289789D1B5433AC77899887FA52E00A177BC`. Per-language scores are in `build/ama10-20lang-tagged-v0.1-tagged-eval.txt` and the prompt-matched baseline in `build/ama10-20lang-fresh-v0.1-tagged-eval.txt`; probe reports are `build/20lang-tagged-probe.json` and `build/20lang-fresh-tagged-prompt-probe.json`.

## 2026-11-17 — rejected five-pass shuffled-page order

To break the repeated language transitions in the round-robin stream, added `scripts/prepare_multilingual_epoch_stream.ps1`. It makes five deterministic, independently seeded permutations of all 225 revision-attributed training pages. Every pass reconstructs the original 392,273-byte corpus exactly, all pages occur once per pass, and validation remains byte-identical (SHA-256 `9B9CBE9340F58A9BB22DF48E65DBC234FBE66E3CC62B11058CE317C0BE4993A4`). The concatenated training stream is 1,961,369 bytes with SHA-256 `CEDB397CEDAADA30010CBF4A535550129991D6D7A39DBAFBD6417B5092130529`; source order and revision metadata are preserved in `data/processed/multilingual_20lang_epoch_shuffle_sources.json`.

Trained from random seed 10 for 20,000 updates using the same width (128), vocabulary (1,024), accumulation (2), AdamW, learning rate (`1e-4`), and inverse-square-root decay (0.1) as the grouped baseline. The trainer's 512-token validation NLL ended at 6.737033 (grouped baseline: 6.639658); full held-out bits/byte improved in only 7/20 languages. Deltas (shuffled minus grouped) were Arabic −0.040, Bengali +0.005, German +0.025, English +0.046, Spanish +0.066, Persian +1.327, French +0.041, Hindi −0.055, Indonesian +0.038, Italian +0.048, Japanese −0.057, Korean −0.090, Dutch +0.040, Polish +0.007, Portuguese +0.051, Russian −0.200, Swedish −0.004, Turkish +0.040, Vietnamese −0.084, and Chinese −0.059. The expected-script supervisor probe passed 0/20 at threshold 0.50, so this checkpoint is rejected; the grouped baseline remains the selected model. The probe ran with the Cloudflare teacher disabled. Checkpoint SHA-256 `0DDC373CABFE3FB0FA379B1465B02D9FF55F07022FBEB5F9CF843E0D2529A4E2`; tokenizer SHA-256 `A8A52D7459D314EEBE5ADA74F55F05FB146B8F418D360089364D88A9B6BEEB60`. Per-language scores are preserved in `build/20lang-epoch-shuffle-eval.txt`, and generation diagnostics in `build/20lang-epoch-shuffle-probe.json`.

## 2026-11-17 — designated dataset source catalog and 503 clarification

Added a registry for all 26 designated training sources in `data/pipeline/datasets.json`, preserving pending access, uncertain source identity, and unresolved official shard availability as explicit states. The companion `data/pipeline/status.jsonl` is a local append-only status index; it is not an external database and has no network connection string. Dataset counters begin at zero because none of these newly designated source shards had been acquired or consumed at catalog creation. The initial text mix target is recorded as 38% English, 29% Chinese, 21% multilingual, and 12% code/math tokens.

The already-running polite LibriSpeech HF collector completed 20 more rows at its established 30-second request interval and committed 80 total examples (64 train, 16 validation) with 80 WAV files and matching provenance entries. The supplied 503 log is an HTTP service-unavailable response from Hugging Face, not a database identifier or endpoint. The existing retry sequence backed off 30, 60, and 120 seconds before recovering. No new fetch was started in response to that 503. This LibriSpeech set is existing ASR work, not FLEURS and not evidence that any of the 26 designated sources participated in model updates.

The detailed shard lifecycle and experiment plan is in `docs/data_acquisition_plan.md`. This catalog step did not download or process any of its designated sources, and it did not initiate training. The existing C11 dataset module currently streams local plain-text files into token caches; it is not yet a general native-format shard pipeline. No designated-source training update or multimodal capability change is claimed.

## 2026-11-17 — C11 dataset shard ledger

Implemented `src/pipeline_ledger.c`, its public C11 interface, and `ama10_pipeline_ledger` CLI. The append-only writer validates dataset IDs against the full 26-source catalog and validates lifecycle states, SHA-256 syntax, acquired/validated byte ordering, and valid/consumed token ordering. It JSON-escapes user fields and uses an exclusive Windows file lock plus `FlushFileBuffers` so separate writers do not interleave records. Added a focused regression test for quoting, control-character escaping, unknown dataset IDs, and invalid counters.

The first full validation run caught a serializer allocation undercount and failed with Windows heap-corruption exit `-1073740940`. Increased the fixed JSON overhead allocation, rebuilt the targeted regression executable, and confirmed `C11 shard ledger append, JSON escaping, and counter validation: ok`. A subsequent full `scripts/build.ps1 -Test` run exited 0. Its output included the new ledger regression and the existing gradient, finite-difference, checkpoint, image, audio/TTS, Unicode/tokenizer, q8 parity, tool, and Cloudflare teacher allowlist checks.

This advances status recording, but is not yet a downloader or native shard reader. No designated source shard was downloaded or processed, no designated tokens were consumed, and no training update used the new registry. The full-scale acquisition and training goal remains active.

## 2026-11-17 — tokenizer encoding throughput improvement

Replaced the tokenizer's production encoding loop, which rescanned the full remaining sequence for every merge rank, with an indexed pair-rank lookup and min-heap of eligible adjacent merges. The heap retains the original merge-rank and left-to-right tie ordering; stale adjacency candidates are discarded by node-version checks. Added a simple reference implementation to `tests/tokenizer_test.c` and verified byte-for-byte token ID parity on the existing UTF-8/emoji/NUL fixture, the complete repeated training fixture, and 128 KiB repeated and deterministic-random byte streams. Strict C11 `-Wall -Wextra -Wpedantic -Werror` build and tokenizer regression passed. The training process already running at implementation time uses the prior executable and is unchanged; no speedup is claimed for that process.

## 2026-11-17 — high-resolution image ingestion and automotive source registration

The C11 image loader now decodes P6 PPM, JPEG, PNG, BMP, and TGA into one RGB8 representation. Image-caption training and generation use the general loader. The high-resolution caption path accepts varying image dimensions and uses overlapping 16×16 tiles (stride 12) plus a global view and four quadrant views. A synthetic 1080×640 RGB input produced 4,770 fine tiles plus five context views (4,775 total; 768 floats per view). This is a resolution/coverage test, not evidence of learned car-scene understanding. Strict C11 builds of the caption train/generate programs succeeded; runtime loader smoke checks decoded 3×2 PPM, PNG, and JPEG fixtures.

Seven automotive candidates are now additive entries in `data/pipeline/datasets.json` and are accepted by the C11 append-only ledger: BDD100K, nuScenes, Waymo Open Dataset, Argoverse 2, KITTI, Audi A2D2, and comma2k19. Each remains `catalogued_pending`; none was downloaded or trained in this pass. Official references document BDD100K's 100K video archive (1.8 TB) and 100K keyframe images (5.3 GB), nuScenes camera images stored as JPEG with exposure timestamps and calibration metadata, Waymo's versioned GCS-hosted TFRecord/modular releases, and Argoverse 2's sensor release with synchronized camera imagery. See [BDD100K download documentation](https://github.com/bdd100k/bdd100k/blob/master/doc/source/download.rst), [nuScenes data format](https://www.nuscenes.org/data-format), [Waymo Open Dataset download](https://waymo.com/open/download/), and [Argoverse 2](https://www.argoverse.org/av2.html). Exact package/revision manifests will be pinned before acquisition.

The existing image-generation trainer remains the experimental 64×64, 8×8 patch, 64-code k-means path. It has not been upgraded by the caption ingestion work; no high-resolution generation quality is claimed. The unified vision interface does not yet encode video timestamps, camera calibration, or temporal frame state. Those are explicit remaining gaps before vehicle-camera training.

## 2026-11-17 — FineWeb-Edu run completion and compute reuse

The existing production run completed normally at 10,000 AdamW updates, width 128, vocabulary 1,024, depth 2, sequence length 24, and accumulation 2. It consumed 480,000 token positions from the 1,427,334,072-token train cache (about 0.034% of available train tokens; repeated/overlap sampling is not counted as unique coverage). Train token NLL fell from 7.030832 to 6.333781; held-out validation token NLL fell from 6.977068 to 6.339499. The deterministic 128-token free continuation had distinct-token ratio 0.0156 and repeated-4gram ratio 0.8960, so continuous generation remains a failure despite the NLL decrease. Checkpoint SHA-256: `87676808DD9D768A1144CD8A0660105E09FECD59CB9BC6021F02DCC73EA96EFF`. The shard remains in `training_in_progress`; this run is partial-corpus optimization and is not a completed pretraining pass.

Source counts are 2,358,493,358 downloaded/validated bytes, 759,857 unique documents, 1,427,334,072 train tokens, and 74,318,180 held-out tokens. There was one within-shard duplicate; all 759,858 rows passed UTF-8 validation and no score filter could be used because the source score field was null. The local append-only ledger now records 480,000 consumed token positions. The hardware run was CPU AdamW; no OpenCL timing was taken in this task segment.

The timed resume retained the original optimizer checkpoint and ran 200 additional updates to step 10,200 on the same FineWeb-Edu source and configuration. C11 QueryPerformanceCounter instrumentation measured 18.699 s in update work (10.696 updates/s; 513.4 input token positions/s), with checkpoint writes included. Tokenizer/checkpoint loading took 7.874 s, cached token-dataset open 0.111 s, the trainer's initial/final 512-token train+validation diagnostics 0.433/0.417 s, and the 128-token generation 0.079 s. This was CPU AdamW. The matched train NLL changed 6.333781→6.329329; the 512-token validation diagnostic changed 6.339499→6.335454. Free-generation repetition remained high (repeated-4gram ratio 0.888). These measurements show where this invocation spent time; they do not claim throughput improvement over a baseline run, nor a full validation scan. Machine-readable timing: `build/fineweb-resume-timing.json`; per-update metrics: `build/fineweb-resume-timing.csv`.
A single earlier pre-optimization observation on the same host/config was 544 updates in 52.1 s (10.441 updates/s). The timed 200-update resumed segment was 10.696 updates/s (+2.4%). This small difference is observational only: one short window per version, no randomized/repeated A/B run, and checkpoint cadence/OS load can affect it. It does not establish a stable optimization gain.

The 10,200-update trainer configuration has width/input-width 128 and vocabulary 1,024. Its eight trainable tensor groups total 327,936 scalar parameters (`Wx` 16,384; three recurrent matrices 49,152; two biases 256; embedding and output head 262,144). This executable is therefore an AMA-10 prototype training run, not a 1.45B-parameter pretraining run. The current host also targets 6 GB VRAM; AdamW training-state memory for 1.45B FP32 parameters alone would exceed that device capacity. No 1.45B-scale training or checkpoint is claimed here.

The next 2,000-update continuation resumed from step 10,200 with the same saved AdamW moments and tokenizer, reaching step 12,200. Training-update time was 194.942 s (10.259 updates/s; 492.5 token positions/s), plus 7.687 s for checkpoint load/fingerprinting, 0.126 s to reopen the valid token caches, 0.427/0.425 s for initial/final 512-token diagnostics, and 0.080 s for generation. Train NLL changed 6.329329→6.318655 and the 512-token validation diagnostic 6.335454→6.313093. However, free generation remained collapsed: distinct-token ratio 0.0156 and repeated-4gram ratio 0.9840. This is not a generation-quality improvement. The sequential cursor consumed 585,600 unique source token positions so far, about 0.041% of the 1,427,334,072-token training cache; the single shard is still far from one pass. Checkpoint SHA-256: `556CF8481AF16471CE19A702D1A7003DB16C91851DAF4E19879C7C6151A4EF35`. Timings are in `build/fineweb-12k-timing.json`, update rows in `build/fineweb-resume-12k.csv`.

### Slow-memory depth scaling diagnosis (analytic, not a training result)

The current slow trace uses `m_next = 0.995 m + 0.005 h` at every recurrent microstep. At fixed depth 2, one token retains `0.995^2 = 0.990025` of the incoming slow state; at depth 4 it retains `0.995^4 = 0.980150`. Thus depth changes the slow-memory time constant as well as compute. A depth-normalized coefficient `rho_d = 0.995^(1/depth)` would keep per-token retention at 0.995 for fixed-depth execution, but changes the learned state transition and requires fresh weights plus CPU/OpenCL forward and BPTT parity updates. This ablation has not yet been trained; it is a next model-mechanism experiment, not an achieved fix.

### Timing comparison correction

The 2,000-update timed segment ran at 10.259 updates/s, below both the 10.696 updates/s short resumed segment and the earlier single-window 10.441 updates/s baseline. Combined across the two optimized segments, update throughput was 10.30 updates/s. The current measurements therefore do not demonstrate a stable throughput gain from the hot-path changes; profiling identifies a repeatable baseline for further work, while performance improvement remains unproven.

## 2026-11-17 — variable-resolution RGB patch-code generation

Replaced the image generator's hard-coded 64-code, 64×64 encode/decode routines with a variable grid derived from image dimensions and codebook patch dimensions. New image-code training uses 16×16 RGB vectors (768 values), a train-only 64-center k-means codebook, variable-length per-image target sequences, and PPM output at an explicit requested size. Older 8×8 `.vq` files still decode at their original 64×64 default. The image-generation test now covers an odd 35×23 input (six codes), a 1080×640 decode (2,720 codes), codebook serialization, and the existing 64×64 path. Strict C11 builds of the codebook test, trainer, and generator passed.

Trained a fresh width-32 shared-core checkpoint for 1,000 AdamW updates over the complete available local Commons split (14 train rows, 6 validation rows), then resumed the same optimizer/checkpoint to 5,000 updates. Validation caption-token NLL fell 5.554833→4.080026; validation image-code NLL fell 5.604289→5.196232. Generated a real 1080×640 PPM from the held-out red-fox prompt using 2,720 sampled codes. Manual review shows a severe patch mosaic, with no recognizable fox. At matched 64×64 size the 5,000-update sample scored 4.329531 dB PSNR and 0.053172 SSIM against the held-out fox image; the 1,000-update sample scored 4.341243 dB and 0.066135, so the longer run did not improve those image metrics. Lowering sampling temperature to 0.05 traded PSNR down to 3.574046 dB for SSIM up to 0.203708 and still did not create a recognizable image. The local images are all 64×64, so the 1080×640 run also forces 2,720-code continuations although training examples contain only 16 spatial codes. This is an output-shape capability, not high-resolution learned visual quality.

Checkpoint: `checkpoints/ama10-commons-variable-vq64-patch16-v0.2.ama`; generated artifact: `build/ama10-red-fox-1080x640-v0.2.png` (PPM master alongside it). The manual review failure is recorded as a failed semantic-generation result. Residual quantization, spatially conditioned decoding, multi-scale targets, and training on actual high-resolution images remain necessary experiments.

## 2026-11-17 — 64 versus 128 visual-code capacity

Added a configurable codebook size (2–254) to the variable-resolution generator. Compared 64 and 128 centers using the same 16×16 patches, 14/6 split, prompt, and sampler. The 64-center 5k checkpoint reported held-out image-code NLL 5.196232 and its 64×64 fox output scored 4.329531 dB PSNR / 0.053172 SSIM. The fresh 128-center model completed 5,000 updates; checkpoint metadata confirms the optimizer at update 5,000. Its code-NLL snapshot at 5,000 (recorded by a one-update resume) was 5.451262, and its matched 64×64 output scored 4.866863 dB / 0.077256. The 128-center output was also generated at 1080×640 and manually reviewed: it remains an unrecognizable patch mosaic. Thus larger codebook capacity modestly improved pixel reconstruction metrics while worsening held-out code NLL; neither capacity delivers semantic generation. The 128-center 1080×640 preview is `build/ama10-red-fox-vq128-1080x640.png`; checkpoint SHA-256 `837E98D3905B50873DBACDEB41141F5CBC35ACE633CD678566044592003E785E`.

## 2026-11-17 — image-token isolation, overlap fusion, and explicit 2D coordinates

Root-cause inspection found that the prior generator used image-code IDs 1–64, colliding with byte-fallback tokens 1–255, and used byte ID 255 as EOS. It also extended 64×64 training sequences (16 non-overlapping patches) to 2,720 codes at 1080×640. The user-reviewed high-resolution fox artifact is a failed semantic-generation result.

Reserved image-code, coordinate, and EOS tokens were added to the tokenizer and checkpoint formats; legacy checkpoint loading remains supported. Text decoding filters special tokens. The pixel decoder/trainer now use 16×16 patches with 12×12 stride and weighted overlap fusion. Each visual code is preceded by normalized eight-bin y/x markers. A new checkpoint was trained from random weights for 5,000 updates on the existing cached Commons data only (14 train rows, 6 validation rows; 11 source files, all 64×64). Metrics: validation text NLL 5.875810→4.176023; image-code teacher-forcing NLL, computed over code IDs only, 4.228424→3.880039 (uniform 64-code baseline ln(64)=4.158883). This is a modest teacher-forcing improvement, not semantic success.

The VQ-only held-out fox reconstruction scored 13.095 dB PSNR / 0.543 SSIM (training fox reconstruction: 22.775 dB / 0.758). Across the 25 held-out fox patch positions, 7/64 codebook entries were used (entropy 1.652/4.159 nats; most frequent code share 36%). Same-seed 64×64 “fox” and “apple” prompts changed 6.2% of channel bytes for the non-training prompts; exact cached training captions changed 38.2%. Despite prompt sensitivity, manual inspection does not show a recognizable fox or apple. The 1080×640 output uses 4,770 codes and remains a dense collage, far beyond the 25-code training grid.

For the matched 64×64 fox prompt, mean RGB jump across 16-pixel boundaries fell from 103.31 with non-overlapping tiles to 23.35 after overlap fusion; after adding coordinate markers it measured 30.58. The overlap-only output had 16 distinct complete 16×16 tiles versus 13 before; the coordinate output also had 16. The 1080×640 coordinate output had 2,680 distinct complete tiles, but this exact-tile statistic does not measure semantic coherence. The overlap decoder improved the boundary statistic; coordinate markers did not improve that metric. These changes do not meet the requested visual quality target.

Artifacts are local: `checkpoints/ama10-commons-coord-overlap-v3.ama` and `build/fox-coord-64.png`, `build/fox-coord-1080x640.png`, `build/apple-coord-64.png`. Strict C11 builds passed for image trainer/generator, tokenizer, checkpoint, and image-codec tests. Tokenizer special-ID round-trip and checkpoint save/load tests passed. No upload to GitHub was made. Remaining blockers are fundamental: only 15 cached image files exist and every one is 64×64; the encoder is still pixel k-means, and global semantic generation fails. High-resolution learned detail cannot be established from this cache, and the task remains incomplete.
