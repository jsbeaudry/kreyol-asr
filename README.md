# kreyol-asr

Fine-tune [`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
for **Haitian Creole**. You pass Hugging Face dataset URLs; you get a trained,
benchmarked, published streaming ASR model.

```bash
kreyol-asr prepare --datasets you/creole-100h:train,you/creole-extra:train
kreyol-asr patch      # register ht-HT, warm-start it from fr-FR
kreyol-asr train      # GPU host
kreyol-asr bench      # baseline vs fine-tuned, all 4 streaming latencies
kreyol-asr push       # -> Hugging Face Hub
```

Or `make all`.

---

## The core idea: Creole gets its own language slot

The base model conditions on language through a **128-slot one-hot vector**
(`num_prompts: 128`). Its shipped `prompt_dictionary` assigns indices **0–104** across
40 locales — Haitian Creole is not among them, but **105–127 are unused**.

So rather than overwriting a language, this pipeline claims free slot **105** for
`ht-HT`, and **warm-starts it from `fr-FR` (slot 8)** — Haitian Creole is
French-lexified Latin script, so the French prompt vector is the best available prior.

This was verified empirically against the real checkpoint, not assumed:

| Check | Result |
|---|---|
| Prompt projection | `prompt_kernel.0.weight`, shape **(2048, 1152)** |
| Language one-hot location | columns **1024:1152** (concatenated *after* the 1024-dim acoustic embedding) |
| Untrained-slot signature | free slots' column norm **0.57** vs trained slots **6.67** — slot 105 really is dead weight |
| After warm start | slot 105 column norm **0.574 → 13.832**, identical to `fr-FR` |
| Collateral damage | **656 of 657 tensors bit-identical**; exactly one column changed; all 121 dictionary entries preserved |

`kreyol-asr patch` re-derives the offset every run by probing which column block shows
the untrained signature, so it fails loudly instead of silently patching the wrong
tensor. Run `kreyol-asr inspect` to see all of this for yourself.

## Traps this repo handles for you

These are the things that silently produce a worse model rather than an error:

1. **`att_context_size` left context is 56, not 70.** The checkpoint's
   `sliding_window: 57` implies 56 frames of left context, but the stock NeMo YAML
   ships `[70, 6]`. `check_att_context()` rejects anything else.
2. **It's a *list* of contexts.** The released model trains on
   `[[56,3],[56,0],[56,6],[56,13]]` so one model serves every latency. Passing a
   single pair specialises it and regresses the other three.
3. **The BPE has no `:` `;` `(` `)` `%` `&` `/` tokens.** Left in transcripts they
   become `<unk>` *training targets*, teaching the model to emit `<unk>`. `prepare`
   maps them to the nearest encodable character and reports every substitution.
4. **Patching the `.nemo` is not enough.** NeMo builds the model from the training
   YAML and *then* loads weights, so `ht-HT` must be in the training config too.
   We generate that config — passing `++...prompt_dictionary.ht-HT=105` would fail
   anyway, since Hydra's override grammar rejects `-` in keys.
5. **Tokenizer artifacts are hash-prefixed** inside the `.nemo`
   (`427ad33c…_tokenizer.model`). `model.tokenizer.dir=` needs plain names.
6. **Never retrain the tokenizer.** Changing the vocab resets the RNN-T joint and
   discards the pretrained decoder. `prepare` prints a coverage verdict so you can
   confirm reuse is safe (real Creole data: 2.4 tokens/word, 0 `<unk>`).

## Where things run

| Step | Your Mac | GPU pod |
|---|---|---|
| `prepare`, `inspect`, `patch`, `push` | yes | yes |
| `train`, `bench` | no — needs CUDA + NeMo | yes |

NeMo needs Linux, an NVIDIA GPU and Python ≤3.12.

## Setup

```bash
cp .env.example .env      # add HF_TOKEN (needed for private datasets and pushing)
make install              # local: uv pip install -e ".[patch,dev]"
```

On a RunPod / Lambda / Vast pod, from the NeMo container:

```bash
export HF_TOKEN=hf_...
bash scripts/pod_bootstrap.sh
```

That verifies the GPU and NeMo install, runs `scripts/smoke.sh` (50 clips, 20 steps —
fails fast on config or manifest problems), then runs the full pipeline in tmux.
`push` is deliberately **not** automatic: look at the benchmark first.

## Configuration

Two files, and normally you only touch the first.

- **`configs/datasets.ht.yaml`** — which HF datasets feed the run. Column names are
  auto-detected (`audio`/`text`/`speaker_id` on real data) and overridable. `weight`
  oversamples a source in train only; val/test stay unweighted so metrics reflect the
  real distribution.
- **`configs/finetune.ht.yaml`** — slot/warm-start, `att_context_size`, LR schedule,
  eval latencies, publish target.

Note the LR: the base recipe's `lr: 2.0` is a Noam *scale factor*, not a learning
rate. For fine-tuning we switch to `CosineAnnealing` at `1e-4`.

## What `prepare` gives you

`data/ht/manifests/{train,val,test}.json` (NeMo JSON-lines, with `target_lang`),
16 kHz mono PCM16 wavs, plus `data_report.md`:

- hours per source and per split, and whether the split was **speaker-disjoint**
  (it says so plainly when it had to fall back to a hash split)
- every dropped clip with a reason
- tokenizer coverage: tokens/word, `<unk>` rate, and the **source substrings** that
  failed to encode — not the useless literal `<unk>`
- a **transcript-style check**. The base model emits punctuated, properly-cased text.
  If your transcripts are lowercase and unpunctuated the report warns you, because
  the baseline WER comparison then unfairly penalises the base model — use
  `bench --normalize-scoring` in that case.

Durations are measured from decoded frames, never trusted from metadata. Audio is
decoded with `soundfile` and resampled with `soxr` rather than going through
`datasets`' decoding, which changed backends in v4 and now needs torchcodec/ffmpeg.

## Benchmarking

`kreyol-asr bench` scores **baseline vs fine-tuned at all four latencies**
(80/320/560/1120 ms). The baseline is the untouched base model prompted with
`fr-FR`, its closest language — without that row, the fine-tune delta is
unmeasurable. Output is `benchmarks/latest/report.md` plus the 20 worst clips per
configuration for eyeballing.

Success looks like a clear WER drop at `[56, 3]` versus baseline.

## Testing

```bash
make test     # 48 tests, no GPU and no 2.4 GB download required
```

The `.nemo` surgery is tested against a stub checkpoint that reproduces the real
one's traps: the (2048, 1152) projection, 48 `pos_bias` decoy tensors of shape
(8, 128) that a naive shape match would grab, near-zero free slots, and
hash-prefixed tokenizer artifacts.

## Known gaps

- **`transformers` `from_pretrained` won't work on the output.** We ship the `.nemo`;
  converting to safetensors needs NVIDIA's converter. The generated model card says
  this rather than implying it works.
- **`bench` shells out to NeMo's streaming inference script.** That script's CLI has
  shipped in both `key=value` and argparse forms; if one is rejected, use
  `--arg-style argparse`.
- The `nvcr.io/nvidia/nemo:26.06` tag is what the model card calls for; the
  Dockerfile verifies it at build time and `pod_bootstrap.sh` clones NeMo if the
  image lacks `examples/`.

## License

Code: Apache-2.0. Models produced inherit **OpenMDW-1.1** from the base checkpoint.
