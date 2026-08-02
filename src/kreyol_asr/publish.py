"""Push the fine-tuned model + a benchmark-backed model card to the Hugging Face Hub."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from rich.console import Console

from . import BASE_MODEL

console = Console()


def push(
    *,
    repo_id: str,
    nemo_path: Path,
    lang_tag: str,
    slot: int,
    warm_start_from: str | None,
    base_model: str = BASE_MODEL,
    benchmark: dict[str, Any] | None = None,
    data_stats: dict[str, Any] | None = None,
    private: bool = True,
    token: str | None = None,
    tokenizer_dir: Path | None = None,
) -> str:
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=token)
    api.create_repo(repo_id, private=private, exist_ok=True)
    console.print(f"Repo [bold]{repo_id}[/bold] (private={private})")

    staging = Path(tempfile.mkdtemp(prefix="kreyol-push-"))
    try:
        shutil.copy2(nemo_path, staging / f"{repo_id.split('/')[-1]}.nemo")

        # processor_config.json with ht-HT registered, so downstream code can resolve
        # the tag the same way NeMo does.
        try:
            src = hf_hub_download(base_model, "processor_config.json", token=token)
            data = json.loads(Path(src).read_text())
            data.setdefault("prompt_dictionary", {})[lang_tag] = slot
            data["prompt_dictionary"][lang_tag.split("-")[0]] = slot
            (staging / "processor_config.json").write_text(
                json.dumps(data, indent=2, ensure_ascii=False))
        except Exception as e:  # non-fatal: the .nemo is self-contained
            console.print(f"[yellow]Could not stage processor_config.json: {e}[/yellow]")

        if tokenizer_dir and Path(tokenizer_dir).exists():
            for p in Path(tokenizer_dir).iterdir():
                if p.is_file():
                    shutil.copy2(p, staging / p.name)

        (staging / "benchmark.json").write_text(json.dumps(benchmark or {}, indent=2))
        (staging / "README.md").write_text(model_card(
            repo_id=repo_id, base_model=base_model, lang_tag=lang_tag, slot=slot,
            warm_start_from=warm_start_from, benchmark=benchmark, data_stats=data_stats))

        api.upload_folder(repo_id=repo_id, folder_path=str(staging),
                          commit_message=f"Add Haitian Creole fine-tune ({lang_tag} @ slot {slot})")
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    url = f"https://huggingface.co/{repo_id}"
    console.print(f"[green]Pushed[/green] {url}")
    console.print(
        "[yellow]Note:[/yellow] this uploads the NeMo checkpoint. `transformers`' "
        "`from_pretrained` additionally needs safetensors weights, which requires "
        "NVIDIA's .nemo->HF converter; until that is available, load via NeMo "
        "(the model card says so explicitly)."
    )
    return url


def model_card(*, repo_id: str, base_model: str, lang_tag: str, slot: int,
               warm_start_from: str | None, benchmark: dict | None,
               data_stats: dict | None) -> str:
    from .evaluate import render_report

    bench_md = render_report(benchmark) if benchmark and benchmark.get("results") else \
        "_No benchmark recorded for this upload._"
    hours = (data_stats or {}).get("total_hours", "n/a")
    sources = "\n".join(f"- `{k}` — {v} h"
                        for k, v in (data_stats or {}).get("per_source_hours", {}).items()) \
        or "- not recorded"

    return f"""---
license: other
license_name: openmdw-1.1
language:
- ht
base_model: {base_model}
tags:
- automatic-speech-recognition
- streaming
- nemo
- haitian-creole
- kreyol
pipeline_tag: automatic-speech-recognition
---

# {repo_id.split('/')[-1]}

Haitian Creole (Kreyòl Ayisyen) fine-tune of
[`{base_model}`](https://huggingface.co/{base_model}) — a 0.6B cache-aware streaming
FastConformer RNN-T with language-ID prompt conditioning.

## What changed

The base model conditions on language through a 128-slot one-hot prompt vector, but its
`prompt_dictionary` only assigns indices 0–104. This fine-tune claims **free slot
{slot}** for `{lang_tag}`{f", warm-started from `{warm_start_from}`" if warm_start_from else ""}.

Pass `target_lang="{lang_tag}"` to get Haitian Creole.
{f"The slot was initialized from `{warm_start_from}` because Haitian Creole is French-lexified Latin script, giving the new language a useful prior instead of starting from noise." if warm_start_from else ""}

The pretrained BPE tokenizer (vocab 13088) is reused unchanged, so the RNN-T joint keeps
its pretrained weights.

## ⚠️ This is a Creole-only model — the other languages are NOT preserved

Claiming a free slot protects the other language *columns* from gradient, and they are
in fact untouched by it. It does **not** protect model *behaviour*: the encoder, decoder
and joint are shared by every language and were fine-tuned on Haitian Creole only, so
the backbone drifted. Measured on this checkpoint, on Haitian Creole audio:

| prompt | output |
|---|---|
| `{lang_tag}` | correct Creole |
| `fr-FR` | **Creole** — byte-identical to `{lang_tag}` on 4 of 5 sampled clips |
| `en-US` | **Creole** |
| `zh-CN` | degraded output — neither Chinese nor Creole |
| `en-US` on the **base** model (control) | English, as expected |

The control matters: the base model steers correctly on the same audio, so this is a
real change in this checkpoint rather than an artifact of the audio being Creole.

Root cause: slot {slot} was warm-started from `{warm_start_from or "n/a"}` and never
diverged from it (`cos = 0.99993` after training), so both prompts feed nearly the same
vector into a backbone that now only knows Creole.

**Do not use this model for any language other than Haitian Creole.** If you need the
base model's multilingual coverage, use
[`{base_model}`](https://huggingface.co/{base_model}) directly. Preserving it through a
fine-tune requires mixing a replay set of the original languages into training.

## Training data

- Total: **{hours} h** of transcribed Haitian Creole speech, 16 kHz mono.

{sources}

## Benchmark

{bench_md}

## Usage

```python
import nemo.collections.asr as nemo_asr

model = nemo_asr.models.ASRModel.restore_from("{repo_id.split('/')[-1]}.nemo")
model.encoder.set_default_att_context_size([56, 3])   # 320 ms chunks
print(model.transcribe(["clip.wav"], target_lang="{lang_tag}"))
```

Streaming inference (matches how the benchmark above was measured):

```bash
python $NEMO_DIR/examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py \\
  model_path={repo_id.split('/')[-1]}.nemo \\
  dataset_manifest=test.json \\
  target_lang={lang_tag} \\
  att_context_size="[56,3]" \\
  decoder_type=rnnt pad_and_drop_preencoded=true strip_lang_tags=false
```

`att_context_size` left context **must be 56** — it matches the encoder's
`sliding_window: 57`. Right context selects latency: 0→80 ms, 3→320 ms, 6→560 ms,
13→1120 ms.

### Loading with `transformers`

Not yet supported: this repo ships the NeMo checkpoint, and converting it to
safetensors requires NVIDIA's `.nemo`→HF converter. Use NeMo as shown above.

## Known limitations

- **Creole only** — see the warning above.
- **Word segmentation dominates the remaining errors.** CER is roughly a third of WER
  (≈4.8% vs ≈15.7% at 320 ms), which is the signature of boundary errors rather than
  acoustic ones: `Si l`→`Sil`, `ou mèt`→`Omèt`, `bò dlo`→`bòd lo`. Proper nouns and
  numbers are the other common failures (`Dariyis`→`Dariyich`, `120`→`12`). A decode-time
  phrase-boosting list or an n-gram LM would target this directly.
- **Training data is partly synthetic.** TTS-generated speech was used for training only
  and excluded from the validation and test splits, so the numbers above reflect real
  recordings.
- **Some sources are machine-transcribed.** Clips whose transcripts exceeded 25
  characters/second — degenerate repetition loops such as `pou pou pou pou …` — were
  filtered out, but milder label noise below that threshold may remain.
- **Clips longer than 18 s were excluded** from training due to a GPU memory limit on a
  46 GB card, not for data-quality reasons. Long-utterance behaviour is therefore less
  well covered.
- **Training stopped early**, at roughly epoch 5 of a planned 15, while validation WER was
  still improving. This checkpoint is not the ceiling for this data.

## License

Inherits [OpenMDW-1.1](https://huggingface.co/{base_model}) from the base model.
"""
