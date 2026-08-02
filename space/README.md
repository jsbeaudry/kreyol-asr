---
title: Haitian Creole streaming ASR
emoji: 🎙️
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
short_description: Haitian Creole streaming ASR (Kreyol)
---

# Haitian Creole streaming ASR

[`jsbeaudry/nemotron-3.5-asr-streaming-0.6b-ht`](https://huggingface.co/jsbeaudry/nemotron-3.5-asr-streaming-0.6b-ht)
— a 0.6B cache-aware streaming FastConformer RNN-T fine-tuned on 45.9 h of Kreyol
Ayisyen, prompted with `ht-HT` (slot 105).

**15.7% WER** at 320 ms on a 385-clip speaker-disjoint test set. The base model,
which has no Creole slot, scores 98.2% on the same clips.

| latency | att_context_size | WER |
|---|---|---:|
| 80 ms | `[56, 0]` | 17.3% |
| 320 ms | `[56, 3]` | 15.7% |
| 560 ms | `[56, 6]` | 14.9% |
| 1120 ms | `[56, 13]` | 14.7% |

## Note

**Haitian Creole only.** Fine-tuning on Creole-only data moved the shared encoder
and decoder, so this model does not retain the base model's other languages —
prompting it with `fr-FR` or `en-US` returns Creole.
