---
title: Haitian Creole speech (ASR + TTS)
emoji: 🗣️
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
short_description: Haitian Creole speech-to-text and text-to-speech
---

# Haitian Creole speech

Two tabs, two RunPod Serverless endpoints: **speech to text** and **text to
speech** for Kreyol. Neither model runs in the Space.

They are separate endpoints because they cannot share a Python environment —
`kani-tts` pins `nemo-toolkit[tts]==2.4.0`, while the ASR model needs
`nemo_toolkit[asr]` from git main.

## Speech to text

Thin client over a RunPod Serverless endpoint running
[`jsbeaudry/nemotron-3.5-asr-streaming-0.6b-ht`](https://huggingface.co/jsbeaudry/nemotron-3.5-asr-streaming-0.6b-ht)
— a 0.6B cache-aware streaming FastConformer RNN-T fine-tuned on 58.8 h of Kreyol.

**10.1% WER** at 320 ms on a 332-clip speaker-disjoint test set. The base model,
which has no Creole slot, scores 98.6% on the same clips.

| latency | att_context_size | WER |
|---|---|---:|
| 80 ms | `[56, 0]` | 11.9% |
| 320 ms | `[56, 3]` | 10.1% |
| 560 ms | `[56, 6]` | 10.3% |
| 1120 ms | `[56, 13]` | 10.1% |

## Text to speech

[`jsbeaudry/haitian-kani-ht-v3`](https://huggingface.co/jsbeaudry/haitian-kani-ht-v3)
— a KaniTTS voice model for Kreyol with eight speakers: `nana`, `deniz`, `mako`,
`mariz`, `klodin`, `jan`, `job`, `leo`. Output is 22050 Hz mono, up to 200
characters per request.

Sampling is adjustable under **Generation settings**:

| knob | default | range | effect |
|---|---:|---|---|
| temperature | 1.0 | 0.1 – 2.0 | higher is more varied, less stable |
| top_p | 0.95 | 0.05 – 1.0 | nucleus sampling cutoff |
| repetition_penalty | 1.1 | 1.0 – 2.0 | higher discourages loops and stuck syllables |

Those are the only three `KaniTTS.__call__` accepts — `max_new_tokens` is fixed
when the model loads, so it cannot be set per request. Out-of-range values are
clamped by the worker, which reports back what it used.

Level varies between speakers — `jan` came back about 9 dB quieter than `nana` on
the same length of text. The UI reports the peak so a quiet result is not
mistaken for a failed one.

## Setup

Under **Settings -> Variables and secrets**:

| name | kind | value |
|---|---|---|
| `RUNPOD_API_KEY` | secret | your RunPod API key |
| `RUNPOD_ENDPOINT` | variable | `9fds364d4gicy0` |
| `RUNPOD_TTS_ENDPOINT` | variable | `90fnsmvwgqfl6y` |

**The key must be scoped to both endpoints.** A RunPod *restricted* key is
limited to the endpoints picked when it was created, so a key that works in one
tab can return 403 in the other. The UI names that case explicitly when it
happens.

No GPU is needed — `cpu-basic` is enough, since inference happens on the worker.

## Cold starts

A worker that has been idle takes ~2 minutes to start (it pulls ~13 GB and imports
NeMo). Requests after that are ~2 s. The UI reports queue/cold time and inference
time separately so you can tell which you are looking at.

## Note

**Haitian Creole only.** Fine-tuning on Creole-only data moved the shared encoder
and decoder, so this model does not retain the base model's other languages —
prompting it with `fr-FR` or `en-US` returns Creole.
