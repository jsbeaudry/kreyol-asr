# RunPod Serverless worker

Haitian Creole streaming ASR as an autoscaling HTTP endpoint.

## Build & push

The model is public, so no build secret is needed:

```bash
docker build -t <you>/kreyol-asr-serverless:v1 .
docker push  <you>/kreyol-asr-serverless:v1
```

If you make the model private again, pass a token at build time instead:

```bash
docker build --build-arg HF_TOKEN=hf_... -t <you>/kreyol-asr-serverless:v1 .
```

The image is ~12-15 GB: torch + CUDA is most of it, the 2.5 GB checkpoint is
baked in on purpose so a cold worker does not spend 30-60 s downloading it.

Your Mac has ~4 GB free, so build on the pod or in CI, not locally.

## Endpoint settings

| setting | value | why |
|---|---|---|
| `gpuPoolIds` | `["AMPERE_48"]` | the A40 pool the model was trained and benchmarked on |
| `workersMin` | `0` | scale to zero — no idle cost |
| `workersMax` | `3` | cap the blast radius |
| `flashboot` | `FLASHBOOT` | biggest lever on cold start; NeMo's import alone is 20-30 s |
| `containerDiskInGb` | `25` | image + HF cache |
| `executionTimeoutMs` | `120000` | a cold worker needs a minute before it decodes |

## Request

```bash
curl -X POST https://api.runpod.ai/v2/<endpoint-id>/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"audio_base64": "'"$(base64 -i clip.wav)"'", "latency_ms": 320}}'
```

`latency_ms` is one of 80 / 320 / 560 / 1120 — measured WER 11.9 / 10.1 / 10.3 / 10.1%
(v2; 332-clip speaker-disjoint test set, exact scoring). Left context is fixed at 56 and is not a parameter: it matches the
encoder's `sliding_window: 57`, and changing it degrades streaming behaviour.

## Response

```json
{"text": "mwen wè nou kanpe isit kòm lòt bò dlo",
 "lang": "ht-HT", "latency_ms": 320, "att_context_size": [56, 3],
 "duration_s": 6.1, "device": "cuda"}
```

Errors come back as `{"error": "..."}` with a 200 — check for the key.

## Cost shape

Warm decode is ~1 s of GPU. A cold start is 60-120 s. That is a ~100x
difference per request, so this is economical for batches and expensive for
one-off calls minutes apart. See the main README for the comparison against a
pod and against the ZeroGPU Space.
