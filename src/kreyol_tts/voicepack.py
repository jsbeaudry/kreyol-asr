"""Build and validate Kokoro voicepacks.

A voicepack is a bare `(510, 1, 256)` float32 tensor — not a dict, not a state dict.
`KPipeline.load_single_voice` does `torch.load(path, weights_only=True)` and uses the
result directly, and `KModel.forward` splits it as:

    ref_s[:, :128]   -> decoder style
    ref_s[:, 128:]   -> duration / prosody predictor

The 510 rows are a **length-conditioned table**: inference selects `pack[len(ps)-1]`,
so row *i* is the style to use for an utterance of *i+1* phonemes. Style genuinely
varies with length — speech rate and final lengthening differ between a one-word reply
and a long sentence — so the rows are not copies of each other.

Calibration against the real packs (measured 2026-08-02):

    pack        ‖row 0‖   ‖row 509‖   cos(row 0, row 509)
    af_heart      7.81       2.24            0.31
    ff_siwis      9.92       2.97            0.53

so expect a ~3.5x norm decay across the length axis and a cosine in the 0.3-0.55 band.
A flat norm profile or a cosine near 1.0 means the length bucketing collapsed; a
cosine near 0 means the style encoder is emitting noise.

Note Kokoro ships **no** `style_encoder` or `predictor_encoder` — they are trained from
scratch in Stage 1 — so pack quality is bounded by encoders trained on the local corpus.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

PACK_ROWS = 510
STYLE_DIM = 128        # decoder style, ref_s[:, :128]
PROSODY_DIM = 128      # duration/prosody predictor, ref_s[:, 128:]
PACK_DIM = STYLE_DIM + PROSODY_DIM

# Calibrated from af_heart / ff_siwis. Deliberately loose — these catch a broken
# encoder, not a merely different-sounding voice.
NORM_DECAY_RANGE = (1.8, 8.0)
COS_RANGE = (0.05, 0.90)

# Reference clips are pooled by phoneme length with a sliding window; a row with
# fewer than this many references falls back to its nearest populated neighbour.
MIN_REFS_PER_ROW = 5
WINDOW_FRAC = 0.20
SMOOTH_WINDOW = 15


def build_pack(embeddings: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Length-bucketed voicepack from per-reference style vectors.

    `embeddings` is (n_refs, 256) — `concat(style_encoder(mel), predictor_encoder(mel))`
    for each reference clip. `lengths` is that clip's phoneme count.

    Kept free of torch and of any model so it can be tested without a checkpoint:
    the bucketing logic is where the bugs live, not the encoder call.
    """
    embeddings = np.asarray(embeddings, dtype=np.float64)
    lengths = np.asarray(lengths, dtype=np.float64)
    if embeddings.ndim != 2 or embeddings.shape[1] != PACK_DIM:
        raise ValueError(f"embeddings must be (n, {PACK_DIM}), got {embeddings.shape}")
    if len(embeddings) != len(lengths):
        raise ValueError(f"{len(embeddings)} embeddings vs {len(lengths)} lengths")
    if len(embeddings) == 0:
        raise ValueError("no reference embeddings")

    pack = np.zeros((PACK_ROWS, PACK_DIM), dtype=np.float64)
    populated = np.zeros(PACK_ROWS, dtype=bool)

    for row in range(PACK_ROWS):
        target = row + 1
        window = max(3.0, target * WINDOW_FRAC)
        sel = np.abs(lengths - target) <= window
        if sel.sum() < MIN_REFS_PER_ROW:
            # Widen to the nearest N references rather than leaving a hole; a hole
            # would later be filled by a neighbour anyway, and this keeps the
            # transition smooth instead of stepped.
            order = np.argsort(np.abs(lengths - target))
            sel = np.zeros(len(lengths), dtype=bool)
            sel[order[:min(MIN_REFS_PER_ROW, len(lengths))]] = True
        pack[row] = embeddings[sel].mean(axis=0)
        populated[row] = True

    # Smooth along the length axis. Bucket boundaries otherwise produce audible
    # prosody jumps when an utterance crosses one.
    if SMOOTH_WINDOW > 1:
        kernel = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
        padded = np.pad(pack, ((SMOOTH_WINDOW // 2, SMOOTH_WINDOW // 2), (0, 0)), mode="edge")
        pack = np.stack([np.convolve(padded[:, d], kernel, mode="valid")[:PACK_ROWS]
                         for d in range(PACK_DIM)], axis=1)

    return pack.astype(np.float32)


def validate_pack(pack) -> dict:
    """Structural + statistical checks. Returns a report; raises only on shape/dtype."""
    import torch

    if not torch.is_tensor(pack):
        raise TypeError(
            f"a voicepack must be a bare tensor, got {type(pack).__name__}. "
            f"`load_single_voice` uses torch.load(...) directly, so a dict will not load."
        )
    if tuple(pack.shape) != (PACK_ROWS, 1, PACK_DIM):
        raise ValueError(f"expected {(PACK_ROWS, 1, PACK_DIM)}, got {tuple(pack.shape)}")
    if pack.dtype != torch.float32:
        raise ValueError(f"expected float32, got {pack.dtype}")
    if not torch.isfinite(pack).all():
        raise ValueError("pack contains NaN or Inf")

    first, last = pack[0].flatten(), pack[-1].flatten()
    n0, n509 = float(first.norm()), float(last.norm())
    decay = n0 / max(n509, 1e-9)
    cos = float(torch.nn.functional.cosine_similarity(first, last, dim=0))

    warnings = []
    if not (NORM_DECAY_RANGE[0] <= decay <= NORM_DECAY_RANGE[1]):
        warnings.append(
            f"norm decay {decay:.2f}x is outside {NORM_DECAY_RANGE}; real packs decay "
            f"~3.5x from row 0 to row 509. A flat profile means the length bucketing "
            f"collapsed and every row is the same style."
        )
    if not (COS_RANGE[0] <= cos <= COS_RANGE[1]):
        warnings.append(
            f"cos(row 0, row 509) = {cos:.3f} is outside {COS_RANGE}; near 1.0 means the "
            f"rows are copies, near 0 means the style encoder is emitting noise."
        )
    return {"shape": tuple(pack.shape), "dtype": str(pack.dtype),
            "norm_row0": n0, "norm_row509": n509, "norm_decay": decay,
            "cos_first_last": cos, "warnings": warnings,
            "verdict": "PASS" if not warnings else "SUSPECT"}


def save_pack(pack: np.ndarray | "object", path: Path | str) -> Path:
    """Write as a bare (510, 1, 256) float32 tensor."""
    import torch

    if isinstance(pack, np.ndarray):
        t = torch.from_numpy(np.ascontiguousarray(pack, dtype=np.float32))
    else:
        t = pack.detach().cpu().float()
    if t.ndim == 2:
        t = t.unsqueeze(1)
    validate_pack(t)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(t, out)   # bare tensor, deliberately not a dict
    return out


def load_pack(path: Path | str):
    import torch

    return torch.load(str(path), weights_only=True)
