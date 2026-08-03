"""Voicepack format contract.

Risk #7 in the plan: a pack that only loads in our own code is a failed deliverable,
and the failure would surface at the very last step of the project. These tests pin
the contract now, before any training exists, so Stage 2 later changes only the
numbers in the tensor and never its shape or type.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from kreyol_tts.voicepack import (PACK_DIM, PACK_ROWS, build_pack,  # noqa: E402
                                  load_pack, save_pack, validate_pack)

rng = np.random.default_rng(1804)


def _plausible_refs(n=60):
    """Reference embeddings whose style drifts with utterance length, as real ones do."""
    lengths = rng.integers(10, 400, size=n).astype(float)
    base = rng.normal(0, 1.0, size=PACK_DIM)
    drift = rng.normal(0, 1.0, size=PACK_DIM)
    embs = np.stack([
        (base + drift * (L / 400.0)) * (8.0 - 5.5 * (L / 400.0)) / max(np.linalg.norm(base), 1e-9)
        + rng.normal(0, 0.05, size=PACK_DIM)
        for L in lengths
    ])
    return embs, lengths


def test_build_pack_shape_and_dtype():
    embs, lens = _plausible_refs()
    pack = build_pack(embs, lens)
    assert pack.shape == (PACK_ROWS, PACK_DIM)
    assert pack.dtype == np.float32


def test_build_pack_rows_are_not_copies():
    """If every row is identical the length conditioning did nothing."""
    embs, lens = _plausible_refs()
    pack = build_pack(embs, lens)
    assert not np.allclose(pack[0], pack[-1]), "row 0 == row 509; bucketing collapsed"


def test_build_pack_is_smooth_along_length():
    """Adjacent rows must not jump, or prosody snaps when an utterance crosses a bucket."""
    embs, lens = _plausible_refs()
    pack = build_pack(embs, lens).astype(np.float64)
    steps = np.linalg.norm(np.diff(pack, axis=0), axis=1)
    spread = np.linalg.norm(pack[0] - pack[-1])
    assert steps.max() < spread, f"max adjacent jump {steps.max():.3f} >= total spread {spread:.3f}"


def test_build_pack_rejects_bad_input():
    with pytest.raises(ValueError):
        build_pack(np.zeros((5, 99)), np.arange(5))
    with pytest.raises(ValueError):
        build_pack(np.zeros((5, PACK_DIM)), np.arange(4))
    with pytest.raises(ValueError):
        build_pack(np.zeros((0, PACK_DIM)), np.zeros(0))


def test_build_pack_survives_a_single_reference():
    pack = build_pack(rng.normal(size=(1, PACK_DIM)), np.array([50.0]))
    assert pack.shape == (PACK_ROWS, PACK_DIM)
    assert np.isfinite(pack).all()


# --- the format contract ----------------------------------------------------

def test_save_produces_a_bare_tensor_not_a_dict(tmp_path):
    """`load_single_voice` does torch.load(...) and uses the result directly."""
    embs, lens = _plausible_refs()
    p = save_pack(build_pack(embs, lens), tmp_path / "kf_test.pt")
    loaded = torch.load(p, weights_only=True)
    assert torch.is_tensor(loaded), f"got {type(loaded).__name__}, not a tensor"
    assert tuple(loaded.shape) == (PACK_ROWS, 1, PACK_DIM)
    assert loaded.dtype == torch.float32


def test_load_pack_roundtrip(tmp_path):
    embs, lens = _plausible_refs()
    p = save_pack(build_pack(embs, lens), tmp_path / "v.pt")
    assert torch.equal(load_pack(p), torch.load(p, weights_only=True))


def test_validate_rejects_a_dict():
    with pytest.raises(TypeError, match="bare tensor"):
        validate_pack({"pack": torch.zeros(PACK_ROWS, 1, PACK_DIM)})


@pytest.mark.parametrize("shape", [(PACK_ROWS, PACK_DIM), (256, 1, PACK_DIM),
                                   (PACK_ROWS, 1, 128)])
def test_validate_rejects_wrong_shape(shape):
    with pytest.raises(ValueError):
        validate_pack(torch.zeros(*shape, dtype=torch.float32))


def test_validate_rejects_wrong_dtype():
    with pytest.raises(ValueError, match="float32"):
        validate_pack(torch.zeros(PACK_ROWS, 1, PACK_DIM, dtype=torch.float64))


def test_validate_rejects_nan():
    t = torch.zeros(PACK_ROWS, 1, PACK_DIM)
    t[3, 0, 7] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        validate_pack(t)


def test_validate_flags_collapsed_rows():
    """Every row identical -> no length conditioning. Must not silently pass."""
    t = torch.ones(PACK_ROWS, 1, PACK_DIM, dtype=torch.float32)
    r = validate_pack(t)
    assert r["verdict"] == "SUSPECT"
    assert any("decay" in w or "cos" in w for w in r["warnings"])


def test_validate_accepts_a_plausible_pack():
    embs, lens = _plausible_refs()
    t = torch.from_numpy(build_pack(embs, lens)).unsqueeze(1)
    r = validate_pack(t)
    assert r["shape"] == (PACK_ROWS, 1, PACK_DIM)
    assert np.isfinite(r["norm_decay"])


# --- calibration against Kokoro's own packs ---------------------------------

@pytest.mark.slow
def test_real_kokoro_packs_satisfy_our_validator():
    """Our thresholds must accept the packs they were calibrated on.

    Guards against tightening the ranges until real Kokoro voices would be rejected.
    """
    hf = pytest.importorskip("huggingface_hub")
    for voice in ("af_heart", "ff_siwis"):
        path = hf.hf_hub_download("hexgrad/Kokoro-82M", f"voices/{voice}.pt")
        r = validate_pack(torch.load(path, weights_only=True))
        assert r["verdict"] == "PASS", f"{voice}: {r['warnings']}"
