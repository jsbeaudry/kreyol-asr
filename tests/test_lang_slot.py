"""Tests for the prompt-slot surgery, using a stub .nemo — no 2.4 GB download."""

import tarfile

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("omegaconf")

from kreyol_asr import HIGHEST_USED_SLOT, NUM_PROMPTS  # noqa: E402
from kreyol_asr.lang_slot import (  # noqa: E402
    extract_tokenizer,
    find_prompt_dictionary,
    find_prompt_weight,
    infer_prompt_offset,
    patch,
    resolve_prompt_projection,
    unpack,
)

# Mirrors the real checkpoint: slots 0-104 used, fr-FR at 8.
STUB_DICT = {f"l{i}-XX": i for i in range(HIGHEST_USED_SLOT + 1)}
STUB_DICT["fr-FR"] = 8
STUB_DICT["en-US"] = 0

PARAM = "prompt_kernel.0.weight"
ACOUSTIC = 1024  # the real checkpoint concatenates 1024 acoustic dims before the one-hot


def make_stub_nemo(tmp_path, param_name=PARAM):
    """A miniature of the real checkpoint, including its traps.

    - prompt projection is (2048, 1024+128): the one-hot is the *trailing* block
    - 48 pos_bias tensors of shape (8, 128) exist as decoys
    - free slots' columns are near zero, as they are in the real weights
    - tokenizer artifacts carry NeMo's 32-hex-char prefix
    """
    from omegaconf import OmegaConf

    work = tmp_path / "work"
    work.mkdir()
    cfg = OmegaConf.create({
        "encoder": {"att_context_size": [[56, 3], [56, 0]]},
        "model_defaults": {"prompt_dictionary": dict(STUB_DICT),
                           "initialize_prompt_feature": True},
    })
    OmegaConf.save(cfg, work / "model_config.yaml")

    torch.manual_seed(0)
    w = torch.randn(2048, ACOUSTIC + NUM_PROMPTS)
    free = [i for i in range(NUM_PROMPTS) if i not in set(STUB_DICT.values())]
    w[:, ACOUSTIC + torch.tensor(free)] *= 0.01  # untrained slots stayed near zero
    state = {param_name: w, f"{param_name.rsplit('.', 2)[0]}.2.weight": torch.randn(1024, 2048)}
    for i in range(24):  # the decoys that a naive shape match would pick up
        state[f"encoder.layers.{i}.self_attn.pos_bias_u"] = torch.randn(8, NUM_PROMPTS)
        state[f"encoder.layers.{i}.self_attn.pos_bias_v"] = torch.randn(8, NUM_PROMPTS)
    torch.save(state, work / "model_weights.ckpt")
    (work / f"{'a'*32}_tokenizer.model").write_bytes(b"stub-spm")
    (work / f"{'b'*32}_vocab.txt").write_text("a\nb\n")

    nemo = tmp_path / "stub.nemo"
    with tarfile.open(nemo, "w") as tar:
        for item in work.iterdir():
            tar.add(item, arcname=item.name)
    return nemo, param_name


def test_finds_prompt_dictionary_anywhere_in_tree(tmp_path):
    nemo, _ = make_stub_nemo(tmp_path)
    bundle = unpack(nemo, tmp_path / "unpacked")
    path, d = find_prompt_dictionary(bundle.config)
    assert path == "model_defaults.prompt_dictionary"
    assert d["fr-FR"] == 8


def test_prompt_weight_is_not_confused_by_pos_bias_decoys(tmp_path):
    nemo, param = make_stub_nemo(tmp_path)
    bundle = unpack(nemo, tmp_path / "unpacked")
    state = torch.load(bundle.weights_path, map_location="cpu", weights_only=False)
    # 48 tensors have shape (8, 128); a naive `shape[1] == 128` match picks those.
    assert sum(1 for v in state.values() if v.ndim == 2 and v.shape[1] == NUM_PROMPTS) == 48
    assert find_prompt_weight(state) == param


def test_offset_inferred_from_untrained_free_slots(tmp_path):
    nemo, param = make_stub_nemo(tmp_path)
    bundle = unpack(nemo, tmp_path / "unpacked")
    w = torch.load(bundle.weights_path, map_location="cpu", weights_only=False)[param]
    assert infer_prompt_offset(w, STUB_DICT) == ACOUSTIC


def test_offset_is_zero_when_input_is_exactly_the_one_hot(tmp_path):
    w = torch.randn(64, NUM_PROMPTS)
    assert infer_prompt_offset(w, STUB_DICT) == 0


def test_offset_probe_rejects_a_layer_with_no_untrained_block():
    # prompt_kernel.2.weight (1024, 2048) never sees the one-hot; its columns are
    # all trained, so no candidate offset shows the untrained signature.
    torch.manual_seed(1)
    with pytest.raises(RuntimeError, match="Cannot identify"):
        infer_prompt_offset(torch.randn(1024, 2048), STUB_DICT)


def test_resolve_skips_second_mlp_layer_and_picks_the_real_projection(tmp_path):
    nemo, param = make_stub_nemo(tmp_path)
    bundle = unpack(nemo, tmp_path / "unpacked")
    state = torch.load(bundle.weights_path, map_location="cpu", weights_only=False)
    assert resolve_prompt_projection(state, STUB_DICT) == (param, ACOUSTIC)


def test_patch_warm_starts_and_preserves_existing_slots(tmp_path):
    from omegaconf import OmegaConf

    nemo, param = make_stub_nemo(tmp_path)
    before = torch.load(unpack(nemo, tmp_path / "pre").weights_path,
                        map_location="cpu", weights_only=False)[param].clone()

    out = tmp_path / "patched.nemo"
    info = patch(nemo, out, tag="ht-HT", slot=105, warm_start_from="fr-FR")

    assert info["slot"] == 105 and info["warm_start_slot"] == 8
    assert info["prompt_offset"] == ACOUSTIC
    bundle = unpack(out, tmp_path / "post")
    after = torch.load(bundle.weights_path, map_location="cpu", weights_only=False)[param]

    # The new slot is a copy of fr-FR, at the correct offset ...
    assert torch.equal(after[:, ACOUSTIC + 105], before[:, ACOUSTIC + 8])
    # ... it actually changed (the stub's free slots start near zero) ...
    assert not torch.equal(after[:, ACOUSTIC + 105], before[:, ACOUSTIC + 105])
    # ... and every other column, acoustic and language alike, is untouched.
    others = [c for c in range(after.shape[1]) if c != ACOUSTIC + 105]
    assert torch.equal(after[:, others], before[:, others])

    cfg = OmegaConf.load(bundle.config_path)
    assert cfg.model_defaults.prompt_dictionary["ht-HT"] == 105
    assert cfg.model_defaults.prompt_dictionary["ht"] == 105
    # Must be True: the flag gates construction of prompt_kernel. False makes
    # NeMo skip building it, and the checkpoint's weights fail to load.
    assert cfg.model_defaults.initialize_prompt_feature is True


def test_patch_without_warm_start_leaves_slot_untouched(tmp_path):
    nemo, param = make_stub_nemo(tmp_path)
    before = torch.load(unpack(nemo, tmp_path / "pre").weights_path,
                        map_location="cpu", weights_only=False)[param].clone()
    out = tmp_path / "cold.nemo"
    patch(nemo, out, tag="ht-HT", slot=110, warm_start_from=None)
    after = torch.load(unpack(out, tmp_path / "post").weights_path,
                       map_location="cpu", weights_only=False)[param]
    assert torch.equal(after, before)


def test_extract_tokenizer_strips_nemo_hash_prefix(tmp_path):
    # NeMo stores artifacts as "<32 hex>_tokenizer.model"; model.tokenizer.dir=
    # expects plain names, so a missed prefix breaks training at startup.
    nemo, _ = make_stub_nemo(tmp_path)
    dest = extract_tokenizer(nemo, tmp_path / "tok")
    names = {p.name for p in dest.iterdir()}
    assert "tokenizer.model" in names and "vocab.txt" in names


def test_rejects_occupied_slot(tmp_path):
    nemo, _ = make_stub_nemo(tmp_path)
    with pytest.raises(ValueError, match="not free"):
        patch(nemo, tmp_path / "x.nemo", tag="ht-HT", slot=8)


def test_rejects_slot_beyond_vector_width(tmp_path):
    nemo, _ = make_stub_nemo(tmp_path)
    with pytest.raises(ValueError, match="not free"):
        patch(nemo, tmp_path / "x.nemo", tag="ht-HT", slot=NUM_PROMPTS)


def test_rejects_unknown_warm_start_language(tmp_path):
    nemo, _ = make_stub_nemo(tmp_path)
    with pytest.raises(RuntimeError, match="not in prompt_dictionary"):
        patch(nemo, tmp_path / "x.nemo", tag="ht-HT", slot=105, warm_start_from="xx-XX")


def test_rejects_duplicate_tag(tmp_path):
    nemo, _ = make_stub_nemo(tmp_path)
    with pytest.raises(RuntimeError, match="already maps"):
        patch(nemo, tmp_path / "x.nemo", tag="fr-FR", slot=105)
