"""The generated training config is what actually registers ht-HT at build time.

Patching the .nemo alone is not enough: NeMo builds the model from the YAML and
only then loads the checkpoint weights into it.
"""

import pytest

pytest.importorskip("omegaconf")

from kreyol_asr.train import CONFIG_DIR, CONFIG_NAME, build_command, materialize_config  # noqa: E402

LANG = {"tag": "ht-HT", "slot": 105, "warm_start_from": "fr-FR"}


def make_fake_nemo(tmp_path, prompt_dict=None):
    from omegaconf import OmegaConf

    conf_dir = tmp_path / CONFIG_DIR
    conf_dir.mkdir(parents=True)
    (tmp_path / "examples/asr").mkdir(parents=True, exist_ok=True)
    (tmp_path / "examples/asr/speech_to_text_finetune.py").touch()
    # Mirrors the real recipe: EVERYTHING lives under `model:`, and the dataloader
    # reads the dictionary through ${model.model_defaults.prompt_dictionary}.
    # A fixture with a top-level model_defaults hides the path bug entirely.
    OmegaConf.save(
        OmegaConf.create({
            "name": "FastConformer-Transducer-BPE-Prompt-Streaming",
            "model": {
                "model_defaults": {
                    "num_prompts": 128,
                    "prompt_dictionary": prompt_dict or {"en-US": 0, "fr-FR": 8},
                    "initialize_prompt_feature": False,
                },
                "encoder": {"att_context_size": [70, 6]},
                "train_ds": {
                    "prompt_dictionary": "${model.model_defaults.prompt_dictionary}",
                },
            },
        }),
        conf_dir / f"{CONFIG_NAME}.yaml",
    )
    return tmp_path


def test_generated_config_registers_the_tag_and_keeps_existing_languages(tmp_path):
    from omegaconf import OmegaConf

    nd = make_fake_nemo(tmp_path)
    conf_dir, name = materialize_config(nd, LANG)
    conf = OmegaConf.load(conf_dir / f"{name}.yaml")

    assert conf.model.model_defaults.prompt_dictionary["ht-HT"] == 105
    assert conf.model.model_defaults.prompt_dictionary["ht"] == 105
    assert conf.model.model_defaults.prompt_dictionary["fr-FR"] == 8  # untouched
    assert conf.model.model_defaults.initialize_prompt_feature is True


def test_generated_config_name_has_no_dash(tmp_path):
    # Hydra's --config-name should not carry a dash-laden locale verbatim.
    nd = make_fake_nemo(tmp_path)
    _, name = materialize_config(nd, LANG)
    assert "-" not in name and name.endswith("ht_HT")


def test_original_config_is_left_intact(tmp_path):
    from omegaconf import OmegaConf

    nd = make_fake_nemo(tmp_path)
    materialize_config(nd, LANG)
    original = OmegaConf.load(nd / CONFIG_DIR / f"{CONFIG_NAME}.yaml")
    assert "ht-HT" not in original.model.model_defaults.prompt_dictionary


def test_command_never_passes_a_dashed_hydra_key(tmp_path, monkeypatch):
    """Hydra's override grammar rejects `-` in key names — this is the regression."""
    nd = make_fake_nemo(tmp_path)
    monkeypatch.setenv("NEMO_DIR", str(nd))
    data = tmp_path / "data"
    (data / "manifests").mkdir(parents=True)
    for split in ("train", "val"):
        (data / "manifests" / f"{split}.json").write_text("{}\n")

    cmd = build_command({"att_context_size": [[56, 3], [56, 0]]}, data,
                        tmp_path / "init.nemo", tmp_path / "tok", lang=LANG)

    overrides = [c for c in cmd if c.startswith(("+", "++"))]
    for o in overrides:
        key = o.split("=", 1)[0]
        assert "-" not in key, f"Hydra cannot parse the key in {o!r}"
    assert not any("prompt_dictionary" in o for o in overrides)
    assert f"--config-name={CONFIG_NAME}_ht_HT" in cmd


def test_command_carries_the_full_multi_latency_context(tmp_path, monkeypatch):
    nd = make_fake_nemo(tmp_path)
    monkeypatch.setenv("NEMO_DIR", str(nd))
    data = tmp_path / "data"
    (data / "manifests").mkdir(parents=True)
    for split in ("train", "val"):
        (data / "manifests" / f"{split}.json").write_text("{}\n")

    cmd = build_command({"att_context_size": [[56, 3], [56, 0], [56, 6], [56, 13]]},
                        data, tmp_path / "init.nemo", tmp_path / "tok", lang=LANG)
    assert "++model.encoder.att_context_size=[[56,3],[56,0],[56,6],[56,13]]" in cmd


def test_noam_only_sched_keys_are_dropped_when_switching_scheduler(tmp_path):
    """NoamAnnealing's d_model/warmup_ratio crash every other NeMo scheduler.

    The stock streaming-prompt YAML ships a Noam sched block; overriding only
    `sched.name` leaves the siblings behind and NeMo raises
    TypeError: WarmupAnnealHoldPolicy.__init__() got an unexpected keyword 'd_model'.
    """
    from omegaconf import OmegaConf

    nd = make_fake_nemo(tmp_path)
    y = nd / CONFIG_DIR / f"{CONFIG_NAME}.yaml"
    conf = OmegaConf.load(y)
    conf.model.optim = {"name": "adamw", "lr": 2.0,
                        "sched": {"name": "NoamAnnealing", "d_model": 512,
                                  "warmup_steps": 10000, "warmup_ratio": None,
                                  "min_lr": 1e-6}}
    OmegaConf.save(conf, y)

    conf_dir, name = materialize_config(nd, LANG, scheduler="CosineAnnealing")
    out = OmegaConf.load(conf_dir / f"{name}.yaml")
    assert "d_model" not in out.model.optim.sched
    assert "warmup_ratio" not in out.model.optim.sched
    assert "min_lr" in out.model.optim.sched, "non-Noam keys must survive"


def test_noam_keys_kept_when_staying_on_noam(tmp_path):
    from omegaconf import OmegaConf

    nd = make_fake_nemo(tmp_path)
    y = nd / CONFIG_DIR / f"{CONFIG_NAME}.yaml"
    conf = OmegaConf.load(y)
    conf.model.optim = {"sched": {"name": "NoamAnnealing", "d_model": 512}}
    OmegaConf.save(conf, y)

    conf_dir, name = materialize_config(nd, LANG, scheduler="NoamAnnealing")
    out = OmegaConf.load(conf_dir / f"{name}.yaml")
    assert out.model.optim.sched.d_model == 512


def test_dictionary_is_written_where_the_dataloader_reads_it(tmp_path):
    """The regression: a top-level `model_defaults` block is an orphan.

    The recipe nests everything under `model:` and the Lhotse prompt dataset
    resolves ${model.model_defaults.prompt_dictionary}. Writing to a top-level
    key leaves the real dictionary stock, and training dies at the first batch
    with "Unknown prompt key: 'ht-HT'".
    """
    from omegaconf import OmegaConf

    nd = make_fake_nemo(tmp_path)
    conf_dir, name = materialize_config(nd, LANG)
    raw = OmegaConf.load(conf_dir / f"{name}.yaml")

    assert "model_defaults" not in raw, "must not create a top-level orphan block"
    assert raw.model.model_defaults.prompt_dictionary["ht-HT"] == 105

    # The interpolation the dataloader actually follows must now see ht-HT.
    resolved = OmegaConf.to_container(raw, resolve=True)
    assert resolved["model"]["train_ds"]["prompt_dictionary"]["ht-HT"] == 105


def test_finder_survives_unresolvable_interpolations(tmp_path):
    """The recipe is full of ${...}; iterating a DictConfig resolves them and throws.

    It also carries `prompt_dictionary: ${model.model_defaults.prompt_dictionary}`
    under train_ds — a reference, not a definition. The finder must skip references
    and must not blow up on interpolations that don't resolve standalone.
    """
    from omegaconf import OmegaConf

    from kreyol_asr.lang_slot import find_prompt_dictionary

    conf = OmegaConf.create({
        "model": {
            "model_defaults": {"prompt_dictionary": {"en-US": 0, "fr-FR": 8}},
            "encoder": {"d_model": "${model.model_defaults.missing_key}"},
            "train_ds": {"prompt_dictionary": "${model.model_defaults.prompt_dictionary}"},
            "validation_ds": {"prompt_dictionary": "${model.model_defaults.prompt_dictionary}"},
        },
    })
    path, d = find_prompt_dictionary(conf)
    assert path == "model.model_defaults.prompt_dictionary"
    assert d == {"en-US": 0, "fr-FR": 8}


def test_finder_ignores_reference_only_entries(tmp_path):
    from omegaconf import OmegaConf

    from kreyol_asr.lang_slot import find_prompt_dictionary

    conf = OmegaConf.create({
        "a": {"prompt_dictionary": "${b.prompt_dictionary}"},
        "b": {"prompt_dictionary": {"en-US": 0}},
    })
    path, _ = find_prompt_dictionary(conf)
    assert path == "b.prompt_dictionary"


def test_smoke_script_uses_a_separate_exp_dir():
    """resume_if_exists=true makes a leftover smoke checkpoint hijack the real run.

    exp_manager would resume from 20 steps over 50 clips and silently ignore
    --init, so the real fine-tune would start from the smoke model instead of the
    patched base checkpoint — and nothing in the logs would call that out.
    """
    import re
    from pathlib import Path as _P

    sh = (_P(__file__).resolve().parents[1] / "scripts/smoke.sh").read_text()
    train = re.search(r"kreyol-asr train(.*?)\n\necho", sh, re.S)
    assert train, "could not find the train invocation in smoke.sh"
    assert "--exp-dir" in train.group(1), "smoke train must not share exp/ with the real run"


def test_train_cli_exposes_exp_dir_override():
    from pathlib import Path as _P

    src = (_P(__file__).resolve().parents[1] / "src/kreyol_asr/cli.py").read_text()
    train_src = src.split("def train(")[1].split("\n@app.command")[0]
    assert "--exp-dir" in train_src
    assert 'tcfg["exp_dir"] = str(exp_dir)' in train_src
