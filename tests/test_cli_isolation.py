"""A smoke run must never be able to clobber a real prepared corpus.

`smoke.sh` and the real pipeline both default to the config's `output_dir`
(data/ht). Without an override the smoke run silently overwrites hours of
prepared audio — which is exactly what nearly happened mid-run.
"""

import re
from pathlib import Path

import pytest

pytest.importorskip("typer")

REPO = Path(__file__).resolve().parents[1]


def test_prepare_exposes_an_output_dir_override():
    from kreyol_asr import cli

    params = {p.name for p in cli.app.registered_commands
              if p.callback.__name__ == "prepare"
              for p in [p]} if False else None  # noqa: F841
    src = (REPO / "src/kreyol_asr/cli.py").read_text()
    prepare_src = src.split("def prepare(")[1].split("def ")[0]
    assert "--output-dir" in prepare_src, "prepare must be able to write somewhere else"
    assert "cfg.output_dir = Path(output_dir)" in prepare_src


def test_smoke_script_isolates_its_output():
    sh = (REPO / "scripts/smoke.sh").read_text()
    assert "SMOKE_DIR" in sh

    prep = re.search(r"kreyol-asr prepare(.*?)\n\necho", sh, re.S)
    assert prep, "could not locate the prepare invocation in smoke.sh"
    assert "--output-dir" in prep.group(1), \
        "smoke.sh prepare must pass --output-dir or it overwrites data/ht"

    train = re.search(r"kreyol-asr train(.*?)\n\necho", sh, re.S)
    assert train, "could not locate the train invocation in smoke.sh"
    assert "--data-dir" in train.group(1), \
        "smoke.sh train must read from SMOKE_DIR, not the real corpus"


def test_smoke_script_does_not_write_the_real_init_checkpoint():
    sh = (REPO / "scripts/smoke.sh").read_text()
    assert "--out checkpoints/smoke-init.nemo" in sh, \
        "smoke patch must not overwrite the real init .nemo"
