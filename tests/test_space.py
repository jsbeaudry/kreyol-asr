"""Static checks on the Gradio Space client.

The Space is a thin client, so these guard the wiring rather than any model: the
two endpoints stay distinct, the gradio pin tracks the declared sdk_version, and
the failure modes we actually hit in production keep their explicit handling.
"""

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPACE = ROOT / "space"
APP = (SPACE / "app.py").read_text()
REQS = (SPACE / "requirements.txt").read_text()
README = (SPACE / "README.md").read_text()
# The comments name the very packages the file exists to exclude ("No torch, no
# NeMo"), so match against real requirement lines only.
REQ_LINES = "\n".join(
    l for l in REQS.splitlines() if l.strip() and not l.strip().startswith("#"))


def test_asr_and_tts_use_separate_endpoints():
    """One env var per endpoint. They cannot be merged: kani-tts pins
    nemo-toolkit[tts]==2.4.0 while ASR needs nemo from git main."""
    assert "RUNPOD_ENDPOINT" in APP
    assert "RUNPOD_TTS_ENDPOINT" in APP
    assert "ASR_BASE" in APP and "TTS_BASE" in APP


def test_both_tabs_exist():
    assert APP.count("gr.Tab(") == 2
    assert "Speech to text" in APP
    assert "Text to speech" in APP


def test_all_eight_voices_are_offered():
    for v in ("nana", "deniz", "mako", "mariz", "klodin", "jan", "job", "leo"):
        assert f'"{v}"' in APP


def test_restricted_key_403_is_explained():
    """A RunPod restricted key is scoped to specific endpoints, so a key that
    works in one tab can 403 in the other. A bare stack trace does not tell the
    user that the fix is in the RunPod console."""
    assert "403" in APP
    assert "restricted" in APP.lower()


def test_gradio_pin_tracks_the_declared_sdk_version():
    """`gradio>=5.0` let pip install 6.x, which dropped
    Textbox(show_copy_button=...) and broke the app at import."""
    sdk = next(l for l in README.splitlines() if l.startswith("sdk_version:"))
    major_minor = ".".join(sdk.split(":")[1].strip().split(".")[:2])
    assert f"gradio~={major_minor}" in REQ_LINES, (
        f"requirements.txt must pin gradio to the {major_minor} line "
        f"declared as sdk_version")


def test_space_stays_a_thin_client():
    """No torch or NeMo here — the whole point is that the Space needs no GPU."""
    for heavy in ("torch", "nemo", "transformers", "kani"):
        assert heavy not in REQ_LINES.lower()


@pytest.mark.parametrize("fn", ["transcribe", "synthesize"])
def test_handlers_return_messages_rather_than_raising(fn):
    node = next(n for n in ast.parse(APP).body
                if isinstance(n, ast.FunctionDef) and n.name == fn)
    src = ast.unparse(node)
    # WorkerError is caught and turned into a user-facing string in both paths.
    assert "except WorkerError" in src
    assert "raise" not in src.replace("raise WorkerError", "")
