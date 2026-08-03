"""The serverless worker has to carry every lesson the pod taught us.

handler.py imports nemo and downloads a checkpoint at import time, so it cannot
be imported here. These check the artefacts as text — which is enough, because
every failure they guard against is a missing line rather than a logic bug.
"""

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SERVERLESS = ROOT / "serverless"

pytestmark = pytest.mark.skipif(not SERVERLESS.exists(), reason="serverless/ not present")

DOCKERFILE = (SERVERLESS / "Dockerfile").read_text()
HANDLER = (SERVERLESS / "handler.py").read_text()

# Comments in these files deliberately *name* the things we must not do, so
# assertions about what the build actually runs have to ignore them.
DOCKER_DIRECTIVES = "\n".join(
    line for line in DOCKERFILE.splitlines() if not line.strip().startswith("#"))


def test_handler_is_valid_python():
    ast.parse(HANDLER)


def test_numba_is_pinned_below_the_cuda_split():
    """numba 0.62 moved CUDA out to numba-cuda; 0.66 breaks NeMo's RNNT kernels with
    'Signature mismatch: 2 argument types given, but function takes 1'."""
    assert 'numba==0.61.2' in DOCKER_DIRECTIVES


def test_does_not_use_the_nemo_container():
    """nvcr.io/nvidia/nemo:26.06 needs driver >= 595.58; RunPod hosts run 570.x
    and the container crash-loops without ever opening a port."""
    assert "nvcr.io/nvidia/nemo" not in DOCKER_DIRECTIVES
    assert "runpod/pytorch" in DOCKER_DIRECTIVES


def test_uses_pep508_install_not_egg_fragment():
    """pip >= 25 rejects `git+...#egg=name[extras]` with 'invalid-egg-fragment' —
    which is exactly what the model card tells you to run."""
    assert "#egg=nemo_toolkit" not in DOCKER_DIRECTIVES
    assert "nemo_toolkit[asr] @ git+" in DOCKER_DIRECTIVES


def test_checkpoint_is_baked_into_the_image():
    """Otherwise every cold worker pays 30-60 s pulling 2.5 GB before serving."""
    assert "hf_hub_download" in DOCKER_DIRECTIVES


def test_language_is_selected_via_manifest_not_target_lang_kwarg():
    """`transcribe(target_lang=...)` sets `default_lang`, which the Lhotse dataset
    ignores. It reads cut.supervisions[0].language, i.e. the manifest `lang` field.
    Bare file paths give 'Unknown prompt key: None'."""
    assert '"lang": LANG' in HANDLER
    assert "MODEL.transcribe(_manifest(" in HANDLER
    assert "target_lang=LANG" not in HANDLER, "the kwarg route is inert for this model"


def test_left_context_is_fixed_at_56_and_not_caller_controlled():
    """56 matches the encoder's sliding_window of 57. Exposing it invites callers
    to break the streaming behaviour the checkpoint was trained for."""
    assert re.search(r"^LEFT_CONTEXT = 56$", HANDLER, re.M)
    assert "left_context" not in HANDLER.split("def handler")[1]


def test_only_supported_latencies_are_accepted():
    assert re.search(r"LATENCY_TO_RIGHT = \{80: 0, 320: 3, 560: 6, 1120: 13\}", HANDLER)
    assert "must be one of" in HANDLER


def test_model_loads_once_at_import_not_per_request():
    """A serverless worker owns its GPU for its lifetime; reloading per call would
    add tens of seconds to every warm request."""
    tree = ast.parse(HANDLER)
    module_level = {n.targets[0].id for n in tree.body
                    if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    assert "MODEL" in module_level
    handler_fn = next(n for n in tree.body
                      if isinstance(n, ast.FunctionDef) and n.name == "handler")
    assert "restore_from" not in ast.unparse(handler_fn)


def test_handler_returns_errors_instead_of_raising():
    """RunPod surfaces an exception as an opaque worker failure; a returned
    {"error": ...} tells the caller what to fix."""
    handler_fn = next(n for n in ast.parse(HANDLER).body
                      if isinstance(n, ast.FunctionDef) and n.name == "handler")
    src = ast.unparse(handler_fn)
    # Both failure paths (bad input, failed decode) must return rather than raise.
    assert src.count("'error'") + src.count('"error"') >= 2
    assert "raise" not in src, "a raise becomes an opaque RunPod worker failure"


def test_audio_url_scheme_is_validated():
    """Fetching an arbitrary scheme from caller-supplied input is an SSRF footgun."""
    assert 'startswith(("http://", "https://"))' in HANDLER


def test_audio_length_is_bounded():
    """Unbounded input against a per-second-billed GPU is a cost incident."""
    assert "MAX_AUDIO_SECONDS" in HANDLER
    assert "limit is" in HANDLER


def test_empty_transcription_is_not_masked_by_a_falsy_or():
    """`Hypothesis.text` is "" when nothing decodes, and "" is falsy.

    `getattr(hyp, "text", None) or str(hyp)` therefore returns the entire
    Hypothesis repr — tensors and all — instead of an empty result, hiding the
    actual problem (silent or truncated audio) behind a wall of object dump.
    """
    assert 'or str(out[0])' not in HANDLER
    assert 'or str(first)' not in HANDLER
    assert 'getattr(first, "text", "")' in HANDLER


def test_empty_result_explains_itself():
    """An empty transcription should say why, not just come back blank."""
    assert '"note"' in HANDLER
    assert "audio_rms" in HANDLER
    assert "silent or near-silent" in HANDLER


def test_too_short_audio_is_rejected_with_a_reason():
    assert "too short to transcribe" in HANDLER


def test_quiet_audio_is_gain_normalised():
    """Training audio peaked near full scale; mic input often sits 20-30 dB lower,
    and an out-of-distribution level makes the RNN-T emit nothing at all."""
    assert "gain = 0.95 / raw_peak" in HANDLER
    assert "1e-4 < raw_peak < 0.5" in HANDLER, "must not amplify silence"
    assert "gain_applied" in HANDLER


# --- TTS worker -------------------------------------------------------------

TTS = ROOT / "serverless-tts"
TTS_DOCKER = (TTS / "Dockerfile").read_text() if TTS.exists() else ""
# Same as DOCKER_DIRECTIVES: the comments name what the image must NOT contain.
TTS_DIRECTIVES = "\n".join(
    l for l in TTS_DOCKER.splitlines() if not l.strip().startswith("#"))
TTS_HANDLER = (TTS / "handler.py").read_text() if TTS.exists() else ""


@pytest.mark.skipif(not TTS.exists(), reason="serverless-tts/ not present")
def test_tts_is_a_separate_image_from_asr():
    """kani-tts pins nemo-toolkit[tts]==2.4.0; the ASR model needs nemo from git
    main. Installed together, pip downgrades NeMo and the ASR model stops loading."""
    assert "kani-tts==1.0.1" in TTS_DIRECTIVES
    assert "nemo_toolkit[asr]" not in TTS_DIRECTIVES, "ASR NeMo must not be in the TTS image"
    assert "kani-tts" not in DOCKER_DIRECTIVES, "TTS must not be in the ASR image"


@pytest.mark.skipif(not TTS.exists(), reason="serverless-tts/ not present")
def test_tts_voice_names_map_to_ids():
    for name in ("nana", "deniz", "mako", "mariz", "klodin", "jan", "job", "leo"):
        assert f'"{name}"' in TTS_HANDLER
    assert '"2ed43a0fa899"' in TTS_HANDLER


@pytest.mark.skipif(not TTS.exists(), reason="serverless-tts/ not present")
def test_tts_bakes_the_model_and_bounds_input():
    assert "snapshot_download" in TTS_DIRECTIVES
    assert "MAX_CHARS" in TTS_HANDLER


@pytest.mark.skipif(not TTS.exists(), reason="serverless-tts/ not present")
def test_tts_handler_returns_errors_rather_than_raising():
    fn = next(n for n in ast.parse(TTS_HANDLER).body
              if isinstance(n, ast.FunctionDef) and n.name == "handler")
    src = ast.unparse(fn)
    assert src.count("'error'") + src.count('"error"') >= 3
    assert "raise" not in src


def test_hf_transfer_is_installed_in_both_images():
    """runpod/pytorch sets HF_HUB_ENABLE_HF_TRANSFER=1, and huggingface_hub raises
    rather than falling back when the package is absent. The ASR image happened to
    get it via NeMo-on-main; the TTS image did not, and its build died at
    snapshot_download."""
    assert "hf_transfer" in DOCKER_DIRECTIVES
    if TTS.exists():
        assert "hf_transfer" in TTS_DIRECTIVES


@pytest.mark.skipif(not TTS.exists(), reason="serverless-tts/ not present")
def test_tts_overrides_nemo_transformers_cap():
    """kani-tts 1.0.1 pins nemo-toolkit[tts]==2.4.0 (transformers<=4.52.0) but its
    model.py imports TransformersKwargs, added in 4.55.0 — a plain install cannot
    import kani_tts at all. The model is also model_type lfm2, which needs >=4.54.0."""
    assert "transformers==4.57.1" in TTS_DIRECTIVES
    # must be installed after kani-tts or pip's resolution puts the cap back
    assert TTS_DIRECTIVES.index("kani-tts==1.0.1") < TTS_DIRECTIVES.index("transformers==4.57.1")


@pytest.mark.skipif(not TTS.exists(), reason="serverless-tts/ not present")
def test_tts_image_smoke_tests_the_import_at_build_time():
    """The broken resolution surfaced only as a silent crash-loop on a live worker.
    A build-time import makes the same failure fail the build instead."""
    assert "import kani_tts" in TTS_DIRECTIVES


@pytest.mark.skipif(not TTS.exists(), reason="serverless-tts/ not present")
def test_tts_exposes_only_the_real_generation_knobs():
    """KaniTTS.__call__ takes exactly temperature, top_p and repetition_penalty.
    max_new_tokens is fixed at model construction and cannot be set per request."""
    assert "GEN_PARAMS" in TTS_HANDLER
    for k in ("temperature", "top_p", "repetition_penalty"):
        assert f'"{k}"' in TTS_HANDLER
    assert "max_new_tokens" not in TTS_HANDLER.split("GEN_PARAMS")[1].split("}")[0]


@pytest.mark.skipif(not TTS.exists(), reason="serverless-tts/ not present")
def test_tts_default_char_limit_is_200():
    """Not higher: past ~250 characters the model silently truncates — the audio
    stops but the request still succeeds — and at 600 it raises outright."""
    assert '"MAX_CHARS", "200"' in TTS_HANDLER


@pytest.mark.skipif(not TTS.exists(), reason="serverless-tts/ not present")
def test_tts_clamps_generation_params_and_echoes_them_back():
    """Out-of-range values should still make audio, and the caller needs to see
    which value was actually used."""
    ns = {}
    src = TTS_HANDLER.split("def _resolve_gen")[1].split("\ndef ")[0]
    exec("GEN_PARAMS = {'temperature': (1.0, 0.1, 2.0), 'top_p': (0.95, 0.05, 1.0),"
         " 'repetition_penalty': (1.1, 1.0, 2.0)}\ndef _resolve_gen" + src, ns)
    f = ns["_resolve_gen"]
    assert f({}) == {"temperature": 1.0, "top_p": 0.95, "repetition_penalty": 1.1}
    assert f({"temperature": 9})["temperature"] == 2.0     # clamped high
    assert f({"temperature": -3})["temperature"] == 0.1    # clamped low
    assert f({"top_p": None})["top_p"] == 0.95             # explicit null -> default
    assert f({"temperature": "0.8"})["temperature"] == 0.8  # numeric strings ok
    with pytest.raises(ValueError):
        f({"temperature": "hot"})
    with pytest.raises(ValueError):
        f({"temperature": float("nan")})
