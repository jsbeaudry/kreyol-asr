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
