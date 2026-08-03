"""Fail loudly, at the top of the log, on the things that otherwise waste a pod.

A pod that boots, clones, installs, and *then* dies on the first Hub call still bills
for the boot and buries the cause under a stack trace. Every check here is cheap
(metadata only, no downloads) and runs before anything expensive.

    python -m kreyol_tts.cli preflight --config configs/tts.ht.yaml
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

# The corpus is ~12 GB of 24 kHz PCM16; refuse to start a multi-hour download that
# cannot possibly fit rather than dying most of the way through it.
MIN_FREE_GB = 20.0


def _check_token() -> tuple[bool, str]:
    from kreyol_asr.config import hf_token

    tok = hf_token()
    if not tok:
        return False, (
            "HF_TOKEN is not set. Every source dataset is private, so prepare cannot "
            "read a single one. On RunPod set it in the POD's environment at deploy "
            "time — editing the template's env after creation does not always merge "
            "into pods built from it."
        )
    return True, f"HF_TOKEN present ({len(tok)} chars)"


def _check_hub_access(repo_id: str) -> tuple[bool, str]:
    """Metadata only — proves the token can see the repo without downloading it."""
    from huggingface_hub import HfApi

    from kreyol_asr.config import hf_token

    try:
        info = HfApi(token=hf_token()).dataset_info(repo_id)
        return True, f"can read {repo_id} (private={info.private})"
    except Exception as e:  # noqa: BLE001 - this is the diagnostic
        return False, (
            f"cannot read {repo_id}: {type(e).__name__}: {str(e)[:200]}\n"
            f"    A 401/RepositoryNotFound here means the token is missing, expired, "
            f"or lacks `read` scope on the private repos — not that the repo is gone."
        )


def _check_disk(out_dir: Path) -> tuple[bool, str]:
    probe = out_dir if out_dir.exists() else out_dir.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free_gb = shutil.disk_usage(probe).free / 1e9
    ok = free_gb >= MIN_FREE_GB
    return ok, (f"{free_gb:.1f} GB free at {probe}"
                + ("" if ok else f" — need at least {MIN_FREE_GB:.0f} GB for the corpus"))


def _check_writable(out_dir: Path) -> tuple[bool, str]:
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / ".preflight"
        p.write_text("x")
        p.unlink()
        return True, f"{out_dir} is writable"
    except OSError as e:
        return False, f"{out_dir} is not writable: {e}"


def _check_imports() -> tuple[bool, str]:
    missing = []
    for mod in ("numpy", "soundfile", "soxr", "scipy", "pyloudnorm", "datasets",
                "pyarrow", "huggingface_hub", "yaml"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    return (not missing), ("all runtime deps importable" if not missing
                           else f"missing: {', '.join(missing)}")


def _check_g2p() -> tuple[bool, str]:
    """The G2P is self-contained — no espeak — so this should never fail. Assert it
    anyway: a silently wrong phonemizer is the worst failure mode in this project."""
    from .g2p import g2p

    got = g2p("Bonjou, koman ou ye?")
    want = "bɔ̃ʒu, komɑ̃ u je?"
    return got == want, (f"g2p OK ({got})" if got == want
                         else f"g2p produced {got!r}, expected {want!r}")


def run(config_path: str | Path) -> list[str]:
    """Return a list of failures; empty means good to go."""
    from .config import TTSDataConfig

    cfg = TTSDataConfig.load(config_path)
    first = cfg.sources[0].repo_id

    checks = [
        ("token", _check_token()),
        ("hub access", _check_hub_access(first)),
        ("deps", _check_imports()),
        ("g2p", _check_g2p()),
        ("output dir", _check_writable(cfg.output_dir)),
        ("disk", _check_disk(cfg.output_dir)),
    ]

    failures = []
    print("=" * 70)
    print("PREFLIGHT")
    print("=" * 70)
    for name, (ok, msg) in checks:
        print(f"  [{'OK  ' if ok else 'FAIL'}] {name:12s} {msg}")
        if not ok:
            failures.append(f"{name}: {msg}")
    print("=" * 70)
    print(f"{len(cfg.sources)} sources -> {cfg.output_dir} @ {cfg.sample_rate} Hz")
    print("PREFLIGHT_OK" if not failures else f"PREFLIGHT_FAILED ({len(failures)})")
    print("=" * 70, flush=True)
    return failures
