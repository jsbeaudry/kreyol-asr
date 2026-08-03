"""Hugging Face Hub access, and error messages you can act on.

Three failure modes cost real debugging time on this project, and all three surface
from `datasets` as the same misleading "couldn't find ... check your connection":

  * A private-storage overage returns 403 on file *reads* while metadata calls keep
    working, so listings succeed and only the download fails.
  * An unwritable `HF_HOME` (e.g. a pod path like /workspace/.hf carried into a
    laptop .env) is not a connection problem either.
  * A duplicated `HF_TOKEN` line in .env silently shadows the real token.

Keeping this in one place is the whole point of `kreyol_common`.
"""

from __future__ import annotations

import os
from pathlib import Path


def _hf_home_problem() -> str | None:
    """Return why HF_HOME can't be used as a cache, or None if it's fine.

    Must actually attempt the write. `os.access()` consults the real uid's
    permission bits and returns True for root on paths root still cannot use —
    and containers run as root, which is exactly where a stale pod path like
    /workspace/.hf on a laptop needs to be caught.
    """
    home = os.environ.get("HF_HOME")
    if not home:
        return None
    probe = Path(home) / ".kreyol_write_test"
    try:
        probe.parent.mkdir(parents=True, exist_ok=True)
        probe.write_text("x")
        probe.unlink()
        return None
    except OSError as e:
        return str(e)


def _probe_repo_access(repo_id: str, token: str | None) -> str:
    """Read one byte of one file, and report what actually goes wrong.

    `datasets` collapses every Hub failure into "Couldn't find ... check your
    connection", which hides 403s. Reading through HfFileSystem surfaces the real
    status line, so the message the user sees names the true cause.
    """
    try:
        from huggingface_hub import HfApi, HfFileSystem

        info = HfApi(token=token).dataset_info(repo_id, files_metadata=True)
        target = next((s.rfilename for s in info.siblings
                       if s.rfilename.endswith((".parquet", ".wav", ".mp3", ".csv"))), None)
        if not target:
            return ""
        with HfFileSystem(token=token).open(f"datasets/{repo_id}/{target}") as fh:
            fh.read(1)
        return ""  # reads are fine; the failure is something else
    except Exception as e:  # noqa: BLE001 - this IS the diagnostic
        return f"{type(e).__name__}: {e}"


def _explain_hub_error(repo_id: str, err: Exception, token: str | None = None) -> RuntimeError:
    """Turn Hub failures into something you can act on."""
    text = f"{type(err).__name__}: {err}"
    if "storage limit" not in text.lower():
        text += " | probe -> " + _probe_repo_access(repo_id, token)
    if "storage limit" in text.lower() or "403" in text and "private" in text.lower():
        return RuntimeError(
            f"{repo_id}: Hugging Face is blocking file reads because the account's "
            f"private-repo storage limit is reached. Metadata still resolves, which is "
            f"why this looks like a permissions bug. Free up private storage or upgrade "
            f"the plan, then re-run. See https://huggingface.co/docs/hub/storage-limits"
        )
    unwritable = _hf_home_problem()
    if unwritable:
        return RuntimeError(
            f"{repo_id}: HF_HOME={os.environ.get('HF_HOME')!r} is not usable, so nothing "
            f"can be cached ({unwritable}). Pod paths like /workspace/.hf do not exist on "
            f"a laptop — comment HF_HOME out in .env when running locally. "
            f"(Underlying error: {text[:160]})"
        )
    if "401" in text or "RepositoryNotFound" in text:
        return RuntimeError(
            f"{repo_id}: not found or not readable with the current token. If it is "
            f"private, set HF_TOKEN in .env. Note a duplicated HF_TOKEN line in .env "
            f"silently shadows the real one. (Underlying error: {text[:160]})"
        )
    return RuntimeError(f"{repo_id}: could not load — {text[:300]}")
