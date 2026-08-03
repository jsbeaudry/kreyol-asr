"""Shared plumbing for the ASR and TTS pipelines.

Extracted from `kreyol_asr.datasets` when `kreyol_tts` needed the same Hub access.
Deliberately *not* a grab-bag of generic utilities: `hub.py` in particular encodes
three specific, expensively-learned Hugging Face failure modes, and forking it would
mean the next one gets diagnosed once and stays broken in the other copy.

`kreyol_asr.datasets` re-exports every moved name, so existing imports and tests keep
working unchanged.
"""

from rich.console import Console

# One console for both pipelines so progress output interleaves sanely.
console = Console()

__all__ = ["console"]
