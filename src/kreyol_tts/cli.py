"""`kreyol-tts` — Kokoro fine-tuning for Haitian Creole.

Separate entry point from `kreyol-asr` because the dependency sets are disjoint: this
needs torch/scipy/pyloudnorm, that needs NeMo. Imports stay inside each command so the
non-GPU verbs (`audit`, `prepare`, `voicepack`, `push`) run on a laptop with only the
`tts` extra installed.
"""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help=__doc__)


def _token() -> str | None:
    from kreyol_asr.config import hf_token
    return hf_token()


@app.command()
def audit(
    config: Path = typer.Option("configs/datasets.ht.yaml", help="dataset config to audit"),
    out: Path = typer.Option("benchmarks/tts", help="where to write audit.md/json"),
    samples: int = typer.Option(40, help="clips sampled per source"),
) -> None:
    """Measure what the corpus actually contains. No GPU, no bulk download.

    Reads parquet footers and a few row groups spread across shards. The number that
    matters is E>8k against stock Kokoro's own 1.66e-02 — a 24 kHz header says
    nothing about whether there is content up there.
    """
    from kreyol_asr.config import DataConfig

    from .audit import run
    run(DataConfig.load(config), _token(), out, n_samples=samples)


@app.command()
def prepare(
    config: Path = typer.Option("configs/tts.ht.yaml"),
    limit: int = typer.Option(0, help="clips per source; 0 = all. Use a small value first."),
) -> None:
    """Build the 24 kHz corpus and StyleTTS 2 manifests."""
    from .config import TTSDataConfig
    from .datasets import prepare as _prepare

    cfg = TTSDataConfig.load(config)
    stats = _prepare(cfg, _token(), limit or None)
    typer.echo(f"\n{stats['total_hours']} h / {stats['total_clips']} clips -> {cfg.output_dir}")
    for name, v in stats["voices"].items():
        typer.echo(f"  {name:8s} tier {v['tier']}  stage2 clips {v['clips_stage2']:4d}  {v['status']}")


@app.command()
def warmstart(
    out: Path = typer.Option("checkpoints/kokoro_ht_init.pth"),
    check: bool = typer.Option(False, "--check", help="also round-trip back and assert bit-identity"),
) -> None:
    """Convert Kokoro's checkpoint into StyleTTS 2's training layout."""
    from .warmstart import fetch_kokoro, roundtrip_check, to_styletts2

    if check:
        roundtrip_check(_token(), workdir=out.parent)
    else:
        to_styletts2(fetch_kokoro(_token()), out)


@app.command()
def voicepack(
    checkpoint: Path = typer.Option(..., help="Stage 2 checkpoint"),
    voice: str = typer.Option(..., help="voice name, e.g. deniz"),
    out: Path = typer.Option(None, help="defaults to voices/<kf_name>.pt"),
) -> None:
    """Extract a (510, 1, 256) voicepack that loads in the stock `kokoro` package."""
    from .voices import pack_name

    target = out or Path("voices") / f"{pack_name(voice)}.pt"
    typer.echo(f"would write {target} from {checkpoint}")
    raise typer.Exit(code=1)  # wired up in Phase 6, once Stage 2 checkpoints exist


if __name__ == "__main__":
    app()
