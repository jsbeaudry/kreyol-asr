"""kreyol-asr — HF dataset URLs in, fine-tuned streaming ASR model out."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from . import BASE_MODEL
from .config import DataConfig, FinetuneConfig, hf_token

app = typer.Typer(add_completion=False, help=__doc__, no_args_is_help=True)
console = Console()


def _exit_without_finalizing(code: int = 0) -> None:
    """Leave the process without running interpreter finalization.

    fsspec/aiohttp — reached through huggingface_hub's HfFileSystem and `datasets`'
    streaming reader — leaves a background thread that touches the GIL while Python
    is finalizing, aborting the process with

        Fatal Python error: PyGILState_Release: thread state ... must be current

    That happens *after* every manifest, wav and report is safely on disk, so the
    work is complete and only teardown is unsafe. Exiting here turns a cosmetic
    SIGABRT (exit 134) into a clean 0, which matters because `smoke.sh` and
    `pod_bootstrap.sh` run under `set -e` and would otherwise abort the pipeline
    after a fully successful prepare.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)

DATA_CFG = typer.Option("configs/datasets.ht.yaml", "--config", "-c", help="Dataset config YAML")
FT_CFG = typer.Option("configs/finetune.ht.yaml", "--config", "-c", help="Fine-tune config YAML")


@app.command()
def prepare(
    config: str = DATA_CFG,
    limit: Optional[int] = typer.Option(None, help="Clips per source — for smoke tests"),
    base_model: str = typer.Option(BASE_MODEL, help="Model whose tokenizer is used for coverage"),
    datasets_override: Optional[str] = typer.Option(
        None, "--datasets", "-d",
        help="Comma-separated HF dataset ids, overriding the config's `sources`. "
             "Use repo_id[:split] e.g. 'me/a:train,me/b:validation'"),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir",
        help="Override the config's output_dir. Use this to keep a smoke run from "
             "overwriting a real one — they otherwise share data/ht."),
):
    """Turn Hugging Face datasets into NeMo manifests + 16 kHz mono wavs."""
    from .config import Source
    from .datasets import prepare as run_prepare

    cfg = DataConfig.load(config)
    if datasets_override:
        srcs = []
        for spec in datasets_override.split(","):
            spec = spec.strip()
            if not spec:
                continue
            repo, _, split = spec.partition(":")
            srcs.append(Source(repo_id=repo, split=split or "train"))
        cfg.sources = srcs
        console.print(f"Overriding sources with: {[s.repo_id for s in srcs]}")

    if output_dir is not None:
        cfg.output_dir = Path(output_dir)
    console.print(f"Output dir: {cfg.output_dir}")

    stats = run_prepare(cfg, base_model=base_model, token=hf_token(), limit=limit)
    console.print(f"\n[bold]{stats['total_hours']} h[/bold] over {stats['total_clips']} clips "
                  f"({stats['split_mode']} split)")
    console.print(f"Tokenizer verdict: {stats['tokenizer_verdict']}")
    _exit_without_finalizing()


@app.command()
def inspect(config: str = FT_CFG):
    """Dump the base .nemo internals — run this before `patch`."""
    from .lang_slot import fetch_base_nemo, inspect as run_inspect

    ft = FinetuneConfig.load(config)
    nemo = fetch_base_nemo(ft.base_model, token=hf_token())
    console.print_json(json.dumps(run_inspect(nemo), indent=2, default=str))


@app.command()
def patch(
    config: str = FT_CFG,
    out: Path = typer.Option(Path("checkpoints/nemotron-3.5-asr-ht-init.nemo"), "--out", "-o"),
    tokenizer_dir: Path = typer.Option(Path("checkpoints/tokenizer")),
    prompt_param: Optional[str] = typer.Option(None, help="Override prompt-projection param name"),
    prompt_offset: Optional[int] = typer.Option(
        None, help="Override where the 128-slot one-hot starts in the projection input"),
):
    """Register the Creole language slot and warm-start it."""
    from .lang_slot import extract_tokenizer, fetch_base_nemo, patch as run_patch

    ft = FinetuneConfig.load(config)
    nemo = fetch_base_nemo(ft.base_model, token=hf_token())
    info = run_patch(
        nemo, out,
        tag=ft.language["tag"], slot=int(ft.language["slot"]),
        warm_start_from=ft.language.get("warm_start_from"), prompt_param=prompt_param,
        prompt_offset=prompt_offset,
    )
    extract_tokenizer(nemo, tokenizer_dir)
    Path(out).with_suffix(".info.json").write_text(json.dumps(info, indent=2))
    console.print_json(json.dumps(info, indent=2))


@app.command()
def train(
    config: str = FT_CFG,
    data_dir: Path = typer.Option(Path("data/ht"), "--data-dir"),
    init: Path = typer.Option(Path("checkpoints/nemotron-3.5-asr-ht-init.nemo"), "--init"),
    tokenizer_dir: Path = typer.Option(Path("checkpoints/tokenizer")),
    max_steps: Optional[int] = typer.Option(None, help="Override max_steps (smoke tests)"),
    devices: Optional[int] = typer.Option(None, help="Override GPU count"),
    exp_dir: Optional[Path] = typer.Option(
        None, "--exp-dir",
        help="Override exp_manager.exp_dir. Give a smoke run its own directory: "
             "resume_if_exists=true will otherwise resume the real run from the "
             "smoke checkpoint, silently ignoring --init."),
    dry_run: bool = typer.Option(False, help="Print the NeMo command without running it"),
):
    """Fine-tune (GPU host only)."""
    from .train import run as run_train

    ft = FinetuneConfig.load(config)
    tcfg = dict(ft.train)
    if max_steps is not None:
        tcfg["max_steps"] = max_steps
    if devices is not None:
        tcfg["devices"] = devices
    if exp_dir is not None:
        tcfg["exp_dir"] = str(exp_dir)
    if not Path(init).exists():
        raise typer.BadParameter(f"{init} not found — run `kreyol-asr patch` first.")
    rc = run_train(tcfg, Path(data_dir), Path(init), Path(tokenizer_dir),
                   lang=ft.language, dry_run=dry_run)
    raise typer.Exit(rc)


@app.command()
def bench(
    config: str = FT_CFG,
    data_dir: Path = typer.Option(Path("data/ht"), "--data-dir"),
    exp_dir: Path = typer.Option(Path("exp"), "--exp-dir"),
    out_dir: Path = typer.Option(Path("benchmarks/latest"), "--out-dir"),
    model: Optional[Path] = typer.Option(None, help="Fine-tuned .nemo (default: newest in exp-dir)"),
    baseline_only: bool = typer.Option(False, help="Score only the untouched base model"),
    skip_baseline: bool = typer.Option(False, help="Score only the fine-tuned model"),
    normalize_scoring: bool = typer.Option(
        False, help="Lowercase + strip punctuation before scoring — use when your "
                    "transcripts are unpunctuated, so the baseline isn't unfairly penalised"),
    arg_style: str = typer.Option("hydra", help="NeMo infer script CLI style: hydra|argparse"),
):
    """Baseline vs fine-tuned WER/CER across every streaming latency."""
    from .evaluate import bench as run_bench
    from .lang_slot import fetch_base_nemo
    from .train import latest_checkpoint

    ft = FinetuneConfig.load(config)
    base = None if skip_baseline else str(fetch_base_nemo(ft.base_model, token=hf_token()))
    tuned = None
    if not baseline_only:
        tuned = str(model) if model else str(latest_checkpoint(Path(exp_dir)))
        console.print(f"Fine-tuned checkpoint: {tuned}")

    run_bench(ft_cfg={"eval": ft.eval}, data_dir=Path(data_dir), out_dir=Path(out_dir),
              base_model_nemo=base, finetuned_nemo=tuned, lang_tag=ft.language["tag"],
              aggressive_scoring=normalize_scoring, arg_style=arg_style)


@app.command()
def push(
    config: str = FT_CFG,
    exp_dir: Path = typer.Option(Path("exp"), "--exp-dir"),
    model: Optional[Path] = typer.Option(None, help="Fine-tuned .nemo (default: newest in exp-dir)"),
    data_dir: Path = typer.Option(Path("data/ht"), "--data-dir"),
    bench_dir: Path = typer.Option(Path("benchmarks/latest"), "--bench-dir"),
    tokenizer_dir: Path = typer.Option(Path("checkpoints/tokenizer")),
    repo: Optional[str] = typer.Option(None, help="Target repo id (overrides config)"),
    public: bool = typer.Option(False, help="Publish publicly instead of private"),
):
    """Upload the model + benchmark-backed model card to the Hub."""
    from .publish import push as run_push
    from .train import latest_checkpoint

    ft = FinetuneConfig.load(config)
    if not hf_token():
        raise typer.BadParameter("HF_TOKEN is not set — cannot push.")
    nemo = Path(model) if model else latest_checkpoint(Path(exp_dir))

    def maybe(path: Path):
        return json.loads(path.read_text()) if path.exists() else None

    run_push(
        repo_id=repo or ft.publish["repo_id"],
        nemo_path=nemo,
        lang_tag=ft.language["tag"],
        slot=int(ft.language["slot"]),
        warm_start_from=ft.language.get("warm_start_from"),
        base_model=ft.base_model,
        benchmark=maybe(Path(bench_dir) / "results.json"),
        data_stats=maybe(Path(data_dir) / "prepare_stats.json"),
        private=not public and ft.publish.get("private", True),
        token=hf_token(),
        tokenizer_dir=Path(tokenizer_dir),
    )


if __name__ == "__main__":
    app()
