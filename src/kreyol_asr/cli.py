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


# --------------------------------------------------------------------------- radio
# Radio Haiti-Inter (Zenodo 17818122) is five steps, not one, because its transcripts
# are another model's output and every step exists to keep them from being trusted by
# default. Grouped so the flat top-level namespace stays about the model, not one corpus.

radio = typer.Typer(add_completion=False, no_args_is_help=True, help=(
    "Radio Haiti-Inter (Zenodo 10.5281/zenodo.17818122, CC-BY-4.0) — ~60 h of archival "
    "Kreyòl radio with MACHINE-GENERATED transcripts. Pipeline: inspect -> ingest -> "
    "agree -> review -> gate. Only `gate` writes the metadata.jsonl that `prepare` will "
    "read, so pseudo-labels cannot reach training ungated."))
app.add_typer(radio, name="radio")

RADIO_ROOT = typer.Option(Path("/workspace/corpora/radio-haiti/raw"), "--root",
                          help="Extracted Zenodo tree: recordings/ eaf/ transcriptions/")
RADIO_DEFAULT_FOLDER = Path("/workspace/corpora/radio-haiti/radio-haiti-inter")
RADIO_FOLDER = typer.Option(RADIO_DEFAULT_FOLDER, "--audiofolder",
                            help="Output of `radio ingest`")
# Name the directory `radio-haiti-inter`, not `segments`: `Source.slug` is the
# directory name, and it becomes the source label in data_report.md and on the card.
RADIO_OUT = typer.Option(RADIO_DEFAULT_FOLDER, "--out", "-o", "--audiofolder",
                         help="Audiofolder to write")
RADIO_AGREE = typer.Option(Path("/workspace/corpora/radio-haiti/agreement"), "--agreement",
                           help="Output of `radio agree`")


@radio.command("inspect")
def radio_inspect(
    root: Path = RADIO_ROOT,
    out: Optional[Path] = typer.Option(None, "--out", help="Default: <root>/../inspect_report.md"),
    base_model: str = typer.Option(BASE_MODEL, help="Model whose tokenizer is used for coverage"),
    audio_sample: int = typer.Option(40, help="Recordings to open for spectral analysis"),
    skip_audio: bool = typer.Option(False, help="Text only — no recordings.zip needed"),
    compare_manifest: Optional[Path] = typer.Option(
        None, "--compare-manifest",
        help="An existing NeMo manifest (data/ht/manifests/train.json) to measure "
             "vocabulary overlap and clitic spelling against"),
):
    """Measure the archive before converting any of it. Read-only."""
    from .radio_haiti import inspect_corpus, render_inspect_report

    stats = inspect_corpus(root, base_model=base_model, audio_sample=audio_sample,
                           skip_audio=skip_audio, compare_manifest=compare_manifest,
                           token=hf_token())
    dest = Path(out) if out else Path(root).parent
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "inspect_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    (dest / "inspect_report.md").write_text(render_inspect_report(stats))
    console.print(f"[green]Wrote[/green] {dest/'inspect_report.md'}")
    console.print(f"{stats['segments']} segments / {stats['segment_hours']} h · "
                  f"tokenizer: {stats['tokenizer_verdict']}")


@radio.command("ingest")
def radio_ingest(
    root: Path = RADIO_ROOT,
    out: Path = RADIO_OUT,
    base_model: str = typer.Option(BASE_MODEL),
    splits: str = typer.Option("train", help="Corpus splits to ingest, comma-separated. "
                                             "Use `test` to build the held-out slice."),
    limit: Optional[int] = typer.Option(None, help="Recordings to process — for smoke tests"),
    restyle: str = typer.Option("clitics", help="none | clitics | clitics+period"),
    no_merge: bool = typer.Option(False, "--no-merge", help="Keep source segmentation as-is"),
    max_s: float = typer.Option(15.0, help="Split above this. 15 leaves margin under "
                                           "prepare's 18 s hard drop."),
    min_s: float = typer.Option(1.0, help="Drop below this after merging"),
    target_s: float = typer.Option(12.0, help="Merge up to this length"),
    pad_ms: int = typer.Option(100, help="Context around each segment, clamped to neighbours"),
):
    """Slice the archive into an audiofolder. Writes metadata.all.jsonl — not trainable yet."""
    from .radio_haiti import ingest as run_ingest

    stats = run_ingest(root, out, base_model=base_model,
                       splits=[s.strip() for s in splits.split(",") if s.strip()],
                       limit=limit, restyle=restyle, merge=not no_merge, max_s=max_s,
                       min_s=min_s, target_s=target_s, pad_ms=pad_ms, token=hf_token())
    console.print(f"{stats['hours']} h over {stats['clips']} clips · "
                  f"tokenizer: {stats['tokenizer_verdict']}")
    _exit_without_finalizing()


@radio.command("agree")
def radio_agree(
    audiofolder: Path = RADIO_FOLDER,
    out: Path = RADIO_AGREE,
    config: str = FT_CFG,
    model: Optional[str] = typer.Option(None, help="Fine-tuned .nemo or repo id "
                                                   "(default: the config's publish repo)"),
    right_context: int = typer.Option(3, help="Single attention context — the gate measures "
                                              "label agreement, not latency"),
    batch_size: int = typer.Option(16),
    min_confidence: float = typer.Option(0.0, help="Pre-filter before spending GPU"),
    max_hours: Optional[float] = typer.Option(None, help="Cap the pass — pilot with ~5 first"),
    device: int = typer.Option(0),
    arg_style: str = typer.Option("hydra", help="NeMo infer script CLI style: hydra|argparse"),
):
    """Transcribe every candidate with our own model and score the disagreement (GPU)."""
    from .radio_gate import agree as run_agree

    ft = FinetuneConfig.load(config)
    stats = run_agree(audiofolder, model or ft.publish["repo_id"], out,
                      lang=ft.language["tag"], right_context=right_context,
                      batch_size=batch_size, min_confidence=min_confidence,
                      max_hours=max_hours, device=device, arg_style=arg_style)
    console.print_json(json.dumps(stats, indent=2))


@radio.command("review")
def radio_review(
    agreement: Path = RADIO_AGREE,
    audiofolder: Path = RADIO_FOLDER,
    out: Path = typer.Option(Path("review"), "--out"),
    n: int = typer.Option(180, help="Clips to sample across CER band x confidence tercile"),
    seed: int = typer.Option(1804),
    no_audio: bool = typer.Option(False, "--no-audio", help="Skip copying the clips"),
):
    """Build a listening sheet — the only check that confidence tracks correctness."""
    from .radio_gate import review as run_review

    console.print_json(json.dumps(
        run_review(agreement, audiofolder, out, n=n, seed=seed, copy_audio=not no_audio),
        indent=2))


@radio.command("gate")
def radio_gate_cmd(
    audiofolder: Path = RADIO_FOLDER,
    agreement: Path = RADIO_AGREE,
    review: Optional[Path] = typer.Option(None, "--review",
                                          help="Filled review.csv — turns the band edges "
                                               "from starting values into measurements"),
    min_confidence: Optional[float] = typer.Option(None, help="Default: p10 of the data"),
    band_confidence: Optional[float] = typer.Option(None, help="Default: p50 of the data"),
    cer_accept_below: float = typer.Option(0.05, help="At or below this, both models agree"),
    cer_band_high: float = typer.Option(0.45, help="Above this, one side is badly wrong"),
    consensus_share: float = typer.Option(
        0.5, help="Keep this fraction of the agreeing (low-information) clips"),
    max_hours: Optional[float] = typer.Option(
        None, help="Blend control. Use this, not `weight` — _apply_weights cannot "
                   "down-weight, only oversample."),
    seed: int = typer.Option(1804),
):
    """Write metadata.jsonl — the only file `prepare` will train on."""
    from .radio_gate import gate as run_gate

    stats = run_gate(audiofolder, agreement, review_csv=review, min_confidence=min_confidence,
                     band_confidence=band_confidence, cer_accept_below=cer_accept_below,
                     cer_band_high=cer_band_high, consensus_share=consensus_share,
                     max_hours=max_hours, seed=seed)
    console.print(f"Report: {agreement/'gate_report.md'}")
    if not review:
        console.print("[yellow]No review sheet[/yellow] — the band edges are starting "
                      "values, not measurements. Run `radio review` before trusting them.")


if __name__ == "__main__":
    app()
