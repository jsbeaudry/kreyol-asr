"""Benchmark: baseline vs fine-tuned, WER/CER across every streaming latency.

Inference is delegated to NeMo's cache-aware streaming script so the numbers reflect
real streaming behaviour, not offline decoding. Scoring is pure Python, so
`bench --score-only` re-scores existing predictions anywhere (including your Mac).
"""

from __future__ import annotations

import json
import re
import string
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from . import LATENCY_MS, LEFT_CONTEXT

console = Console()

INFER_SCRIPT = "examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py"
_PUNCT = str.maketrans("", "", string.punctuation + "«»…")


_LANG_TAG = re.compile(r"<[a-z]{2,3}(?:-[A-Za-z]{2,4})?>")


def _norm_for_scoring(text: str, aggressive: bool) -> str:
    text = unicodedata.normalize("NFC", text).strip()
    # NeMo can emit the prompt tag inline, e.g. "... contribution. <fr-FR>".
    # It is never part of a reference transcript, so leaving it in inflates WER
    # by one token on every clip that carries it.
    text = _LANG_TAG.sub(" ", text)
    if aggressive:
        text = text.lower().translate(_PUNCT)
    return re.sub(r"\s+", " ", text).strip()


def tags_emitted(hyps: list[str]) -> dict[str, int]:
    """Which language tags the model actually produced.

    Scoring strips these so a formatting artifact doesn't inflate WER — but that
    also destroys the evidence. If a run prompted with `ht-HT` comes back emitting
    `<fr-FR>`, the prompt conditioning is not taking effect and the WER would look
    respectable while the model is still decoding as French. Count them so that
    failure is visible instead of silently normalized away.
    """
    from collections import Counter

    counts: Counter[str] = Counter()
    for h in hyps:
        counts.update(_LANG_TAG.findall(h or ""))
    return dict(counts)


def score(pairs: list[tuple[str, str]], aggressive: bool = False) -> dict[str, Any]:
    """WER + CER over (reference, hypothesis) pairs."""
    import jiwer

    raw_hyps = [h for _, h in pairs]
    refs = [_norm_for_scoring(r, aggressive) for r, _ in pairs]
    hyps = [_norm_for_scoring(h, aggressive) for _, h in pairs]
    keep = [(r, h) for r, h in zip(refs, hyps) if r]
    if not keep:
        raise RuntimeError("No non-empty references to score against.")
    refs, hyps = [r for r, _ in keep], [h for _, h in keep]
    return {
        "wer": round(jiwer.wer(refs, hyps), 5),
        "cer": round(jiwer.cer(refs, hyps), 5),
        "clips": len(refs),
        "tags_emitted": tags_emitted(raw_hyps),
    }


def worst_clips(rows: list[dict], n: int, aggressive: bool) -> list[dict]:
    import jiwer

    scored = []
    for i, r in enumerate(rows):
        ref = _norm_for_scoring(r.get("text", ""), aggressive)
        hyp = _norm_for_scoring(r.get("pred_text", ""), aggressive)
        if not ref:
            continue
        # NeMo's streaming output carries only {pred_text, text, wer} — no path.
        # Fall back to the row index so the worst-clips report still points
        # somewhere useful instead of raising KeyError.
        scored.append({"audio_filepath": r.get("audio_filepath", f"<row {i}>"),
                       "ref": r.get("text", ""),
                       "hyp": r.get("pred_text", ""), "wer": round(jiwer.wer(ref, hyp), 4)})
    return sorted(scored, key=lambda x: -x["wer"])[:n]


def run_inference(model_path: str, manifest: Path, out_json: Path, *, target_lang: str,
                  att_context: list[int], batch_size: int = 8, device: int = 0,
                  nemo_dir: Path | None = None, arg_style: str = "hydra",
                  extra: list[str] | None = None) -> Path:
    """Drive NeMo's cache-aware streaming inference script.

    `arg_style` exists because NeMo has shipped this script with both a `key=value`
    and an argparse interface across releases. If one errors out, switch styles in
    the CLI (`--arg-style argparse`) instead of editing code.
    """
    import os

    nd = Path(nemo_dir or os.environ.get("NEMO_DIR", "/opt/NeMo"))
    script = nd / INFER_SCRIPT
    if not script.exists():
        raise RuntimeError(f"{script} not found. Set NEMO_DIR to a NeMo git checkout.")

    # NeMo's streaming infer script treats `output_path` as a DIRECTORY and picks
    # the filename itself:
    #     hyp_json = os.path.join(cfg.output_path, fname)
    #     os.makedirs(cfg.output_path, exist_ok=True)
    # Passing a file path makes it mkdir that path and write inside, so reading it
    # back fails with "IsADirectoryError". Give it a directory and find the file.
    out_root = out_json.with_suffix("")
    out_root.mkdir(parents=True, exist_ok=True)
    att = f"[{att_context[0]},{att_context[1]}]"
    if arg_style == "hydra":
        args = [
            f"model_path={model_path}",
            f"dataset_manifest={manifest.resolve()}",
            f"output_path={out_root.resolve()}",
            f"target_lang={target_lang}",
            f"att_context_size={att}",
            "decoder_type=rnnt",
            "pad_and_drop_preencoded=true",
            f"batch_size={batch_size}",
            f"cuda={device}",
            "strip_lang_tags=false",
        ]
    else:
        args = [
            "--asr_model", model_path,
            "--dataset_manifest", str(manifest.resolve()),
            "--output_path", str(out_root.resolve()),
            "--att_context_size", att,
            "--batch_size", str(batch_size),
            "--device", f"cuda:{device}",
        ]
    cmd = [sys.executable, str(script), *args, *(extra or [])]
    console.print("  " + " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        raise RuntimeError(
            f"Inference failed (exit {rc}). If the script rejected the argument names, "
            f"retry with --arg-style {'argparse' if arg_style == 'hydra' else 'hydra'}."
        )
    return locate_predictions(out_root)


def locate_predictions(out_root: Path) -> Path:
    """Find the manifest NeMo actually wrote inside `out_root`.

    The filename is derived from the model and manifest names (e.g.
    `streaming_out_nemotron-3.5-asr-streaming-0_test.json`), so it cannot be
    predicted reliably — glob for it instead.
    """
    if out_root.is_file():
        return out_root
    found = sorted(out_root.glob("**/*.json"))
    if not found:
        raise RuntimeError(
            f"Inference reported success but wrote no JSON under {out_root}. "
            f"Check the NeMo inference log above."
        )
    if len(found) > 1:
        console.print(f"[yellow]{len(found)} prediction files in {out_root}; "
                      f"using {found[0].name}[/yellow]")
    return found[0]


def read_predictions(path: Path) -> list[dict]:
    """Read NeMo's output manifest (JSON-lines, or a JSON array)."""
    raw = Path(path).read_text().strip()
    if raw.startswith("["):
        rows = json.loads(raw)
    else:
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    for r in rows:
        if "pred_text" not in r:
            for alt in ("pred", "prediction", "transcript", "hypothesis"):
                if alt in r:
                    r["pred_text"] = r[alt]
                    break
    return rows


def bench(
    *,
    ft_cfg: dict[str, Any],
    data_dir: Path,
    out_dir: Path,
    base_model_nemo: str | None,
    finetuned_nemo: str | None,
    lang_tag: str,
    aggressive_scoring: bool = False,
    arg_style: str = "hydra",
    nemo_dir: Path | None = None,
) -> dict[str, Any]:
    ev = ft_cfg.get("eval", {})
    test_manifest = Path(data_dir) / "manifests" / "test.json"
    if not test_manifest.exists():
        raise RuntimeError(f"{test_manifest} missing — run `kreyol-asr prepare` first.")

    contexts = [list(c) for c in ev.get("att_context_sizes", [[LEFT_CONTEXT, r] for r in LATENCY_MS])]
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    targets = []
    if base_model_nemo:
        # The base model has no Creole slot, so prompt it with its closest language.
        targets.append(("baseline", base_model_nemo, ev.get("baseline_lang", "fr-FR")))
    if finetuned_nemo:
        targets.append(("finetuned", finetuned_nemo, lang_tag))

    for name, model_path, target_lang in targets:
        for att in contexts:
            tag = f"{name}_{att[0]}-{att[1]}"
            console.print(f"[bold]{tag}[/bold] (target_lang={target_lang}, "
                          f"{LATENCY_MS.get(att[1], '?')}ms)")
            preds = run_inference(model_path, test_manifest, out_dir / f"{tag}.json",
                                  target_lang=target_lang, att_context=att,
                                  batch_size=ev.get("batch_size", 8), nemo_dir=nemo_dir,
                                  arg_style=arg_style)
            rows = read_predictions(preds)
            m = score([(r.get("text", ""), r.get("pred_text", "")) for r in rows], aggressive_scoring)
            results.append({"model": name, "att_context_size": att,
                            "latency_ms": LATENCY_MS.get(att[1]), "target_lang": target_lang,
                            **m})

            # Loud, because it is the difference between "the fine-tune did not
            # help" and "the fine-tune was never actually used".
            emitted = m.get("tags_emitted") or {}
            stray = {t: c for t, c in emitted.items()
                     if t.strip("<>").lower() != target_lang.lower()}
            if stray:
                console.print(
                    f"  [red]WARNING[/red]: prompted with {target_lang} but the model "
                    f"emitted {stray}. The prompt conditioning is not taking effect — "
                    f"WER below is not measuring {target_lang}."
                )
            elif emitted:
                console.print(f"  tags emitted: {emitted}")
            (out_dir / f"{tag}.worst.json").write_text(json.dumps(
                worst_clips(rows, ev.get("worst_n", 20), aggressive_scoring),
                indent=2, ensure_ascii=False))

    report = {
        "results": results,
        "scoring": "lowercase+nopunct" if aggressive_scoring else "exact",
        "test_clips": results[0]["clips"] if results else 0,
    }
    (out_dir / "results.json").write_text(json.dumps(report, indent=2))
    (out_dir / "report.md").write_text(render_report(report))
    _print_table(results)
    console.print(f"[green]Wrote[/green] {out_dir/'report.md'}")
    return report


def _print_table(results: list[dict]) -> None:
    t = Table(title="WER / CER by streaming latency")
    for col in ("model", "att_context", "latency", "target_lang", "WER", "CER"):
        t.add_column(col)
    for r in results:
        t.add_row(r["model"], str(r["att_context_size"]), f"{r['latency_ms']}ms",
                  r["target_lang"], f"{r['wer']:.4f}", f"{r['cer']:.4f}")
    console.print(t)


def render_report(report: dict) -> str:
    rows = report["results"]
    base = {r["latency_ms"]: r for r in rows if r["model"] == "baseline"}
    ft = {r["latency_ms"]: r for r in rows if r["model"] == "finetuned"}
    lines = [
        "# Benchmark",
        "",
        f"- Test clips: {report.get('test_clips', 'n/a')}",
        f"- Scoring: `{report['scoring']}`",
        "",
        "| latency | att_context_size | baseline WER | finetuned WER | Δ WER | finetuned CER |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for lat in sorted(set(list(base) + list(ft)), key=lambda x: (x is None, x)):
        b, f = base.get(lat), ft.get(lat)
        att = (f or b)["att_context_size"]
        cell = lambda r, k: f"{r[k]:.4f}" if r else "—"  # noqa: E731
        delta = f"{(f['wer'] - b['wer']):+.4f}" if (b and f) else "—"
        lines.append(f"| {lat} ms | `{att}` | {cell(b, 'wer')} | {cell(f, 'wer')} "
                     f"| {delta} | {cell(f, 'cer')} |")
    lines += ["", "Baseline = untouched base model prompted with its closest existing "
                  "language; fine-tuned = the new `ht-HT` slot.", ""]
    return "\n".join(lines) + "\n"
