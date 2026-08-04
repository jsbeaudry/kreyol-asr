"""Decide which Radio Haiti pseudo-labels are worth training on.

The corpus transcripts come from a model scoring ~21% CER on its own test set — very
plausibly worse than the fine-tune in this repo. Training on them unfiltered distils
that model's errors. Two independent signals decide instead:

  1. the corpus's own per-segment confidence, cut at a percentile of the *measured*
     distribution (it is compressed — p5 0.874, p50 0.962 — so an absolute threshold
     like "0.9" would be arbitrary);
  2. agreement with our current published checkpoint, as per-segment CER.

The CER is kept as a *band*, not "near zero". Near-zero-only would train the model on
what it already gets right: it biases toward easy audio and cannot move WER. Everything
would distil the weaker teacher. The band, conjoined with confidence, is the compromise
— and the review sheet, not intuition, is what sets its edges.
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from . import LEFT_CONTEXT
from .evaluate import _norm_for_scoring, read_predictions, run_inference
from .radio_haiti import histogram, percentiles, unstyle_clitics

from kreyol_common import console

# CER band edges. Starting values — `gate --review` recalculates label accuracy per
# band from human verdicts and tells you where to move them.
CER_CONSENSUS = 0.05
CER_BAND_HIGH = 0.45
# Fraction of the agreeing clips to keep. Per-clip, deterministic by hash.
CONSENSUS_SHARE = 0.5
BANDS = (0.0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.45, 0.60, 1.01)


def norm_for_agreement(text: str) -> str:
    """Normalize away orthography so the CER measures errors, not spelling conventions.

    `_norm_for_scoring(aggressive=True)` handles case and punctuation, but not the
    difference that dominates here: we emit `n'ap`, the corpus writes `n ap`. Stripping
    punctuation turns `n'ap` into `nap`, which still mismatches `n ap` — so the clitic
    has to be un-joined *before* punctuation is stripped, or the gate rejects perfectly
    good labels for a reason that has nothing to do with the audio.
    """
    return _norm_for_scoring(unstyle_clitics((text or "").lower()), aggressive=True)


def segment_cer(reference: str, hypothesis: str) -> float:
    import jiwer

    ref, hyp = norm_for_agreement(reference), norm_for_agreement(hypothesis)
    if not ref:
        return 1.0
    return float(jiwer.cer(ref, hyp))


def resolve_model(target: str, token: str | None = None) -> str:
    """A `.nemo` path stays as-is; a Hub repo id is downloaded first.

    NeMo's streaming inference script takes `model_path=` and expects a file on
    disk. Handing it `jsbeaudry/…-ht` fails deep inside NeMo with a message about
    a missing file, which points nowhere near the actual cause.
    """
    if Path(target).exists():
        return str(target)
    if "/" not in target or target.endswith(".nemo"):
        raise RuntimeError(
            f"{target} does not exist and is not a Hub repo id. Pass a .nemo path "
            f"or an org/name repo."
        )
    from .lang_slot import fetch_base_nemo

    try:
        return str(fetch_base_nemo(target, token=token))
    except Exception as e:  # noqa: BLE001 - re-raised with the cause named
        hint = "" if token else (
            " HF_TOKEN is not set, and the published checkpoint is private by "
            "default (see `publish.private` in the fine-tune config) — export it "
            "on this host and retry."
        )
        raise RuntimeError(f"Could not fetch {target}: {e}.{hint}") from e


def read_candidates(audiofolder: Path) -> list[dict]:
    meta = audiofolder / "metadata.all.jsonl"
    if not meta.exists():
        raise RuntimeError(f"{meta} missing — run `kreyol-asr radio ingest` first.")
    return [json.loads(l) for l in meta.read_text().splitlines() if l.strip()]


def _keep_fraction(key: str, fraction: float, salt: str = "") -> bool:
    """Deterministic per-clip subsampling — same input, same subset, every run."""
    h = hashlib.md5((salt + key).encode()).hexdigest()[:8]
    return int(h, 16) / 0xFFFFFFFF < fraction


# --------------------------------------------------------------------------- agree


def build_manifest(rows: list[dict], audiofolder: Path, out: Path, *, lang: str) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for r in rows:
            fh.write(json.dumps({
                "audio_filepath": str((audiofolder / r["file_name"]).resolve()),
                "duration": r["duration"],
                "text": r["text"],
                "lang": lang,
                "target_lang": lang,
            }, ensure_ascii=False) + "\n")
    return out


def agree(audiofolder: Path, model: str, out: Path, *, lang: str = "ht-HT",
          right_context: int = 3, batch_size: int = 16, min_confidence: float = 0.0,
          max_hours: float | None = None, seed: int = 1804, device: int = 0,
          nemo_dir: Path | None = None, arg_style: str = "hydra",
          token: str | None = None) -> dict[str, Any]:
    """Transcribe every candidate with our own model and score the disagreement.

    One attention context only. The gate measures label agreement, not latency
    behaviour, so the other three contexts `bench` uses would be pure GPU waste.
    """
    # Resolve before reading candidates: a missing token or typo'd repo should fail
    # in seconds, not after building a manifest for 30k clips.
    model_path = resolve_model(model, token=token)
    rows = read_candidates(audiofolder)
    before = len(rows)
    rows = [r for r in rows if r["confidence"] >= min_confidence]
    if max_hours:
        rng = random.Random(seed)
        rng.shuffle(rows)
        acc, capped = 0.0, []
        for r in rows:
            if acc / 3600 >= max_hours:
                break
            capped.append(r)
            acc += r["duration"]
        rows = sorted(capped, key=lambda r: r["file_name"])
    if not rows:
        raise RuntimeError(f"No candidates left after min_confidence={min_confidence}.")
    console.print(f"Agreement pass over {len(rows)}/{before} clips "
                  f"({sum(r['duration'] for r in rows) / 3600:.2f} h)")

    out.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(rows, audiofolder, out / "candidates.json", lang=lang)
    preds = run_inference(model_path, manifest, out / "preds", target_lang=lang,
                          att_context=[LEFT_CONTEXT, right_context], batch_size=batch_size,
                          device=device, nemo_dir=nemo_dir, arg_style=arg_style)
    hyps = read_predictions(preds)

    # NeMo's streaming output carries no audio path (see evaluate.worst_clips), so the
    # only join available is manifest order. That is safe *if* the counts match — and a
    # silent length mismatch would pair every clip with the wrong hypothesis, so refuse.
    if len(hyps) != len(rows):
        raise RuntimeError(
            f"Inference returned {len(hyps)} rows for {len(rows)} clips. The join is "
            f"positional (NeMo emits no audio_filepath), so a mismatch would misalign "
            f"every hypothesis. Re-run the pass rather than trusting this output."
        )

    scored = []
    for r, h in zip(rows, hyps):
        ours = h.get("pred_text", "") or ""
        cer = segment_cer(r["text"], ours)
        # An empty hypothesis scores CER 1.0, but that is not evidence about the
        # label — it is our decoder producing nothing. Measured on the first 5 h:
        # 11.7% of clips, and strongly length-dependent (23.6% of clips under 2 s
        # versus 1.5% above 10 s), i.e. a cache-aware streaming warm-up artifact.
        # Treated as disagreement it would strip short clips from the corpus
        # wholesale, which is the opposite of what the evidence supports.
        scored.append({**r, "cer": round(cer, 5), "ours": ours, "band": _band(cer),
                       "cer_reliable": bool(ours.strip())})

    path = out / "per_segment.jsonl"
    with open(path, "w") as fh:
        for s in scored:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")
    console.print(f"[green]Wrote[/green] {path}")
    return {"clips": len(scored), "hours": round(sum(s["duration"] for s in scored) / 3600, 3),
            "cer": percentiles([s["cer"] for s in scored]),
            "band_counts": dict(Counter(s["band"] for s in scored))}


def _band(cer: float) -> str:
    for lo, hi in zip(BANDS, BANDS[1:]):
        if lo <= cer < hi:
            return f"{lo:.2f}-{hi:.2f}"
    return f">{BANDS[-2]:.2f}"


# -------------------------------------------------------------------------- review


def review(agreement: Path, audiofolder: Path, out: Path, *, n: int = 180,
           seed: int = 1804, copy_audio: bool = True) -> dict[str, Any]:
    """A sheet to check by ear before committing GPU hours.

    Sampled across CER band x confidence tercile with equal counts per cell, not
    uniformly: the cells that decide the thresholds are the rare ones, and a uniform
    sample would barely contain them.
    """
    import csv as _csv

    rows = [json.loads(l) for l in (agreement / "per_segment.jsonl").read_text().splitlines()
            if l.strip()]
    if not rows:
        raise RuntimeError(f"{agreement}/per_segment.jsonl is empty — run `radio agree` first.")

    confs = sorted(r["confidence"] for r in rows)
    t1, t2 = confs[len(confs) // 3], confs[2 * len(confs) // 3]
    tercile = lambda c: "low" if c < t1 else ("mid" if c < t2 else "high")  # noqa: E731

    cells: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        cells.setdefault((r["band"], tercile(r["confidence"])), []).append(r)

    rng = random.Random(seed)
    per_cell = max(1, n // max(len(cells), 1))
    sample: list[dict] = []
    for key in sorted(cells):
        pool = sorted(cells[key], key=lambda r: r["file_name"])
        rng.shuffle(pool)
        for r in pool[:per_cell]:
            sample.append({**r, "conf_tercile": key[1]})

    out.mkdir(parents=True, exist_ok=True)
    fields = ["file_name", "recording_id", "timestamp", "duration", "confidence",
              "conf_tercile", "cer", "band", "corpus_text", "our_text", "verdict"]
    with open(out / "review.csv", "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in sample:
            w.writerow({
                "file_name": r["file_name"],
                "recording_id": r["recording_id"],
                "timestamp": f"{r['start_ms'] / 1000:.1f}-{r['end_ms'] / 1000:.1f}",
                "duration": r["duration"],
                "confidence": r["confidence"],
                "conf_tercile": r["conf_tercile"],
                "cer": r["cer"],
                "band": r["band"],
                "corpus_text": r["text"],
                "our_text": r["ours"],
                "verdict": "",
            })

    if copy_audio:
        clips = out / "clips"
        clips.mkdir(exist_ok=True)
        for r in sample:
            src = audiofolder / r["file_name"]
            if src.exists():
                shutil.copy2(src, clips / Path(r["file_name"]).name)

    (out / "review.md").write_text(_render_review(sample, t1, t2))
    # The CSV is for re-gating; this is for the ears. Reviewing means listening, and
    # a page with inline players beats opening 151 wavs by hand.
    (out / "index.html").write_text(_render_review_html(sample))
    console.print(f"[green]Wrote[/green] {out/'review.csv'} ({len(sample)} clips)")
    console.print("Fill the `verdict` column with: corpus | ours | both-ok | both-bad | unclear, "
                  "then pass it back via `radio gate --review`.")
    return {"sampled": len(sample), "cells": len(cells),
            "confidence_terciles": [round(t1, 4), round(t2, 4)]}


def _render_review(sample: list[dict], t1: float, t2: float) -> str:
    lines = [
        "# Radio Haiti — label review",
        "",
        "For each clip: listen, then say which transcript is right.",
        "",
        "- `corpus` — the corpus transcript matches the audio (our model is wrong)",
        "- `ours` — our model matches the audio (the corpus label is wrong)",
        "- `both-ok` — both acceptable, differences are spelling only",
        "- `both-bad` — neither matches; the audio is unusable or the speech is not Kreyòl",
        "- `unclear` — you cannot tell",
        "",
        "This is what decides the gate thresholds. `corpus` and `both-ok` are labels worth "
        "training on; `ours` and `both-bad` are label noise. The CER band where "
        "`corpus`+`both-ok` falls below ~60% is where `--cer-band-high` belongs.",
        "",
        f"Confidence terciles: low < {t1:.4f} <= mid < {t2:.4f} <= high",
        "",
        "| clip | cer | conf | corpus | ours |",
        "|---|---:|---:|---|---|",
    ]
    for r in sample:
        lines.append(f"| `{Path(r['file_name']).name}` | {r['cer']:.3f} | {r['confidence']:.3f} "
                     f"| {r['text']} | {r['ours']} |")
    return "\n".join(lines) + "\n"


def read_verdicts(path: Path) -> dict[str, str]:
    import csv as _csv

    with open(path, newline="") as fh:
        return {r["file_name"]: (r.get("verdict") or "").strip().lower()
                for r in _csv.DictReader(fh) if (r.get("verdict") or "").strip()}


def band_accuracy(rows: list[dict], verdicts: dict[str, str]) -> dict[str, dict]:
    """Share of human-checked clips per band whose *corpus* label was usable."""
    acc: dict[str, Counter] = {}
    for r in rows:
        v = verdicts.get(r["file_name"])
        if not v:
            continue
        c = acc.setdefault(r["band"], Counter())
        c["checked"] += 1
        if v in ("corpus", "both-ok"):
            c["label_ok"] += 1
    return {b: {"checked": c["checked"], "label_ok": c["label_ok"],
                "accuracy": round(c["label_ok"] / c["checked"], 3) if c["checked"] else None}
            for b, c in sorted(acc.items())}


# ---------------------------------------------------------------------------- gate


def gate(audiofolder: Path, agreement: Path, *, review_csv: Path | None = None,
         min_confidence: float | None = None, band_confidence: float | None = None,
         cer_accept_below: float = CER_CONSENSUS, cer_band_high: float = CER_BAND_HIGH,
         consensus_share: float = CONSENSUS_SHARE, max_hours: float | None = None,
         max_chars_per_second: float = 25.0, seed: int = 1804) -> dict[str, Any]:
    """Write `metadata.jsonl` — the only file `_iter_localdir` will train on."""
    rows = [json.loads(l) for l in (agreement / "per_segment.jsonl").read_text().splitlines()
            if l.strip()]
    if not rows:
        raise RuntimeError(f"{agreement}/per_segment.jsonl is empty — run `radio agree` first.")

    confs = [r["confidence"] for r in rows]
    pct = percentiles(confs)
    if min_confidence is None:
        min_confidence = pct["p10"]
    if band_confidence is None:
        band_confidence = pct["p50"]

    verdicts = read_verdicts(review_csv) if review_csv else {}
    accuracy = band_accuracy(rows, verdicts) if verdicts else {}

    tiers: Counter = Counter()
    hours: Counter = Counter()
    accepted: list[dict] = []
    for r in rows:
        cps = len(r["text"]) / r["duration"] if r["duration"] else float("inf")
        if not r["text"].strip():
            tier = "reject_empty"
        elif cps > max_chars_per_second:
            tier = "reject_chars_per_second"
        elif r["confidence"] < min_confidence:
            tier = "reject_low_confidence"
        elif not r.get("cer_reliable", bool((r.get("ours") or "").strip())):
            # Our decoder emitted nothing, so the CER of 1.0 says nothing about the
            # transcript. Fall back to the one signal that still applies. Kept in its
            # own tier so the volume stays visible rather than blending into the band.
            tier = ("unscored" if r["confidence"] >= band_confidence
                    else "reject_unscored_low_confidence")
        elif r["cer"] > cer_band_high:
            tier = "reject_disagreement"
        elif r["cer"] <= cer_accept_below:
            # Safe but low-information: we already produce this text. Subsampled so the
            # accepted set does not fill up with clips that cannot teach anything. This
            # is a per-clip rate, not an exact share of hours — the realised share lands
            # near it but is not pinned to it, and the report prints what it actually was.
            tier = ("consensus" if _keep_fraction(r["file_name"], consensus_share, str(seed))
                    else "reject_consensus_capped")
        elif r["confidence"] >= band_confidence:
            tier = "informative"
        else:
            tier = "reject_band_low_confidence"
        tiers[tier] += 1
        hours[tier] += r["duration"] / 3600
        if tier in ("consensus", "informative", "unscored"):
            accepted.append(r)

    accepted.sort(key=lambda r: (-r["confidence"], r["file_name"]))
    if max_hours:
        capped, acc = [], 0.0
        for r in accepted:
            if acc / 3600 >= max_hours:
                tiers["reject_over_max_hours"] += 1
                hours["reject_over_max_hours"] += r["duration"] / 3600
                continue
            capped.append(r)
            acc += r["duration"]
        accepted = capped
    accepted.sort(key=lambda r: r["file_name"])

    if not accepted:
        raise RuntimeError("The gate accepted nothing — loosen --cer-band-high or "
                           "--min-confidence, and check gate_report.md for why.")

    keys = ("file_name", "text", "speaker", "recording_id", "corpus_split", "start_ms",
            "end_ms", "duration", "confidence", "cer", "band", "origin", "cer_reliable")
    meta = audiofolder / "metadata.jsonl"
    with open(meta, "w") as fh:
        for r in accepted:
            fh.write(json.dumps({k: r[k] for k in keys if k in r}, ensure_ascii=False) + "\n")

    stats = _gate_stats(rows, accepted, tiers, hours, accuracy, pct,
                        min_confidence=min_confidence, band_confidence=band_confidence,
                        cer_accept_below=cer_accept_below, cer_band_high=cer_band_high,
                        consensus_share=consensus_share, max_hours=max_hours)
    (agreement / "gate_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    (agreement / "gate_report.md").write_text(render_gate_report(stats))
    console.print(f"[green]Accepted[/green] {stats['accepted_clips']} clips / "
                  f"{stats['accepted_hours']} h -> {meta}")
    return stats


def _gate_stats(rows, accepted, tiers, hours, accuracy, pct, **thresholds) -> dict[str, Any]:
    from .text import style_stats

    return {
        "thresholds": {k: (round(v, 6) if isinstance(v, float) else v)
                       for k, v in thresholds.items()},
        "confidence_percentiles": pct,
        "candidates": len(rows),
        "candidate_hours": round(sum(r["duration"] for r in rows) / 3600, 3),
        "accepted_clips": len(accepted),
        "accepted_hours": round(sum(r["duration"] for r in accepted) / 3600, 3),
        "tiers": {k: {"clips": v, "hours": round(hours[k], 3)} for k, v in tiers.most_common()},
        "cer_histogram": histogram([r["cer"] for r in rows], 20, 0.0, 1.0),
        "accepted_cer": percentiles([r["cer"] for r in accepted]),
        "accepted_style": style_stats([r["text"] for r in accepted]),
        "accepted_speakers": len({r["speaker"] for r in accepted}),
        "top_recording_hours": dict(Counter(
            {k: round(v, 3) for k, v in _hours_by(accepted, "recording_id").items()}
        ).most_common(10)),
        "label_accuracy_by_band": accuracy,
    }


def _hours_by(rows: list[dict], key: str) -> dict[str, float]:
    acc: Counter = Counter()
    for r in rows:
        acc[r[key]] += r["duration"] / 3600
    return dict(acc)


def render_gate_report(s: dict[str, Any]) -> str:
    t, style = s["thresholds"], s["accepted_style"]
    total = s["accepted_hours"] or 1.0
    lines = [
        "# Radio Haiti — gate report",
        "",
        f"- Candidates: **{s['candidates']}** clips / **{s['candidate_hours']} h**",
        f"- Accepted: **{s['accepted_clips']}** clips / **{s['accepted_hours']} h** "
        f"across **{s['accepted_speakers']}** speaker keys",
        "",
        "## Thresholds",
        "",
        f"- `min_confidence` **{t['min_confidence']}** (p10 of the measured distribution)",
        f"- `band_confidence` **{t['band_confidence']}** (p50)",
        f"- consensus at CER <= **{t['cer_accept_below']}**, subsampled to "
        f"**{t['consensus_share']:.0%}** of eligible clips",
        f"- informative band **{t['cer_accept_below']} < CER <= {t['cer_band_high']}**",
        f"- hour cap: **{t['max_hours'] or 'none'}**",
        "",
        "## Where every candidate went",
        "",
        "| tier | clips | hours |",
        "|---|---:|---:|",
    ]
    for name, v in s["tiers"].items():
        lines.append(f"| `{name}` | {v['clips']} | {v['hours']} |")
    lines += ["", "## CER distribution (all candidates)", "", "| bin | clips |", "|---|---:|"]
    lines += [f"| {b} | {c} |" for b, c in s["cer_histogram"] if c]

    if s["label_accuracy_by_band"]:
        lines += ["", "## Human verdicts — is the corpus label usable?", "",
                  "| band | checked | label OK | accuracy |", "|---|---:|---:|---:|"]
        for band, v in s["label_accuracy_by_band"].items():
            lines.append(f"| {band} | {v['checked']} | {v['label_ok']} | {v['accuracy']} |")
        lines += ["", "> Put `--cer-band-high` at the band where accuracy drops below ~0.60.", ""]
    else:
        lines += ["", "> No review sheet supplied. The band edges above are starting values, "
                      "not measurements — run `radio review`, fill the verdicts, and re-gate "
                      "with `--review` before trusting them.", ""]

    lines += ["", "## Concentration", "",
              "Hours from the top recordings — a single dominant programme is a diversity "
              "risk, not a win:", ""]
    for rid, h in s["top_recording_hours"].items():
        lines.append(f"- `{rid}`: {h} h ({h / total:.1%})")
    lines += [
        "",
        "## Accepted transcript style",
        "",
        f"- cased **{style['cased_ratio']:.1%}** · punctuated **{style['punctuated_ratio']:.1%}**",
        "",
        "Blend is controlled by `--max-hours` here, not by `weight` in the dataset config: "
        "`_apply_weights` rounds and clamps to at least one copy, so `weight: 0.5` emits a "
        "full copy rather than half.",
        "",
    ]
    return "\n".join(lines) + "\n"


_REVIEW_CSS = """
body{font:15px/1.5 -apple-system,system-ui,sans-serif;margin:0;background:#f6f6f7;color:#111}
header{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:12px 20px;z-index:9}
h1{font-size:17px;margin:0 0 4px} .sub{color:#666;font-size:13px;max-width:820px}
#bar{margin-top:8px;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
button{padding:6px 12px;border:1px solid #bbb;border-radius:6px;background:#fff;cursor:pointer}
button:hover{background:#eee} #done{font-variant-numeric:tabular-nums;color:#666;font-size:13px}
main{max-width:860px;margin:0 auto;padding:16px}
.card{background:#fff;border:1px solid #e2e2e4;border-radius:10px;padding:14px;margin:12px 0}
.card.answered{border-color:#8bc34a;background:#fcfffa}
.meta{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:8px}
.n{font-weight:600;color:#888}
.rec{color:#999;font-size:12px;margin-left:auto;font-family:ui-monospace,monospace}
.pill{background:#eef;border-radius:20px;padding:2px 9px;font-size:12px;color:#446}
audio{width:100%;height:34px;margin:4px 0 10px}
.t{display:flex;gap:10px;margin:5px 0}
.t b{flex:0 0 56px;color:#888;font-size:12px;text-transform:uppercase;padding-top:2px}
.t span{flex:1}
.v{display:flex;gap:14px;flex-wrap:wrap;margin-top:10px;padding-top:10px;
   border-top:1px solid #f0f0f0;font-size:13px}
.v label{cursor:pointer;user-select:none}
"""

_REVIEW_JS = """
const K='rh-verdicts';
let V=JSON.parse(localStorage.getItem(K)||'{}');
function paint(){
  document.querySelectorAll('.card').forEach(c=>{
    const f=c.querySelector('.v').dataset.file;
    if(V[f]){c.classList.add('answered');
      const r=c.querySelector('input[value="'+V[f]+'"]'); if(r) r.checked=true;}
  });
  document.getElementById('done').textContent=
    Object.keys(V).length+' / '+document.querySelectorAll('.card').length+' done';
}
document.addEventListener('change',e=>{
  if(e.target.type!=='radio')return;
  const c=e.target.closest('.card');
  V[c.querySelector('.v').dataset.file]=e.target.value;
  localStorage.setItem(K,JSON.stringify(V)); paint();
});
function dl(){
  let out='file_name,verdict\\n';
  for(const k of Object.keys(V)) out+=k+','+V[k]+'\\n';
  const a=document.createElement('a');
  a.href=URL.createObjectURL(new Blob([out],{type:'text/csv'}));
  a.download='verdicts.csv'; a.click();
}
paint();
"""

_VERDICTS = [("corpus", "corpus right"), ("ours", "ours right"), ("both-ok", "both ok"),
             ("both-bad", "both bad"), ("unclear", "unclear")]


def _render_review_html(sample: list[dict]) -> str:
    """A listening page. Verdicts live in localStorage and export as the CSV `gate` reads.

    Deliberately dependency-free and offline: it is served off the pod by
    `python -m http.server`, so anything fetched from a CDN would simply not load.
    """
    import html as _h

    ordered = sorted(sample, key=lambda r: (r["cer"], r["file_name"]))
    cards = []
    for i, r in enumerate(ordered):
        name = _h.escape(r["file_name"].split("/")[-1])
        ours = _h.escape(r["ours"]) or "<i>(our model returned nothing)</i>"
        radios = "".join(
            f'<label><input type="radio" name="v{i}" value="{v}">{lbl}</label>'
            for v, lbl in _VERDICTS)
        cards.append(
            f'<div class="card"><div class="meta"><span class="n">#{i + 1}</span>'
            f'<span class="pill">CER {r["cer"]:.3f}</span>'
            f'<span class="pill">conf {r["confidence"]:.3f}</span>'
            f'<span class="pill">{_h.escape(r.get("conf_tercile", ""))}</span>'
            f'<span class="pill">{r["duration"]:.1f}s</span>'
            f'<span class="rec">{_h.escape(r["recording_id"])[:8]} @ '
            f'{r["start_ms"] / 1000:.0f}s</span></div>'
            f'<audio controls preload="none" src="clips/{name}"></audio>'
            f'<div class="t"><b>corpus</b><span>{_h.escape(r["text"])}</span></div>'
            f'<div class="t"><b>ours</b><span>{ours}</span></div>'
            f'<div class="v" data-file="{_h.escape(r["file_name"])}">{radios}</div></div>')

    return (
        "<!doctype html><meta charset=utf-8><title>Radio Haiti — label review</title>"
        f"<style>{_REVIEW_CSS}</style>"
        "<header><h1>Radio Haiti — label review</h1><div class=sub>"
        "Listen, then say which transcript matches the audio. <b>corpus right</b> and "
        "<b>both ok</b> count as usable labels; <b>ours right</b> and <b>both bad</b> are "
        "label noise. The CER band where usable labels fall below ~60% is where "
        "<code>--cer-band-high</code> belongs. Sorted by CER, lowest first. Saved in this "
        "browser as you go.</div><div id=bar>"
        "<button onclick=\"dl()\">Download verdicts CSV</button>"
        "<button onclick=\"if(confirm('Clear all verdicts?')){localStorage.clear();"
        "location.reload()}\">Reset</button><span id=done></span></div></header>"
        f"<main>{''.join(cards)}</main><script>{_REVIEW_JS}</script>")
