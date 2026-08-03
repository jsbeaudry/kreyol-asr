"""Move weights between Kokoro's inference layout and StyleTTS 2's training layout.

Kokoro has no published training code, so fine-tuning means running its architecture
through StyleTTS 2 and warm-starting from the released checkpoint. The two disagree
about layout, and about which modules exist at all.

Measured on `kokoro-v1_0.pth` (2026-08-02):

    {component: {"module.<path>": tensor}}    5 components, 81.8M params
      decoder       375 tensors   53.3M   iSTFTNet
      predictor     122 tensors   16.2M   LSTM + shared + F0/N + duration_proj
      bert           25 tensors    6.3M   PL-BERT
      text_encoder   24 tensors    5.6M
      bert_encoder    2 tensors    0.4M

StyleTTS 2 wants `{"net": {component: {"<path>": tensor}}}` — same components, no
`module.` prefix, plus several modules Kokoro does not ship at all.

**The gap that matters.** Kokoro contains no `style_encoder` and no
`predictor_encoder`. It does not need them: at inference the style vector comes from
a voicepack (`ref_s`), not from encoding a mel. But StyleTTS 2 *training* needs both,
so they start from random init. Three consequences worth stating plainly:

  * The warm start covers the acoustic path only, never the style path.
  * Stage 1's first steps look worse than a full warm start would predict.
  * Extracted voicepacks are only as good as encoders trained on the local corpus.

So do not trim Stage 1 epochs to save GPU time — that is the budget the style
encoders need.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console

from . import KOKORO_CKPT, KOKORO_REPO

console = Console()

# Present in the released checkpoint; these are what the warm start actually buys.
KOKORO_COMPONENTS = ("bert", "bert_encoder", "predictor", "decoder", "text_encoder")

# Needed by StyleTTS 2 training, absent from Kokoro. Verified absent, not assumed.
#   style_encoder / predictor_encoder — mel -> style vector, needed only in training
#   diffusion                         — Kokoro dropped the style diffusion sampler
#   mpd / msd / wd                    — discriminators, training-only by definition
TRAIN_FROM_SCRATCH = ("style_encoder", "predictor_encoder", "diffusion",
                      "mpd", "msd", "wd")

# Both embeddings are sized by the phoneme vocab, so a vocab change breaks two
# tensors, not one. Kreyòl needs no additions — every phoneme is already in
# Kokoro's 114 populated ids — but assert it rather than trust it.
VOCAB_SIZED = {
    "text_encoder": ("embedding.weight", 512),
    "bert": ("embeddings.word_embeddings.weight", 128),
}
N_TOKEN = 178

# An iSTFTNet generator upsamples in 2 stages (kernels 20, 12); a HiFi-GAN uses 4.
# Counting the layers catches a wrong `decoder.type` that a config string would not:
# training HiFi-GAN by accident produces weights that never load into KModel, and
# the failure surfaces only at the very end of the project.
ISTFTNET_UPSAMPLE_LAYERS = 2


def fetch_kokoro(token: str | None = None) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(KOKORO_REPO, KOKORO_CKPT, token=token))


def _strip_module(sd: dict[str, Any]) -> dict[str, Any]:
    """Drop the `module.` DataParallel prefix Kokoro's tensors were saved with."""
    return {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}


def _add_module(sd: dict[str, Any]) -> dict[str, Any]:
    return {k if k.startswith("module.") else f"module.{k}": v for k, v in sd.items()}


def _n_params(sd: dict[str, Any]) -> int:
    import torch

    return sum(v.numel() for v in sd.values() if torch.is_tensor(v))


def audit_checkpoint(ckpt: dict[str, Any]) -> dict[str, Any]:
    """Everything we assert about the checkpoint, computed in one place."""
    import torch

    report: dict[str, Any] = {"components": {}, "missing": [], "problems": []}

    for name in KOKORO_COMPONENTS:
        if name not in ckpt:
            report["problems"].append(f"expected component {name!r} is absent")
            continue
        sd = ckpt[name]
        report["components"][name] = {"tensors": len(sd), "params": _n_params(sd)}

    report["missing"] = [m for m in TRAIN_FROM_SCRATCH
                         if m not in ckpt and not any(m in k for sd in ckpt.values() for k in sd)]

    for comp, (suffix, dim) in VOCAB_SIZED.items():
        sd = ckpt.get(comp, {})
        key = next((k for k in sd if k.endswith(suffix)), None)
        if key is None:
            report["problems"].append(f"{comp}: no vocab-sized tensor ending {suffix!r}")
            continue
        shape = tuple(sd[key].shape)
        report.setdefault("vocab", {})[comp] = shape
        if shape != (N_TOKEN, dim):
            report["problems"].append(
                f"{comp}.{suffix} is {shape}, expected {(N_TOKEN, dim)} — the vocab "
                f"changed, so the embedding can no longer be loaded wholesale"
            )

    dec = ckpt.get("decoder", {})
    ups = {k.split(".ups.")[1].split(".")[0] for k in dec if ".ups." in k}
    report["upsample_layers"] = len(ups)
    if len(ups) != ISTFTNET_UPSAMPLE_LAYERS:
        report["problems"].append(
            f"decoder has {len(ups)} upsample layers, expected "
            f"{ISTFTNET_UPSAMPLE_LAYERS} — this does not look like iSTFTNet. A "
            f"HiFi-GAN decoder will not load into kokoro's KModel."
        )

    report["total_params"] = sum(c["params"] for c in report["components"].values())
    del torch
    return report


def render_audit(report: dict[str, Any]) -> str:
    lines = [f"{'component':16s} {'tensors':>8s} {'params':>13s}", "-" * 40]
    for name, c in report["components"].items():
        lines.append(f"{name:16s} {c['tensors']:8d} {c['params']:13,d}")
    lines += ["-" * 40,
              f"{'loaded':16s} {'':8s} {report['total_params']:13,d} "
              f"({report['total_params']/1e6:.1f}M)"]
    if report["missing"]:
        lines += ["", "train from scratch (absent from Kokoro):",
                  "  " + ", ".join(report["missing"])]
    if v := report.get("vocab"):
        lines += ["", "vocab-sized tensors: "
                  + ", ".join(f"{k}{tuple(s)}" for k, s in v.items())]
    lines.append(f"decoder upsample layers: {report['upsample_layers']} "
                 f"(iSTFTNet expects {ISTFTNET_UPSAMPLE_LAYERS})")
    return "\n".join(lines)


def to_styletts2(ckpt_path: Path | str, out_path: Path | str,
                 strict: bool = True) -> dict[str, Any]:
    """Kokoro checkpoint -> StyleTTS 2 `{"net": ...}` training checkpoint."""
    import torch

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    report = audit_checkpoint(ckpt)
    console.print(render_audit(report))

    if report["problems"]:
        for p in report["problems"]:
            console.print(f"[red]problem[/red] {p}")
        if strict:
            raise RuntimeError(
                f"{len(report['problems'])} structural problem(s) in {ckpt_path}. "
                f"A half-loaded warm start is indistinguishable from slow training, "
                f"so this refuses rather than proceeding."
            )

    net = {name: _strip_module(ckpt[name]) for name in KOKORO_COMPONENTS if name in ckpt}
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"net": net}, out)
    console.print(f"[green]Wrote[/green] {out}  "
                  f"({report['total_params']/1e6:.1f}M params, "
                  f"{len(report['missing'])} module(s) left to train)")
    return report


def to_kokoro(ckpt_path: Path | str, out_path: Path | str) -> dict[str, Any]:
    """StyleTTS 2 training checkpoint -> Kokoro's layout, for publishing.

    The inverse of `to_styletts2`. Modules that only exist for training are dropped:
    KModel does not know about them, and shipping them would bloat the artifact and
    invite the wrong thing to be loaded.
    """
    import torch

    obj = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    net = obj.get("net", obj)

    out_sd, dropped = {}, []
    for name, sd in net.items():
        if name in KOKORO_COMPONENTS:
            out_sd[name] = _add_module(sd)
        else:
            dropped.append(name)

    absent = [c for c in KOKORO_COMPONENTS if c not in out_sd]
    if absent:
        raise RuntimeError(
            f"{ckpt_path}: missing component(s) {absent} that KModel requires. "
            f"Publishing this would produce a repo that cannot be loaded."
        )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_sd, out)
    total = sum(_n_params(sd) for sd in out_sd.values())
    console.print(f"[green]Wrote[/green] {out}  ({total/1e6:.1f}M params)")
    if dropped:
        console.print(f"  dropped training-only modules: {', '.join(sorted(dropped))}")
    return {"components": sorted(out_sd), "dropped": sorted(dropped), "params": total}


def roundtrip_check(token: str | None = None,
                    workdir: Path | str = "checkpoints") -> dict[str, Any]:
    """Kokoro -> StyleTTS 2 -> Kokoro, asserting every tensor survives bit-identical.

    Runs before any training code exists. If the conversion is lossy, everything
    downstream inherits the loss silently, and the symptom would not appear until
    the published repo fails to load at the very end of the project.
    """
    import torch

    work = Path(workdir)
    src = fetch_kokoro(token)
    st2 = work / "kokoro_ht_init.pth"
    back = work / "kokoro_roundtrip.pth"

    report = to_styletts2(src, st2)
    to_kokoro(st2, back)

    original = torch.load(str(src), map_location="cpu", weights_only=False)
    restored = torch.load(str(back), map_location="cpu", weights_only=False)

    mismatches = []
    checked = 0
    for comp in KOKORO_COMPONENTS:
        a, b = original.get(comp, {}), restored.get(comp, {})
        if set(a) != set(b):
            mismatches.append(f"{comp}: key sets differ "
                              f"(+{sorted(set(b)-set(a))[:3]} -{sorted(set(a)-set(b))[:3]})")
            continue
        for k in a:
            if torch.is_tensor(a[k]):
                checked += 1
                if not torch.equal(a[k], b[k]):
                    mismatches.append(f"{comp}.{k}: tensor changed")

    if mismatches:
        for m in mismatches[:10]:
            console.print(f"[red]mismatch[/red] {m}")
        raise RuntimeError(f"round-trip is lossy: {len(mismatches)} mismatch(es)")

    console.print(f"[green]Round-trip clean[/green] — {checked} tensors bit-identical")
    report["roundtrip_tensors"] = checked
    return report
