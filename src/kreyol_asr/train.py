"""Wrap NeMo's speech_to_text_finetune.py with the streaming-prompt recipe."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from rich.console import Console

from . import LATENCY_MS, LEFT_CONTEXT

console = Console()

CONFIG_DIR = "examples/asr/conf/fastconformer/cache_aware_streaming"
CONFIG_NAME = "fastconformer_transducer_bpe_streaming_prompt"
FINETUNE_SCRIPT = "examples/asr/speech_to_text_finetune.py"

# What the released checkpoint itself was trained with (read from its model_config.yaml).
# Training on all four keeps one model usable at every latency.
DEFAULT_ATT_CONTEXTS = [[LEFT_CONTEXT, 3], [LEFT_CONTEXT, 0],
                        [LEFT_CONTEXT, 6], [LEFT_CONTEXT, 13]]


def nemo_dir() -> Path:
    """NeMo git checkout — the pip package does not ship examples/."""
    d = Path(os.environ.get("NEMO_DIR", "/opt/NeMo"))
    if not (d / FINETUNE_SCRIPT).exists():
        raise RuntimeError(
            f"{d/FINETUNE_SCRIPT} not found. Set NEMO_DIR to a NeMo git checkout "
            f"(the pip wheel omits examples/). Inside the NeMo container it is /opt/NeMo."
        )
    return d


def check_att_context(att: list) -> list[list[int]]:
    """Guard the single easiest way to silently ruin this fine-tune.

    Two traps:
    1. config.json gives encoder.sliding_window=57 -> 56 frames of left context, but
       the stock NeMo YAML ships [70, 6]. The wrong left context degrades the exact
       streaming behaviour the checkpoint was built around.
    2. The checkpoint trains on a *list* of contexts —
       [[56,3],[56,0],[56,6],[56,13]] — so one model serves every latency. Passing a
       single pair silently specialises the model to that latency and regresses the
       other three.

    Accepts a single pair or a list of pairs; always returns a list of pairs.
    """
    pairs = [list(att)] if att and isinstance(att[0], int) else [list(a) for a in att]
    for left, right in pairs:
        if left != LEFT_CONTEXT:
            raise ValueError(
                f"att_context_size left context is {left}, but this checkpoint uses "
                f"{LEFT_CONTEXT} (config.json sliding_window=57). Fix configs/finetune.ht.yaml."
            )
        if right not in LATENCY_MS:
            raise ValueError(
                f"right context {right} unsupported; checkpoint allows "
                f"{sorted(LATENCY_MS)} (= {list(LATENCY_MS.values())} ms)."
            )
    return pairs


# `NoamAnnealing` needs these; every other NeMo scheduler rejects them outright.
# The stock streaming-prompt YAML ships a Noam sched block, so switching the
# scheduler by name alone leaves them behind and NeMo dies with
#   TypeError: WarmupAnnealHoldPolicy.__init__() got an unexpected keyword 'd_model'
NOAM_ONLY_SCHED_KEYS = ("d_model", "warmup_ratio")


def materialize_config(nd: Path, lang: dict[str, Any],
                       scheduler: str | None = None) -> tuple[Path, str]:
    """Write a training YAML with the Creole slot already registered.

    The model is BUILT from this YAML and only then are the .nemo weights loaded
    into it — so patching `prompt_dictionary` inside the .nemo is not enough. It
    also has to be in the training config, or `target_lang=ht-HT` in our manifests
    will not resolve.

    We generate a file rather than passing `++model_defaults.prompt_dictionary.ht-HT=105`
    because Hydra's override grammar does not accept `-` in key names. The copy is
    written next to the original so any relative `defaults:` still resolve.
    """
    from omegaconf import OmegaConf

    src_dir = nd / CONFIG_DIR
    src = src_dir / f"{CONFIG_NAME}.yaml"
    if not src.exists():
        raise RuntimeError(f"{src} not found — is NEMO_DIR pointing at a full NeMo checkout?")

    from .lang_slot import find_prompt_dictionary

    conf = OmegaConf.load(src)
    tag, slot = lang["tag"], int(lang["slot"])

    # Find where this YAML actually keeps the dictionary. In the .nemo's
    # model_config.yaml it is top-level `model_defaults.prompt_dictionary`, but in
    # the training recipe everything is nested under `model:`, so the real path is
    # `model.model_defaults.prompt_dictionary` — and the dataloader reads it through
    # `${model.model_defaults.prompt_dictionary}` interpolations. Writing to the
    # wrong path silently creates an orphan block and training dies at the first
    # batch with "Unknown prompt key: 'ht-HT'".
    dict_path, existing = find_prompt_dictionary(conf)
    OmegaConf.update(conf, f"{dict_path}.{tag}", slot, force_add=True)
    OmegaConf.update(conf, f"{dict_path}.{tag.split('-')[0]}", slot, force_add=True)
    console.print(f"  registered {tag}={slot} at {dict_path} "
                  f"({len(existing)} existing entries preserved)")
    defaults_path = dict_path.rsplit(".", 1)[0]
    # MUST be true. This flag gates *construction* of the prompt MLP, not weight
    # re-initialization:
    #     if self.cfg.model_defaults.get('initialize_prompt_feature', False):
    #         self.initialize_prompt_feature()      # rnnt_bpe_models_prompt.py:106
    # With it false, `prompt_kernel` is never built and loading the checkpoint dies
    # with "Unexpected key(s) in state_dict: prompt_kernel.0.weight, ...".
    # The warm-started column survives regardless: NeMo constructs the module first,
    # then `init_from_nemo_model` loads our patched weights over it.
    OmegaConf.update(conf, f"{defaults_path}.initialize_prompt_feature", True, force_add=True)

    # Strip scheduler keys that only NoamAnnealing understands when we're moving off
    # it. Hydra's `++model.optim.sched.name=` replaces the name but leaves siblings.
    if scheduler and "noam" not in scheduler.lower():
        sched = (OmegaConf.select(conf, "model.optim.sched")
                 or OmegaConf.select(conf, "optim.sched"))
        if sched is not None:
            for key in NOAM_ONLY_SCHED_KEYS:
                if key in sched:
                    sched.pop(key)
                    console.print(f"  dropped model.optim.sched.{key} "
                                  f"(Noam-only, incompatible with {scheduler})")

    name = f"{CONFIG_NAME}_{tag.replace('-', '_')}"
    try:
        out = src_dir / f"{name}.yaml"
        OmegaConf.save(conf, out)
    except OSError:  # read-only NeMo install
        out = Path("conf") / f"{name}.yaml"
        out.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(conf, out)
        console.print(f"[yellow]NeMo conf dir not writable; wrote {out}. If the config "
                      f"used relative `defaults:`, copy it back manually.[/yellow]")
    console.print(f"Training config: {out} ({tag} -> slot {slot})")
    return out.parent.resolve(), name


def build_command(cfg: dict[str, Any], data_dir: Path, init_nemo: Path,
                  tokenizer_dir: Path, lang: dict[str, Any] | None = None) -> list[str]:
    nd = nemo_dir()
    att = check_att_context(cfg.get("att_context_size", DEFAULT_ATT_CONTEXTS))
    att_str = "[" + ",".join(f"[{a},{b}]" for a, b in att) + "]"
    manifests = Path(data_dir) / "manifests"
    for split in ("train", "val"):
        if not (manifests / f"{split}.json").exists():
            raise RuntimeError(f"Missing {manifests/f'{split}.json'} — run `kreyol-asr prepare` first.")

    exp_dir = Path(cfg.get("exp_dir", "exp"))
    ov = [
        f"+init_from_nemo_model={Path(init_nemo).resolve()}",
        f"++model.tokenizer.dir={Path(tokenizer_dir).resolve()}",
        "++model.tokenizer.type=bpe",
        f"++model.train_ds.manifest_filepath={(manifests/'train.json').resolve()}",
        f"++model.validation_ds.manifest_filepath={(manifests/'val.json').resolve()}",
        "++model.train_ds.use_lhotse=true",
        "++model.validation_ds.use_lhotse=true",
        f"++model.train_ds.batch_duration={cfg.get('batch_duration', 400)}",
        f"++model.train_ds.num_workers={cfg.get('num_workers', 8)}",
        f"++model.validation_ds.num_workers={cfg.get('num_workers', 8)}",
        f"++model.encoder.att_context_size={att_str}",
        f"++model.optim.name={cfg.get('optimizer', 'adamw')}",
        f"++model.optim.lr={cfg.get('lr', 1e-4)}",
        f"++model.optim.weight_decay={cfg.get('weight_decay', 1e-3)}",
        f"++model.optim.sched.name={cfg.get('scheduler', 'CosineAnnealing')}",
        f"++model.optim.sched.warmup_steps={cfg.get('warmup_steps', 2000)}",
        f"++model.optim.sched.min_lr={cfg.get('min_lr', 1e-6)}",
        f"++trainer.max_steps={cfg.get('max_steps', 30000)}",
        "++trainer.max_epochs=-1",  # step budget, not epochs (iterable Lhotse dataset)
        f"++trainer.devices={cfg.get('devices', 1)}",
        f"++trainer.precision={cfg.get('precision', 'bf16')}",
        f"++trainer.val_check_interval={cfg.get('val_check_interval', 1000)}",
        f"++exp_manager.exp_dir={exp_dir.resolve()}",
        f"++exp_manager.checkpoint_callback_params.save_top_k={cfg.get('save_top_k', 3)}",
        "++exp_manager.resume_if_exists=true",
        "++exp_manager.resume_ignore_no_checkpoint=true",
    ]
    ov += [str(x) for x in cfg.get("extra_overrides", []) or []]

    if lang:
        conf_dir, conf_name = materialize_config(
            nd, lang, scheduler=cfg.get("scheduler", "CosineAnnealing"))
    else:
        conf_dir, conf_name = (nd / CONFIG_DIR).resolve(), CONFIG_NAME

    cmd = [
        sys.executable, str(nd / FINETUNE_SCRIPT),
        f"--config-path={conf_dir}",
        f"--config-name={conf_name}",
        *ov,
    ]
    return cmd


def run(cfg: dict[str, Any], data_dir: Path, init_nemo: Path, tokenizer_dir: Path,
        lang: dict[str, Any] | None = None, dry_run: bool = False) -> int:
    cmd = build_command(cfg, data_dir, init_nemo, tokenizer_dir, lang=lang)
    console.print("[bold]NeMo command:[/bold]")
    console.print("  " + " \\\n    ".join(shlex.quote(c) for c in cmd))
    if dry_run:
        console.print("[yellow]--dry-run: not executing.[/yellow]")
        return 0
    return subprocess.call(cmd)


def latest_checkpoint(exp_dir: Path) -> Path:
    """Newest .nemo written by exp_manager."""
    cands = sorted(Path(exp_dir).rglob("*.nemo"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise RuntimeError(f"No .nemo checkpoint under {exp_dir}. Did training finish?")
    return cands[0]
