"""Give Haitian Creole its own language slot in the Nemotron 3.5 prompt vector.

The checkpoint conditions on language via a 128-dim one-hot vector fed through a
2-layer MLP (`num_prompts: 128`, `prompt_intermediate_size: 2048`). The shipped
`prompt_dictionary` only assigns indices **0-104**, so **105-127 are unused**.

So instead of overwriting an existing language, we:

1. register `ht-HT -> 105` in `prompt_dictionary`;
2. warm-start it by copying the *input column* of `fr-FR` (index 8) in the prompt
   MLP's first Linear — Haitian Creole is French-lexified Latin script, so the
   French prompt vector is the best available prior;
3. set `initialize_prompt_feature: true` — this gates whether NeMo *constructs* the
   prompt MLP at all (rnnt_bpe_models_prompt.py:106), so `false` makes loading the
   checkpoint fail with "Unexpected key(s) in state_dict: prompt_kernel.0.weight".

Everything is asserted: columns 0-104 must come out bit-identical.
"""

from __future__ import annotations

import json
import re
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console

from . import HIGHEST_USED_SLOT, NUM_PROMPTS

console = Console()

WEIGHT_NAMES = ("model_weights.ckpt", "model_weights.pt")
CONFIG_NAMES = ("model_config.yaml", "model_config.yml")


@dataclass
class NemoBundle:
    """An unpacked .nemo (it is just a tar archive)."""

    root: Path
    config_path: Path
    weights_path: Path

    @property
    def config(self) -> Any:
        from omegaconf import OmegaConf

        return OmegaConf.load(self.config_path)


def fetch_base_nemo(repo_id: str, token: str | None = None) -> Path:
    """Download the .nemo checkpoint from the Hub (cached)."""
    from huggingface_hub import hf_hub_download, list_repo_files

    files = list_repo_files(repo_id, token=token)
    nemo_files = [f for f in files if f.endswith(".nemo")]
    if not nemo_files:
        raise RuntimeError(f"No .nemo file in {repo_id}. Files: {files}")
    console.print(f"Downloading [bold]{nemo_files[0]}[/bold] from {repo_id} …")
    return Path(hf_hub_download(repo_id, nemo_files[0], token=token))


def unpack(nemo_path: Path, dest: Path | None = None) -> NemoBundle:
    dest = Path(dest or tempfile.mkdtemp(prefix="nemo-"))
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(nemo_path) as tar:
        try:
            tar.extractall(dest, filter="data")
        except TypeError:  # filter= landed in 3.10.12/3.11.4/3.12
            tar.extractall(dest)

    def find(names: tuple[str, ...], what: str) -> Path:
        for name in names:
            hits = list(dest.rglob(name))
            if hits:
                return hits[0]
        listing = sorted(p.name for p in dest.rglob("*") if p.is_file())
        raise RuntimeError(f"No {what} ({'/'.join(names)}) inside the .nemo. Found: {listing}")

    return NemoBundle(dest, find(CONFIG_NAMES, "model config"), find(WEIGHT_NAMES, "weights"))


def find_prompt_dictionary(cfg: Any) -> tuple[str, dict[str, int]]:
    """Locate the real `prompt_dictionary` definition in a config tree.

    Two traps, both hit on NeMo's training recipe:

    * Iterating a DictConfig *resolves* `${...}` interpolations, and this YAML has
      plenty that don't resolve standalone — walking it raises ConfigKeyError. So
      convert with ``resolve=False`` and search plain data.
    * `train_ds`/`validation_ds` carry
      ``prompt_dictionary: ${model.model_defaults.prompt_dictionary}`` — references,
      not definitions. Unresolved they are plain strings, so we keep only dict
      values and land on the one real definition.
    """
    from omegaconf import DictConfig, ListConfig, OmegaConf

    if isinstance(cfg, (DictConfig, ListConfig)):
        plain = OmegaConf.to_container(cfg, resolve=False)
    else:
        plain = cfg

    found: list[tuple[str, dict]] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                sub = f"{path}.{k}" if path else str(k)
                if k == "prompt_dictionary" and isinstance(v, dict) and v:
                    found.append((sub, v))
                else:
                    walk(v, sub)

    walk(plain, "")
    if not found:
        raise RuntimeError(
            "No `prompt_dictionary` definition in the config. This checkpoint may not be "
            "the prompt-conditioned variant — run `kreyol-asr inspect` and check, or fall "
            "back to `language.warm_start_from: fr-FR` with `language.tag: fr-FR`."
        )
    # Prefer the richest definition if a config somehow carries more than one.
    found.sort(key=lambda t: -len(t[1]))
    return found[0]


def resolve_prompt_projection(state: dict, prompt_dict: dict[str, int],
                              num_prompts: int = NUM_PROMPTS) -> tuple[str, int]:
    """Pick the parameter that consumes the one-hot, and where the one-hot sits.

    Candidates are validated rather than guessed: only the true projection has a
    block of columns where the never-used language slots stayed untrained. On the
    real checkpoint `prompt_kernel.0.weight` (2048, 1152) passes at offset 1024,
    while the second MLP layer `prompt_kernel.2.weight` (1024, 2048) fails.
    """
    errors: list[str] = []
    for cand in prompt_weight_candidates(state, num_prompts):
        try:
            offset = infer_prompt_offset(state[cand], prompt_dict, num_prompts)
        except RuntimeError as e:
            errors.append(f"{cand}: {e}")
            continue
        return cand, offset
    raise RuntimeError(
        "Could not identify the prompt projection. Tried:\n  " + "\n  ".join(errors) +
        "\nRun `kreyol-asr inspect`, then pass --prompt-param and --prompt-offset."
    )


def prompt_weight_candidates(state: dict, num_prompts: int = NUM_PROMPTS) -> list[str]:
    """Find the Linear that consumes the language one-hot.

    In this checkpoint it is `prompt_kernel.0.weight` with shape (2048, 1152): the
    128-dim one-hot is *concatenated* with the 1024-dim acoustic embedding, so the
    input dim is 1152, not 128. Matching on `shape[1] == 128` instead picks up 48
    `pos_bias_u/v` tensors of shape (8, 128), which are attention biases.
    """
    cands = [k for k, v in state.items()
             if hasattr(v, "ndim") and v.ndim == 2 and v.shape[1] >= num_prompts
             and "prompt" in k.lower()]
    if not cands:
        # Fall back to an exact-width match, excluding known decoys.
        cands = [k for k, v in state.items()
                 if hasattr(v, "ndim") and v.ndim == 2 and v.shape[1] == num_prompts
                 and "pos_bias" not in k]
    if not cands:
        raise RuntimeError(
            "Could not locate the prompt projection. Run `kreyol-asr inspect` to dump "
            "parameter names/shapes, then pass --prompt-param explicitly."
        )

    def layer_index(name: str) -> int:
        nums = [int(p) for p in name.split(".") if p.isdigit()]
        return nums[0] if nums else 10**6

    # Try the earliest MLP layer first — that is the one fed the raw one-hot.
    return sorted(cands, key=lambda k: (layer_index(k), k))


def find_prompt_weight(state: dict, num_prompts: int = NUM_PROMPTS) -> str:
    """First candidate by layer order (kept for callers that don't need the offset)."""
    return prompt_weight_candidates(state, num_prompts)[0]


def infer_prompt_offset(weight, prompt_dict: dict[str, int],
                        num_prompts: int = NUM_PROMPTS) -> int:
    """Locate the one-hot block inside a concatenated input.

    Decided empirically rather than assumed: slots no shipped language maps to were
    never activated during pretraining, so their columns stayed near zero. Whichever
    candidate offset shows that signature is the language block. On the real
    checkpoint the trailing block gives free/used column-norm ratio 0.086 while the
    leading (acoustic) block gives 0.99.
    """
    width = int(weight.shape[1])
    if width == num_prompts:
        return 0
    used = sorted(set(prompt_dict.values()))
    free = [i for i in range(num_prompts) if i not in set(used)]
    if not free:
        offset = width - num_prompts
        console.print(f"[yellow]No free slots to probe with; assuming offset {offset}[/yellow]")
        return offset

    scores: dict[int, float] = {}
    for offset in {width - num_prompts, 0}:
        blk = weight[:, offset:offset + num_prompts]
        nu = blk[:, used].norm(dim=0).mean().item()
        nf = blk[:, free].norm(dim=0).mean().item()
        scores[offset] = nf / nu if nu else float("inf")
    best = min(scores, key=scores.get)
    console.print("one-hot block offset probe (free/used column-norm ratio): " +
                  ", ".join(f"{o}->{r:.3f}" for o, r in sorted(scores.items())))
    if scores[best] > 0.5:
        raise RuntimeError(
            f"Cannot identify the language one-hot block in a width-{width} input: no "
            f"candidate offset shows untrained free slots (best ratio {scores[best]:.3f}). "
            f"Pass --prompt-offset explicitly after inspecting the checkpoint."
        )
    console.print(f"Language one-hot occupies columns [{best}:{best + num_prompts}]")
    return best


def patch(
    nemo_path: Path,
    out_path: Path,
    *,
    tag: str = "ht-HT",
    slot: int = 105,
    warm_start_from: str | None = "fr-FR",
    prompt_param: str | None = None,
    prompt_offset: int | None = None,
) -> dict[str, Any]:
    """Write a new .nemo with `tag` registered at `slot`, warm-started."""
    import torch
    from omegaconf import OmegaConf

    if not (HIGHEST_USED_SLOT < slot < NUM_PROMPTS):
        raise ValueError(
            f"slot {slot} is not free: indices 0-{HIGHEST_USED_SLOT} are taken by shipped "
            f"languages and the vector is only {NUM_PROMPTS} wide. Use {HIGHEST_USED_SLOT+1}"
            f"-{NUM_PROMPTS-1}."
        )

    bundle = unpack(nemo_path)
    cfg = bundle.config
    dict_path, prompt_dict = find_prompt_dictionary(cfg)
    console.print(f"prompt_dictionary at [cyan]{dict_path}[/cyan] "
                  f"({len(prompt_dict)} entries, max index {max(prompt_dict.values())})")

    if tag in prompt_dict:
        raise RuntimeError(f"{tag!r} already maps to slot {prompt_dict[tag]}; nothing to do.")
    taken = {v: k for k, v in prompt_dict.items()}
    if slot in taken:
        raise RuntimeError(f"slot {slot} is already used by {taken[slot]!r}.")

    src_slot = None
    if warm_start_from:
        if warm_start_from not in prompt_dict:
            raise RuntimeError(f"warm_start_from={warm_start_from!r} is not in prompt_dictionary.")
        src_slot = prompt_dict[warm_start_from]

    # --- config edits -------------------------------------------------------
    OmegaConf.update(cfg, f"{dict_path}.{tag}", slot, force_add=True)
    short = tag.split("-")[0]
    if short not in prompt_dict:
        OmegaConf.update(cfg, f"{dict_path}.{short}", slot, force_add=True)
    # Ensure the prompt MLP is constructed. This flag gates module creation, not
    # weight re-initialization — with it false the checkpoint's prompt_kernel.*
    # tensors have nowhere to load and both training and `bench` fail on restore.
    OmegaConf.update(cfg, "model_defaults.initialize_prompt_feature", True, force_add=True)
    OmegaConf.save(cfg, bundle.config_path)

    # --- weight warm-start --------------------------------------------------
    state = torch.load(bundle.weights_path, map_location="cpu", weights_only=False)
    inner = state.get("state_dict", state) if isinstance(state, dict) else state

    if prompt_param and prompt_offset is not None:
        param, offset = prompt_param, prompt_offset
    elif prompt_param:
        param = prompt_param
        offset = infer_prompt_offset(inner[param], prompt_dict)
    else:
        param, offset = resolve_prompt_projection(inner, prompt_dict)
        if prompt_offset is not None:
            offset = prompt_offset
    w = inner[param]
    console.print(f"prompt projection [cyan]{param}[/cyan] shape={tuple(w.shape)} "
                  f"offset={offset}")

    before = w.detach().clone()
    if src_slot is not None:
        dst_col, src_col = offset + slot, offset + src_slot
        w[:, dst_col] = w[:, src_col]

        # Assertions: the copy landed, and nothing that ships with the model moved.
        assert torch.equal(w[:, dst_col], before[:, src_col]), "warm-start copy failed"
        untouched = [c for c in range(w.shape[1]) if c != dst_col]
        assert torch.equal(w[:, untouched], before[:, untouched]), \
            "columns other than the new slot changed — aborting"

        gained = w[:, dst_col].norm().item()
        console.print(f"[green]Warm-started[/green] slot {slot} ({tag}) from slot {src_slot} "
                      f"({warm_start_from}); column norm "
                      f"{before[:, dst_col].norm().item():.3f} -> {gained:.3f}")
    else:
        console.print(f"[yellow]Slot {slot} left at its initial values (no warm start)[/yellow]")

    torch.save(state, bundle.weights_path)

    # --- repack -------------------------------------------------------------
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_path, "w") as tar:
        for item in sorted(bundle.root.iterdir()):
            tar.add(item, arcname=item.name)
    shutil.rmtree(bundle.root, ignore_errors=True)

    info = {
        "out": str(out_path),
        "tag": tag,
        "slot": slot,
        "warm_start_from": warm_start_from,
        "warm_start_slot": src_slot,
        "prompt_param": param,
        "prompt_offset": offset,
        "prompt_dict_path": dict_path,
    }
    console.print(f"[green]Wrote[/green] {out_path}")
    return info


def extract_tokenizer(nemo_path: Path, dest: Path) -> Path:
    """Pull the tokenizer out of the .nemo into a directory.

    The streaming-prompt training config declares `tokenizer.dir: ???` (mandatory),
    and we deliberately reuse the pretrained BPE — retraining it would reset the
    RNN-T joint. So point `model.tokenizer.dir` at what we extract here.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    bundle = unpack(nemo_path)
    patterns = ("*tokenizer*", "*.model", "*vocab*", "*merges*")
    copied: list[str] = []
    for pat in patterns:
        for p in bundle.root.rglob(pat):
            if not p.is_file() or p.name in CONFIG_NAMES:
                continue
            # NeMo stores artifacts hash-prefixed, e.g.
            # `427ad33c...._tokenizer.model`. `model.tokenizer.dir=` expects plain
            # `tokenizer.model` / `vocab.txt`, so strip the prefix.
            name = re.sub(r"^[0-9a-f]{32}_", "", p.name)
            shutil.copy2(p, dest / name)
            copied.append(name)
    shutil.rmtree(bundle.root, ignore_errors=True)
    if not copied:
        raise RuntimeError(f"No tokenizer files found inside {nemo_path}")
    console.print(f"Tokenizer -> {dest} ({', '.join(sorted(set(copied)))})")
    return dest


def patch_processor_config(path: Path, tag: str, slot: int) -> None:
    """Mirror the slot into processor_config.json for the HF-side artifact."""
    data = json.loads(Path(path).read_text())
    data.setdefault("prompt_dictionary", {})[tag] = slot
    data["prompt_dictionary"][tag.split("-")[0]] = slot
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False))


def inspect(nemo_path: Path) -> dict[str, Any]:
    """Dump what's actually inside the checkpoint. Run this before `patch`."""
    import torch

    bundle = unpack(nemo_path)
    cfg = bundle.config
    report: dict[str, Any] = {"files": sorted(p.name for p in bundle.root.rglob("*") if p.is_file())}

    try:
        dict_path, prompt_dict = find_prompt_dictionary(cfg)
        used = sorted(set(prompt_dict.values()))
        report["prompt_dictionary_path"] = dict_path
        report["prompt_dictionary_size"] = len(prompt_dict)
        report["used_slots"] = f"{min(used)}-{max(used)}"
        report["free_slots"] = [i for i in range(NUM_PROMPTS) if i not in set(used)]
        report["fr_FR_slot"] = prompt_dict.get("fr-FR")
    except RuntimeError as e:
        report["prompt_dictionary_error"] = str(e)

    state = torch.load(bundle.weights_path, map_location="cpu", weights_only=False)
    inner = state.get("state_dict", state) if isinstance(state, dict) else state
    report["prompt_named_params"] = {
        k: tuple(v.shape) for k, v in inner.items() if "prompt" in k.lower()
    }
    try:
        _, pd = find_prompt_dictionary(cfg)
        param, offset = resolve_prompt_projection(inner, pd)
        report["prompt_param"] = param
        report["prompt_param_shape"] = tuple(inner[param].shape)
        report["prompt_offset"] = offset
    except RuntimeError as e:
        report["prompt_param_error"] = str(e)
    report["att_context_size"] = _dig(cfg, "encoder.att_context_size")
    report["att_context_probs"] = _dig(cfg, "encoder.att_context_probs")
    report["tokenizer"] = _dig(cfg, "tokenizer")
    shutil.rmtree(bundle.root, ignore_errors=True)
    return report


def _dig(cfg: Any, dotted: str) -> Any:
    from omegaconf import OmegaConf

    node = cfg
    for part in dotted.split("."):
        if node is None:
            return None
        try:
            node = node.get(part) if hasattr(node, "get") else None
        except Exception:
            return None
    try:
        return OmegaConf.to_container(node) if node is not None and hasattr(node, "keys") else node
    except Exception:
        return node
