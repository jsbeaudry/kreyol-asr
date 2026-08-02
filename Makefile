.PHONY: help install install-gpu prepare inspect patch train bench push all smoke test docker clean

DATA_CFG    ?= configs/datasets.ht.yaml
FT_CFG      ?= configs/finetune.ht.yaml
DATA_DIR    ?= data/ht
EXP_DIR     ?= exp
PATCHED     ?= checkpoints/nemotron-3.5-asr-ht-init.nemo
NEMO_IMAGE  ?= nvcr.io/nvidia/nemo:26.06

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:      ## Local (macOS ok): data prep, inspect, scoring, publish
	uv pip install -e ".[patch,dev]"

install-gpu:  ## On the GPU pod, inside the NeMo container
	pip install -e ".[dev]"

prepare:      ## HF dataset URLs -> wavs + NeMo manifests + data_report.md
	kreyol-asr prepare --config $(DATA_CFG)

inspect:      ## Print .nemo internals (prompt_dictionary, prompt-MLP params)
	kreyol-asr inspect --config $(FT_CFG)

patch:        ## Register ht-HT at slot 105, warm-started from fr-FR
	kreyol-asr patch --config $(FT_CFG) --out $(PATCHED)

train:        ## Fine-tune (GPU pod only)
	kreyol-asr train --config $(FT_CFG) --data-dir $(DATA_DIR) --init $(PATCHED)

bench:        ## Baseline vs fine-tuned WER/CER across all 4 streaming latencies
	kreyol-asr bench --config $(FT_CFG) --data-dir $(DATA_DIR) --exp-dir $(EXP_DIR)

push:         ## Upload model + benchmark-backed model card to the Hub
	kreyol-asr push --config $(FT_CFG) --exp-dir $(EXP_DIR)

all: prepare patch train bench push  ## Full pipeline

smoke:        ## ~10min end-to-end proof on the pod before committing to a full run
	./scripts/smoke.sh

test:
	pytest -q

docker:
	docker build --build-arg NEMO_IMAGE=$(NEMO_IMAGE) -t kreyol-asr:latest .

clean:
	rm -rf $(DATA_DIR) $(EXP_DIR) checkpoints benchmarks
