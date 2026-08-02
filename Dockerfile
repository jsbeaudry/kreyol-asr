# Training image for RunPod / Lambda / Vast.
# The NeMo container already ships CUDA, PyTorch, NeMo and — importantly — the
# examples/ tree, which the pip package does not include.
ARG NEMO_IMAGE=nvcr.io/nvidia/nemo:26.06
FROM ${NEMO_IMAGE}

ENV NEMO_DIR=/opt/NeMo \
    HF_HOME=/workspace/.hf \
    PYTHONUNBUFFERED=1

WORKDIR /workspace/kreyol-asr

# Fail loudly at build time if this image doesn't have what we need, rather than
# 20 minutes into a training run.
RUN python -c "import nemo; print('NeMo', nemo.__version__)" && \
    test -f ${NEMO_DIR}/examples/asr/speech_to_text_finetune.py \
      || (echo "ERROR: ${NEMO_DIR}/examples/asr/speech_to_text_finetune.py missing. \
Clone NeMo and set NEMO_DIR, or pick a different --build-arg NEMO_IMAGE." && exit 1)

RUN apt-get update && apt-get install -y --no-install-recommends \
        sox libsndfile1 ffmpeg jq tmux && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e .

COPY configs ./configs
COPY scripts ./scripts
COPY Makefile ./
RUN chmod +x scripts/*.sh

CMD ["bash"]
