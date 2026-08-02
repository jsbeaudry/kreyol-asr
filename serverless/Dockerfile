# Serverless worker for Haitian Creole streaming ASR.
#
# Base image is the same one the model was trained under: CUDA 12.8 / torch 2.8.
# NOT nvcr.io/nvidia/nemo — that container requires NVIDIA driver >= 595.58 and
# crash-loops on RunPod hosts, which run 570.x.
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf \
    HF_HUB_DISABLE_TELEMETRY=1

RUN pip install --no-cache-dir Cython packaging && \
    # The model card's `git+...#egg=nemo_toolkit[asr]` is rejected by pip >= 25
    # ("invalid-egg-fragment"); this is the PEP 508 spelling.
    pip install --no-cache-dir "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@main" && \
    pip install --no-cache-dir "transformers>=5.13.0" && \
    # NeMo's RNNT CUDA kernels predate numba 0.62's numba-cuda split. A fresh
    # install pulls 0.66 and every decode dies with
    #   TypeError: Signature mismatch: 2 argument types given, but function takes 1
    pip install --no-cache-dir "numba==0.61.2" && \
    # Explicit, not transitive: the base image sets
    # HF_HUB_ENABLE_HF_TRANSFER=1 and huggingface_hub hard-errors without it.
    pip install --no-cache-dir hf_transfer && \
    pip install --no-cache-dir runpod soundfile soxr

# Bake the checkpoint into the image. A cold worker otherwise spends 30-60 s
# pulling 2.5 GB from the Hub before it can serve its first request.
ARG MODEL_REPO=jsbeaudry/nemotron-3.5-asr-streaming-0.6b-ht
ARG MODEL_FILE=nemotron-3.5-asr-streaming-0.6b-ht.nemo
ARG HF_TOKEN=
ENV MODEL_REPO=${MODEL_REPO} MODEL_FILE=${MODEL_FILE}
RUN python -c "\
import os; from huggingface_hub import hf_hub_download; \
hf_hub_download(os.environ['MODEL_REPO'], os.environ['MODEL_FILE'], token=os.environ.get('HF_TOKEN') or None)"

COPY handler.py /handler.py
CMD ["python", "-u", "/handler.py"]
