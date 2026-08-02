# Serverless worker for Haitian Creole TTS (KaniTTS).
#
# Deliberately a SEPARATE image from the ASR worker: kani-tts pins
# nemo-toolkit[tts]==2.4.0, while the ASR model needs nemo_toolkit[asr] from
# git main (3.1.0) for its prompt-conditioned RNNT. Those cannot coexist in one
# Python environment, so they get one endpoint each.
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf \
    HF_HUB_DISABLE_TELEMETRY=1

# kani-tts requires torch>=2.8.0, which the base image already satisfies.
RUN pip install --no-cache-dir "kani-tts==1.0.1" && \
    pip install --no-cache-dir runpod soundfile

# Bake the voice model in so a cold worker does not download it per start.
ARG MODEL_REPO=jsbeaudry/haitian-kani-ht-v3
ENV MODEL_REPO=${MODEL_REPO}
RUN python -c "\
import os; from huggingface_hub import snapshot_download; \
snapshot_download(os.environ['MODEL_REPO'])"

COPY handler.py /handler.py
CMD ["python", "-u", "/handler.py"]
