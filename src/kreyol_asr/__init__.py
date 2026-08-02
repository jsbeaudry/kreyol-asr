"""Fine-tune NVIDIA Nemotron 3.5 streaming ASR for Haitian Creole."""

__version__ = "0.1.0"

BASE_MODEL = "nvidia/nemotron-3.5-asr-streaming-0.6b"

# From the checkpoint's config.json / processor_config.json.
NUM_PROMPTS = 128  # size of the one-hot language vector
HIGHEST_USED_SLOT = 104  # prompt_dictionary uses 0..104 -> 105..127 are free
SAMPLE_RATE = 16000
VOCAB_SIZE = 13088

# config.json encoder.sliding_window = 57 => 56 frames of left context.
# The stock NeMo streaming-prompt YAML ships [70, 6], which does not match.
LEFT_CONTEXT = 56
# right-context frames -> emitted chunk latency
LATENCY_MS = {0: 80, 3: 320, 6: 560, 13: 1120}
