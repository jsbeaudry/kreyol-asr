"""Fine-tune Kokoro-82M (StyleTTS 2 + iSTFTNet) for Haitian Creole.

Separate from `kreyol_asr` because the dependency sets are disjoint: TTS needs
torch/librosa/misaki, ASR needs NeMo. Shared Hub and audio helpers currently come
from `kreyol_asr.datasets`; Phase 1 extracts them into `kreyol_common`.
"""

# Kokoro synthesizes at 24 kHz. The iSTFTNet stack is upsample_rates [10, 6] x
# gen_istft_hop_size 5 = 300x, exactly StyleTTS 2's hop_length at sr=24000 — that
# equality is what lines 80 fps mel up with the decoder. Changing it invalidates
# every pretrained decoder filter, i.e. the whole point of warm-starting.
SAMPLE_RATE = 24000

KOKORO_REPO = "hexgrad/Kokoro-82M"
KOKORO_CKPT = "kokoro-v1_0.pth"

# Bandwidth tiers. A 24 kHz container can still hold 8 kHz of content (something
# upstream upsampled it); a vocoder trained on that learns a spectral cliff, and
# resampling cannot undo it. So bandwidth is measured, not inferred from the header.
#   A: Stage 1 + Stage 2
#   B: Stage 1; Stage 2 only when the speaker has no Tier A audio
#   C: Stage 1 only, reduced weight — never a Stage 2 reference
#   D: natively narrowband; Stage 1 only
TIER_A_MIN_E8K = 5.0e-3
TIER_B_MIN_E8K = 5.0e-4
MIN_NATIVE_SR = 22050

__all__ = ["SAMPLE_RATE", "KOKORO_REPO", "KOKORO_CKPT",
           "TIER_A_MIN_E8K", "TIER_B_MIN_E8K", "MIN_NATIVE_SR"]
