/**
 * Browser-side audio plumbing.
 *
 * TTS comes back as base64 WAV per segment and is joined here, so a long
 * passage becomes one clip. ASR goes the other way: whatever the browser
 * recorded is converted to the 16 kHz mono PCM16 the worker expects, rather
 * than trusting it to decode a WebM/Opus blob.
 */

export const ASR_SAMPLE_RATE = 16000;
export const SEGMENT_GAP_S = 0.15; // silence between joined segments

export function base64ToBytes(b64: string): Uint8Array {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

export function bytesToBase64(bytes: Uint8Array): string {
  // Chunked: String.fromCharCode(...bytes) blows the argument limit on
  // anything longer than a second or two of audio.
  let binary = "";
  const CHUNK = 0x8000;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    binary += String.fromCharCode(...bytes.subarray(i, i + CHUNK));
  }
  return btoa(binary);
}

/** Mono float samples from any container the browser can decode. */
export async function decodeToMono(
  bytes: Uint8Array,
): Promise<{ samples: Float32Array; sampleRate: number }> {
  const ctx = new AudioContext();
  try {
    const copy = new ArrayBuffer(bytes.byteLength);
    new Uint8Array(copy).set(bytes);
    const buffer = await ctx.decodeAudioData(copy);
    return { samples: mixToMono(buffer), sampleRate: buffer.sampleRate };
  } finally {
    void ctx.close();
  }
}

function mixToMono(buffer: AudioBuffer): Float32Array {
  if (buffer.numberOfChannels === 1) return buffer.getChannelData(0).slice();
  const out = new Float32Array(buffer.length);
  for (let c = 0; c < buffer.numberOfChannels; c += 1) {
    const data = buffer.getChannelData(c);
    for (let i = 0; i < data.length; i += 1) out[i] += data[i]! / buffer.numberOfChannels;
  }
  return out;
}

/** Resample with an OfflineAudioContext — no hand-written interpolation. */
export async function resample(
  samples: Float32Array,
  from: number,
  to: number,
): Promise<Float32Array> {
  if (from === to) return samples;
  const frames = Math.max(1, Math.round((samples.length * to) / from));
  const offline = new OfflineAudioContext(1, frames, to);
  const source = offline.createBufferSource();
  const buffer = offline.createBuffer(1, samples.length, from);
  // copyToChannel wants a Float32Array over a plain ArrayBuffer; the decoded
  // samples may be typed as ArrayBufferLike, which no longer matches.
  const backing = new Float32Array(samples.length);
  backing.set(samples);
  buffer.copyToChannel(backing, 0);
  source.buffer = buffer;
  source.connect(offline.destination);
  source.start();
  return (await offline.startRendering()).getChannelData(0).slice();
}

/** Join TTS segments with a short silence so sentences breathe. */
export function joinSegments(chunks: Float32Array[], sampleRate: number): Float32Array {
  if (chunks.length === 0) return new Float32Array(0);
  if (chunks.length === 1) return chunks[0]!;
  const gap = Math.round(SEGMENT_GAP_S * sampleRate);
  const total =
    chunks.reduce((n, c) => n + c.length, 0) + gap * (chunks.length - 1);
  const out = new Float32Array(total);
  let offset = 0;
  chunks.forEach((chunk, i) => {
    if (i > 0) offset += gap; // the gap is already zeroed
    out.set(chunk, offset);
    offset += chunk.length;
  });
  return out;
}

/** 16-bit PCM WAV. */
export function encodeWav(samples: Float32Array, sampleRate: number): Uint8Array {
  const bytes = new Uint8Array(44 + samples.length * 2);
  const view = new DataView(bytes.buffer);
  const ascii = (offset: number, text: string) => {
    for (let i = 0; i < text.length; i += 1) view.setUint8(offset + i, text.charCodeAt(i));
  };

  ascii(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  ascii(8, "WAVE");
  ascii(12, "fmt ");
  view.setUint32(16, 16, true); // PCM chunk size
  view.setUint16(20, 1, true); // format: PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate
  view.setUint16(32, 2, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  ascii(36, "data");
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i += 1) {
    const clamped = Math.max(-1, Math.min(1, samples[i]!));
    view.setInt16(offset, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true);
    offset += 2;
  }
  return bytes;
}

/** Whatever the browser captured -> base64 16 kHz mono WAV for the worker. */
export async function toAsrBase64(blob: Blob): Promise<{ base64: string; seconds: number }> {
  const bytes = new Uint8Array(await blob.arrayBuffer());
  const { samples, sampleRate } = await decodeToMono(bytes);
  const resampled = await resample(samples, sampleRate, ASR_SAMPLE_RATE);
  return {
    base64: bytesToBase64(encodeWav(resampled, ASR_SAMPLE_RATE)),
    seconds: resampled.length / ASR_SAMPLE_RATE,
  };
}

export function wavBlobUrl(samples: Float32Array, sampleRate: number): string {
  const wav = encodeWav(samples, sampleRate);
  return URL.createObjectURL(new Blob([wav as BlobPart], { type: "audio/wav" }));
}

export function peakOf(samples: Float32Array): number {
  let peak = 0;
  for (let i = 0; i < samples.length; i += 1) {
    const v = Math.abs(samples[i]!);
    if (v > peak) peak = v;
  }
  return peak;
}

export function formatTime(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) seconds = 0;
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}
