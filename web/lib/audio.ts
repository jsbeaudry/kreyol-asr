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

/**
 * Parse a PCM WAV ourselves.
 *
 * decodeAudioData resamples to the *device* rate — 44.1 kHz on most machines —
 * which mislabels the worker's 22.05 kHz output and, worse, overshoots on
 * interpolation: a clip that peaked at 0.98 comes back at 1.002 and clips when
 * re-encoded. The worker always sends PCM WAV, so read it exactly.
 *
 * Returns null for anything that is not a PCM/float WAV, so callers can fall
 * back to the browser decoder.
 */
function parseWav(
  bytes: Uint8Array,
): { samples: Float32Array; sampleRate: number } | null {
  if (bytes.length < 44) return null;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const tag = (o: number) =>
    String.fromCharCode(view.getUint8(o), view.getUint8(o + 1), view.getUint8(o + 2), view.getUint8(o + 3));
  if (tag(0) !== "RIFF" || tag(8) !== "WAVE") return null;

  let format = 0;
  let channels = 0;
  let sampleRate = 0;
  let bits = 0;
  let dataOffset = -1;
  let dataLength = 0;

  // Walk the chunks: LIST/fact can sit between "fmt " and "data".
  let offset = 12;
  while (offset + 8 <= bytes.length) {
    const id = tag(offset);
    const size = view.getUint32(offset + 4, true);
    const body = offset + 8;
    if (id === "fmt ") {
      format = view.getUint16(body, true);
      channels = view.getUint16(body + 2, true);
      sampleRate = view.getUint32(body + 4, true);
      bits = view.getUint16(body + 14, true);
    } else if (id === "data") {
      dataOffset = body;
      dataLength = Math.min(size, bytes.length - body);
    }
    offset = body + size + (size % 2); // chunks are word-aligned
  }
  if (dataOffset < 0 || !sampleRate || !channels) return null;

  let interleaved: Float32Array;
  if (format === 1 && bits === 16) {
    const count = Math.floor(dataLength / 2);
    interleaved = new Float32Array(count);
    for (let i = 0; i < count; i += 1) {
      interleaved[i] = view.getInt16(dataOffset + i * 2, true) / 0x8000;
    }
  } else if (format === 3 && bits === 32) {
    const count = Math.floor(dataLength / 4);
    interleaved = new Float32Array(count);
    for (let i = 0; i < count; i += 1) {
      interleaved[i] = view.getFloat32(dataOffset + i * 4, true);
    }
  } else {
    return null; // compressed or exotic: let the browser handle it
  }

  if (channels === 1) return { samples: interleaved, sampleRate };
  const frames = Math.floor(interleaved.length / channels);
  const mono = new Float32Array(frames);
  for (let f = 0; f < frames; f += 1) {
    let sum = 0;
    for (let c = 0; c < channels; c += 1) sum += interleaved[f * channels + c]!;
    mono[f] = sum / channels;
  }
  return { samples: mono, sampleRate };
}

/** Mono float samples from any container the browser can decode. */
export async function decodeToMono(
  bytes: Uint8Array,
): Promise<{ samples: Float32Array; sampleRate: number }> {
  const parsed = parseWav(bytes);
  if (parsed) return parsed;
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

function rms(x: Float32Array): number {
  if (x.length === 0) return 0;
  let sum = 0;
  for (let i = 0; i < x.length; i += 1) sum += x[i]! * x[i]!;
  return Math.sqrt(sum / x.length);
}

/**
 * Drop near-silent head and tail.
 *
 * Each segment is generated independently and comes with its own leading and
 * trailing padding, so joining them raw produces pauses of uneven length that
 * read as a stumble between sentences.
 */
export function trimSilence(x: Float32Array, sampleRate: number): Float32Array {
  const level = rms(x);
  if (level === 0) return x;
  const floor = Math.max(level * 0.05, 1e-4);
  const win = Math.max(1, Math.round(0.005 * sampleRate)); // 5 ms
  const loud = (at: number) => {
    let sum = 0;
    const end = Math.min(at + win, x.length);
    for (let i = at; i < end; i += 1) sum += Math.abs(x[i]!);
    return sum / Math.max(1, end - at) > floor;
  };

  let start = 0;
  while (start < x.length && !loud(start)) start += win;
  let stop = x.length;
  while (stop > start && !loud(Math.max(0, stop - win))) stop -= win;
  if (stop <= start) return x; // all quiet — leave it alone

  // Keep a little padding so consonants are not clipped off.
  const pad = Math.round(0.02 * sampleRate);
  return x.slice(Math.max(0, start - pad), Math.min(x.length, stop + pad));
}

/**
 * Match each segment's level to the median of the set.
 *
 * Independent generations vary in loudness — measured at ~21% RMS difference
 * (about 1.7 dB) between two segments of one passage, which is plainly audible
 * as the voice "changing" mid-clip. Gain is capped so a genuinely quiet or
 * near-empty segment cannot be pumped up into noise.
 */
export function matchLevels(chunks: Float32Array[]): Float32Array[] {
  const levels = chunks.map(rms).filter((v) => v > 1e-5);
  if (levels.length < 2) return chunks;
  const sorted = [...levels].sort((a, b) => a - b);
  const mid = sorted.length % 2
    ? sorted[(sorted.length - 1) / 2]!
    : (sorted[sorted.length / 2 - 1]! + sorted[sorted.length / 2]!) / 2;

  return chunks.map((chunk) => {
    const level = rms(chunk);
    if (level < 1e-5) return chunk;
    const gain = Math.min(Math.max(mid / level, 0.5), 2.0);
    if (Math.abs(gain - 1) < 0.02) return chunk; // not worth touching
    const out = new Float32Array(chunk.length);
    for (let i = 0; i < chunk.length; i += 1) out[i] = chunk[i]! * gain;
    return out;
  });
}

/**
 * Join TTS segments: trim each, match levels, then butt them together with a
 * short silence and a few milliseconds of fade so the seam has no click.
 */
export function joinSegments(chunks: Float32Array[], sampleRate: number): Float32Array {
  if (chunks.length === 0) return new Float32Array(0);
  if (chunks.length === 1) return chunks[0]!;

  const prepared = matchLevels(chunks.map((c) => trimSilence(c, sampleRate)));
  const gap = Math.round(SEGMENT_GAP_S * sampleRate);
  const fade = Math.min(Math.round(0.006 * sampleRate), ...prepared.map((c) => c.length >> 1));

  const total =
    prepared.reduce((n, c) => n + c.length, 0) + gap * (prepared.length - 1);
  const out = new Float32Array(total);
  let offset = 0;
  prepared.forEach((chunk, i) => {
    if (i > 0) offset += gap; // already zeroed
    out.set(chunk, offset);
    // Ramp the seam edges only; the very start and end are left untouched.
    if (fade > 0) {
      if (i > 0) {
        for (let k = 0; k < fade; k += 1) out[offset + k]! *= k / fade;
      }
      if (i < prepared.length - 1) {
        const tail = offset + chunk.length - fade;
        for (let k = 0; k < fade; k += 1) out[tail + k]! *= 1 - k / fade;
      }
    }
    offset += chunk.length;
  });

  // Trimming and level matching can push the sum past full scale.
  const peak = peakOf(out);
  if (peak > 0.99) {
    const scale = 0.99 / peak;
    for (let i = 0; i < out.length; i += 1) out[i]! *= scale;
  }
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
