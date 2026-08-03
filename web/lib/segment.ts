/**
 * Split text into pieces that each fit one TTS request.
 *
 * The worker takes 120 characters per call, so anything longer is cut here,
 * synthesised one segment at a time and joined back into a single clip.
 *
 * Port of the splitter in space/app.py — keep the two in step.
 */

export const MAX_CHARS = 120; // the worker's per-request limit
export const MAX_TOTAL_CHARS = 5000; // guard on one submission

const SENTENCE_END = /(?<=[.!?…])\s+/;
const CLAUSE_END = /(?<=[,;:])\s+/;

/** Last resort for a clause with no punctuation left to break on. */
function hardWrap(unit: string, limit: number): string[] {
  const out: string[] = [];
  let line = "";
  for (const word of unit.split(" ")) {
    if (word.length > limit) {
      // one unsplittable token
      if (line) {
        out.push(line);
        line = "";
      }
      for (let i = 0; i < word.length; i += limit) out.push(word.slice(i, i + limit));
    } else if (!line) {
      line = word;
    } else if (line.length + 1 + word.length <= limit) {
      line = `${line} ${word}`;
    } else {
      out.push(line);
      line = word;
    }
  }
  if (line) out.push(line);
  return out;
}

/**
 * A single sentence longer than the worker allows. Break at clause punctuation
 * first so the cut lands somewhere a speaker would already pause.
 */
function breakDown(unit: string, limit: number): string[] {
  if (unit.length <= limit) return [unit];
  const out: string[] = [];
  let buf = "";
  for (const part of unit.split(CLAUSE_END)) {
    if (part.length > limit) {
      if (buf) {
        out.push(buf);
        buf = "";
      }
      out.push(...hardWrap(part, limit));
    } else if (!buf) {
      buf = part;
    } else if (buf.length + 1 + part.length <= limit) {
      buf = `${buf} ${part}`;
    } else {
      out.push(buf);
      buf = part;
    }
  }
  if (buf) out.push(buf);
  return out;
}

/**
 * Sentence boundaries first, since that is where a natural pause already is.
 * Consecutive sentences are then packed back together up to the limit, because
 * every segment is a separate round trip to a GPU worker.
 */
export function splitForTts(text: string, limit: number = MAX_CHARS): string[] {
  const normalised = (text ?? "").split(/\s+/).filter(Boolean).join(" ");
  if (!normalised) return [];

  const pieces: string[] = [];
  for (const sentence of normalised.split(SENTENCE_END)) {
    const trimmed = sentence.trim();
    if (trimmed) pieces.push(...breakDown(trimmed, limit));
  }

  const packed: string[] = [];
  for (const piece of pieces) {
    const last = packed[packed.length - 1];
    if (last !== undefined && last.length + 1 + piece.length <= limit) {
      packed[packed.length - 1] = `${last} ${piece}`;
    } else {
      packed.push(piece);
    }
  }
  return packed;
}
