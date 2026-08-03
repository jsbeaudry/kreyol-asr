#!/usr/bin/env bash
# Radio Haiti-Inter (Zenodo 10.5281/zenodo.17818122, CC-BY-4.0) -> extracted tree.
#
#   bash scripts/fetch_radio_haiti.sh
#   ONLY=transcriptions,eaf bash scripts/fetch_radio_haiti.sh   # text only, ~140 MB
#
# ~6.2 GB down, ~8.6 GB extracted. Provision 60 GB free on the pod: `radio ingest`
# writes sliced clips, and `prepare` then writes its own copy of every one of them
# under data/ht/wav/, so the segment audio lives on disk twice.
#
# Not for the Mac. The audio alone is 6 GB.
set -euo pipefail

RADIO_DIR="${RADIO_DIR:-/workspace/corpora/radio-haiti}"
RECORD="${RECORD:-17818122}"
ONLY="${ONLY:-recordings,eaf,transcriptions}"
KEEP_ZIPS="${KEEP_ZIPS:-0}"

ZIP_DIR="$RADIO_DIR/zips"
RAW_DIR="$RADIO_DIR/raw"
mkdir -p "$ZIP_DIR" "$RAW_DIR"

# Pinned from the Zenodo API. A truncated 6 GB download that unzips partially would
# look like a smaller corpus rather than a failure, and we would train on it.
md5_of() {
  case "$1" in
    recordings.zip)     echo "0d5e520e06574651d75261210f24cc22" ;;
    eaf.zip)            echo "38a5de184ab358dff2140ddbb5ef71ea" ;;
    transcriptions.zip) echo "d8083623b47286486dca204a4266f89e" ;;
  esac
}

# macOS ships `md5`, Linux `md5sum`. The pod is Linux; handle both so the script is
# testable wherever it is read.
md5_file() {
  if command -v md5sum >/dev/null 2>&1; then md5sum "$1" | awk '{print $1}'
  else md5 -q "$1"; fi
}

echo "==> Zenodo record $RECORD -> $RADIO_DIR"
for name in ${ONLY//,/ }; do
  zip="$name.zip"
  want="$(md5_of "$zip")"
  if [ -z "$want" ]; then
    echo "unknown component '$name' (expected: recordings, eaf, transcriptions)" >&2
    exit 2
  fi

  if [ -d "$RAW_DIR/$name" ] && [ -n "$(ls -A "$RAW_DIR/$name" 2>/dev/null)" ]; then
    echo "  $name/ already extracted — skipping"
    continue
  fi

  echo "  downloading $zip"
  # -C - resumes a partial file, so an interrupted 6 GB pull is not restarted.
  curl -fL -C - --retry 3 --retry-delay 5 \
    -o "$ZIP_DIR/$zip" \
    "https://zenodo.org/api/records/$RECORD/files/$zip/content"

  echo -n "  verifying $zip ... "
  got="$(md5_file "$ZIP_DIR/$zip")"
  if [ "$got" != "$want" ]; then
    echo "FAILED"
    echo "    expected $want" >&2
    echo "    got      $got" >&2
    echo "    Delete $ZIP_DIR/$zip and re-run; a resumed download can carry over a" >&2
    echo "    corrupt prefix that -C - will not notice." >&2
    exit 1
  fi
  echo "ok"

  echo "  extracting $zip"
  unzip -q -o "$ZIP_DIR/$zip" -d "$RAW_DIR"
  [ "$KEEP_ZIPS" = "1" ] || rm -f "$ZIP_DIR/$zip"
done

echo "==> Contents"
for d in recordings eaf transcriptions; do
  [ -d "$RAW_DIR/$d" ] || continue
  printf '  %-16s %s files\n' "$d/" "$(find "$RAW_DIR/$d" -type f | wc -l | tr -d ' ')"
done
du -sh "$RAW_DIR" 2>/dev/null || true

cat <<EOF

Next:
  kreyol-asr radio inspect --root $RAW_DIR --compare-manifest data/ht/manifests/train.json

Read inspect_report.md before ingesting. The transcripts in this corpus are machine
generated (Havard et al., Interspeech 2025); nothing downstream trusts them by default.
CC-BY-4.0 — attribution is required in the model card, see src/kreyol_asr/publish.py.
EOF
