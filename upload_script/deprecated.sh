#!/usr/bin/env bash
set -euo pipefail

# ---------------- CONFIG ----------------
REMOTE="crg"
REMOTE_BASE_PATH="DATA"
SOURCE_DIR="./"
# ----------------------------------------

# ---------------- ARG CHECK ----------------
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <game letter A/B/C> <person name> [--dry-run]"
  exit 1
fi

GAME_LETTER="$(echo "$1" | tr '[:lower:]' '[:upper:]')"
PERSON="$2"
DRY_RUN="${3:-}"

# ---------------- GAME MAPPING ----------------
case "$GAME_LETTER" in
  A) GAME_FOLDER="Game A"; GAME_TAG="game1" ;;
  B) GAME_FOLDER="Game B"; GAME_TAG="game2" ;;
  C) GAME_FOLDER="Game C"; GAME_TAG="game3" ;;
  *)
    echo "Invalid game letter. Use A, B, or C."
    exit 1
    ;;
esac

DEST_BASE="${REMOTE}:${REMOTE_BASE_PATH}/${GAME_FOLDER}"

echo "Routing files for:"
echo "  Game folder: ${GAME_FOLDER}"
echo "  Filename match: ${GAME_TAG}"
echo "  Name suffix: (${PERSON})"
[[ "$DRY_RUN" == "--dry-run" ]] && echo "  Mode: DRY RUN"
echo

# ---------------- CREATE DEST FOLDERS ----------------
rclone mkdir "${DEST_BASE}/Audio"
rclone mkdir "${DEST_BASE}/Eye Tracking"
rclone mkdir "${DEST_BASE}/Game States"

# ---------------- GLOB FILES ----------------
shopt -s nullglob

AUDIO_FILES=("${SOURCE_DIR}"/*"${GAME_TAG}"*.webm)
EYE_JSON_FILES=("${SOURCE_DIR}"/*"${GAME_TAG}"*eye*.json)
STATE_JSON_FILES=("${SOURCE_DIR}"/*"${GAME_TAG}"*state*.json)

# ---------------- COPY AUDIO ----------------
for f in "${AUDIO_FILES[@]}"; do
  base=$(basename "$f")
  ext="${base##*.}"
  name="${base%.*}"
  dest="${DEST_BASE}/Audio/${name}(${PERSON}).${ext}"   # no space before (
  [[ "$DRY_RUN" == "--dry-run" ]] \
    && echo "[DRY-RUN] $f -> $dest" \
    || rclone copyto "$f" "$dest" -v
done

# ---------------- COPY EYE-TRACKING JSON ----------------
for f in "${EYE_JSON_FILES[@]}"; do
  base=$(basename "$f")
  ext="${base##*.}"
  name="${base%.*}"
  dest="${DEST_BASE}/Eye Tracking/${name}(${PERSON}).${ext}"  # no space
  [[ "$DRY_RUN" == "--dry-run" ]] \
    && echo "[DRY-RUN] $f -> $dest" \
    || rclone copyto "$f" "$dest" -v
done

# ---------------- COPY GAME STATE JSON ----------------
for f in "${STATE_JSON_FILES[@]}"; do
  base=$(basename "$f")
  ext="${base##*.}"
  name="${base%.*}"
  dest="${DEST_BASE}/Game States/${name}(${PERSON}).${ext}"  # no space
  [[ "$DRY_RUN" == "--dry-run" ]] \
    && echo "[DRY-RUN] $f -> $dest" \
    || rclone copyto "$f" "$dest" -v
done

echo "Done."

