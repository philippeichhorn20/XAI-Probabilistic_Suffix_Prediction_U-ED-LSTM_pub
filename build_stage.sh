#!/usr/bin/env bash
# Builds a minimum payload for running
# src/notebooks/training_variational_dropout/BPIC17/remote_training.py
# on a remote server.
#
# The stage at $STAGE is transport-only — never edit files there.
# Iterate in the source repo, re-run this script to refresh the stage,
# then rsync the stage to the server.
set -euo pipefail

SRC="$( cd "$(dirname "$0")" && pwd )"
STAGE="${STAGE:-$HOME/IdeaProjects/thesis-server-upload}"

mkdir -p "$STAGE"

# --- Python packages (mirror with --delete so renames/removals propagate) ---
for pkg in model loss trainer event_log_loader; do
    rsync -av --delete \
        --exclude='__pycache__/' \
        --include='*/' --include='*.py' --exclude='*' \
        "$SRC/src/$pkg/" "$STAGE/src/$pkg/"
done

# --- Entry point ---
mkdir -p "$STAGE/src/notebooks/training_variational_dropout/BPIC17"
rsync -av \
    "$SRC/src/notebooks/training_variational_dropout/BPIC17/remote_training.py" \
    "$STAGE/src/notebooks/training_variational_dropout/BPIC17/remote_training.py"

# --- Encoded data (large; no --delete so unrelated pkls in stage survive) ---
mkdir -p "$STAGE/encoded_data"
rsync -av --progress \
    "$SRC/encoded_data/BPIC_2017_all_5_train.pkl" \
    "$SRC/encoded_data/BPIC_2017_all_5_val.pkl" \
    "$STAGE/encoded_data/"

# --- Environment ---
rsync -av "$SRC/Pipfile" "$SRC/Pipfile.lock" "$STAGE/"

echo
echo "Stage built at $STAGE"
du -sh "$STAGE"
