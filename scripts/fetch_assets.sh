#!/usr/bin/env bash
# Fetch the Franka Emika Panda model from MuJoCo Menagerie (Apache-2.0) into assets/menagerie.
set -euo pipefail
cd "$(dirname "$0")/.."
DEST=assets/menagerie
if [ -d "$DEST/franka_emika_panda" ]; then echo "already fetched"; exit 0; fi
mkdir -p assets
git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie "$DEST"
git -C "$DEST" sparse-checkout set franka_emika_panda
