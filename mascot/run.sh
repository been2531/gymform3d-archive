#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BLENDER="${BLENDER:-/home/box/opt/blender-4.2.3-linux-x64/blender}"
OUT="${OUT:-$ROOT/output/mascot/2026-11-17}"
MODE="${1:-stills}"
cd "$ROOT"
nice -n 19 "$BLENDER" -b -P mascot/build_mascot.py -- --out "$OUT" --mode "$MODE"
if [[ "$MODE" == "preview" || "$MODE" == "stills" || "$MODE" == "all" ]]; then
  python3 mascot/compose.py --out "$OUT" --mode stills
fi
if [[ "$MODE" == "reel" || "$MODE" == "all" ]]; then
  python3 mascot/compose.py --out "$OUT" --mode reel --fps 24
fi
