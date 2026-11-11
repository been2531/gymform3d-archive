#!/usr/bin/env bash
# Run a pilot script inside headless Blender 4.x (system bpy, not pip).
set -euo pipefail
SCRIPT="$1"
shift
exec blender --background --python "$SCRIPT" -- "$@"
