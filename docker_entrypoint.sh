#!/bin/bash
# CUA Agent entrypoint — starts virtual desktop then runs agent
# Matches Vision Claude architecture: Xvfb → Openbox → Chromium → xdotool

set -e

DISPLAY=:99
SCREEN_WIDTH=${SCREEN_WIDTH:-1280}
SCREEN_HEIGHT=${SCREEN_HEIGHT:-720}
BRAIN_URL=${BRAIN_URL:-http://localhost:8080/v1}
BRAIN_TYPE=${BRAIN_TYPE:-evocua}

echo "╔═══════════════════════════════════════════════╗"
echo "║  Mantis CUA Agent — Starting                  ║"
echo "╠═══════════════════════════════════════════════╣"
echo "║  Display:  ${DISPLAY} (${SCREEN_WIDTH}x${SCREEN_HEIGHT})"
echo "║  Brain:    ${BRAIN_TYPE} @ ${BRAIN_URL}"
echo "║  Browser:  chromium"
echo "╚═══════════════════════════════════════════════╝"

# 1. Start Xvfb (virtual display)
echo "Starting Xvfb..."
Xvfb ${DISPLAY} -screen 0 ${SCREEN_WIDTH}x${SCREEN_HEIGHT}x24 -ac -nolisten tcp &
sleep 1
export DISPLAY=${DISPLAY}

# 2. Start Openbox (window manager — handles maximize, focus, etc.)
echo "Starting Openbox..."
openbox &
sleep 1

# 3. Run the CUA agent
echo "Starting agent..."
python run_local.py \
  --brain-url "${BRAIN_URL}" \
  --brain-type "${BRAIN_TYPE}" \
  --env-type xdotool \
  --human-speed \
  --output /app/results/run.json \
  "$@"

echo "Agent finished. Results: /app/results/run.json"
