#!/usr/bin/env bash
# Resumable driver for the component-tagged calibration (design 2026-10-12). Run from the repository root:
#   bash configs/run_component_tagging.sh [workers]
# A (config, energy) batch whose events.npz exists is skipped. Cheap particles first; electrons last (200 GeV is the longest batch).
set -u
WORKERS="${1:-14}"
mkdir -p data/geant4_component_logs
run() {
  local name="$1"; shift
  for energy in "$@"; do
    out="data/geant4_component_${name}/train/E${energy}GeV"
    if [ -f "$out/events.npz" ]; then echo "skip ${name} ${energy}"; continue; fi
    echo "$(date -u +%FT%TZ) start ${name} ${energy} GeV"
    PYTHONPATH=src .venv/Scripts/python -m ams_ecal.geant4_simulation.geant4_backend run \
      --sample train --energy "$energy" --workers "$WORKERS" \
      --pilot-config "configs/geant4_component_${name}.yaml" \
      > "data/geant4_component_logs/${name}_${energy}.log" 2>&1
    echo "$(date -u +%FT%TZ) end ${name} ${energy} GeV exit $?"
  done
}
run muon 5 10 20 50 100 200
run proton 5 10 20 50 100 200
run pion 5 10 20 50 100 200
run proton_qbbc 10 100
run electron 5 10 20 50 100 200
echo "$(date -u +%FT%TZ) ALLDONE"
