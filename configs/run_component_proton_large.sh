#!/usr/bin/env bash
# Resumable driver for the larger component-tagged proton run (researcher's approval 2026-10-15). From the repository root:
#   bash configs/run_component_proton_large.sh [workers]
# A (config, energy) batch whose events.npz exists is skipped. Cheapest energy first.
set -u
WORKERS="${1:-14}"
mkdir -p data/geant4_component_logs
for energy in 10 20 50 100; do
  out="data/geant4_component_proton_large/train/E${energy}GeV"
  if [ -f "$out/events.npz" ]; then echo "skip proton_large ${energy}"; continue; fi
  echo "$(date -u +%FT%TZ) start proton_large ${energy} GeV"
  PYTHONPATH=src .venv/Scripts/python -m ams_ecal.geant4_simulation.geant4_backend run \
    --sample train --energy "$energy" --workers "$WORKERS" \
    --pilot-config configs/geant4_component_proton_large.yaml \
    > "data/geant4_component_logs/proton_large_${energy}.log" 2>&1
  echo "$(date -u +%FT%TZ) end proton_large ${energy} GeV exit $?"
done
echo "$(date -u +%FT%TZ) ALLDONE proton_large"
