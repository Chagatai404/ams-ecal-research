#!/usr/bin/env bash
# Resumable driver for the EXP-015 realistic-regime pilot (approved 2026-11-09). From the repository root:
#   bash configs/run_realistic_regime_pilot.sh [workers] [rung]
# rung 0: smoke test, 10 events per band per species into data/geant4_realistic_regime_pilot_rung0_{electron,proton}.
# rung 1: the full configs (300 electrons and 1000 protons per band), cheapest bands first, logs in
#         data/geant4_realistic_regime_pilot_logs/. A batch whose events.npz exists is skipped.
set -u
WORKERS="${1:-14}"
RUNG="${2:-0}"
PY=".venv/Scripts/python"
LOGS=data/geant4_realistic_regime_pilot_logs
mkdir -p "$LOGS"
run() {  # species, energy (GeV), output directory, extra args
  species="$1"; energy="$2"; out="$3"; shift 3
  if [ -f "$out/events.npz" ]; then echo "skip $species $energy"; return; fi
  echo "$(date -u +%FT%TZ) start $species ${energy} GeV"
  PYTHONPATH=src "$PY" -m ams_ecal.geant4_simulation.geant4_backend run \
    --sample pilot --energy "$energy" --workers "$WORKERS" \
    --pilot-config "configs/geant4_realistic_regime_pilot_${species}.yaml" --out "$out" "$@" \
    > "$LOGS/rung${RUNG}_${species}_${energy}.log" 2>&1
  echo "$(date -u +%FT%TZ) end $species ${energy} GeV exit $?"
}
if [ "$RUNG" = "0" ]; then
  for energy in 50 100 200 400 800; do
    run electron "$energy" "data/geant4_realistic_regime_pilot_rung0_electron/E${energy}GeV" --events 10
  done
  for energy in 50 100 200 400 800 2000; do
    run proton "$energy" "data/geant4_realistic_regime_pilot_rung0_proton/E${energy}GeV" --events 10
  done
elif [ "$RUNG" = "1" ]; then
  # Cheapest bands first (protons are about an order of magnitude cheaper per event than electrons).
  for energy in 50 100 200 400 800 2000; do
    run proton "$energy" "data/geant4_realistic_regime_pilot_proton/pilot/E${energy}GeV"
  done
  for energy in 50 100 200 400 800; do
    run electron "$energy" "data/geant4_realistic_regime_pilot_electron/pilot/E${energy}GeV"
  done
else
  echo "unknown rung $RUNG (use 0 or 1)"; exit 2
fi
echo "$(date -u +%FT%TZ) ALLDONE realistic_regime_pilot rung $RUNG"
