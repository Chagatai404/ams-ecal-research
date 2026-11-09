#!/usr/bin/env bash
# Overnight subset (about 12 h on 14 workers) of the powered realistic-regime production. Researcher 2026-11-09: "30 hours are a
# bit too much for 1 run, plan it for overnight for now." Same configs, chunks and seeds as configs/run_realistic_regime_powered.sh,
# so the chunks left out here can be run later by that full driver (it skips every chunk whose events.npz exists).
#   bash configs/run_realistic_regime_powered_overnight.sh [workers]
# Protons and electrons are interleaved so a shortened night still grows both classes of the hard subset.
# Planned: protons [0.8,2) TeV 2500 (c01), [2,4) TeV 2 x 1200, [4,8) TeV 2 x 600; electrons [0.8,2) TeV 5 x 150, [0.4,0.8) TeV 500.
set -u
WORKERS="${1:-14}"
PY=".venv/Scripts/python"
LOGS=data/geant4_realistic_regime_powered_logs
mkdir -p "$LOGS"
say() { echo "$*" | tee -a "$LOGS/driver.log"; }
run() {  # species, chunk, band lower edge (GeV), events
  species="$1"; chunk="$2"; energy="$3"; events="$4"
  out="data/geant4_realistic_regime_powered_${species}/${chunk}/E${energy}GeV"
  if [ -f "$out/events.npz" ]; then say "skip $species $chunk ${energy}"; return; fi
  say "$(date -u +%FT%TZ) start $species $chunk ${energy} GeV ($events events) [overnight]"
  PYTHONPATH=src "$PY" -m ams_ecal.geant4_simulation.geant4_backend run \
    --sample "$chunk" --energy "$energy" --events "$events" --workers "$WORKERS" \
    --pilot-config "configs/geant4_realistic_regime_powered_${species}.yaml" --out "$out" \
    > "$LOGS/${species}_${chunk}_${energy}.log" 2>&1
  status=$?
  say "$(date -u +%FT%TZ) end $species $chunk ${energy} GeV exit $status"
}
run proton c01 800 2500
run proton c03 2000 1200
run electron c03 800 150
run proton c08 4000 600
run electron c04 800 150
run proton c04 2000 1200
run electron c05 800 150
run proton c09 4000 600
run electron c06 800 150
run electron c07 800 150
run electron c01 400 500
say "$(date -u +%FT%TZ) ALLDONE realistic_regime_powered overnight"
