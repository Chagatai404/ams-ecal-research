#!/usr/bin/env bash
# Resumable driver for the powered realistic-regime production (researcher's go 2026-11-09: "do this overnight today";
# "Just generate the data we need for this pass. Especially the proton."). From the repository root:
#   bash configs/run_realistic_regime_powered.sh [workers]
# Protons first (bands 0.8-2, 2-4, 4-8 TeV), then electrons (0.8-2 TeV, then 0.4-0.8 TeV). Each chunk is one batch with its
# own base seed (configs/geant4_realistic_regime_powered_{proton,electron}.yaml); a chunk whose events.npz exists is skipped,
# so the script can be restarted after an interruption and only the unfinished chunk is redone. One batch at a time (parallel
# jobs slowed the pilot's 800 GeV electron batch).
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
  say "$(date -u +%FT%TZ) start $species $chunk ${energy} GeV ($events events)"
  PYTHONPATH=src "$PY" -m ams_ecal.geant4_simulation.geant4_backend run \
    --sample "$chunk" --energy "$energy" --events "$events" --workers "$WORKERS" \
    --pilot-config "configs/geant4_realistic_regime_powered_${species}.yaml" --out "$out" \
    > "$LOGS/${species}_${chunk}_${energy}.log" 2>&1
  status=$?
  say "$(date -u +%FT%TZ) end $species $chunk ${energy} GeV exit $status"
}
# Protons: [800,2000) 2 x 2500; [2000,4000) 5 x 1200; [4000,8000) 4 x 600.
run proton c01 800 2500
run proton c03 2000 1200
run proton c08 4000 600
run proton c02 800 2500
run proton c04 2000 1200
run proton c09 4000 600
run proton c05 2000 1200
run proton c10 4000 600
run proton c06 2000 1200
run proton c11 4000 600
run proton c07 2000 1200
# Electrons: [800,2000) 12 x 150; [400,800) 2 x 500.
for chunk in c03 c04 c05 c06 c07 c08; do run electron "$chunk" 800 150; done
run electron c01 400 500
for chunk in c09 c10 c11 c12 c13 c14; do run electron "$chunk" 800 150; done
run electron c02 400 500
say "$(date -u +%FT%TZ) ALLDONE realistic_regime_powered"
