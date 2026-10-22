#!/usr/bin/env bash
# Resumable driver for the EXP-011 high-energy pilot. NOT YET LAUNCHED: it needs the researcher's go. From the repository root:
#   bash configs/run_highenergy_pilot.sh [workers]
# Before launching: python -m ams_ecal.geant4_simulation.seed_collision_check must report no collision with an existing set of the
# same particle, energy and geometry. A (config, energy) batch whose events.npz exists is skipped. Cheapest first.
set -u
WORKERS="${1:-14}"
mkdir -p data/geant4_highenergy_pilot_logs
run() {  # config, output directory, sample, energy (GeV)
  out="$2/$3/E${4}GeV"
  if [ -f "$out/events.npz" ]; then echo "skip $1 ${4}"; return; fi
  echo "$(date -u +%FT%TZ) start $1 ${4} GeV"
  PYTHONPATH=src .venv/Scripts/python -m ams_ecal.geant4_simulation.geant4_backend run \
    --sample "$3" --energy "$4" --workers "$WORKERS" --pilot-config "configs/$1.yaml" \
    > "data/geant4_highenergy_pilot_logs/$1_${4}.log" 2>&1
  echo "$(date -u +%FT%TZ) end $1 ${4} GeV exit $?"
}
for energy in 500 1000 2000; do run geant4_highenergy_pilot_proton data/geant4_highenergy_pilot_proton pilot "$energy"; done
for energy in 200 400 800; do run geant4_highenergy_pilot_electron data/geant4_highenergy_pilot_electron pilot "$energy"; done
run geant4_highenergy_pilot_electron_extended data/geant4_highenergy_pilot_electron_extended pilot 1000
echo "$(date -u +%FT%TZ) ALLDONE highenergy_pilot"
