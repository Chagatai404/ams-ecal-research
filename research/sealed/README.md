# Sealed Geant4 validation sets

Sealed 2026-10-04. Never opened for analysis. Opened only through `ams_ecal.validation.sealed_set.open_set`
(final flag, stated purpose, committed hash manifest, opening ledger `opening_ledger.jsonl`).

| set | config | events | energies (GeV) | base seed |
|---|---|---|---|---|
| proton | `configs/geant4_proton_sealed.yaml` | 3000 per energy | 10, 14, 20, 30, 50, 70, 100 | 20271001 |
| electron | `configs/geant4_electron_sealed.yaml` | 1000 per energy | 10, 14, 20, 30, 50, 70, 100 | 20271002 |

Generated from commit `7f75f73` (clean tree, in a separate git worktree) with Geant4 11.4.1, FTFP_BERT.
The manifests (`proton_manifest.json`, `electron_manifest.json`) hold the SHA-256 of every file.

## How it was generated, and one incident

* 2026-10-02: proton (all seven energies) and electron 10-50 GeV ran in one background script.
* The script died at electron 70 GeV (a Windows `fork: Resource temporarily unavailable` error,
  when the Claude Code session ended). No partial batch directory existed.
* 2026-10-04: electron 70 and 100 GeV were rerun with the identical config, seeds and commit.
  Seeds are a pure function of (base seed, energy), so the rerun is the same event stream a
  clean run would have produced.
* Logs: `generation_log_part1.txt`, `generation_log_part2.txt`.

## What was read before sealing (disclosed)

An integrity check read, from each batch, only `metadata.json`, the event count, the per-event
seeds (compared with `event_seeds(base_seed, energy, n)`), the git commit and a finiteness flag
of the readout grid. No energy, hit, depth or shape value was inspected or computed. That read
did not go through `open_set`, so it is not in the ledger; it is recorded here instead.
