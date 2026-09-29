"""Truth-level first inelastic interaction of the primary, from Geant4 records.

OPERATIONAL DEFINITION. The first inelastic interaction is the earliest point,
in global time, at which a Geant4 *hadronic* process whose name ends in
``Inelastic`` (``protonInelastic`` for a proton primary) created a secondary
whose parent is the primary track. The products of that one interaction are
the secondaries sharing its time and position.

This is read from the secondaries rather than from how the primary track ends,
because it then does not depend on whether a model kills the projectile or lets
it continue. Explicitly NOT counted as the first interaction:

* ionization delta rays (``hIoni``), ``hBrems``, ``hPairProd``, Coulomb
  scattering - electromagnetic processes;
* multiple scattering and transportation - they create no secondaries at all;
* hadronic *elastic* scattering (``hadElastic``), whose recoil nuclei are counted
  separately as ``n_elastic_recoils``;
* any other hadronic process (for example a separate charge-exchange process in
  some physics lists) - counted as ``n_other_hadronic`` and reported, never
  silently merged with inelastic.

LIMITATION. An inelastic interaction that produced no secondary at all would be
invisible here. Geant4's inelastic models destroy the projectile, so at least
the outgoing leading particle is a secondary; the Geant4 integration tests check
that every inelastic end of the primary is seen.

Pure Python: the Geant4 backend fills ``SecondaryRecord`` objects, and this
module decides, so the definition is unit-tested without Geant4.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from math import isclose, nan

PI0_PDG = 111
PHOTON_PDG = 22


@dataclass(frozen=True, slots=True)
class SecondaryRecord:
    """One secondary as created, with the process that created it."""

    parent_id: int
    creator_process: str
    creator_is_hadronic: bool
    global_time_ns: float
    x_mm: float
    y_mm: float
    z_mm: float
    kinetic_energy_mev: float
    total_energy_mev: float
    pdg: int
    charge: float


def is_hadronic_inelastic(process_name: str, is_hadronic: bool) -> bool:
    """Return whether a creator process counts as a hadronic inelastic one."""

    return bool(is_hadronic) and process_name.endswith("Inelastic")


def is_hadronic_elastic(process_name: str, is_hadronic: bool) -> bool:
    return bool(is_hadronic) and process_name == "hadElastic"


@dataclass(frozen=True, slots=True)
class FirstInteraction:
    """Truth summary of the primary's first inelastic interaction."""

    occurred: bool
    x_mm: float
    y_mm: float
    z_mm: float
    process: str
    n_secondaries: int
    n_charged: int
    leading_kinetic_energy_mev: float
    pi0_gamma_energy_mev: float
    n_elastic_recoils: int
    n_other_hadronic: int


def first_inelastic_interaction(
    records: Iterable[SecondaryRecord], primary_track_id: int = 1
) -> FirstInteraction:
    """Apply the operational definition to one event's secondary records."""

    own = [record for record in records if record.parent_id == primary_track_id]
    inelastic = [
        r for r in own if is_hadronic_inelastic(r.creator_process, r.creator_is_hadronic)
    ]
    elastic_recoils = sum(
        is_hadronic_elastic(r.creator_process, r.creator_is_hadronic) for r in own
    )
    other_hadronic = sum(
        r.creator_is_hadronic
        and not is_hadronic_inelastic(r.creator_process, True)
        and not is_hadronic_elastic(r.creator_process, True)
        for r in own
    )

    if not inelastic:
        return FirstInteraction(
            occurred=False,
            x_mm=nan,
            y_mm=nan,
            z_mm=nan,
            process="",
            n_secondaries=0,
            n_charged=0,
            leading_kinetic_energy_mev=nan,
            pi0_gamma_energy_mev=nan,
            n_elastic_recoils=elastic_recoils,
            n_other_hadronic=other_hadronic,
        )

    first = min(inelastic, key=lambda r: r.global_time_ns)
    products = [
        r
        for r in inelastic
        if isclose(r.global_time_ns, first.global_time_ns, rel_tol=0.0, abs_tol=1e-9)
        and isclose(r.x_mm, first.x_mm, rel_tol=0.0, abs_tol=1e-6)
        and isclose(r.y_mm, first.y_mm, rel_tol=0.0, abs_tol=1e-6)
        and isclose(r.z_mm, first.z_mm, rel_tol=0.0, abs_tol=1e-6)
    ]
    return FirstInteraction(
        occurred=True,
        x_mm=first.x_mm,
        y_mm=first.y_mm,
        z_mm=first.z_mm,
        process=first.creator_process,
        n_secondaries=len(products),
        n_charged=sum(r.charge != 0 for r in products),
        leading_kinetic_energy_mev=max(r.kinetic_energy_mev for r in products),
        pi0_gamma_energy_mev=sum(
            r.total_energy_mev for r in products if r.pdg in (PI0_PDG, PHOTON_PDG)
        ),
        n_elastic_recoils=elastic_recoils,
        n_other_hadronic=other_hadronic,
    )
