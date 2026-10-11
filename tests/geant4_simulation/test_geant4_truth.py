from math import isnan

from ams_ecal.geant4_simulation.geant4_truth import (
    SecondaryRecord,
    first_inelastic_interaction,
    is_hadronic_inelastic,
)


def record(process, hadronic=True, t=1.0, z=10.0, parent=1, ke=100.0, pdg=211, charge=1.0):
    return SecondaryRecord(
        parent_id=parent,
        creator_process=process,
        creator_is_hadronic=hadronic,
        global_time_ns=t,
        x_mm=0.0,
        y_mm=0.0,
        z_mm=z,
        kinetic_energy_mev=ke,
        total_energy_mev=ke + (0.0 if pdg == 22 else 135.0),
        pdg=pdg,
        charge=charge,
    )


def test_no_secondaries_means_no_interaction() -> None:
    truth = first_inelastic_interaction([])
    assert not truth.occurred
    assert isnan(truth.z_mm) and truth.process == "" and truth.n_secondaries == 0


def test_electromagnetic_secondaries_are_not_an_interaction() -> None:
    truth = first_inelastic_interaction(
        [
            record("hIoni", hadronic=False, pdg=11, charge=-1.0),
            record("hBrems", hadronic=False, pdg=22, charge=0.0),
            record("hPairProd", hadronic=False, pdg=11, charge=-1.0),
            record("CoulombScat", hadronic=False, pdg=1000822080, charge=82.0),
        ]
    )
    assert not truth.occurred


def test_elastic_recoils_are_counted_but_are_not_an_interaction() -> None:
    truth = first_inelastic_interaction(
        [record("hadElastic", pdg=1000822080), record("hadElastic", t=2.0)]
    )
    assert not truth.occurred
    assert truth.n_elastic_recoils == 2


def test_the_earliest_inelastic_vertex_is_the_first_interaction() -> None:
    first = [
        record("protonInelastic", t=0.2, z=40.0, ke=900.0),
        record("protonInelastic", t=0.2, z=40.0, ke=50.0, pdg=111, charge=0.0),
        record("protonInelastic", t=0.2, z=40.0, ke=7.0, pdg=22, charge=0.0),
    ]
    later = [record("protonInelastic", t=0.5, z=90.0, ke=5000.0)]
    truth = first_inelastic_interaction(later + first + [record("hIoni", False, t=0.1)])
    assert truth.occurred and truth.z_mm == 40.0
    assert truth.process == "protonInelastic"
    assert truth.n_secondaries == 3 and truth.n_charged == 1
    assert truth.leading_kinetic_energy_mev == 900.0
    assert truth.pi0_gamma_energy_mev == (50.0 + 135.0) + 7.0


def test_only_the_primarys_own_secondaries_count() -> None:
    truth = first_inelastic_interaction([record("protonInelastic", parent=7)])
    assert not truth.occurred


def test_a_hadronic_flag_is_required() -> None:
    assert is_hadronic_inelastic("protonInelastic", True)
    assert not is_hadronic_inelastic("protonInelastic", False)
    assert not is_hadronic_inelastic("hadElastic", True)


def test_other_hadronic_processes_are_reported_separately() -> None:
    truth = first_inelastic_interaction([record("chargeExchange")])
    assert not truth.occurred
    assert truth.n_other_hadronic == 1
