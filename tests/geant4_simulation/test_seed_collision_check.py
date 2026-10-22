from ams_ecal.geant4_simulation.seed_collision_check import find_collisions


def _set(config, particle, energy, seeds, geometry="ams"):
    return {
        "config": config,
        "sample": "s",
        "particle": particle,
        "geometry": geometry,
        "energy_gev": energy,
        "seeds": tuple(seeds),
    }


def test_a_shared_seed_is_reported_and_classified():
    planned = [_set("planned.yaml", "e-", 200.0, [1, 2, 3])]
    existing = [
        _set("a.yaml", "e-", 200.0, [3, 9]),
        _set("b.yaml", "proton", 200.0, [2]),
    ]
    found = find_collisions(planned, existing)
    assert {(c["seed"], c["same_particle_energy_geometry"]) for c in found} == {
        (3, True),
        (2, False),
    }


def test_no_shared_seed_gives_no_collision():
    planned = [_set("planned.yaml", "e-", 200.0, [1, 2])]
    assert find_collisions(planned, [_set("a.yaml", "e-", 200.0, [5, 6])]) == []


def test_two_planned_sets_are_checked_against_each_other():
    planned = [
        _set("p1.yaml", "e-", 200.0, [7]),
        _set("p2.yaml", "e-", 200.0, [7]),
    ]
    assert len(find_collisions(planned, [])) == 2
