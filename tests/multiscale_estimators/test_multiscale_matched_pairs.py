import numpy as np

from ams_ecal.multiscale_estimators import multiscale_matched_pairs as mpair


def populations(seed=0, n_e=600, n_p=800):
    rng = np.random.default_rng(seed)
    # electrons deposit more for the same occupancy; the supports overlap only partly
    cells_e = rng.lognormal(6.6, 0.25, n_e)
    energy_e = cells_e * rng.lognormal(3.0, 0.2, n_e)
    cells_p = rng.lognormal(6.2, 0.35, n_p)
    energy_p = cells_p * rng.lognormal(2.5, 0.35, n_p)
    return np.column_stack([cells_e, energy_e]), np.column_stack([cells_p, energy_p])


def test_the_matching_is_one_to_one_and_every_pair_is_inside_the_caliper():
    ke, kp = populations()
    ie, ip = mpair.match_pairs(ke, kp, 0.25)
    assert (
        len(ie) == len(set(ie.tolist()))
        and len(ip) == len(set(ip.tolist()))
        and len(ie) > 0
    )
    ze, zp, _ = mpair.standardized_logs(ke, kp)
    assert (np.linalg.norm(ze[ie] - zp[ip], axis=1) <= 0.25 + 1e-12).all()


def test_a_wider_caliper_matches_at_least_as_many_pairs_and_matching_balances_the_keys():
    ke, kp = populations(1)
    narrow, wide = mpair.match_pairs(ke, kp, 0.1), mpair.match_pairs(ke, kp, 0.5)
    assert len(wide[0]) >= len(narrow[0])
    ie, ip = mpair.match_pairs(ke, kp, 0.25)
    before = np.abs(np.log(ke).mean(axis=0) - np.log(kp).mean(axis=0))
    after = np.abs(np.array(mpair.balance(ke, kp, ie, ip)))
    assert after.max() < 0.1 and before.max() > 0.3


def test_identical_keys_are_matched_completely():
    k = np.column_stack([np.linspace(400, 900, 150), np.linspace(1e4, 9e4, 150)])
    ie, ip = mpair.match_pairs(k, k.copy(), 0.25)
    assert len(ie) == 150 and (k[ie] == k[ip]).all()


def test_the_pair_swap_test_holds_its_level_under_the_null_and_detects_a_shift():
    rng = np.random.default_rng(3)
    hits_null = hits_shift = 0
    for _ in range(80):
        base = rng.normal(size=(150, 1))
        e, p = (
            base + 0.3 * rng.normal(size=(150, 1)),
            base + 0.3 * rng.normal(size=(150, 1)),
        )
        t = mpair.pair_swap_test(e, p, 199, rng)
        hits_null += (1 + (t["null"][:, 0] >= t["observed"][0]).sum()) / 200 <= 0.05
        t = mpair.pair_swap_test(e + 0.5, p, 199, rng)
        hits_shift += (1 + (t["null"][:, 0] >= t["observed"][0]).sum()) / 200 <= 0.05
    assert (
        hits_null / 80 <= 0.05 + 3 * np.sqrt(0.05 * 0.95 / 80) and hits_shift / 80 > 0.9
    )


def test_a_pair_with_a_missing_member_is_removed_from_that_column_in_both_classes():
    rng = np.random.default_rng(4)
    e, p = rng.normal(size=(150, 2)), rng.normal(size=(150, 2))
    e[:20, 0] = np.nan
    t = mpair.pair_swap_test(e, p, 99, rng)
    assert t["valid_pairs"].tolist() == [130, 150]
