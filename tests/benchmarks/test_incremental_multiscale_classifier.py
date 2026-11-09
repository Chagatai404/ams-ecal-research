"""Fast synthetic tests of the EXP-018 incremental-classifier machinery (no Geant4 data)."""

import math

import numpy as np
import pytest
from scipy import stats
from sklearn.metrics import roc_auc_score

from ams_ecal.benchmarks import incremental_multiscale_classifier as inc


def test_hard_subset_requires_572_cells_and_a_neighbour_of_the_other_class_in_both_keys():
    # electrons: three overlapping, one far above everything, one below the 572-cell cut that has a proton twin
    cells_e = [600, 650, 700, 1500, 300, 620]
    energy_e = [
        1000,
        1100,
        1200,
        90000,
        500,
        40000,
    ]  # the last has a cells-neighbour but no energy-neighbour
    cells_p = [610, 640, 5000, 300, 5200]
    energy_p = [1010, 1090, 1e7, 510, 1e7]
    keep_e, keep_p = inc.hard_subset_masks(cells_e, energy_e, cells_p, energy_p)
    assert keep_e.tolist() == [True, True, True, False, False, False]
    assert keep_p.tolist() == [True, True, False, False, False]
    # a proton that fails (a) cannot serve as the neighbour that rescues an electron
    keep_e, keep_p = inc.hard_subset_masks(
        [300, 800], [500, 5000], [300, 5000], [500, 1e7]
    )
    assert not keep_e.any() and not keep_p.any()


def test_hard_subset_uses_only_the_keys_so_it_is_blind_to_anything_else():
    rng = np.random.default_rng(1)
    cells_e, cells_p = rng.integers(500, 900, 300), rng.integers(500, 900, 300)
    energy_e, energy_p = (
        2.0 * cells_e * rng.uniform(0.9, 1.1, 300),
        2.0 * cells_p * rng.uniform(0.9, 1.1, 300),
    )
    first = inc.hard_subset_masks(cells_e, energy_e, cells_p, energy_p)
    again = inc.hard_subset_masks(
        cells_e.copy(), energy_e.copy(), cells_p.copy(), energy_p.copy()
    )
    assert all(np.array_equal(a, b) for a, b in zip(first, again, strict=True))
    assert first[0].sum() > 0 and first[1].sum() > 0
    assert (cells_e[first[0]] >= inc.MINIMUM_OCCUPIED_CELLS).all()


def test_nadeau_bengio_variance_matches_the_formula_on_a_known_case():
    rng = np.random.default_rng(0)
    deltas = rng.normal(-0.02, 0.05, 100)
    iv = inc.nadeau_bengio_interval(
        deltas, n_test=80.0, n_train=320.0, folds=5, repeats=20
    )
    s2 = float(np.var(deltas, ddof=1))
    assert iv["variance_of_mean"] == pytest.approx((0.01 + 0.25) * s2)
    assert iv["degrees_of_freedom"] == 99
    half = stats.t.ppf(0.975, 99) * math.sqrt((0.01 + 0.25) * s2)
    assert iv["upper"] - iv["lower"] == pytest.approx(2 * half)
    assert iv["mean"] == pytest.approx(deltas.mean()) and iv[
        "half_width"
    ] == pytest.approx(half)
    # a constant vector has zero spread; the wrong number of folds is refused
    assert inc.nadeau_bengio_interval(np.full(100, -0.1), 80, 320)[
        "half_width"
    ] == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError):
        inc.nadeau_bengio_interval(deltas[:99], 80, 320)


def test_the_corrected_interval_is_wider_than_the_naive_one():
    deltas = np.random.default_rng(3).normal(0.0, 0.05, 100)
    corrected = inc.nadeau_bengio_interval(deltas, 80, 320)["half_width"]
    naive = stats.t.ppf(0.975, 99) * deltas.std(ddof=1) / math.sqrt(100)
    assert corrected > 2.0 * naive


@pytest.mark.parametrize(
    ("lower", "upper", "decision"),
    [
        (-0.10, -0.01, inc.DECISION_ADDS),
        (
            -0.10,
            0.00,
            inc.DECISION_NONE,
        ),  # upper limit exactly 0 does not satisfy "< 0"
        (-0.05, 0.05, inc.DECISION_NONE),
        (0.00, 0.05, inc.DECISION_NONE),  # lower limit exactly 0 does not satisfy "> 0"
        (0.01, 0.05, inc.DECISION_WORSE),
    ],
)
def test_the_primary_decision_rule_table(lower, upper, decision):
    assert inc.decide(lower, upper) == decision


def test_the_negative_control_and_overall_status_table():
    assert inc.negative_control_status(-0.05, 0.05) == inc.NEGATIVE_CONTROL_PASSED
    assert inc.negative_control_status(-0.05, -0.001) == inc.NEGATIVE_CONTROL_INVALID
    assert inc.negative_control_status(0.001, 0.05) == inc.NEGATIVE_CONTROL_WORSE
    passed = inc.NEGATIVE_CONTROL_PASSED
    assert (
        inc.overall_status(inc.DECISION_ADDS, inc.NEGATIVE_CONTROL_INVALID, True)
        == "invalid"
    )
    assert "not passed" in inc.overall_status(
        inc.DECISION_ADDS, inc.NEGATIVE_CONTROL_WORSE, True
    )
    assert "same-class" in inc.overall_status(inc.DECISION_ADDS, passed, False)
    assert inc.overall_status(inc.DECISION_ADDS, passed, True) == inc.DECISION_ADDS
    assert inc.overall_status(inc.DECISION_NONE, passed, True) == inc.DECISION_NONE


def _signal_problem(n=240, seed=0):
    rng = np.random.default_rng(seed)
    y = np.r_[np.ones(n // 2, int), np.zeros(n // 2, int)]
    conventional = rng.normal(size=(n, 4))
    multiscale = rng.normal(size=(n, 18)) + 2.0 * y[:, None]  # strongly informative
    return conventional, multiscale, y


def test_the_permuted_multiscale_columns_lose_their_link_to_the_label_but_keep_their_rows():
    _, multiscale, y = _signal_problem()
    permuted = inc.permute_multiscale(multiscale, repeat=0)
    assert not np.array_equal(permuted, multiscale)
    # joint permutation: the rows are the same set of 18-vectors
    assert sorted(map(tuple, permuted.round(8))) == sorted(
        map(tuple, multiscale.round(8))
    )
    assert roc_auc_score(y, multiscale[:, 0]) > 0.9
    assert abs(roc_auc_score(y, permuted[:, 0]) - 0.5) < 0.1
    assert np.array_equal(
        permuted, inc.permute_multiscale(multiscale, repeat=0)
    )  # one fixed permutation per repeat
    assert not np.array_equal(permuted, inc.permute_multiscale(multiscale, repeat=1))


def test_the_pipeline_sees_a_real_signal_and_the_permuted_control_does_not():
    conventional, multiscale, y = _signal_problem()
    sets = [
        {
            "A": conventional,
            "B": np.column_stack([conventional, multiscale]),
            "B_permuted": np.column_stack(
                [conventional, inc.permute_multiscale(multiscale, r)]
            ),
        }
        for r in range(3)
    ]
    results = inc.cross_fit(sets, y, "hgb", n_jobs=1)
    # compare() needs the registered 20 x 5 folds, so the per-fold rows are used directly here
    rows = [row for r in results for row in r["rows"]]
    delta_real = np.mean([row["logloss_B"] - row["logloss_A"] for row in rows])
    delta_control = np.mean(
        [row["logloss_B_permuted"] - row["logloss_A"] for row in rows]
    )
    assert delta_real < -0.3  # the informative multiscale block helps a lot
    assert (
        delta_control > -0.05
    )  # the permuted block cannot help (it may hurt slightly)


def test_a_column_that_copies_the_label_in_the_test_rows_only_does_not_help_on_average():
    # Any nonzero weight on a column that is a perfect label copy in the test rows gives a high AUC for that draw, with a random sign,
    # so the claim is about the average over draws: a model trained on noise cannot systematically exploit the test-fold label.
    aucs = {"hgb": [], "logistic": []}
    leaky = []
    for seed in range(25):
        rng = np.random.default_rng(seed)
        n_train, n_test = 120, 60
        y_train, y_test = (
            (rng.uniform(size=n_train) < 0.5).astype(int),
            (rng.uniform(size=n_test) < 0.5).astype(int),
        )
        x_train = rng.normal(
            size=(n_train, 4)
        )  # the last column is pure noise in the training rows
        x_test = np.column_stack(
            [rng.normal(size=(n_test, 3)), y_test + 0.01 * rng.normal(size=n_test)]
        )
        for kind, values in aucs.items():
            values.append(
                roc_auc_score(
                    y_test, inc.fit_predict_fold(kind, x_train, y_train, x_test)
                )
            )
        x_train_leaky = np.column_stack(
            [x_train[:, :3], y_train + 0.01 * rng.normal(size=n_train)]
        )
        leaky.append(
            roc_auc_score(
                y_test, inc.fit_predict_fold("hgb", x_train_leaky, y_train, x_test)
            )
        )
    for kind, values in aucs.items():
        assert abs(np.mean(values) - 0.5) < 0.1, kind
    assert (
        np.mean(leaky) > 0.95
    )  # the same column present in the training rows IS exploited, so the check could have failed


def test_the_training_rows_never_contain_a_test_row(monkeypatch):
    conventional, _, y = _signal_problem(n=100)
    row_id = np.arange(len(y), dtype=float)
    seen = []

    def spy(kind, x_train, y_train, x_test, random_state=0):
        seen.append((set(x_train[:, -1]), set(x_test[:, -1])))
        return np.full(len(x_test), 0.5)

    monkeypatch.setattr(inc, "fit_predict_fold", spy)
    inc.run_repeat(
        {"A": np.column_stack([conventional, row_id])},
        y,
        "hgb",
        fold_seed=inc.SEED,
        repeat=0,
    )
    assert len(seen) == inc.N_FOLDS
    assert all(not (train & test) for train, test in seen)
    assert set().union(*(test for _, test in seen)) == set(row_id)


def test_pure_noise_features_give_no_gain_and_chance_auc_through_the_whole_cross_fit():
    rng = np.random.default_rng(11)
    y = np.r_[np.ones(60, int), np.zeros(60, int)]
    noise_c, noise_m = rng.normal(size=(120, 4)), rng.normal(size=(120, 18))
    sets = [{"A": noise_c, "B": np.column_stack([noise_c, noise_m])} for _ in range(3)]
    results = inc.cross_fit(sets, y, "hgb", n_jobs=1)
    summary = inc.score_summary(results, y, ["A", "B"])
    assert (
        abs(summary["A"]["auc_mean"] - 0.5) < 0.12
        and abs(summary["B"]["auc_mean"] - 0.5) < 0.12
    )


def test_every_event_is_scored_out_of_fold_exactly_once_per_repeat_with_shared_folds():
    conventional, multiscale, y = _signal_problem(n=120)
    sets = {"A": conventional, "B": np.column_stack([conventional, multiscale])}
    result = inc.run_repeat(sets, y, "hgb", fold_seed=inc.SEED, repeat=0)
    assert all(np.isfinite(scores).all() for scores in result["oof"].values())
    assert sum(row["n_test"] for row in result["rows"]) == len(y)
    assert all(row["n_train"] + row["n_test"] == len(y) for row in result["rows"])
    assert len(result["rows"]) == inc.N_FOLDS


def test_the_class_balanced_log_loss_is_ln2_for_uninformative_scores():
    y = np.r_[np.ones(10, int), np.zeros(90, int)]
    assert inc.fold_log_loss(np.full(100, 0.5), y) == pytest.approx(math.log(2))
    # many correct protons cannot hide wrong electrons
    assert inc.fold_log_loss(np.r_[np.full(10, 0.01), np.zeros(90)], y) > 2.0


def test_the_hanley_mcneil_null_standard_error_and_the_same_class_null_on_noise():
    assert inc.hanley_mcneil_null_se(100, 100) == pytest.approx(
        math.sqrt(201 / (12 * 100 * 100))
    )
    rng = np.random.default_rng(2)
    null = inc.same_class_null(
        rng.normal(size=(120, 4)), rng.normal(size=(120, 18)), repeats=2
    )
    assert null["passed"] and null["pseudo_class_sizes"] == [60, 60]
    assert set(null["auc_per_repeat"]) == {"A", "B"}


def test_score_summary_counts_protons_above_the_electron_order_statistic_and_bounds_zero_counts():
    y = np.r_[np.ones(200, int), np.zeros(100, int)]
    scores = np.r_[np.linspace(0.5, 1.0, 200), np.linspace(0.0, 0.4, 100)]
    summary = inc.score_summary([{"oof": {"A": scores}}], y, ["A"])["A"]
    assert summary["protons_passing_per_repeat"] == [0]
    assert summary["auc_mean"] == pytest.approx(1.0)
    assert summary["false_positive_rate_clopper_pearson_upper_95_per_repeat"][
        0
    ] == pytest.approx(1 - 0.05 ** (1 / 100))
