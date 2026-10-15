import numpy as np

from ams_ecal.detector import sampling_response_em_share as em


class Constant:
    """Stands in for a fitted table: every cell gets a fixed value."""

    def __init__(self, value: float) -> None:
        self.value = value

    def apply(self, deposit, rng):
        return np.full(deposit.shape, self.value, np.float32)


def part(bins):
    n = len(bins)
    return {"deposit": np.ones((n, 18, 72), np.float32), "share_bin": np.asarray(bins), "index": np.arange(n)}


def test_every_event_is_predicted_with_the_table_of_its_own_share_bin():
    models = [Constant(float(g)) for g in range(em.N_BINS)]
    p = part([0, 3, 5, 3, 1])
    out = em._predict(models, p, np.ones(5, bool), np.random.default_rng(0))
    assert [float(out[i, 0, 0]) for i in range(5)] == [0.0, 3.0, 5.0, 3.0, 1.0]


def test_a_single_table_ignores_the_share_bin():
    out = em._predict([Constant(7.0)], part([0, 4, 5]), np.ones(3, bool), np.random.default_rng(0))
    assert (out == 7.0).all()


def test_the_registered_share_bins_are_six_and_fixed():
    assert em.SHARE_EDGES == (0.3, 0.5, 0.65, 0.8, 0.9) and em.N_BINS == 6
    assert list(np.digitize([0.0, 0.3, 0.64, 0.7, 0.85, 0.95, 1.0], em.SHARE_EDGES)) == [0, 1, 2, 3, 4, 5, 5]
