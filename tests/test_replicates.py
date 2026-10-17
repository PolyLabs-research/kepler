"""Seed-to-seed error bars: whole replicate curves resampled, because points along one run are
correlated and a mean curve's scatter understates the uncertainty. Known answers, seeded."""
import numpy as np
import pytest

from kepler import TIME_LAWS, compare_replicates, fit_replicates, predict

T = np.arange(0.0, 200.0, 2.0)


def curves(seed, replicates=20, beta=None):
    """Each replicate settles to its own level c with its own amplitude, plus noise correlated
    over about ten generations; the mean settles to 0.2 with time constant 25."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(replicates):
        a = 0.3 * (1 + 0.2 * rng.standard_normal())
        c = 0.2 + 0.01 * rng.standard_normal()
        noise, e = np.zeros(len(T)), 0.0
        for i in range(len(T)):
            e = 0.9 * e + 0.004 * rng.standard_normal()
            noise[i] = e
        shape = np.exp(-T / 25.0) if beta is None else np.exp(-((T / 25.0) ** beta))
        out.append(c + a * shape + noise)
    return np.array(out)


def test_seed_to_seed_errors_are_wider_than_the_scatter_suggests():
    result = fit_replicates("relaxation", T, curves(1), resamples=150)
    assert result.replicates == 20 and result.resamples >= 140
    for p, truth in {"c": 0.2, "a": 0.3, "tau": 25.0}.items():
        assert result.value(p) == pytest.approx(truth, abs=4 * result.error(p)), p
        low, high = result.intervals[p]
        assert low < result.value(p) < high
    # Correlated points: the fit's own error bar for the level is several times too small.
    naive = dict(zip(result.fit.param_names, result.naive_stderr))
    assert result.error("c") > 3 * naive["c"]
    y, low, high = predict(result.fit, [300.0])
    assert low[0] < y[0] < high[0]


def test_the_intervals_cover_the_truth():
    hits = 0
    for seed in range(25):
        result = fit_replicates("relaxation", T, curves(100 + seed), resamples=60, seed=seed)
        hits += abs(result.value("c") - 0.2) <= 1.96 * result.error("c")
    assert hits >= 21  # about 95% of 25, allowing for chance


def test_the_seeds_choose_the_law():
    exponential = compare_replicates(TIME_LAWS, T, curves(1), resamples=60)
    assert exponential.best == "relaxation" and exponential.share > 0.6
    assert exponential.ranking.best.name == "relaxation"
    stretched = compare_replicates(TIME_LAWS, T, curves(2, beta=0.5), resamples=60)
    assert stretched.best == "stretched_exponential" and stretched.share > 0.8
    assert sum(stretched.shares.values()) == pytest.approx(1.0)


def test_replicates_must_be_whole_curves():
    with pytest.raises(ValueError, match="one row of 100 values"):
        fit_replicates("relaxation", T, np.zeros((5, 99)))
    with pytest.raises(ValueError, match="at least two"):
        fit_replicates("relaxation", T, np.zeros((1, 100)))
