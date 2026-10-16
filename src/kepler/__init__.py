"""
Kepler's classical core: fit, rank and predict closed-form laws with error bars.

    from kepler import fit, rank, predict

    fits = [fit(name, L, T, sigma=T_err) for name in ("logarithmic", "power", "linear")]
    ranking = rank(fits)
    T_hat, low, high = predict(ranking.best.fit, [48, 64, 96, 128])

Works on any (x, y) data; needs numpy, scipy and sympy only. The specialist
transformers (galileo_minimal/, the `models` extra) and the MCP server (the
`mcp` extra) are separate and optional.
"""
from kepler.design import Separation, separation
from kepler.fitting import Fit, fit, information, parse_skeleton, predict, predict_covariance
from kepler.joint import GroupFit, Sharing, compare_sharing, fit_groups
from kepler.laws import FSS_LAWS, GROWTH_LAWS, LAWS, PROBABILITY_LAWS, Law, get_law
from kepler.ranking import MIN_R2, OCCAM_WINDOW, RankEntry, Ranking, bic, occam_order, rank

__version__ = "0.1.0"

__all__ = [
    "FSS_LAWS",
    "Fit",
    "GROWTH_LAWS",
    "GroupFit",
    "LAWS",
    "Law",
    "MIN_R2",
    "OCCAM_WINDOW",
    "PROBABILITY_LAWS",
    "RankEntry",
    "Ranking",
    "Separation",
    "Sharing",
    "bic",
    "compare_sharing",
    "fit",
    "fit_groups",
    "get_law",
    "information",
    "occam_order",
    "parse_skeleton",
    "predict",
    "predict_covariance",
    "rank",
    "separation",
]
