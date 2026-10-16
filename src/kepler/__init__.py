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
from kepler.fitting import Fit, fit, parse_skeleton, predict
from kepler.laws import GROWTH_LAWS, LAWS, Law, get_law
from kepler.ranking import MIN_R2, OCCAM_WINDOW, RankEntry, Ranking, bic, occam_order, rank

__version__ = "0.1.0"

__all__ = [
    "Fit",
    "GROWTH_LAWS",
    "LAWS",
    "Law",
    "MIN_R2",
    "OCCAM_WINDOW",
    "RankEntry",
    "Ranking",
    "bic",
    "fit",
    "get_law",
    "occam_order",
    "parse_skeleton",
    "predict",
    "rank",
]
