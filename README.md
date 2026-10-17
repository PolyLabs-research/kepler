# Kepler

Kepler finds closed-form laws in numerical data. It has two halves:

- **The `kepler` library** (`src/kepler`) is the classical half. It fits a law's constants
  with error bars, ranks rival laws, predicts with a confidence band, and says before an
  experiment whether the design can decide between them. It also finds exact laws in integer
  data, conserved quantities, and known numbers among fitted constants. It needs numpy, scipy
  and sympy only.
- **Galileo Trainer** (`galileo_minimal/`) trains transformer models that propose formulas from
  data, and an MCP server through which a language model can propose an equation and have
  `kepler` fit it.

## The `kepler` library

```bash
pip install -e .              # the library
pip install -e ".[models]"    # + torch, for galileo_minimal/train.py and friends
pip install -e ".[mcp]"       # + mcp, for galileo_minimal/mcp_server.py
pytest                        # the library's tests (tests/); galileo_minimal has its own
```

### Fit, rank, predict

```python
from kepler import fit, rank, predict, LAWS

# Constants with standard errors; sigma weights each point, space="log" fits log y.
fits = [fit(name, L, T, sigma=T_err) for name in ("logarithmic", "power", "exponential")]
ranking = rank(fits)                      # BIC, 6-point Occam window, eligibility reasons
best = ranking.best.fit                   # None when no law fits: that is an answer too
best.params, best.stderr                  # e.g. (27.2, 0.350), (0.98, 0.013)
T_hat, low, high = predict(best, [48, 64, 128])   # 95% band from the covariance
```

- `fit(skeleton, x, y, sigma=None, space="linear")` takes a law name from `kepler.LAWS`
  (constant, logarithmic, linear, power, exponential, exponential_area, logistic,
  shifted_power, and the probability laws below), a `Law`, or an equation whose Float literals
  are the constants (`"20.0*x**0.5"`). A zero literal is refused: SymPy drops `0.0 + x` to `x`.
- `rank(fits, n, occam_window=6.0, min_r2=0.5)` is the rule `GalileoTools.discover_best` uses,
  now shared code. It refuses to compare fits of different data, weights or spaces.
- `predict(fit, x, level=0.95)` returns the curve and its confidence band. A held-out
  measurement's own error is the caller's to add. `predict_covariance(fit, x)` gives the full
  covariance of predictions at several x (correlated through the shared constants), for a joint
  test of held-out points.
- The MCP server gains `fit_skeleton`: a language model proposes an equation and gets its
  constants fitted. `verify_equation` never refits.

### Comparing laws, and designing the experiment

- Model comparison is likelihood-based: with absolute sigmas a fit's BIC is chi-square +
  k ln n (known errors), otherwise n ln(RSS/n) + k ln n; `Fit.aicc` is the small-sample AIC.
  With `inflate`, `rank` divides every chi-square by one shared overdispersion factor (the
  best law's reduced chi-square, at least 1), so errors that are too small cannot make a
  verdict look decisive.
- `separation(truth, rival, x, relative=0.1, space="log", truth_params={...})` says before a
  run whether a design can tell two laws apart: the expected BIC difference, the factor by
  which the seeds must grow to reach a margin, and (with `simulations=N`) the chance of a
  decisive right or wrong verdict. `separation(..., likelihood="binomial", trials=20)` sizes a
  count design the same way.

### Counts, and times that are only bounds

- Counts are fitted by maximum likelihood, not least squares on fractions:
  `fit("logistic_p", rho, survived, likelihood="binomial", trials=seeds)`. Its -2 ln L is
  exact, its BIC counts every seed, `rss` is the deviance, and bands stay inside (0, 1).
  `likelihood="poisson"` takes counts over an `exposure`. `PROBABILITY_LAWS` are a constant,
  a step (`logistic_p`) and a rise then fall (`window`: too sparse to start, too dense to last).
  Perfectly separated outcomes (every row all-or-nothing) are flagged, not trusted.
- Times that are sometimes only bounds (a run that hit its cap before the event) take
  `likelihood="censored"`: one row per run, `censored` marking the bounds, each time
  log-normal about the law with a fitted spread (`Fit.nuisance["log_spread"]`). Dropping the
  bounds, or reading them as times, biases a growth law low; this uses both.

### Several variables, and several groups

- Laws of several variables take `x` as a dict: `fit("fss_logistic", {"rho": rho, "L": L},
  survived, likelihood="binomial", trials=seeds)` fits one surface over density and size,
  rho*(L) = r + a L^-b with a step that sharpens as L^s, so r is the threshold at infinite
  size with an error bar. `FSS_LAWS` rank it against a fixed threshold (`sharpening`) and
  against no transition at all (`crossover`, the same curve at every size).
- One law across groups: `fit_groups("power", L, T, lattice, shared=("b",), sigma=err)` fits
  one exponent for every lattice and an amplitude per lattice (`a[hexagon]`), in one
  likelihood, so the shared constant's error bar uses all the data. `compare_sharing` fits every
  pattern (nothing shared, each subset, everything) and says which the data support, by BIC
  with an Occam window and by a likelihood-ratio p-value against sharing nothing: "is the
  exponent universal across lattices?" `GroupFit.fit_for("hexagon")` predicts like any fit.

### Laws over time, and distributions

- Laws over time (`TIME_LAWS`): `relaxation` c + a e^(-x/tau), `stretched_exponential`, and
  `algebraic_relaxation` c + a (1 + x/tau)^-b. For curves measured once per seed,
  `fit_replicates(law, t, Y)` (one row of Y per seed) takes the error bars from resampling whole
  seeds, since points along one run are correlated and the scatter about the mean curve
  understates them; `compare_replicates(TIME_LAWS, t, Y)` says how often each law is chosen
  across those resamples.
- Distributions of sizes and times (`kepler.distributions`): `fit_distribution(samples,
  "power_law", xmin, discrete=True)` for power law, truncated power law, exponential and
  lognormal, continuous or on the integers; `find_xmin` picks the tail by the KS distance;
  `compare_distributions` gives Vuong's test, or a likelihood-ratio test for a power law inside
  a truncated one; `goodness_of_fit` is the bootstrap p-value that can rule a family out
  (Clauset, Shalizi and Newman 2009).

### Exact laws, conserved quantities and known constants

- Exact laws for integer data (periods, counts): `find_integer_laws(n, y)` tries polynomials,
  laws per residue class, gcd/lcm forms, exponentials, linear recurrences, periodic sequences,
  and products of number-theoretic features, in exact arithmetic. The features include the odd
  part of n, its 2-adic part 2^v(n), and the multiplicative order and suborder of 2 modulo the
  odd part (ord and sord, the least j with 2^j = +-1), with 2^ord - 1 and 2^sord - 1: the
  quantities periods of linear (XOR) automata are built from. It also includes the periods of
  the 1D linear rules 60, 90 and 150 on n and on odd(n) cells, computed exactly, so that a law
  can tie a lattice rule to a 1D one. Each law says how many points
  determined it and how many more confirmed it; laws that agree on the data but part beyond it
  are reported as rivals, and `integer_laws.disagreement` names the next n that would decide.
- Conserved quantities: `find_invariants(runs, modulus=2)` finds the integer combinations of
  counts that never change in any run (exact null space, reduced by LLL to short vectors), and
  those conserved only mod a prime. `Invariants.beyond(known)` counts what the known
  bookkeeping identities do not explain.
- Is a constant a known number? `identify(0.3341, 0.0053)` lists closed forms (p/q, p/q times
  pi, e, sqrt 2, ..., square roots) and named constants (`KNOWN`: 2D percolation and Ising
  exponents, directed percolation, KPZ, coarsening, percolation thresholds, and the Gumbel law's
  shape constants for extreme values) within the error
  bar, simplest first, each with `p_chance`, the chance that a match that simple and that close
  would turn up anyway at that error bar. `identify_fit(fit)` does every constant of a fit.

`galileo_minimal/expression.py`, `normalize.py` and `refine.py` are now shims that hand back
the package modules, so the scripts and tests there run unchanged from inside that folder.

## What it has found

[lattice-lab](https://github.com/tripptytrip/lattice-lab) uses the library as the fitting half
of a law-discovery loop over [LatticeLife](https://github.com/PolyLabs-research/gameoflife_advanced)
cellular automata. Each law's predictions are registered before the runs that test them. In that
loop:

- **Exact period laws** from `find_integer_laws`. For the square XOR rule B1357/S1357 it found
  P(n) = 2^v(n) (2^sord(n) - 1), and for the parity rules on the hexagonal and triangular
  lattices 2 * 2^v(n) (2^sord(n) - 1). Each was determined by one size and confirmed by the
  rest. Registered tests confirmed the square law at every n up to 200. They confirmed the
  triangles' law at 154 sizes and refuted the hexagon's at 3 of 70.
- **A conserved quantity** from `find_invariants`: under the XOR rule, cumulative births are
  always even. Found in the data, it then had a short proof.
- **A settling-time law** chosen by `rank` with a held-out test: on the hexagonal lattice, B2/S34
  soups settle in time that grows with the logarithm of the torus size, not with a power of it.
  A registered replication on tori up to 1024 confirmed it and rejected the power law. So did a
  registered test of eight random rules on four lattice types. Across those eight, the
  fluctuations share one shape and one ratio of spread to growth, both consistent with the
  Gumbel law for the largest of many light-tailed times. With more soups, B2/S34's ratio is the
  same; an apparent exception in the first replication was a low draw. The Gumbel constants were added to
  `KNOWN` for that test; at its error bars, `identify` finds them among hundreds of equally
  good matches, so it cannot yet name the class unaided.
- **A gap, then closed.** For odd parts of at least 5, the hexagon's correct period is 2 * 2^v(n)
  times rule 90's period on the odd part: two hexagon generations are three passes of the 1D
  rule 90. `find_integer_laws` had no feature for another automaton's period, so it could not
  have proposed that law; the refutation and the proof came from lattice-lab's registered test
  and reasoning added afterwards. The search now includes the periods of the 1D linear rules
  (60, 90, 150, on n and on odd(n) cells). In a registered replay it lists the corrected law,
  `2*P90(n)`, beside the formula on the discovery data, names n = 74 as the size that decides,
  and on the data up to 200 keeps the right law on every lattice. That shows capability, in
  hindsight; simplicity alone would have picked the rule-90 law on every lattice, wrongly for
  three of them.

The write-ups, with what was known before and what was not, are lattice-lab's
[docs/xor-periods.md](https://github.com/tripptytrip/lattice-lab/blob/ccr-ab9f0bb1-qb20sx/docs/xor-periods.md),
[docs/parity-lattices.md](https://github.com/tripptytrip/lattice-lab/blob/ccr-ab9f0bb1-qb20sx/docs/parity-lattices.md)
and [docs/settling-time.md](https://github.com/tripptytrip/lattice-lab/blob/ccr-ab9f0bb1-qb20sx/docs/settling-time.md).

## Galileo Trainer

`galileo_minimal/` trains specialist transformers (polynomial, autoregressive, transcendental
and multivariate physics laws) that predict a formula from data, with clean-up of the predicted
coefficients, an evaluation script and an MCP server. Run it from that folder:

```bash
cd galileo_minimal
chmod +x setup.sh
./setup.sh --dev        # auto-detects CUDA (or use --cpu / --cuda)
source .venv/bin/activate
python train.py --model polynomial
python test.py --model polynomial
```

[galileo_minimal/README.md](galileo_minimal/README.md) covers the models, training and testing
options, hardware notes, troubleshooting, Windows setup and how to add a generator.

## License

MIT
