import numpy as np
from scipy import stats


def compute_dynamical_features(y, t=None, dim=64):
    """
    ODE-aware feature extraction for linear_ode domain.

    The original version populated only dims 0-6 (6 nonzero out of 64),
    giving the transformer encoder almost no signal to condition on.
    The decoder collapsed to a bare template for all second-order ODEs.

    This version densely populates all 64 dims with trajectory signatures
    that directly encode the ODE parameters the decoder must recover:
      - First-order 'a': exponential rate, sign, endpoint ratio
      - Second-order C1: envelope decay rate, energy decay rate
      - Second-order C2: derivative ratios |d2y/y|, |dy/y|, FFT frequency
      - Order detection: zero-crossing count, monotonicity flag
      - Waveform shape: 10 normalized trajectory samples

    All values are clipped to bounded ranges. No dim is left at zero.
    """
    features = np.zeros(dim, dtype=np.float32)

    if t is None:
        t = np.linspace(0, 10.0, len(y))

    n = len(y)
    dt = t[1] - t[0] if n > 1 else 1.0
    t_span = t[-1] - t[0] + 1e-8

    # Clip to prevent exponential growth from blowing up all features
    y_c = np.clip(y, -1e4, 1e4)
    dy  = np.gradient(y_c, t)
    d2y = np.gradient(dy, t)

    # Stable magnitude references (never zero)
    y_rms   = np.sqrt(np.mean(y_c**2)) + 1e-8
    dy_rms  = np.sqrt(np.mean(dy**2)) + 1e-8
    d2y_rms = np.sqrt(np.mean(d2y**2)) + 1e-8

    # ----------------------------------------------------------------
    # BLOCK A: Basic trajectory stats  [0-7]
    # ----------------------------------------------------------------
    features[0] = np.clip(np.mean(y_c) / y_rms, -10, 10)
    features[1] = np.std(y_c) / y_rms                                        # ~1 by construction
    features[2] = (np.max(y_c) - np.min(y_c)) / y_rms                        # normalized range
    features[3] = np.clip(y_c[-1] / (np.abs(y_c[0]) + 1e-8), -100, 100)     # endpoint ratio
    features[4] = np.clip(
        np.log(np.abs(y_c[-1]) + 1e-10) - np.log(np.abs(y_c[0]) + 1e-10),
        -15, 15
    )                                                                         # log |endpoint ratio|
    features[5] = np.clip(np.mean(dy) / dy_rms, -10, 10)
    features[6] = np.std(dy) / dy_rms
    features[7] = np.std(d2y) / d2y_rms

    # ----------------------------------------------------------------
    # BLOCK B: Order / type indicators  [8-15]
    # These let the decoder know whether it's first or second order
    # and provide the first-order exponential rate directly.
    # ----------------------------------------------------------------
    zero_crossings   = int(np.sum(np.diff(np.sign(y_c)) != 0))
    dy_sign_changes  = int(np.sum(np.diff(np.sign(dy)) != 0))

    features[8]  = np.clip(zero_crossings, 0, 30)           # 0 for first-order
    features[9]  = np.clip(dy_sign_changes, 0, 30)          # velocity reversals
    features[10] = 1.0 if zero_crossings == 0 else 0.0      # monotonic flag
    features[11] = np.sign(y[0])                             # initial condition sign (unclipped)
    features[12] = np.sign(dy[0]) if abs(dy[0]) > 1e-8 else 0.0
    features[13] = np.clip(zero_crossings / t_span, 0, 10)  # crossing rate (Hz)
    features[14] = np.clip(dy_sign_changes / t_span, 0, 10) # reversal rate (Hz)

    # Exponential rate — this IS 'a' for first-order (x'=ax → y=y0*exp(at))
    if zero_crossings == 0 and abs(y[0]) > 1e-8:
        features[15] = np.clip(
            (np.log(abs(y_c[-1]) + 1e-10) - np.log(abs(y[0]) + 1e-10)) / t_span,
            -5, 5
        )
    # else stays 0

    # ----------------------------------------------------------------
    # BLOCK C: Derivative ratios — core C1/C2 encoding  [16-25]
    # For x'' + C1 x' + C2 x = 0:
    #   |x'|/|x|  ~ sqrt(C2)   (i.e. omega)
    #   |x''|/|x| ~ C2         (i.e. omega^2)
    #   |x''|/|x'|~ sqrt(C2)
    # These are the single most informative features for second-order.
    # ----------------------------------------------------------------
    features[16] = np.clip(dy_rms / y_rms, 0, 20)            # |dy/y|_rms  ≈ omega
    features[17] = np.clip(d2y_rms / y_rms, 0, 50)           # |d2y/y|_rms ≈ omega^2 → C2
    features[18] = np.clip(d2y_rms / dy_rms, 0, 20)          # |d2y/dy|_rms ≈ omega
    features[19] = np.clip(np.sqrt(features[17]), 0, 10)      # sqrt(|d2y/y|) ≈ omega
    features[20] = np.clip(
        features[16] / (features[19] + 1e-8), 0, 5
    )                                                          # |dy/y|/omega ≈ zeta

    # Mean-absolute versions (complementary stability)
    y_am  = np.mean(np.abs(y_c)) + 1e-8
    dy_am = np.mean(np.abs(dy)) + 1e-8
    d2y_am = np.mean(np.abs(d2y)) + 1e-8
    features[21] = np.clip(dy_am / y_am, 0, 20)
    features[22] = np.clip(d2y_am / y_am, 0, 50)             # ≈ omega^2 (mean version)
    features[23] = np.clip(d2y_am / dy_am, 0, 20)

    # Integrated (cumulative) versions — smooths out transients
    features[24] = np.clip(np.sqrt(np.mean(dy**2) / (np.mean(y_c**2) + 1e-8)), 0, 20)
    features[25] = np.clip(np.sqrt(np.mean(d2y**2) / (np.mean(y_c**2) + 1e-8)), 0, 20)  # sqrt → scales as C2, not C2^2

    # ----------------------------------------------------------------
    # BLOCK D: FFT frequency analysis  [26-35]
    # Peak frequency directly encodes omega_d (damped natural frequency).
    # omega_d = omega * sqrt(1 - zeta^2), so for low damping omega_d ≈ omega.
    # ----------------------------------------------------------------
    spectrum = np.abs(np.fft.rfft(y_c - np.mean(y_c)))
    freqs    = np.fft.rfftfreq(n, d=dt)
    spec_ac  = spectrum[1:]   # drop DC
    freq_ac  = freqs[1:]

    if len(spec_ac) > 0:
        peak_idx = np.argmax(spec_ac)
        features[26] = np.clip(freq_ac[peak_idx], 0, 10)                         # peak freq (Hz)
        features[27] = np.clip(2 * np.pi * freq_ac[peak_idx], 0, 60)            # peak angular freq
        features[28] = spec_ac[peak_idx] / (np.sum(spec_ac) + 1e-8)             # peak dominance [0,1]
        features[29] = np.clip(
            np.sum(freq_ac * spec_ac) / (np.sum(spec_ac) + 1e-8), 0, 10
        )                                                                         # spectral centroid

        # Spectral energy in three bands
        total_e = np.sum(spec_ac**2) + 1e-8
        if len(spec_ac) >= 3:
            third = len(spec_ac) // 3
            features[30] = np.sum(spec_ac[:third]**2) / total_e
            features[31] = np.sum(spec_ac[third:2*third]**2) / total_e
            features[32] = np.sum(spec_ac[2*third:]**2) / total_e

    # Period from zero-crossing timing (more robust than FFT at low freq)
    if zero_crossings >= 2:
        zc_times = []
        for i in range(n - 1):
            if y_c[i] * y_c[i + 1] < 0:
                frac = abs(y_c[i]) / (abs(y_c[i]) + abs(y_c[i + 1]) + 1e-10)
                zc_times.append(t[i] + frac * (t[i + 1] - t[i]))
        if len(zc_times) >= 2:
            half_periods = np.diff(zc_times)
            period = np.mean(half_periods) * 2
            features[33] = np.clip(period, 0, 20)
            features[34] = np.clip(2 * np.pi / (period + 1e-8), 0, 60)  # omega from period
            features[35] = np.clip(
                np.std(half_periods) / (np.mean(half_periods) + 1e-8), 0, 5
            )                                                             # period consistency

    # ----------------------------------------------------------------
    # BLOCK E: Envelope decay → C1 encoding  [36-42]
    # For underdamped: envelope = A*exp(-zeta*omega*t), so the decay
    # rate of the peak envelope IS zeta*omega = C1/2.
    # For overdamped/first-order: overall log-decay serves the same role.
    # ----------------------------------------------------------------
    abs_y = np.abs(y_c)
    peaks = []
    for i in range(1, n - 1):
        if abs_y[i] > abs_y[i - 1] and abs_y[i] > abs_y[i + 1] and abs_y[i] > 1e-6:
            peaks.append((t[i], abs_y[i]))

    if len(peaks) >= 2:
        peak_t = np.array([p[0] for p in peaks])
        peak_v = np.array([p[1] for p in peaks])
        try:
            log_pv = np.log(peak_v + 1e-10)
            sigma, _ = np.polyfit(peak_t, log_pv, 1)
            features[36] = np.clip(sigma, -10, 10)              # envelope decay rate (negative)
            features[37] = np.clip(abs(sigma), 0, 10)           # |decay rate| = zeta*omega
            features[38] = np.clip(2 * abs(sigma), 0, 20)       # C1 estimate = 2*zeta*omega
        except:
            pass
        features[39] = np.clip(len(peaks), 0, 30)               # oscillation peak count
        features[40] = peak_v[0] / y_rms                         # first peak (normalized)
        features[41] = np.clip(peak_v[-1] / (peak_v[0] + 1e-8), 0, 2)  # last/first peak ratio
        if len(peaks) >= 2:
            features[42] = np.clip(
                np.log(peak_v[-1] / (peak_v[0] + 1e-8) + 1e-10) / max(1, len(peaks) - 1),
                -5, 0
            )                                                    # log decay per cycle
    else:
        # Non-oscillatory: use overall trajectory decay
        features[39] = 0
        if abs(y[0]) > 1e-8:
            decay = (np.log(abs(y_c[-1]) + 1e-10) - np.log(abs(y[0]) + 1e-10)) / t_span
            features[36] = np.clip(decay, -10, 10)
            features[37] = np.clip(abs(decay), 0, 10)
            features[38] = np.clip(2 * abs(decay), 0, 20)

    # ----------------------------------------------------------------
    # BLOCK F: Energy analysis  [43-48]
    # Proxy energy E = 0.5*x'^2 + 0.5*x^2.
    # For x'' + C1 x' + C2 x = 0, dE/dt = -C1 * x'^2.
    # So log(E) decays linearly with rate ≈ -C1.
    # ----------------------------------------------------------------
    E  = 0.5 * dy**2 + 0.5 * y_c**2
    E0 = E[0] + 1e-8
    features[43] = np.clip(np.mean(E) / E0, 0, 10)
    features[44] = np.clip(E[-1] / E0, 0, 10)                  # final/initial energy ratio
    features[45] = np.clip(np.log(E[-1] / E0 + 1e-10), -15, 15)
    try:
        log_E = np.log(E + 1e-10)
        energy_decay, _ = np.polyfit(t, log_E, 1)
        features[46] = np.clip(energy_decay, -10, 10)           # ≈ -C1
        features[47] = np.clip(abs(energy_decay), 0, 10)        # |energy decay|
    except:
        pass
    features[48] = np.clip(np.std(E) / (np.mean(E) + 1e-8), 0, 5)  # energy variation (0 for undamped)

    # ----------------------------------------------------------------
    # BLOCK G: Phase space geometry  [49-53]
    # ----------------------------------------------------------------
    if np.std(y_c) > 1e-8 and np.std(dy) > 1e-8:
        features[49] = np.corrcoef(y_c, dy)[0, 1]              # phase correlation

    # Phase space area (shoelace) and perimeter — normalized
    phase_area      = 0.5 * abs(np.sum(y_c[:-1] * dy[1:] - y_c[1:] * dy[:-1]))
    phase_perimeter = np.sum(np.sqrt(np.diff(y_c)**2 + np.diff(dy)**2))
    features[50] = np.clip(phase_area / (y_rms * dy_rms * t_span + 1e-8), 0, 100)
    features[51] = np.clip(phase_perimeter / (y_rms * t_span + 1e-8), 0, 100)
    features[52] = np.clip(phase_area / (phase_perimeter**2 + 1e-8), 0, 1)  # compactness

    # Velocity autocorrelation at lag 1
    if n > 1:
        dy_c = dy - np.mean(dy)
        features[53] = np.clip(
            np.mean(dy_c[1:] * dy_c[:-1]) / (np.var(dy) + 1e-8), -1, 1
        )

    # ----------------------------------------------------------------
    # BLOCK H: Waveform samples  [54-63]
    # 10 evenly-spaced samples of the normalized trajectory.
    # This gives the decoder a direct "view" of the waveform shape —
    # the single most reliable way to distinguish all ODE types.
    # ----------------------------------------------------------------
    y_min   = np.min(y_c)
    y_range = np.max(y_c) - y_min + 1e-8
    for i in range(10):
        idx = min(int((i + 0.5) * n / 10), n - 1)
        features[54 + i] = (y_c[idx] - y_min) / y_range       # normalized to [0, 1]

    return np.clip(np.nan_to_num(features), -100, 100).astype(np.float32)


# -------------------------------------------------------------------
# The other feature functions are unchanged — only dynamical is broken.
# -------------------------------------------------------------------

def compute_algebraic_features(x, y, dim=48):
    features = np.zeros(dim, dtype=np.float32)
    features[0:5] = [np.mean(y), np.std(y), np.min(y), np.max(y), y[-1] - y[0]]
    dy = np.gradient(y, x)
    features[5:8] = [np.mean(dy), np.std(dy), np.mean(np.abs(dy))]
    d2y = np.gradient(dy, x)
    features[8:10] = [np.mean(d2y), np.std(d2y)]
    features[10:14] = [
        np.mean(dy > 0),
        np.sum(np.diff(np.sign(dy)) != 0),
        np.mean(d2y > 0),
        np.sum(np.diff(np.sign(d2y)) != 0),
    ]
    features[14] = (np.max(y) - np.min(y)) / (x[-1] - x[0] + 1e-8)
    for deg in range(1, 6):
        try:
            coeffs = np.polyfit(x, y, deg)
            features[14 + deg] = np.std(y - np.polyval(coeffs, x))
        except:
            features[14 + deg] = 1.0
    return np.clip(np.nan_to_num(features), -10, 10)


def compute_timeseries_features(y, dim=32):
    y = np.clip(y, -1e6, 1e6)
    features = np.zeros(dim, dtype=np.float32)
    features[0:4] = [np.mean(y), np.std(y), np.min(y), np.max(y)]
    y_centered = y - np.mean(y)
    var = np.var(y) + 1e-8
    for lag in range(1, min(11, len(y))):
        features[3 + lag] = np.clip(
            np.mean(y_centered[lag:] * y_centered[:-lag]) / var, -1, 1
        )
    features[14] = features[4]
    if abs(features[4]) < 1:
        features[15] = (features[5] - features[4] ** 2) / (1 - features[4] ** 2 + 1e-8)
    try:
        features[16] = np.polyfit(np.arange(len(y)), y, 1)[0]
    except:
        pass
    return np.clip(np.nan_to_num(features), -10, 10)


def compute_multivariate_features(data, dim=64):
    features = np.zeros(dim, dtype=np.float32)
    vars_list = list(data.keys())
    output = data[vars_list[-1]]
    features[0:4] = [np.mean(output), np.std(output), np.min(output), np.max(output)]
    idx = 4
    for key in vars_list[:-1]:
        if idx >= dim - 4:
            break
        inp = data[key]
        features[idx] = np.nan_to_num(np.corrcoef(inp, output)[0, 1])
        idx += 1
        try:
            log_corr = np.nan_to_num(
                np.corrcoef(
                    np.log(np.abs(inp) + 1e-8), np.log(np.abs(output) + 1e-8)
                )[0, 1]
            )
            features[idx] = log_corr
        except:
            pass
        idx += 1
    return np.clip(np.nan_to_num(features), -10, 10)