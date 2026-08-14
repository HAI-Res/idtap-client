"""Sample-level DSP kernels for offline synthesis.

These are faithful ports of the IDTAP web app's AudioWorklet synths
(karplusStrong2, chikaris4, sarangi, klattSynth2/klatt-syn), with two
deliberate improvements over the originals:

- fractional (linearly interpolated) delay-line reads, so string tuning is
  exact at all pitches (the worklets quantized delay length to whole samples);
- the actual output sample rate is used everywhere (the sarangi worklet
  hardcoded 48 kHz).

Control-rate parameter curves (f0, gain, cutoff, formants) are passed in as
arrays sampled every ``hop`` samples; kernels interpolate f0 linearly between
control frames and hold other parameters, mirroring how the web app drove
AudioParams with setValueCurveAtTime.

All kernels are numba-jitted when numba is available (see _nb.py).
"""
from __future__ import annotations

import math

import numpy as np

from ._nb import njit


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

@njit(cache=True)
def _ctrl_interp(arr, i, hop):
    """Linearly interpolate a control-rate array at sample index i."""
    pos = i / hop
    k = int(pos)
    if k >= arr.shape[0] - 1:
        return arr[arr.shape[0] - 1]
    frac = pos - k
    return arr[k] * (1.0 - frac) + arr[k + 1] * frac


@njit(cache=True)
def _ctrl_hold(arr, i, hop):
    """Sample-and-hold a control-rate array at sample index i."""
    k = int(i / hop)
    if k >= arr.shape[0]:
        k = arr.shape[0] - 1
    return arr[k]


# Extended Karplus-Strong string constants. The round-trip gain stays below
# unity at both extremes, so the loop is stable however it is driven: a
# freely ringing string decays slowly, a damped one quickly, and neither
# can accumulate.
# Seconds to -60 dB. A sitar's main string rings for a long time — the note
# has to carry across a whole meend — so the free end of this range is much
# longer than a guitar's would be.
KS_T60_MAX = 24.0         # ringing freely
KS_T60_MIN = 0.15         # fully damped, i.e. a hand on the string
# Stiffness allpass coefficient. Real strings are dispersive — high
# partials travel faster — which stretches the harmonic series and gives a
# plucked steel string its shimmer. 0 disables it.
KS_STIFFNESS = 0.22
# Loop lowpass coefficient at full ring; higher is darker.
KS_BRIGHTNESS = 0.35


@njit(cache=True)
def pink_burst(n, attack_n, amp, seed):
    """Pink-noise burst (Paul Kellet filter), linear attack ramp.

    Port of sendBurst() in Synths.vue (amp is pre-doubled there; caller
    passes the final amplitude here).

    The mean is removed before returning. Kellet's lowest band is very
    nearly an integrator, so the raw burst carries a large DC offset — and
    a plucked string cannot have net displacement anyway, so injecting one
    is unphysical. It is also unbounded: the string loop's filter has unity
    DC gain, so any offset integrates for the rest of the render.
    """
    np.random.seed(seed)
    out = np.empty(n, dtype=np.float64)
    b0 = 0.0
    b1 = 0.0
    b2 = 0.0
    b3 = 0.0
    b4 = 0.0
    b5 = 0.0
    b6 = 0.0
    for i in range(n):
        white = np.random.random() * 2.0 - 1.0
        b0 = 0.99886 * b0 + white * 0.0555179
        b1 = 0.99332 * b1 + white * 0.0750759
        b2 = 0.96900 * b2 + white * 0.1538520
        b3 = 0.86650 * b3 + white * 0.3104856
        b4 = 0.55000 * b4 + white * 0.5329522
        b5 = -0.7616 * b5 - white * 0.0168980
        out[i] = (b0 + b1 + b2 + b3 + b4 + b5 + b6 + white * 0.5362) * 0.11
        b6 = white * 0.115926
    ramp_n = attack_n if attack_n < n else n
    for i in range(ramp_n):
        out[i] *= i / ramp_n
    mean = 0.0
    for i in range(n):
        mean += out[i]
    mean /= n
    for i in range(n):
        out[i] = (out[i] - mean) * amp
    return out


@njit(cache=True)
def dc_blocker(x, sr):
    """First-order highpass at ~5 Hz (the web app's DC-offset biquads)."""
    out = np.empty_like(x)
    rc = 1.0 / (2.0 * math.pi * 5.0)
    dt = 1.0 / sr
    alpha = rc / (rc + dt)
    y1 = 0.0
    x1 = 0.0
    for i in range(x.shape[0]):
        y1 = alpha * (y1 + x[i] - x1)
        x1 = x[i]
        out[i] = y1
    return out


@njit(cache=True)
def tracking_lowpass(x, freq_ctrl, sr, hop):
    """Biquad lowpass whose cutoff tracks a control-rate frequency array.

    Mirrors the web app's lpNode (BiquadFilterNode, Q = default 1) whose
    frequency followed 8x the sitar f0. Coefficients are recomputed per
    control frame.
    """
    out = np.empty_like(x)
    n = x.shape[0]
    b0 = 0.0
    b1 = 0.0
    b2 = 0.0
    a1 = 0.0
    a2 = 0.0
    x1 = 0.0
    x2 = 0.0
    y1 = 0.0
    y2 = 0.0
    last_k = -1
    q = 1.0
    nyq = sr * 0.499
    for i in range(n):
        k = int(i / hop)
        if k >= freq_ctrl.shape[0]:
            k = freq_ctrl.shape[0] - 1
        if k != last_k:
            f = freq_ctrl[k]
            if f < 10.0:
                f = 10.0
            if f > nyq:
                f = nyq
            w0 = 2.0 * math.pi * f / sr
            alpha = math.sin(w0) / (2.0 * q)
            cw = math.cos(w0)
            a0 = 1.0 + alpha
            b0 = ((1.0 - cw) / 2.0) / a0
            b1 = (1.0 - cw) / a0
            b2 = ((1.0 - cw) / 2.0) / a0
            a1 = (-2.0 * cw) / a0
            a2 = (1.0 - alpha) / a0
            last_k = k
        y = b0 * x[i] + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        x2 = x1
        x1 = x[i]
        y2 = y1
        y1 = y
        out[i] = y
    return out


# ---------------------------------------------------------------------------
# Karplus-Strong string (sitar main/jor string, chikari strings)
# ---------------------------------------------------------------------------

@njit(cache=True)
def ks_string(f0_ctrl, cutoff_ctrl, excitation, sr, hop, amp,
              stiffness=KS_STIFFNESS, brightness=KS_BRIGHTNESS):
    """Extended Karplus-Strong plucked string (Jaffe & Smith 1983).

    The worklet's loop could not be made safe by tuning it: its loop filter
    had a DC gain of exactly 1, so the delay loop was a perfect integrator
    at DC and any bias grew without bound — a half-hour render reached
    1e158 — while its "dampen" set the filter coefficient to zero, which
    freezes the filter's state rather than damping the string, injecting a
    constant. This is the standard extended form instead, which is stable
    by construction rather than by clamping:

    - the loop filter is a one-pole lowpass scaled by an explicit gain, so
      the round-trip gain is below unity at every frequency, and decay time
      rather than filter state is what damping controls;
    - a first-order allpass adds stiffness, stretching the partials into
      the inharmonic shimmer that a plucked steel string actually has;
    - delay length is compensated for the phase delay of both filters, so
      the string is in tune at every pitch and damping setting.

    ``cutoff_ctrl`` keeps its old meaning as a control curve: 1 is a freely
    ringing string and 0 a fully damped one.
    """
    n = excitation.shape[0]
    out = np.empty(n, dtype=np.float64)
    size = 8192
    buf = np.zeros(size, dtype=np.float64)
    wp = 0
    lp = 0.0            # loop lowpass state
    ap_x1 = 0.0         # allpass input history
    ap_y1 = 0.0         # allpass output history
    two_pi = 2.0 * math.pi
    for i in range(n):
        f0 = _ctrl_interp(f0_ctrl, i, hop)
        if f0 < 20.0:
            f0 = 20.0
        damp_ctrl = _ctrl_hold(cutoff_ctrl, i, hop)
        if damp_ctrl < 0.0:
            damp_ctrl = 0.0
        elif damp_ctrl > 1.0:
            damp_ctrl = 1.0

        # Round-trip gain set from a target decay time rather than chosen as
        # a bare coefficient: the signal passes the filter once per period,
        # so g^(f0*T60) = 1e-3 gives the gain for a given T60. Always < 1,
        # so the loop cannot run away, and a damped string decays instead of
        # freezing. Higher notes decaying faster in real time falls out of
        # this naturally, as on a real string.
        t60 = KS_T60_MIN + (KS_T60_MAX - KS_T60_MIN) * damp_ctrl
        g = 10.0 ** (-3.0 / (f0 * t60))
        if g > 0.99999:
            g = 0.99999
        # brightness of the loop lowpass (0 = no filtering, -> 1 = dark)
        b = brightness * (1.0 - 0.5 * damp_ctrl)
        if b < 0.0:
            b = 0.0
        elif b > 0.9:
            b = 0.9

        w = two_pi * f0 / sr
        # phase delay of the one-pole lowpass (1-b)/(1 - b z^-1)
        pd = math.atan2(b * math.sin(w), 1.0 - b * math.cos(w)) / w
        # phase delay of the stiffness allpass (a + z^-1)/(1 + a z^-1)
        a = stiffness
        if a != 0.0:
            num = math.sin(w) * (1.0 - a * a)
            den = 2.0 * a + (1.0 + a * a) * math.cos(w)
            pd += math.atan2(num, den) / w
        d = sr / f0 - pd
        # never let compensation eat more than most of the period
        min_d = 2.0
        if d < min_d:
            d = min_d
        if d > size - 4:
            d = size - 4.0

        rpos = wp - d
        if rpos < 0.0:
            rpos += size
        r0 = int(rpos)
        frac = rpos - r0
        r1 = r0 + 1
        if r1 >= size:
            r1 -= size
        delayed = buf[r0] * (1.0 - frac) + buf[r1] * frac

        # stiffness allpass: disperses partials, sharpening the attack into
        # the metallic shimmer of a plucked steel string
        if a != 0.0:
            ap_y = a * delayed + ap_x1 - a * ap_y1
            ap_x1 = delayed
            ap_y1 = ap_y
            delayed = ap_y

        # loop lowpass with explicit gain: DC gain is g, hence stable
        lp = (1.0 - b) * delayed + b * lp
        x = excitation[i] + g * lp
        buf[wp] = x
        wp += 1
        if wp >= size:
            wp = 0
        out[i] = amp * x
    return out



# --- bowed sarangi string -------------------------------------------------
# Where the bow crosses the string, as a fraction of its length from the
# bridge. Sarangi bowing sits close to the bridge, which brightens the tone.
SARANGI_BOW_POSITION = 0.13
# Bridge reflection: lossy and inverting. Below unity so the string decays
# when the bow leaves it.
SARANGI_BRIDGE_GAIN = 0.985
SARANGI_BRIDGE_DAMP = 0.45      # how dark the bridge reflection is
SARANGI_NUT_GAIN = 0.99
# Bow speed at full bow control, and the slope of the friction curve, which
# together set how readily the hair breaks away into slipping.
SARANGI_BOW_SPEED = 0.22
SARANGI_FRICTION_SLOPE = 3.0
# Sympathetic strings. A real sarangi has dozens; a smaller bank tuned to
# scale degrees gives the same halo far more cheaply.
SARANGI_N_TARAF = 11
SARANGI_TARAF_DRIVE = 0.06      # how hard the bridge drives them
SARANGI_TARAF_T60 = 2.5         # seconds; they ring on undamped
SARANGI_TARAF_MIX = 0.5
SARANGI_BODY_MIX = 0.35         # parchment belly resonance in the output
SARANGI_OUT_GAIN = 2.0

# ---------------------------------------------------------------------------
# Sarangi (port of sarangi.worklet.js)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _biquad_bandpass_coeffs(sr, freq, q):
    omega = 2.0 * math.pi * freq / sr
    beta = math.sin(omega) / (2.0 * q)
    b0 = (math.sin(omega) / 2.0) / (1.0 + beta)
    b2 = -b0
    a1 = -2.0 * math.cos(omega) / (1.0 + beta)
    a2 = (1.0 - beta) / (1.0 + beta)
    return b0, 0.0, b2, a1, a2


@njit(cache=True)
def _bow_friction(delta_v, force):
    """Bow-hair friction as a function of bow-string velocity difference.

    The bow alternately sticks to the string and slips, and it is that
    stick-slip cycle — not any filtered noise — that makes a bowed string
    speak. While the relative velocity is small the hair grips and the
    string is dragged with the bow; past a breakaway velocity the grip
    collapses and friction falls off steeply. This is the standard
    memoryless friction curve of McIntyre, Schumacher & Woodhouse (1983),
    in the form Smith uses for the digital waveguide bow.
    """
    a = abs(delta_v * force)
    c = (a + 0.75) ** -4.0
    if c > 1.0:
        c = 1.0
    return c


@njit(cache=True)
def sarangi_string(f0_ctrl, bow_ctrl, gain_ctrl, n, sr, hop, seed,
                   bow_position=SARANGI_BOW_POSITION,
                   n_sympathetic=SARANGI_N_TARAF):
    """Bowed sarangi string: friction-driven digital waveguide.

    The worklet this replaces excited a feedback loop with bandpassed
    noise, which is not bowing — it is a resonator hissed at, and it can
    never produce Helmholtz motion. Here the string is two waveguide
    delay lines meeting at the bow, and the bow is a nonlinear scattering
    junction between them: at every sample the friction curve decides
    whether the hair is gripping the string or slipping across it. The
    sawtooth-like Helmholtz corner that gives a bowed string its tone
    emerges from that, as it does on the instrument.

    A sarangi also carries dozens of undamped sympathetic strings, and
    the resulting halo is most of what makes it recognizable, so a bank of
    them is driven from the bridge.

    ``bow_ctrl`` is bow force/speed (0 = off the string), ``gain_ctrl``
    the output level.
    """
    np.random.seed(seed)
    out = np.empty(n, dtype=np.float64)
    size = 8192
    # string either side of the bow: nut-side and bridge-side
    neck = np.zeros(size, dtype=np.float64)
    bridge = np.zeros(size, dtype=np.float64)
    wp_n = 0
    wp_b = 0
    bridge_lp = 0.0

    # body resonances (the sarangi's parchment belly), as in the original
    res_freqs = np.array([185.0, 275.0, 405.0, 460.0, 530.0])
    n_res = 5
    rc = np.zeros((n_res, 5), dtype=np.float64)
    for j in range(n_res):
        b0, b1, b2, a1, a2 = _biquad_bandpass_coeffs(sr, res_freqs[j], 1.0)
        rc[j, 0] = b0
        rc[j, 1] = b1
        rc[j, 2] = b2
        rc[j, 3] = a1
        rc[j, 4] = a2
    rx1 = np.zeros(n_res, dtype=np.float64)
    rx2 = np.zeros(n_res, dtype=np.float64)
    ry1 = np.zeros(n_res, dtype=np.float64)
    ry2 = np.zeros(n_res, dtype=np.float64)

    # sympathetic strings: comb resonators fed from the bridge, tuned to
    # scale degrees around the played fundamental
    ratios = np.array([0.5, 0.5625, 0.6667, 0.75, 0.8333, 1.0,
                       1.125, 1.3333, 1.5, 1.6667, 2.0])
    n_symp = n_sympathetic
    if n_symp > 11:
        n_symp = 11
    symp = np.zeros((11, size), dtype=np.float64)
    symp_wp = np.zeros(11, dtype=np.int64)
    symp_lp = np.zeros(11, dtype=np.float64)

    for i in range(n):
        f0 = _ctrl_interp(f0_ctrl, i, hop)
        if f0 < 20.0:
            f0 = 20.0
        bow = _ctrl_interp(bow_ctrl, i, hop)
        if bow < 0.0:
            bow = 0.0

        period = sr / f0
        d_bridge = period * bow_position
        d_neck = period - d_bridge
        if d_bridge < 2.0:
            d_bridge = 2.0
        if d_neck < 2.0:
            d_neck = 2.0
        if d_neck > size - 4:
            d_neck = size - 4.0

        # travelling waves arriving at the bow from each side
        rpos = wp_b - d_bridge
        if rpos < 0.0:
            rpos += size
        r0 = int(rpos)
        frac = rpos - r0
        r1 = r0 + 1
        if r1 >= size:
            r1 -= size
        from_bridge = bridge[r0] * (1.0 - frac) + bridge[r1] * frac

        rpos = wp_n - d_neck
        if rpos < 0.0:
            rpos += size
        r0 = int(rpos)
        frac = rpos - r0
        r1 = r0 + 1
        if r1 >= size:
            r1 -= size
        from_neck = neck[r0] * (1.0 - frac) + neck[r1] * frac

        # bridge is lossy and inverting, nut is a near-perfect inversion
        bridge_lp = (1.0 - SARANGI_BRIDGE_DAMP) * from_bridge \
            + SARANGI_BRIDGE_DAMP * bridge_lp
        refl_bridge = -SARANGI_BRIDGE_GAIN * bridge_lp
        refl_neck = -SARANGI_NUT_GAIN * from_neck

        # the bow: stick or slip, decided every sample
        string_vel = refl_bridge + refl_neck
        bow_vel = bow * SARANGI_BOW_SPEED
        delta_v = bow_vel - string_vel
        coeff = _bow_friction(delta_v, SARANGI_FRICTION_SLOPE)
        injected = delta_v * coeff

        neck[wp_n] = refl_bridge + injected
        bridge[wp_b] = refl_neck + injected
        wp_n += 1
        if wp_n >= size:
            wp_n = 0
        wp_b += 1
        if wp_b >= size:
            wp_b = 0

        # bridge force drives body and sympathetic strings
        drive = refl_bridge

        symp_sum = 0.0
        for s in range(n_symp):
            sd = period / ratios[s]
            if sd < 2.0:
                sd = 2.0
            if sd > size - 4:
                sd = size - 4.0
            wp_s = symp_wp[s]
            rpos = wp_s - sd
            if rpos < 0.0:
                rpos += size
            r0 = int(rpos)
            frac = rpos - r0
            r1 = r0 + 1
            if r1 >= size:
                r1 -= size
            sv = symp[s, r0] * (1.0 - frac) + symp[s, r1] * frac
            symp_lp[s] = 0.7 * sv + 0.3 * symp_lp[s]
            g_s = 10.0 ** (-3.0 / ((f0 * ratios[s]) * SARANGI_TARAF_T60))
            val = drive * SARANGI_TARAF_DRIVE + g_s * symp_lp[s]
            symp[s, wp_s] = val
            symp_wp[s] = wp_s + 1
            if symp_wp[s] >= size:
                symp_wp[s] = 0
            symp_sum += val
        if n_symp > 0:
            symp_sum /= n_symp

        sig = drive + SARANGI_TARAF_MIX * symp_sum

        res = 0.0
        for j in range(n_res):
            y = (rc[j, 0] * sig + rc[j, 1] * rx1[j] + rc[j, 2] * rx2[j]
                 - rc[j, 3] * ry1[j] - rc[j, 4] * ry2[j])
            rx2[j] = rx1[j]
            rx1[j] = sig
            ry2[j] = ry1[j]
            ry1[j] = y
            res += y
        body = SARANGI_BODY_MIX * res + (1.0 - SARANGI_BODY_MIX) * sig

        out[i] = body * _ctrl_interp(gain_ctrl, i, hop) * SARANGI_OUT_GAIN
    return out


# ---------------------------------------------------------------------------
# Klatt voice (port of chdh's klatt-syn as bundled in klattSynth2.worklet.js)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _db_to_lin(db):
    if db <= -99.0 or np.isnan(db):
        return 0.0
    return 10.0 ** (db / 20.0)


# --- extra_ctrl row layout -------------------------------------------------
# klatt_voice() takes the many per-frame source/nasal parameters as a single
# 2D float64 array of shape (N_EXTRA_CTRL_ROWS, n_frames). Rows are sampled
# (held, not interpolated) at the start of each F0 period, exactly like the
# oral formant arrays. Levels are in dB; <= -99 means "off" (linear 0).
ROW_CASC_VOICING_DB = 0        # cascade branch voicing level
ROW_CASC_ASPIRATION_DB = 1     # cascade branch aspiration (glottis noise)
ROW_CASC_ASPIRATION_MOD = 2    # cascade aspiration modulation, 0 .. 1
ROW_NASAL_FORMANT_FREQ = 3     # nasal formant freq (Hz), <= 0 = off
ROW_NASAL_FORMANT_BW = 4       # nasal formant bandwidth (Hz)
ROW_NASAL_ANTIFORMANT_FREQ = 5  # cascade nasal antiformant freq, <= 0 = off
ROW_NASAL_ANTIFORMANT_BW = 6   # cascade nasal antiformant bandwidth
ROW_PAR_VOICING_DB = 7         # parallel branch voicing level
ROW_PAR_ASPIRATION_DB = 8      # parallel branch aspiration level
ROW_PAR_ASPIRATION_MOD = 9     # parallel aspiration modulation, 0 .. 1
ROW_FRICATION_DB = 10          # parallel frication noise level
ROW_FRICATION_MOD = 11         # frication modulation, 0 .. 1
ROW_PAR_BYPASS_DB = 12         # parallel bypass level
ROW_PAR_NASAL_FORMANT_DB = 13  # parallel nasal formant level (peak gain)
N_EXTRA_CTRL_ROWS = 14

OFF_DB = -99.0

# resonator / antiresonator modes
_MODE_PASSTHROUGH = 0
_MODE_ACTIVE = 1
_MODE_MUTE = 2


def default_extra_ctrl(n_frames, cascade_voicing_db=0.0,
                       cascade_aspiration_db=-25.0,
                       cascade_aspiration_mod=0.5):
    """Neutral extra_ctrl array: cascade branch only, nasals and parallel off.

    Reproduces the web worklet's defaults (parallelEnabled = 0, no nasal
    formant/antiformant), i.e. exactly the behaviour of the cascade-only
    kernel.
    """
    a = np.zeros((N_EXTRA_CTRL_ROWS, n_frames), dtype=np.float64)
    a[ROW_CASC_VOICING_DB, :] = cascade_voicing_db
    a[ROW_CASC_ASPIRATION_DB, :] = cascade_aspiration_db
    a[ROW_CASC_ASPIRATION_MOD, :] = cascade_aspiration_mod
    # nasal formant / antiformant frequencies of 0 mean "passthrough"
    a[ROW_PAR_VOICING_DB, :] = OFF_DB
    a[ROW_PAR_ASPIRATION_DB, :] = OFF_DB
    a[ROW_PAR_ASPIRATION_MOD, :] = 0.5
    a[ROW_FRICATION_DB, :] = OFF_DB
    a[ROW_FRICATION_MOD, :] = 0.5
    a[ROW_PAR_BYPASS_DB, :] = OFF_DB
    a[ROW_PAR_NASAL_FORMANT_DB, :] = OFF_DB
    return a


def default_par_formant_db_ctrl(n_frames):
    """Neutral (all muted) parallel oral formant level array, shape (6, N)."""
    return np.full((6, n_frames), OFF_DB, dtype=np.float64)


@njit(cache=True)
def _reson_coeffs(f, bw, sr):
    """klatt-syn Resonator.set() with dcGain = 1; returns (a, b, c, r)."""
    r = math.exp(-math.pi * bw / sr)
    c = -(r * r)
    b = 2.0 * r * math.cos(2.0 * math.pi * f / sr)
    a = 1.0 - b - c
    return a, b, c, r


@njit(cache=True)
def _antireson_coeffs(f, bw, sr):
    """klatt-syn AntiResonator.set(); returns (a, b, c) with the 1/a0 inversion."""
    r = math.exp(-math.pi * bw / sr)
    c0 = -(r * r)
    b0 = 2.0 * r * math.cos(2.0 * math.pi * f / sr)
    a0 = 1.0 - b0 - c0
    if a0 == 0.0:
        return 0.0, 0.0, 0.0
    return 1.0 / a0, -b0 / a0, -c0 / a0


@njit(cache=True)
def klatt_voice(f0_ctrl, gain_ctrl, formant_ctrl, bw_ctrl, extra_ctrl,
                par_formant_db_ctrl, n, sr, hop,
                flutter_level, open_phase_ratio, breathiness_db,
                seed, flutter_offset):
    """Klatt synthesizer (cascade + parallel branch), KLGLOTT88 source.

    Faithful port of klatt-syn's Generator with the natural glottal source
    and tiltDb = 0 / gainDb = 0 (per-sample ``gain_ctrl`` replaces gainLin,
    matching the web worklet's extGain).

    Args:
        f0_ctrl: (n_frames,) fundamental frequency in Hz, 0 = silence.
        gain_ctrl: (n_frames,) linear output gain, interpolated per sample.
        formant_ctrl, bw_ctrl: (6, n_frames) oral formant frequencies and
            bandwidths in Hz, shared by both branches (as in klatt-syn).
            freq <= 0 (or bw <= 0) means passthrough in the cascade branch
            and mute in the parallel branch.
        extra_ctrl: (N_EXTRA_CTRL_ROWS, n_frames), see the ROW_* constants.
        par_formant_db_ctrl: (6, n_frames) parallel oral formant peak levels
            in dB; <= -99 mutes the resonator.
        n, sr, hop: sample count, sample rate, control-array hop in samples.
        flutter_level, open_phase_ratio, breathiness_db: frame-constant
            parameters (as in the worklet).
        seed, flutter_offset: RNG seed and flutter time offset.

    The cascade branch is always computed. The parallel branch is only
    computed when at least one of its source levels (voicing, aspiration,
    frication) is nonzero — this replaces klatt-syn's ``parallelEnabled``
    flag and avoids a separate boolean array.

    Noise draw order per sample (fixed, for reproducibility): breathiness
    white noise (only within the glottal open phase), cascade aspiration
    white noise (always, even when the level is 0), then — only while the
    parallel branch is active — parallel aspiration and parallel frication
    white noise. Each of the three LP noise sources keeps its own filter
    state, as klatt-syn uses three independent LpNoiseSource instances.
    """
    np.random.seed(seed)
    out = np.empty(n, dtype=np.float64)

    # noise sources: LpFilter1 matched to klatt-syn's LpNoiseSource. All
    # three sources share these coefficients but keep separate state.
    old_b = 0.75
    g = (1.0 - old_b) / math.sqrt(
        1.0 - 2.0 * old_b * math.cos(2.0 * math.pi * 1000.0 / 10000.0)
        + old_b ** 2)
    extra_gain = 2.5 * (sr / 10000.0) ** 0.33
    w = 2.0 * math.pi * 1000.0 / sr
    q = (1.0 - g ** 2 * math.cos(w)) / (1.0 - g ** 2)
    asp_b = q - math.sqrt(q ** 2 - 1.0)
    asp_a = (1.0 - asp_b) * extra_gain
    asp_casc_y1 = 0.0
    asp_par_y1 = 0.0
    fric_par_y1 = 0.0

    # output LP filter: Resonator.set(0, sr/2)
    r_out = math.exp(-math.pi * (sr / 2.0) / sr)
    ob = 2.0 * r_out  # cos(0) = 1
    oc = -(r_out ** 2)
    oa = 1.0 - ob - oc
    oy1 = 0.0
    oy2 = 0.0

    # cascade oral formant resonators
    fa = np.zeros(6, dtype=np.float64)
    fb = np.zeros(6, dtype=np.float64)
    fc = np.zeros(6, dtype=np.float64)
    fy1 = np.zeros(6, dtype=np.float64)
    fy2 = np.zeros(6, dtype=np.float64)
    fmode = np.zeros(6, dtype=np.int64)

    # cascade nasal formant resonator
    nfa = 1.0
    nfb = 0.0
    nfc = 0.0
    nfy1 = 0.0
    nfy2 = 0.0
    nfmode = _MODE_PASSTHROUGH

    # cascade nasal antiformant (FIR)
    naa = 1.0
    nab = 0.0
    nac = 0.0
    nax1 = 0.0
    nax2 = 0.0
    namode = _MODE_PASSTHROUGH

    # parallel nasal formant resonator
    pna = 0.0
    pnb = 0.0
    pnc = 0.0
    pny1 = 0.0
    pny2 = 0.0
    pnmode = _MODE_MUTE

    # parallel oral formant resonators
    pa = np.zeros(6, dtype=np.float64)
    pb = np.zeros(6, dtype=np.float64)
    pc = np.zeros(6, dtype=np.float64)
    py1 = np.zeros(6, dtype=np.float64)
    py2 = np.zeros(6, dtype=np.float64)
    pmode = np.zeros(6, dtype=np.int64)
    for j in range(6):
        pmode[j] = _MODE_MUTE

    # parallel differencing filter state
    diff_x1 = 0.0

    breathiness_lin = _db_to_lin(breathiness_db)

    # frame state (refreshed at the start of every F0 period)
    cascade_voicing_lin = 0.0
    cascade_aspiration_lin = 0.0
    cascade_aspiration_mod = 0.0
    par_voicing_lin = 0.0
    par_aspiration_lin = 0.0
    par_aspiration_mod = 0.0
    frication_lin = 0.0
    frication_mod = 0.0
    par_bypass_lin = 0.0
    par_enabled = False

    # natural glottal source state
    gs_x = 0.0
    gs_a = 0.0
    gs_b = 0.0

    period_length = 0
    open_phase_length = 0
    pos_in_period = 0
    nyq = sr * 0.499

    n_extra = extra_ctrl.shape[1]
    n_pardb = par_formant_db_ctrl.shape[1]

    for i in range(n):
        if period_length == 0 or pos_in_period >= period_length:
            # --- start new period: pick up frame params from control arrays
            k = int(i / hop)
            if k >= f0_ctrl.shape[0]:
                k = f0_ctrl.shape[0] - 1
            kx = k if k < n_extra else n_extra - 1
            kp = k if k < n_pardb else n_pardb - 1
            f0 = f0_ctrl[k]
            # flutter modulation
            if flutter_level > 0.0 and f0 > 0.0:
                t = i / sr + flutter_offset
                wt = 2.0 * math.pi * t
                fm = (math.sin(12.7 * wt) + math.sin(7.1 * wt)
                      + math.sin(4.7 * wt))
                f0 = f0 * (1.0 + fm * flutter_level / 50.0)
            if f0 > 0.0:
                period_length = int(round(sr / f0))
                if period_length < 1:
                    period_length = 1
            else:
                period_length = 1
            if period_length > 1:
                open_phase_length = int(round(
                    period_length * open_phase_ratio))
            else:
                open_phase_length = 0
            pos_in_period = 0

            # glottal source period start (natural / KLGLOTT88)
            gs_x = 0.0
            if open_phase_length > 0:
                gs_b = -5.0 / (open_phase_length ** 2)
                gs_a = -gs_b * open_phase_length / 3.0
            else:
                gs_b = 0.0
                gs_a = 0.0

            # --- source levels
            cascade_voicing_lin = _db_to_lin(
                extra_ctrl[ROW_CASC_VOICING_DB, kx])
            cascade_aspiration_lin = _db_to_lin(
                extra_ctrl[ROW_CASC_ASPIRATION_DB, kx])
            cascade_aspiration_mod = extra_ctrl[ROW_CASC_ASPIRATION_MOD, kx]
            par_voicing_lin = _db_to_lin(extra_ctrl[ROW_PAR_VOICING_DB, kx])
            par_aspiration_lin = _db_to_lin(
                extra_ctrl[ROW_PAR_ASPIRATION_DB, kx])
            par_aspiration_mod = extra_ctrl[ROW_PAR_ASPIRATION_MOD, kx]
            frication_lin = _db_to_lin(extra_ctrl[ROW_FRICATION_DB, kx])
            frication_mod = extra_ctrl[ROW_FRICATION_MOD, kx]
            par_bypass_lin = _db_to_lin(extra_ctrl[ROW_PAR_BYPASS_DB, kx])
            par_enabled = (par_voicing_lin > 0.0 or par_aspiration_lin > 0.0
                           or frication_lin > 0.0)

            # --- cascade nasal antiformant (AntiResonator)
            naf = extra_ctrl[ROW_NASAL_ANTIFORMANT_FREQ, kx]
            nabw = extra_ctrl[ROW_NASAL_ANTIFORMANT_BW, kx]
            if naf > 0.0 and nabw > 0.0 and naf < nyq:
                naa, nab, nac = _antireson_coeffs(naf, nabw, sr)
                namode = _MODE_ACTIVE
            else:
                naa = 1.0
                nab = 0.0
                nac = 0.0
                nax1 = 0.0
                nax2 = 0.0
                namode = _MODE_PASSTHROUGH

            # --- nasal formant (shared freq/bw for both branches)
            nff = extra_ctrl[ROW_NASAL_FORMANT_FREQ, kx]
            nfbw = extra_ctrl[ROW_NASAL_FORMANT_BW, kx]
            nasal_on = nff > 0.0 and nfbw > 0.0 and nff < nyq
            if nasal_on:
                nfa, nfb, nfc, _r = _reson_coeffs(nff, nfbw, sr)
                nfmode = _MODE_ACTIVE
            else:
                nfa = 1.0
                nfb = 0.0
                nfc = 0.0
                nfy1 = 0.0
                nfy2 = 0.0
                nfmode = _MODE_PASSTHROUGH

            # parallel nasal formant: peak-gain adjusted
            pn_gain = _db_to_lin(extra_ctrl[ROW_PAR_NASAL_FORMANT_DB, kx])
            if nasal_on and pn_gain > 0.0:
                _a, pnb, pnc, pnr = _reson_coeffs(nff, nfbw, sr)
                pna = pn_gain * (1.0 - pnr)
                pnmode = _MODE_ACTIVE
            else:
                pna = 0.0
                pnb = 0.0
                pnc = 0.0
                pny1 = 0.0
                pny2 = 0.0
                pnmode = _MODE_MUTE

            # --- oral formants, both branches
            for j in range(6):
                f = formant_ctrl[j, k]
                bw = bw_ctrl[j, k]
                usable = f > 0.0 and bw > 0.0 and f < nyq
                if usable:
                    fa[j], fb[j], fc[j], r = _reson_coeffs(f, bw, sr)
                    fmode[j] = _MODE_ACTIVE
                else:
                    r = 0.0
                    fa[j] = 1.0
                    fb[j] = 0.0
                    fc[j] = 0.0
                    fy1[j] = 0.0
                    fy2[j] = 0.0
                    fmode[j] = _MODE_PASSTHROUGH
                peak_gain = _db_to_lin(par_formant_db_ctrl[j, kp])
                if usable and peak_gain > 0.0:
                    pb[j] = fb[j]
                    pc[j] = fc[j]
                    if j >= 1:
                        # compensate the differencing filter for F2 .. F6
                        wf = 2.0 * math.pi * f / sr
                        diff_gain = math.sqrt(2.0 - 2.0 * math.cos(wf))
                        filter_gain = peak_gain / diff_gain
                    else:
                        filter_gain = peak_gain
                    pa[j] = filter_gain * (1.0 - r)
                    pmode[j] = _MODE_ACTIVE
                else:
                    pa[j] = 0.0
                    pb[j] = 0.0
                    pc[j] = 0.0
                    py1[j] = 0.0
                    py2[j] = 0.0
                    pmode[j] = _MODE_MUTE

        # --- glottal source sample
        if open_phase_length > 0 and pos_in_period < open_phase_length:
            gs_a += gs_b
            gs_x += gs_a
            voice = gs_x
        else:
            gs_x = 0.0
            voice = 0.0

        # breathiness during open phase
        if pos_in_period < open_phase_length:
            voice += (np.random.random() * 2.0 - 1.0) * breathiness_lin

        second_half = pos_in_period >= period_length / 2.0

        # --- cascade branch
        cascade_voice = voice * cascade_voicing_lin
        if second_half:
            asp_mod = cascade_aspiration_mod
        else:
            asp_mod = 0.0
        white = np.random.random() * 2.0 - 1.0
        asp_casc_y1 = asp_a * white + asp_b * asp_casc_y1
        aspiration = asp_casc_y1 * cascade_aspiration_lin * (1.0 - asp_mod)
        v = cascade_voice + aspiration
        # nasal antiformant (the degenerate a0 == 0 case yields a = b = c = 0,
        # which is equivalent to klatt-syn's mute state)
        if namode == _MODE_ACTIVE:
            y = naa * v + nab * nax1 + nac * nax2
            nax2 = nax1
            nax1 = v
            v = y
        # nasal formant
        if nfmode == _MODE_ACTIVE:
            y = nfa * v + nfb * nfy1 + nfc * nfy2
            nfy2 = nfy1
            nfy1 = y
            v = y
        # oral formants
        for j in range(6):
            if fmode[j] != _MODE_ACTIVE:
                continue  # passthrough
            y = fa[j] * v + fb[j] * fy1[j] + fc[j] * fy2[j]
            fy2[j] = fy1[j]
            fy1[j] = y
            v = y

        # --- parallel branch
        if par_enabled:
            par_voice = voice * par_voicing_lin
            if second_half:
                pmod = par_aspiration_mod
            else:
                pmod = 0.0
            white = np.random.random() * 2.0 - 1.0
            asp_par_y1 = asp_a * white + asp_b * asp_par_y1
            par_asp = asp_par_y1 * par_aspiration_lin * (1.0 - pmod)
            source = par_voice + par_asp
            source_diff = source - diff_x1
            diff_x1 = source
            if second_half:
                fmod_now = frication_mod
            else:
                fmod_now = 0.0
            white = np.random.random() * 2.0 - 1.0
            fric_par_y1 = asp_a * white + asp_b * fric_par_y1
            fric = fric_par_y1 * frication_lin * (1.0 - fmod_now)
            source2 = source_diff + fric
            pv = 0.0
            # nasal formant is directly applied to source
            if pnmode == _MODE_ACTIVE:
                y = pna * source + pnb * pny1 + pnc * pny2
                pny2 = pny1
                pny1 = y
                pv += y
            # F1 is directly applied to source
            if pmode[0] == _MODE_ACTIVE:
                y = pa[0] * source + pb[0] * py1[0] + pc[0] * py2[0]
                py2[0] = py1[0]
                py1[0] = y
                pv += y
            # F2 .. F6 are applied to source difference + frication
            for j in range(1, 6):
                if pmode[j] != _MODE_ACTIVE:
                    continue
                y = pa[j] * source2 + pb[j] * py1[j] + pc[j] * py2[j]
                py2[j] = py1[j]
                py1[j] = y
                if j % 2 == 0:
                    pv += y
                else:
                    pv -= y
            pv += par_bypass_lin * source2
            v += pv

        # output LP + per-sample gain
        y = oa * v + ob * oy1 + oc * oy2
        oy2 = oy1
        oy1 = y
        out[i] = y * _ctrl_interp(gain_ctrl, i, hop)
        pos_in_period += 1
    return out
