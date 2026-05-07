import numpy as np
import pytest
from scipy.integrate import solve_ivp

from profinder import find_profiles, get_example_data, synthetic_glider_pressure


def test_get_example_data() -> None:
    data = get_example_data()
    assert isinstance(data, np.ndarray)
    assert np.isfinite(data).all()


def test_synthetic_glider_pressure() -> None:
    pressure = synthetic_glider_pressure()
    assert pressure.shape == (200,)


def test_find_profiles() -> None:
    pressure = get_example_data()
    peaks_kwargs = {"height": 15, "distance": 200, "width": 200, "prominence": 15}
    segments = find_profiles(pressure, peaks_kwargs=peaks_kwargs, min_pressure=3.0)
    assert len(segments) == 12

    segments = find_profiles(
        pressure, peaks_kwargs=peaks_kwargs, min_pressure=3.0
    )  # , apply_speed_threshold=True, time=np.arange(0, pressure.size/32, 1/32))
    assert len(segments) == 12

    pressure = synthetic_glider_pressure(
        n_points=200, max_p=500.0, intermediate_p=200.0, n_cycles=5
    )
    peaks_kwargs = {"height": 100, "distance": 5, "width": 5, "prominence": 100}
    segments = find_profiles(pressure, apply_smoothing=False, peaks_kwargs=peaks_kwargs)
    assert len(segments) == 6

    # Handling missing
    np.random.seed(14123)

    pressure = synthetic_glider_pressure(
        n_points=1000, max_p=500.0, intermediate_p=200.0, n_cycles=5
    )

    # Add NaN
    pressure[::8] = np.nan
    pressure[::9] = np.nan
    indices = np.random.choice(pressure.size, 50, replace=False)
    pressure[indices] = np.nan

    peaks_kwargs = {"height": 100, "distance": 5, "width": 5, "prominence": 100}
    segments = find_profiles(
        pressure,
        apply_smoothing=False,
        peaks_kwargs=peaks_kwargs,
        missing="drop",
    )
    assert len(segments) == 6


def test_velocity() -> None:
    # Crazy test using a fake VMP
    # Physical parameters
    mv = 14.0  # mass VMP (kg)
    mw = 11.0  # mass water displaced (kg)
    L = 1  # hull length (m)
    g = -9.81  # gravity (m/s^2)
    Cd = 3  # drag coefficient (-)
    Tmax = 120.0  # max tension (N)
    tension_tau = 8.0  # tension ramp-up time constant (s)
    tension_on = 100.0  # time when tension starts (s)

    # Time parameters
    total_time = 200  # (s)
    dt = 1 / 60  # Interpolation time step 60 Hz

    def instrument_ode(t, y):
        z, w = y
        # Tension ramps up after tension_on
        if t < tension_on:
            T = 0.0
        else:
            T = Tmax * (1 - np.exp(-(t - tension_on) / tension_tau))
        dwdt = g * (mv - mw) / mv - (mw / mv) * (Cd / L) * w * np.abs(w) + T / mv
        return [w, dwdt]

    class HitSurface:
        terminal: bool = True
        direction: int = 1  # Only trigger when crossing zero from below

        def __call__(self, t: float, y: np.ndarray) -> float:  # type: ignore[override]
            return float(y[0])

    hit_surface = HitSurface()

    sol = solve_ivp(
        instrument_ode,
        [0, total_time],
        [0.0, 0.0],  # Initial condition [z, w]
        events=hit_surface,
        vectorized=False,
    )

    t_uniform = np.arange(0, sol.t[-1], dt)
    z = np.interp(t_uniform, sol.t, sol.y[0])
    w = np.interp(t_uniform, sol.t, sol.y[1])

    segments_speed = find_profiles(
        -z,
        apply_speed_threshold=True,
        velocity=-w,
        min_speed=0.9,
        direction="down",
    )
    assert len(segments_speed) == 1


def test_challenging_profiles() -> None:
    """Challenging case: false starts both scattered and immediately before real profiles,
    gradually deepening profiles, a mid-descent pause, and near-surface start checks.

    The synthetic time series (1 Hz) contains:
    - Two scattered surface wobbles (2-3 m) early on, must NOT become profiles.
    - A false start (wobble to ~4 m) immediately before EACH real profile, separated
      from the true descent by only ~15 samples at the surface.  The algorithm must
      not mistake these for the beginning of the profile.
    - Four real profiles to increasing depths: 10, 50, 100, 400 m.
    - Profile 3 includes a ~90-sample flat pause at 40 m (winch stop) before
      continuing to 100 m.
    - Gaussian noise (σ = 0.15 m) added throughout, clipped at zero.

    Trough detection relies on the sensitive _default_troughs_kwargs, which correctly
    detects the brief surface return (prominence ≈ 4 m) between a false start and the
    real descent without requiring explicit troughs_kwargs here.

    Assertions checked per profile:
    1. Peak depth reaches at least 80 % of nominal maximum.
    2. Profile start (ds) is within 2 m of the surface.
    3. Profile start index is not more than 25 samples before the true descent begins
       (i.e. the algorithm did not anchor to a false start or a much earlier trough).
    """
    np.random.seed(99)

    def ramp(start: float, end: float, n: int) -> np.ndarray:
        return np.linspace(start, end, n)

    def flat(val: float, n: int) -> np.ndarray:
        return np.full(n, val)

    # Build the time series while tracking where each true descent begins.
    segs: list = []
    true_starts: list = []

    def cur() -> int:
        return sum(len(s) for s in segs)

    # Opening flat
    segs += [flat(0, 30)]
    # Scattered false starts (well before any profile)
    segs += [ramp(0, 3, 6), ramp(3, 0, 6), flat(0, 20)]
    segs += [ramp(0, 2, 5), ramp(2, 0, 5), flat(0, 20)]

    # Profile 1 (10 m): false start immediately before true descent
    segs += [ramp(0, 4, 10), ramp(4, 0, 10), flat(0, 15)]
    true_starts.append(cur())
    segs += [ramp(0, 10, 60), ramp(10, 0, 60)]
    segs += [flat(0, 120)]

    # Profile 2 (50 m): false start immediately before true descent
    segs += [ramp(0, 4, 10), ramp(4, 0, 10), flat(0, 15)]
    true_starts.append(cur())
    segs += [ramp(0, 50, 150), ramp(50, 0, 150)]
    segs += [flat(0, 120)]

    # Profile 3 (100 m): false start immediately before, plus ~90-sample pause at 40 m
    segs += [ramp(0, 4, 10), ramp(4, 0, 10), flat(0, 15)]
    true_starts.append(cur())
    segs += [ramp(0, 40, 120), flat(40, 90), ramp(40, 100, 120), ramp(100, 0, 200)]
    segs += [flat(0, 120)]

    # Profile 4 (400 m): false start immediately before true descent
    segs += [ramp(0, 4, 10), ramp(4, 0, 10), flat(0, 15)]
    true_starts.append(cur())
    segs += [ramp(0, 400, 600), ramp(400, 0, 600)]
    segs += [flat(0, 60)]

    pressure = np.concatenate(segs)
    pressure += np.random.normal(0, 0.15, pressure.size)
    pressure = np.clip(pressure, 0.0, None)

    peaks_kwargs = {
        "height": 8,  # wobbles (2-4 m) are below this; real profiles are not
        "prominence": 8,
        "distance": 80,
        "width": 10,
    }

    found = find_profiles(pressure, peaks_kwargs=peaks_kwargs, min_pressure=0.5)

    assert len(found) == 4, f"Expected 4 profiles, found {len(found)}"

    expected_min_depths = [8.0, 40.0, 80.0, 320.0]
    for i, ((ds, de, us, ue), min_depth) in enumerate(zip(found, expected_min_depths)):
        # 1. Peak depth check
        peak_depth = pressure[ds : ue + 1].max()
        assert peak_depth >= min_depth, (
            f"Profile {i + 1}: peak depth {peak_depth:.1f} m < expected min {min_depth} m"
        )

        # 2. Start must be near the surface (not stuck mid-depth in a false start)
        assert pressure[ds] <= 2.0, (
            f"Profile {i + 1}: start pressure {pressure[ds]:.2f} m > 2.0 m "
            f"(false start not correctly handled?)"
        )

        # 3. Start index must not be more than 25 samples before the true descent.
        #    A larger gap would indicate the algorithm anchored to a false start or an
        #    earlier trough rather than the surface return just before the real dive.
        assert ds >= true_starts[i] - 25, (
            f"Profile {i + 1}: ds={ds} is {true_starts[i] - ds} samples before "
            f"true descent start {true_starts[i]} (expected ≤ 25)"
        )


@pytest.mark.parametrize(
    "fs,label",
    [
        (0.1, "glider_realtime"),
        (1.0, "ctd_1hz"),
        (32.0, "fast_ctd"),
        (64.0, "microstructure"),
    ],
)
def test_sampling_rate_robustness(fs: float, label: str) -> None:
    """Profile detection is robust across typical oceanographic sampling rates.

    Rates tested:
    - 0.1 Hz  glider transmitting real-time (1 sample per 10 s)
    - 1 Hz    CTD on recovery (standard archive resolution)
    - 32 Hz   fast CTD
    - 64 Hz   microstructure profiler

    All sample-count parameters (distance, width) are scaled linearly with fs
    so that their physical duration in seconds stays constant.  The per-sample
    signal amplitude shrinks as 1/fs while noise stays fixed, so the monotone-run
    constraint (min_pressure_change) is disabled (set to 0.0) and run_length
    reduced to 2; peak detection relies entirely on the scaled peaks_kwargs.
    In a real deployment at high sampling rates the user would typically low-pass
    filter the pressure series before calling find_profiles.
    """
    np.random.seed(42)

    def n(duration_s: float) -> int:
        return max(1, round(duration_s * fs))

    def ramp(start: float, end: float, duration_s: float) -> np.ndarray:
        return np.linspace(start, end, n(duration_s))

    def flat(val: float, duration_s: float) -> np.ndarray:
        return np.full(n(duration_s), val)

    # Physical scenario identical to test_challenging_profiles but durations are
    # expressed in seconds so they scale correctly at every sampling rate.
    segs: list = []
    segs += [flat(0, 60)]
    # Scattered false starts
    segs += [ramp(0, 3, 15), ramp(3, 0, 15), flat(0, 60)]
    segs += [ramp(0, 2, 10), ramp(2, 0, 10), flat(0, 60)]
    # Profile 1 (10 m): false start immediately before, then 60 s surface return
    segs += [ramp(0, 4, 15), ramp(4, 0, 15), flat(0, 60)]
    segs += [ramp(0, 10, 60), ramp(10, 0, 60)]
    segs += [flat(0, 180)]
    # Profile 2 (50 m)
    segs += [ramp(0, 4, 15), ramp(4, 0, 15), flat(0, 60)]
    segs += [ramp(0, 50, 150), ramp(50, 0, 150)]
    segs += [flat(0, 180)]
    # Profile 3 (100 m): pause at 40 m
    segs += [ramp(0, 4, 15), ramp(4, 0, 15), flat(0, 60)]
    segs += [ramp(0, 40, 120), flat(40, 90), ramp(40, 100, 120), ramp(100, 0, 200)]
    segs += [flat(0, 180)]
    # Profile 4 (400 m)
    segs += [ramp(0, 4, 15), ramp(4, 0, 15), flat(0, 60)]
    segs += [ramp(0, 400, 600), ramp(400, 0, 600)]
    segs += [flat(0, 60)]

    pressure = np.concatenate(segs)
    # Low noise (0.05 m) keeps peak heights well above the detection threshold
    # at all sampling rates without requiring smoothing.
    pressure += np.random.normal(0, 0.05, pressure.size)
    pressure = np.clip(pressure, 0.0, None)

    # Physical reference durations (seconds) converted to samples:
    #   inter-profile gap  ≥ 180 s  →  distance
    #   minimum profile half-width ≥ 30 s  →  width
    peaks_kwargs = {
        "height": 8,
        "prominence": 8,
        "distance": max(4, round(180 * fs)),
        "width": max(2, round(30 * fs)),
    }
    troughs_kwargs = {
        "prominence": 2,
        "distance": max(2, round(5 * fs)),
        "width": max(1, round(5 * fs)),
    }

    found = find_profiles(
        pressure,
        peaks_kwargs=peaks_kwargs,
        troughs_kwargs=troughs_kwargs,
        min_pressure=0.5,
        min_pressure_change=0.0,
        run_length=2,
    )

    assert len(found) == 4, (
        f"fs={fs} Hz ({label}): expected 4 profiles, found {len(found)}"
    )

    expected_min_depths = [8.0, 40.0, 80.0, 320.0]
    for i, ((ds, de, us, ue), min_depth) in enumerate(zip(found, expected_min_depths)):
        peak_depth = pressure[ds : ue + 1].max()
        assert peak_depth >= min_depth, (
            f"fs={fs} Hz ({label}), profile {i + 1}: "
            f"peak depth {peak_depth:.1f} m < expected {min_depth} m"
        )
        assert pressure[ds] <= 2.0, (
            f"fs={fs} Hz ({label}), profile {i + 1}: "
            f"start pressure {pressure[ds]:.2f} m > 2.0 m (false start not rejected?)"
        )
