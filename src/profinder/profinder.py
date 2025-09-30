from __future__ import annotations

import warnings
from typing import Any, List, Literal, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import find_peaks, savgol_filter

# Type aliases for clarity
FloatArray = NDArray[np.floating]
ProfileTuple = Tuple[int, int, int, int]

__all__ = ["find_profiles", "find_segment"]

_default_peaks_kwargs = {"height": 25, "distance": 200, "width": 200, "prominence": 25}


def _contiguous_regions(condition: ArrayLike) -> NDArray[np.int_]:
    """Return start/stop index pairs for contiguous True regions.

    Parameters
    ----------
    condition : array_like
            Array of boolean values.

    Returns
    -------
    regions : ndarray
            Array of indices demarking the start and end of contiguous True regions in condition.
            Shape is (N, 2) where N is the number of regions.

    """

    condition = np.asarray(condition)
    d = np.diff(condition)
    (regions,) = d.nonzero()
    regions += 1

    if condition[0]:
        regions = np.r_[0, regions]

    if condition[-1]:
        regions = np.r_[regions, condition.size]

    regions.shape = (-1, 2)
    return regions


def _validate_pressure(
    pressure: ArrayLike,
    apply_smoothing: bool,
    window_length: int,
    polyorder: int,
) -> FloatArray:
    """Validate and (optionally) smooth pressure input.

    Parameters
    ----------
    pressure : array_like
        Raw pressure (or depth) observations.
    apply_smoothing : bool
        Whether to smooth prior to further processing.
    window_length : int
        Savitzky-Golay window length (must be odd and >= polyorder+2).
    polyorder : int
        Polynomial order for Savitzky-Golay.

    Returns
    -------
    pressure_sanitised : ndarray
        Validated (and possibly smoothed) pressure as float array.
    """
    pressure = np.asarray(pressure, dtype=float)

    if (~np.isfinite(pressure)).any():
        raise ValueError("Input pressure data contains non-finite values.")

    if apply_smoothing:
        # Ensure window_length is valid for savgol_filter to appease type checkers / runtime
        if window_length % 2 == 0:
            raise ValueError("window_length must be odd for Savitzky-Golay filter.")
        if window_length <= polyorder:
            raise ValueError(
                "window_length must be greater than polyorder for Savitzky-Golay filter."
            )
        pressure = savgol_filter(
            pressure, window_length=window_length, polyorder=polyorder
        )

    return pressure


def _find_segment(
    pressure: ArrayLike,
    apply_min_pressure: bool = True,
    min_pressure: float = -1.0,
    apply_speed_threshold: bool = False,
    time: Optional[ArrayLike] = None,
    velocity: Optional[ArrayLike] = None,
    min_speed: float = 0.2,
    direction: Literal["up", "down"] = "down",
) -> Tuple[int, int]:
    """Find the start and end indices of a single down or up segment.

    Parameters
    ----------
    pressure : array_like
        Pressure (or depth) samples for a single expected up/down cycle.
    apply_min_pressure : bool
        Enforce a minimum pressure threshold.
    min_pressure : float
        Threshold for *pressure > min_pressure* to be considered valid.
    apply_speed_threshold : bool
        If True, enforce a minimum absolute speed condition.
    time : array_like, optional
        Time vector (same length as pressure). Required if velocity not supplied.
    velocity : array_like, optional
        Pre-computed velocity (same length as pressure). Overrides gradient(time).
    min_speed : float
        Minimum absolute velocity magnitude to be considered valid.
    direction : {'up','down'}
        Direction in which the speed threshold is enforced.

    Returns
    -------
    (start, end) : tuple[int, int]
        Index bounds (inclusive start, exclusive end) of the longest valid segment.
    """

    p = np.asarray(pressure, dtype=float)
    good_pressure = np.full(p.shape, True, dtype=bool)
    good_velocity = np.full(p.shape, True, dtype=bool)

    if apply_min_pressure:
        good_pressure = p > float(min_pressure)

    if apply_speed_threshold:
        if velocity is None:
            if time is None:
                raise ValueError("time must be provided if velocity is not supplied.")
            v = np.gradient(p, np.asarray(time, dtype=float))
        else:
            v = np.asarray(velocity, dtype=float)

        if direction == "down":
            good_velocity = v > min_speed
        elif direction == "up":
            good_velocity = v < -min_speed
        else:
            raise ValueError(
                "direction must be 'up' or 'down' when applying speed threshold."
            )

    good_data = good_pressure & good_velocity

    if np.all(good_data):
        return 0, p.size - 1

    if not good_data.any():
        raise RuntimeError("No valid segment found.")

    regions = _contiguous_regions(good_data)

    if regions.shape[0] > 1:
        warnings.warn("Multiple valid segments found, choosing the longest one.")

    region_size = np.diff(regions, axis=1).ravel()
    idx_longest = int(np.argmax(region_size))
    start = int(regions[idx_longest, 0])
    end = int(regions[idx_longest, 1])
    return start, end


def _find_profiles(
    pressure: FloatArray,
    peaks_kwargs: dict[str, Any],
    troughs_kwargs: dict[str, Any],
    min_pressure: float = -1.0,
    run_length: int = 4,
    min_pressure_change: float = 0.0,
    apply_speed_threshold: bool = False,
    time: Optional[ArrayLike] = None,
    velocity: Optional[ArrayLike] = None,
    min_speed: float = 0.2,
    direction: Literal["up", "down", "both"] = "down",
) -> List[ProfileTuple]:
    """
    Find profile segments in a pressure time series from a profiling instrument.

    Parameters
    ----------
    pressure : array_like
        1D array of pressure values sampled continuously.
    peaks_kwargs : dict, optional
        Dictionary of keyword arguments to pass to scipy.signal.find_peaks for peak detection.
    troughs_kwargs : dict, optional
        If not specified, the peaks_kwargs will be used.
    min_pressure : float, optional
        All profiles must start or end at greater than min_pressure (default: -1.0).
    min_pressure_change : float, optional
        Minimum absolute pressure change per sample to confirm descent or ascent (default: 0.01).
    run_length : int, optional
        Number of consecutive samples required to confirm descent/ascent (default: 4).
    apply_speed_threshold : bool, optional
        If True, apply a speed threshold based on the velocity data (default: False).
    time : array_like
        1D array of time values corresponding to the pressure samples. Only needed if applying a speed threshold.
    velocity : array_like
        1D array of velocity values corresponding to the pressure samples. Only needed if applying a speed threshold.
    min_speed : float, optional
        Minimum speed (change in pressure per unit time) to classify ascent/descent (default: 0.1).
    direction: str, optional
        Direction of profiling to apply speed threshold ("up", "down", or "both", default: "down").

    Returns
    -------
    profiles : list of tuple
        List of (down_start, down_end, up_start, up_end) index tuples for each detected profile.
        If no speed threshold is applied the middle two indices will be identical (peak).

    """

    ndata = pressure.size
    diffs = np.diff(pressure)

    peaks, _ = find_peaks(pressure, **peaks_kwargs)
    troughs, _ = find_peaks(pressure.max() - pressure, **troughs_kwargs)

    profiles = []
    for peak_idx in peaks:
        # Ensure peak is an int to keep mypy happy
        peak = int(peak_idx)

        # Find surface point before peak
        trough_before = troughs[troughs < peak]
        ds = int(trough_before[-1]) if len(trough_before) > 0 else 0

        # Move start forward to first robust descent
        for i in range(ds, peak - run_length):
            if np.all(diffs[i : i + run_length] > min_pressure_change):
                ds = i
                break

        # Move start to adjust for min_pressure
        while ds + 1 < peak and pressure[ds + 1] < min_pressure:
            ds += 1

        # Find surface point after peak
        trough_after = troughs[troughs > peak]
        ue = int(trough_after[0]) if len(trough_after) > 0 else ndata - 1

        # Move end backward to last robust ascent
        for i in range(ue, peak + run_length, -1):
            if i - run_length >= peak and np.all(
                diffs[i - run_length : i] < -min_pressure_change
            ):
                ue = i
                break

        while ue - 1 > peak and pressure[ue - 1] < min_pressure:
            ue -= 1

        # Ensure a robust run_length of increasing pressure toward the peak
        de = peak
        for i in range(peak, ds + run_length, -1):
            if np.all(diffs[i - run_length : i] > min_pressure_change):
                de = i
                break

        # Ensure a robust run_length of decreasing pressure away from the peak
        us = peak
        for i in range(peak, ue - run_length + 1):
            if np.all(diffs[i : i + run_length] < -min_pressure_change):
                us = i
                break

        # (down_start, down_end, up_start, up_end)
        profiles.append((ds, de, us, ue))

    if not apply_speed_threshold:
        return profiles

    def refine_segment(
        segment_pressure: FloatArray,
        seg_time: Optional[ArrayLike],
        seg_velocity: Optional[ArrayLike],
        seg_direction: Literal["up", "down"],
    ) -> Tuple[int, int]:
        return _find_segment(
            segment_pressure,
            apply_min_pressure=False,
            apply_speed_threshold=True,
            min_speed=min_speed,
            direction=seg_direction,
            time=seg_time,
            velocity=seg_velocity,
        )

    # ds = down_start, de = down_end, etc.
    for idx, (ds, de, us, ue) in enumerate(profiles):
        # Down portion refinement
        if direction in ("down", "both"):
            ds_off, de_off = refine_segment(
                pressure[ds:de],
                None if time is None else np.asarray(time)[ds:de],
                None if velocity is None else np.asarray(velocity)[ds:de],
                "down",
            )
            ds_new, de_new = ds + ds_off, ds + de_off
        else:
            ds_new, de_new = ds, de

        # Up portion refinement
        if direction in ("up", "both"):
            us_off, ue_off = refine_segment(
                pressure[us:ue],
                None if time is None else np.asarray(time)[us:ue],
                None if velocity is None else np.asarray(velocity)[us:ue],
                "up",
            )
            us_new, ue_new = us + us_off, us + ue_off
        else:
            us_new, ue_new = us, ue

        profiles[idx] = (ds_new, de_new, us_new, ue_new)

    return profiles


def find_segment(
    pressure: ArrayLike,
    apply_smoothing: bool = False,
    window_length: int = 9,
    polyorder: int = 2,
    apply_min_pressure: bool = True,
    min_pressure: float = -1.0,
    apply_speed_threshold: bool = False,
    time: Optional[ArrayLike] = None,
    velocity: Optional[ArrayLike] = None,
    min_speed: float = 0.2,
    direction: Literal["up", "down"] = "down",
) -> Tuple[int, int]:
    """Find the start and end indices of a single down or up segment

    Parameters
    ----------
    pressure : array_like
        Pressure (or depth) samples for a single expected up/down cycle.
    apply_smoothing, window_length, polyorder : see :func:`find_profiles`.
    apply_min_pressure : bool
        Enforce a minimum pressure threshold.
    min_pressure : float
        Threshold for *pressure > min_pressure* to be considered valid.
    apply_speed_threshold : bool
        If True, enforce a minimum absolute speed condition.
    time : array_like, optional
        Time vector (same length as pressure). Required if velocity not supplied.
    velocity : array_like, optional
        Pre-computed velocity (same length as pressure). Overrides gradient(time).
    min_speed : float
        Minimum absolute velocity magnitude to be considered valid.
    direction : {'up','down'}
        Direction in which the speed threshold is enforced.

    Returns
    -------
    (start, end) : tuple[int, int]
        Index bounds (inclusive start, exclusive end) of the longest valid segment.
    """

    if apply_speed_threshold and (time is None and velocity is None):
        raise ValueError(
            "Time or velocity data must be provided if apply_speed_threshold is True."
        )
    elif time is not None and (~np.isfinite(time)).all():
        raise ValueError("Input time data contains non-finite values.")
    elif velocity is not None and (~np.isfinite(velocity)).all():
        raise ValueError("Input velocity data contains non-finite values.")

    p = _validate_pressure(pressure, apply_smoothing, window_length, polyorder)

    if not apply_min_pressure and not apply_speed_threshold:
        return 0, p.size - 1

    start, end = _find_segment(
        p,
        apply_min_pressure=apply_min_pressure,
        min_pressure=min_pressure,
        apply_speed_threshold=apply_speed_threshold,
        time=time,
        velocity=velocity,
        min_speed=min_speed,
        direction=direction,
    )

    return start, end


def find_profiles(
    pressure: ArrayLike,
    apply_smoothing: bool = False,
    window_length: int = 9,
    polyorder: int = 2,
    min_pressure: float = -1.0,
    peaks_kwargs: Optional[dict[str, Any]] = None,
    troughs_kwargs: Optional[dict[str, Any]] = None,
    run_length: int = 4,
    min_pressure_change: float = 0.01,
    apply_speed_threshold: bool = False,
    time: Optional[ArrayLike] = None,
    velocity: Optional[ArrayLike] = None,
    min_speed: float = 0.2,
    direction: Literal["up", "down", "both"] = "down",
    missing: Literal["raise", "drop"] = "raise",
) -> List[ProfileTuple]:
    """
    Find profile segments in a pressure time series from a profiling instrument.

    Parameters
    ----------
    pressure : array_like
        1D array of pressure values sampled continuously.
    apply_smoothing : bool, optional
        If True (default: False), apply Savitzky-Golay smoothing to the pressure data before analysis.
        If False, use the raw pressure data.
    window_length : int, optional
        Window length for Savitzky-Golay smoothing filter (default: 8 points).
    polyorder : int, optional
        Polynomial order for Savitzky-Golay filter (default: 2).
    peaks_kwargs : dict, optional
        Dictionary of keyword arguments to pass to scipy.signal.find_peaks for peak detection.
    troughs_kwargs : dict, optional
        If not specified, the peaks_kwargs will be used.
    min_pressure : float, optional
        All profiles must start or end at greater than min_pressure (default: -1.0).
    run_length : int, optional
        Number of consecutive samples required to confirm descent/ascent (default: 4).
    min_pressure_change : float, optional
        Minimum absolute pressure change per sample to confirm descent or ascent (default: 0.01).
    apply_speed_threshold : bool, optional
        If True, apply a speed threshold based on the velocity data (default: False).
    time : array_like
        1D array of time values corresponding to the pressure samples. Only needed if applying a speed threshold.
    velocity : array_like
        1D array of velocity values corresponding to the pressure samples. Only needed if applying a speed threshold.
    min_speed : float, optional
        Minimum speed (change in pressure per unit time) to classify ascent/descent (default: 0.1).
    direction: str, optional
        Direction of profiling to apply speed threshold ("up", "down", or "both", default: "down").
    missing : {'raise','drop'}
        How to handle missing/non-finite values in the input pressure data.
        'raise' (default) will raise an error if any non-finite values are found.
        'drop' will remove any non-finite values from the input data before processing.

    Returns
    -------
    profiles : list of tuple
        List of (down_start, down_end, up_start, up_end) index tuples for each detected profile.
        If no speed threshold is applied the middle two indices will be identical (peak).

    """

    if apply_speed_threshold and (time is None and velocity is None):
        raise ValueError(
            "Time or velocity data must be provided if apply_speed_threshold is True."
        )
    elif time is not None and (~np.isfinite(time)).all():
        raise ValueError("Input time data contains non-finite values.")
    elif velocity is not None and (~np.isfinite(velocity)).all():
        raise ValueError("Input velocity data contains non-finite values.")

    if peaks_kwargs is None:
        peaks_kwargs = _default_peaks_kwargs
    if troughs_kwargs is None:
        troughs_kwargs = peaks_kwargs
    if "height" not in peaks_kwargs:
        raise ValueError("peaks_kwargs must contain 'height' key for peak detection.")

    p = np.asarray(pressure, dtype=float)

    pressure_contains_nan = (~np.isfinite(p)).any()

    if pressure_contains_nan and missing == "raise":
        raise ValueError("Input p data contains non-finite values.")
    elif pressure_contains_nan and missing == "drop":
        valid_indices = np.isfinite(p)
        # Keep mapping from cleaned indices back to original indices
        valid_idx = np.where(valid_indices)[0]
        p = p[valid_indices]
        if time is not None:
            time = np.asarray(time, dtype=float)[valid_indices]
        if velocity is not None:
            velocity = np.asarray(velocity, dtype=float)[valid_indices]

    if apply_smoothing:
        if pressure_contains_nan:
            warnings.warn(
                "Pressure data containing non-finite values will be smoothed. Data may no longer be evenly spaced."
            )
        # Ensure window_length is valid for savgol_filter to appease type checkers / runtime
        if window_length % 2 == 0:
            raise ValueError("window_length must be odd for Savitzky-Golay filter.")
        if window_length <= polyorder:
            raise ValueError(
                "window_length must be greater than polyorder for Savitzky-Golay filter."
            )
        p = savgol_filter(p, window_length=window_length, polyorder=polyorder)

    profiles = _find_profiles(
        p,
        min_pressure=min_pressure,
        peaks_kwargs=peaks_kwargs,
        troughs_kwargs=troughs_kwargs,
        run_length=run_length,
        min_pressure_change=min_pressure_change,
        apply_speed_threshold=apply_speed_threshold,
        time=time,
        velocity=velocity,
        min_speed=min_speed,
        direction=direction,
    )

    # Dropping NaNs shifts the indices, which needs to be corrected here.
    if (missing == "drop") and (len(profiles) > 0):
        profiles = [
            (
                valid_idx[ds],
                valid_idx[de],
                valid_idx[us],
                valid_idx[ue],
            )
            for (ds, de, us, ue) in profiles
        ]

    return profiles
