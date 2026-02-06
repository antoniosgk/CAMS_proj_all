'''
This file contains the functions that can be used
in order to retrieve the vertical indexing of the stations
'''

import numpy as np
import metpy.calc as mpcalc
from metpy.units import units
from metpy.constants import Rd, g  
import warnings

   
def metpy_compute_heights(p_prof_Pa, T_prof_K, RH=None, z0=0.0):
    """
    Compute geometric heights from pressure + temperature using the hypsometric equation
    (MetPy version), optionally using RH to compute virtual temperature.

    Supports:
      - 1D profiles: (nlev,)
      - 2D profiles: (nlev, ncol)

    Parameters
    ----------
    p_prof_Pa : array-like
        Pressure profile in Pa.
    T_prof_K : array-like
        Temperature profile in K (same shape as p_prof_Pa).
    RH : array-like, optional
        Relative humidity (0–1 or 0–100). If provided, used to compute virtual temperature.
    z0 : float or array-like, optional
        Surface height ASL [m]. For 1D p/T, z0 is scalar.
        For 2D p/T, z0 can be scalar (same for all columns) or 1D (ncol) matching columns.

    Returns
    -------
    z : ndarray
        Heights in meters, same shape as p_prof_Pa.
    """

    p_arr = np.asarray(p_prof_Pa, dtype=float)
    T_arr = np.asarray(T_prof_K, dtype=float)

    if p_arr.shape != T_arr.shape:
        raise ValueError("p_prof_Pa and T_prof_K must have the same shape.")

    RH_arr = None
    if RH is not None:
        RH_arr = np.asarray(RH, dtype=float)
        if RH_arr.shape != p_arr.shape:
            raise ValueError("RH must have the same shape as p_prof_Pa and T_prof_K.")

    # Pressure must be positive for logs
    p_floor = 1.0e-6  # Pa
    p_arr = np.where(np.isfinite(p_arr) & (p_arr > 0.0), p_arr, np.nan)
    p_arr = np.where(np.isfinite(p_arr), np.maximum(p_arr, p_floor), np.nan)

    # Temperature finite check
    T_arr = np.where(np.isfinite(T_arr), T_arr, np.nan)

    # Attach units
    p = p_arr * units.pascal
    T = T_arr * units.kelvin

    # ----- Virtual temperature: RH provided or not -----
    if RH_arr is None:
        print("INFO: No RH provided; using dry temperature T for virtual temperature.")
        Tv = T
    else:
        print("INFO: Using RH (relative humidity) for virtual temperature.")

        # Auto-detect RH convention: if max > 1.5 assume percent
        rh_max = np.nanmax(RH_arr)
        if np.isfinite(rh_max) and rh_max > 1.5:
            RH_frac = RH_arr / 100.0
        else:
            RH_frac = RH_arr

        RH_frac = np.clip(RH_frac, 0.0, 1.0)
        RH_q = RH_frac * units.dimensionless

        # Mask invalid regimes where saturation vapor pressure exceeds total pressure
        e_s = mpcalc.saturation_vapor_pressure(T)  # Pa
        valid = np.isfinite(p.magnitude) & np.isfinite(T.magnitude) & (p > e_s)

        mr = np.zeros_like(p_arr, dtype=float)  # dry fallback

        if np.any(valid):
            p_valid = p[valid]
            T_valid = T[valid]
            RH_valid = RH_q[valid]

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Saturation mixing ratio is undefined*",
                    category=UserWarning,
                )
                mr_valid = mpcalc.mixing_ratio_from_relative_humidity(p_valid, T_valid, RH_valid)

            mr_valid = mr_valid.to("kg/kg").magnitude
            mr_valid = np.where(np.isfinite(mr_valid) & (mr_valid >= 0.0), mr_valid, 0.0)
            mr[valid] = mr_valid

        mixing_ratio = mr * units("kg/kg")
        Tv = mpcalc.virtual_temperature(T, mixing_ratio)

    # Ensure Tv finite (fallback to T where needed)
    Tv_mag = Tv.to("kelvin").magnitude
    T_mag = T.to("kelvin").magnitude
    Tv_mag = np.where(np.isfinite(Tv_mag), Tv_mag, T_mag)
    Tv = Tv_mag * units.kelvin

    # If still any NaNs, replace with benign value to keep integration stable
    Tv_mag = Tv.to("kelvin").magnitude
    if np.any(~np.isfinite(Tv_mag)):
        Tv_mag = np.where(np.isfinite(Tv_mag), Tv_mag, 250.0)
        Tv = Tv_mag * units.kelvin

    # ----- Hypsometric integration -----
    if p.ndim == 1:
        nlev = p.shape[0]
        z = np.empty(nlev) * units.meter

        k_surf = int(np.nanargmax(p.magnitude))
        z[k_surf] = float(z0) * units.meter

        # upward (toward lower p)
        for k in range(k_surf - 1, -1, -1):
            Tv_layer = 0.5 * (Tv[k + 1] + Tv[k])
            ratio = p[k + 1] / p[k]
            dz = (Rd * Tv_layer / g) * np.log(ratio)
            z[k] = z[k + 1] + dz

        # downward (if any p > surface)
        for k in range(k_surf + 1, nlev):
            Tv_layer = 0.5 * (Tv[k - 1] + Tv[k])
            ratio = p[k - 1] / p[k]
            dz = (Rd * Tv_layer / g) * np.log(ratio)
            z[k] = z[k - 1] + dz

        return z.magnitude

    elif p.ndim == 2:
        nlev, ncol = p.shape
        z = np.empty_like(p.magnitude) * units.meter

        k_surf = np.nanargmax(p.magnitude, axis=0)  # (ncol,)

        z0_arr = np.asarray(z0, dtype=float)
        if z0_arr.ndim == 0:
            z_surf = np.full(ncol, z0_arr) * units.meter
        elif z0_arr.shape == (ncol,):
            z_surf = z0_arr * units.meter
        else:
            raise ValueError("For 2D p/T, z0 must be scalar or shape (ncol,)")

        # anchor each column at its surface
        for j in range(ncol):
            z[k_surf[j], j] = z_surf[j]

        # integrate upward (k decreasing)
        for k in range(nlev - 2, -1, -1):
            mask = k < k_surf
            if not np.any(mask):
                continue
            Tv_layer = 0.5 * (Tv[k + 1, mask] + Tv[k, mask])
            ratio = p[k + 1, mask] / p[k, mask]
            dz = (Rd * Tv_layer / g) * np.log(ratio)
            z[k, mask] = z[k + 1, mask] + dz

        # integrate downward (k increasing)
        for k in range(1, nlev):
            mask = k > k_surf
            if not np.any(mask):
                continue
            Tv_layer = 0.5 * (Tv[k - 1, mask] + Tv[k, mask])
            ratio = p[k - 1, mask] / p[k, mask]
            dz = (Rd * Tv_layer / g) * np.log(ratio)
            z[k, mask] = z[k - 1, mask] + dz

        return z.magnitude

    else:
        raise ValueError("metpy_compute_heights currently supports only 1D or 2D p/T profiles.")


def metpy_find_level_index(p_prof_Pa, T_prof_K, station_alt_m,
                           qv=None, RH=None, z_surf_model=0.0):
    p_arr = np.asarray(p_prof_Pa)
    T_arr = np.asarray(T_prof_K)
    RH_arr = np.asarray(RH) if RH is not None else None

    if p_arr.shape != T_arr.shape:
        raise ValueError("p_prof_Pa and T_prof_K must have the same shape.")

    # Height profile(s) ASL
    z_prof = metpy_compute_heights(
        p_prof_Pa=p_arr,
        T_prof_K=T_arr,
        RH=RH_arr,
        z0=z_surf_model,
    )

    p_hPa_prof = p_arr / 100.0

    # ----- 1D CASE -----
    if p_arr.ndim == 1:
        # safer debug (ignores NaN)
        print(f"DEBUG: p_prof range (hPa): {np.nanmin(p_hPa_prof):.1f} → {np.nanmax(p_hPa_prof):.1f}")
        print(f"DEBUG: z_prof range (m): {np.nanmin(z_prof):.1f} → {np.nanmax(z_prof):.1f}")
        print("DEBUG: few levels near surface (by max pressure index):")

        k_surf = int(np.argmax(p_arr))
        for k in range(max(0, k_surf - 5), min(len(z_prof), k_surf + 3)):
            print(f"  k={k:2d}: p={p_hPa_prof[k]:7.2f} hPa, z={z_prof[k]:8.1f} m")

        # ---- FIX: NaN-safe selection ----
        diff = np.abs(z_prof - station_alt_m)
        diff[~np.isfinite(diff)] = np.inf
        vertical_idx = int(np.argmin(diff))

        # If everything was NaN, argmin returns 0 with infs too; guard it
        #if not np.isfinite(diff[vertical_idx]):
           # raise ValueError("All height differences are non-finite (z_prof contains only NaNs/Infs).")

        if diff[vertical_idx] > 1500.0:
            print(f"WARNING: nearest model level is {diff[vertical_idx]:.0f} m away from station altitude")

        p_std = mpcalc.height_to_pressure_std(station_alt_m * units.meter)
        print("DEBUG: Standard atmosphere pressure at station height:",
              p_std.to("hectopascal"))

        p_hPa = p_arr[vertical_idx] / 100.0
        return vertical_idx, p_hPa, z_prof[vertical_idx]

    # ----- 2D CASE: (nlev, ncol) -----
    elif p_arr.ndim == 2:
        nlev, ncol = p_arr.shape

        diff = np.abs(z_prof - station_alt_m)
        diff[~np.isfinite(diff)] = np.inf

        vertical_idx = np.argmin(diff, axis=0)  # (ncol,)

        cols = np.arange(ncol)
        p_sel_hPa = p_hPa_prof[vertical_idx, cols]
        z_sel_m = z_prof[vertical_idx, cols]

        print(f"DEBUG (2D): p_prof range (hPa): {np.nanmin(p_hPa_prof):.1f} → {np.nanmax(p_hPa_prof):.1f}")
        print(f"DEBUG (2D): z_prof range (m): {np.nanmin(z_prof):.1f} → {np.nanmax(z_prof):.1f}")

        sample_col = 0
        k_surf_sample = int(np.argmax(p_arr[:, sample_col]))
        print(f"DEBUG (2D): sample column {sample_col}, few levels near surface:")
        for k in range(max(0, k_surf_sample - 2), min(nlev, k_surf_sample + 3)):
            print(f"  k={k:2d}: p={p_hPa_prof[k, sample_col]:7.2f} hPa, z={z_prof[k, sample_col]:8.1f} m")

        diff_min = np.min(diff, axis=0)  # per column
        n_bad = np.sum(~np.isfinite(diff_min))
        if n_bad > 0:
            raise ValueError(f"{n_bad} column(s) have no finite z_prof (all NaN/Inf).")

        n_far = np.sum(diff_min > 1500.0)
        if n_far > 0:
            print(f"WARNING: {n_far} column(s) have nearest level > 1500 m away from station altitude.")

        p_std = mpcalc.height_to_pressure_std(station_alt_m * units.meter)
        print("DEBUG: Standard atmosphere pressure at station height:",
              p_std.to("hectopascal"))

        return vertical_idx, p_sel_hPa, z_sel_m

    else:
        raise ValueError("metpy_find_level_index currently supports only 1D or 2D p/T profiles.")


    
def altitude_to_pressure_ISA(z_m):
    """Convert altitude (m) to pressure (Pa) using standard barometric formula (ISA troposphere).
       Good approximation for typical surface altitudes."""
    # Constants
    p0 = 101325.0       # Pa
    T0 = 288.15         # K
    g = 9.80665         # m/s2
    L = 0.0065          # K/m
    R = 287.05          # J/(kg K)
    # Avoid negative base when very high z: clip to realistic domain
    term = 1.0 - L * z_m / T0
    if term <= 0:
        return 0.0
    exponent = g / (R * L)
    return p0 * (term ** exponent)

def pressure_to_height(p_hPa, T_K):
    """
    Compute geometric height using the hypsometric equation.

    Parameters
    ----------
    p_hPa : 1D array
        Pressure profile in hPa.
    T_K : 1D array
        Temperature profile in K.

    Returns
    -------
    z : 1D array
        Heights in meters of each model level.
    """
    p_hPa = np.asarray(p_hPa)
    T_K   = np.asarray(T_K)

    # Avoid divide-by-zero / negative pressures
    p_hPa = np.clip(p_hPa, 1e-6, None)

    # Hypsometric equation (relative to 1000 hPa)
    # z = (Rd * T / g) * ln(p0 / p)
    return (Rd * T_K / g) * np.log(1000.0 / p_hPa)


def geos_find_level_index(p_prof_Pa, T_prof_K, station_alt):
    """
    Find the model level closest to station altitude using PL and T profiles.

    Parameters
    ----------
    p_prof_Pa : 1D array
        Pressure profile in Pa at model level midpoints.
    T_prof_K : 1D array
        Temperature profile in K at the same levels.
    station_alt : float
        Station altitude in meters.

    Returns
    -------
    idx : int
        Index of the closest model level.
    p_level_hPa : float
        Pressure of that level (hPa).
    z_level_m : float
        Height of that level (m).
    """
    p_prof_Pa = np.asarray(p_prof_Pa).squeeze()
    T_prof_K  = np.asarray(T_prof_K).squeeze()

    if p_prof_Pa.ndim != 1 or T_prof_K.ndim != 1:
        raise ValueError(f"Expected 1D profiles, got shapes {p_prof_Pa.shape} and {T_prof_K.shape}")

    # Pa -> hPa
    p_prof_hPa = p_prof_Pa / 100.0

    # Height of each model level from hypsometric equation
    z_prof = pressure_to_height(p_prof_hPa, T_prof_K)

    # Level closest in height to station altitude
    idx = int(np.argmin(np.abs(z_prof - station_alt)))

    return idx, float(p_prof_hPa[idx]), float(z_prof[idx])
