import numpy as np
import pandas as pd


def safe_slice(low, high, maxN):
    """
    Safe slice helper:
    - low, high are like Python slice bounds (high is exclusive)
    - maxN is the size of the axis
    """
    return slice(max(low, 0), min(high, maxN))



def compute_ring_sector_masks(ii, jj, Ny, Nx, radii):
    """
    Build ring sectors around (ii, jj) for arbitrary radii.

    radii: list/tuple of integers, e.g. [1,2,3,4]
           Sector k is the square (radius=radii[k]) minus square (radius=radii[k-1]).
           The first sector is the (2*r+1)x(2*r+1) square including the center.

    Returns: list of masks [S1, S2, ...]
    """
    masks = []
    prev = np.zeros((Ny, Nx), dtype=bool)

    for r in radii:
        box = np.zeros((Ny, Nx), dtype=bool)
        box[safe_slice(ii - r, ii + r + 1, Ny),
            safe_slice(jj - r, jj + r + 1, Nx)] = True  #True 

        ring = box & (~prev)
        masks.append(ring)
        prev = box

    return masks

def sector_table(mask, lats_small, lons_small, data_arr, var_name):
    """
    Build a DataFrame with indices and values for a given sector mask.

    mask      : boolean 2D array (Ny, Nx)
    lats_small: 1D or 2D array of latitudes matching mask shape
    lons_small: 1D or 2D array of longitudes matching mask shape
    data_arr  : 2D array (Ny, Nx) with variable values
    var_name  : column name for the variable
    """
    iy, ix = np.where(mask)

    # Handle 1D vs 2D lat/lon
    if lats_small.ndim == 1 and lons_small.ndim == 1:
        lat_vals = lats_small[iy]
        lon_vals = lons_small[ix]
    else:
        lat_vals = lats_small[iy, ix]
        lon_vals = lons_small[iy, ix]

    vals = data_arr[iy, ix]

    return pd.DataFrame({
        "lat_idx": iy,
        "lon_idx": ix,
        "lat": lat_vals,
        "lon": lon_vals,
        var_name: vals,
    })




def sector_stats(df, var_name):
    """
    Compute summary stats for a sector:
    mean, std, CV, median, IQR (Q3-Q1), n
    """
    vals = pd.to_numeric(df[var_name], errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        return {"n": 0, "mean": np.nan, "std": np.nan, "cv": np.nan,
                "median": np.nan, "q1": np.nan, "q3": np.nan, "iqr": np.nan}

    mean = float(np.mean(vals))
    std = float(np.std(vals))
    cv = float(std / mean) if mean != 0 else np.nan

    q1 = float(np.percentile(vals, 25))
    median = float(np.percentile(vals, 50))
    q3 = float(np.percentile(vals, 75))
    iqr = float(q3 - q1)

    return {"n": int(vals.size), "mean": mean, "std": std, "cv": cv,
            "median": median, "q1": q1, "q3": q3, "iqr": iqr}

def compute_sector_tables_generic(ii, jj, lats_small, lons_small, data_arr, var_name, radii):
    Ny, Nx = data_arr.shape
    masks = compute_ring_sector_masks(ii, jj, Ny, Nx, radii)

    dfs = [sector_table(m, lats_small, lons_small, data_arr, var_name) for m in masks]
    return dfs, masks

def cumulative_sector_masks(sector_masks):
    """
    Build cumulative sector masks.

    Input
    -----
    sector_masks : list of boolean masks
        [S1, S2, S3, ...] where each is a disjoint ring

    Returns
    -------
    cumulative_masks : list of boolean masks
        [C1, C2, C3, ...]
        Ck = union of S1 ... Sk
    """
    cumulative_masks = []
    running = np.zeros_like(sector_masks[0], dtype=bool)

    for S in sector_masks:
        running = running | S
        cumulative_masks.append(running.copy())

    return cumulative_masks

def compute_cumulative_sector_tables(
    sector_masks,
    lats_small,
    lons_small,
    data_arr,
    var_name,
):
    """
    Build DataFrames for cumulative sectors.
    """
    cumulative_masks = cumulative_sector_masks(sector_masks)

    dfs = []
    for k, mask in enumerate(cumulative_masks, start=1):
        df = sector_table(mask, lats_small, lons_small, data_arr, var_name)
        dfs.append(df)

    return dfs, cumulative_masks

def weighted_quantile(x, w, q):
    """
    Weighted quantile for 1D arrays.
    q in [0, 1].
    """
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)

    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x = x[m]
    w = w[m]
    if x.size == 0:
        return np.nan

    order = np.argsort(x)
    x = x[order]
    w = w[order]

    cw = np.cumsum(w)
    cw = cw / cw[-1]
    return float(np.interp(q, cw, x))



def sector_stats_unweighted(df, var_col):
    x = df[var_col].to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0, "mean": np.nan, "std": np.nan, "cv": np.nan,
                "median": np.nan, "q25": np.nan, "q75": np.nan, "iqr": np.nan}

    mean = float(np.mean(x))
    std = float(np.std(x, ddof=0))
    median = float(np.median(x))
    q25 = float(np.quantile(x, 0.25))
    q75 = float(np.quantile(x, 0.75))
    iqr = q75 - q25
    cv = float(std / mean) if np.isfinite(mean) and mean != 0 else np.nan

    return {"n": int(x.size), "mean": mean, "std": std, "cv": cv,
            "median": median, "q25": q25, "q75": q75, "iqr": iqr}


def sector_stats_weighted(df, var_name, w_col="w_area"):
    vals = pd.to_numeric(df[var_name], errors="coerce").to_numpy(dtype=float)
    w = pd.to_numeric(df[w_col], errors="coerce").to_numpy(dtype=float)

    m = np.isfinite(vals) & np.isfinite(w) & (w > 0)
    vals = vals[m]
    w = w[m]

    if vals.size == 0:
        return {"n": 0,
                "mean_w": np.nan, "std_w": np.nan, "cv_w": np.nan,
                "median_w": np.nan, "q1_w": np.nan, "q3_w": np.nan, "iqr_w": np.nan}

    wsum = float(np.sum(w))
    mean_w = float(np.sum(w * vals) / wsum)

    var_w = float(np.sum(w * (vals - mean_w) ** 2) / wsum)
    std_w = float(np.sqrt(var_w))
    cv_w = float(std_w / mean_w) if mean_w != 0 else np.nan

    q1_w = float(weighted_quantile(vals, w, 0.25))
    med_w = float(weighted_quantile(vals, w, 0.50))
    q3_w = float(weighted_quantile(vals, w, 0.75))
    iqr_w = float(q3_w - q1_w)

    return {"n": int(vals.size),
            "mean_w": mean_w, "std_w": std_w, "cv_w": cv_w,
            "median_w": med_w, "q1_w": q1_w, "q3_w": q3_w, "iqr_w": iqr_w}
# calculation.py
import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    """
    Vectorized haversine distance (km).
    lat/lon in degrees.
    """
    lat1 = np.deg2rad(lat1)
    lon1 = np.deg2rad(lon1)
    lat2 = np.deg2rad(lat2)
    lon2 = np.deg2rad(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_KM * c

def build_distance_dataframe(
    lats_small,
    lons_small,
    data_arr,
    lat_s,
    lon_s,
    var_name,
    w_area=None,
):
    """
    Create DataFrame with distance (km) from station for each grid cell.
    """
    if lats_small.ndim == 1 and lons_small.ndim == 1:
        LON2D, LAT2D = np.meshgrid(lons_small, lats_small)
    else:
        LAT2D = lats_small
        LON2D = lons_small

    dist_km = haversine_km(lat_s, lon_s, LAT2D, LON2D)

    df = pd.DataFrame({
        "lat": LAT2D.ravel(),
        "lon": LON2D.ravel(),
        "distance_km": dist_km.ravel(),
        var_name: data_arr.ravel(),
    })

    if w_area is not None:
        df["w_area"] = w_area.ravel()

    return df
def stats_by_distance_bins(
    df,
    var_name,
    dist_bins_km,
    w_col=None,
):
    """
    Compute stats per distance bin.
    """
    records = []

    for dmax in dist_bins_km:
        sub = df[df["distance_km"] <= dmax]

        if w_col is None:
            stats = sector_stats_unweighted(sub, var_name)
        else:
            stats = sector_stats_weighted(sub, var_name, w_col=w_col)

        stats["dmax_km"] = dmax
        records.append(stats)

    return pd.DataFrame(records)


def add_distance_bins(df, radii_km):
    """
    Add distance-bin labels to a distance DataFrame.

    radii_km: e.g. [5, 10, 20, 50, 100]
    Produces bins: (0–5], (5–10], ...
    """
    radii_km = np.asarray(radii_km, dtype=float)
    edges = np.concatenate(([0.0], radii_km))

    labels = [f"{int(edges[i])}–{int(edges[i+1])}"
              for i in range(len(edges) - 1)]

    df = df.copy()
    df["bin_label"] = pd.cut(
        df["distance_km"],
        bins=edges,
        labels=labels,
        include_lowest=True,
        right=True,
    )

    df["dmax_km"] = df["bin_label"].map(
        {lab: radii_km[i] for i, lab in enumerate(labels)}
    )

    return df



def cumulative_mean_ratio_to_center(cum_dfs, var_name, center_value, labels=None, w_col=None):
    """
    cum_dfs: list of cumulative DataFrames [C1, C2, ...]
    center_value: scalar (species at central grid cell)
    w_col: None for unweighted, or weight column name (e.g. "w_area") for weighted mean
    """
    if labels is None:
        labels = [f"C{k}" for k in range(1, len(cum_dfs) + 1)]

    rows = [{"label": "C0", "ratio": 1.0}]  # center/center

    for lab, df in zip(labels, cum_dfs):
        vals = pd.to_numeric(df[var_name], errors="coerce").to_numpy(dtype=float)

        if w_col is None:
            vals = vals[np.isfinite(vals)]
            mean_val = np.nanmean(vals) if vals.size else np.nan
        else:
            w = pd.to_numeric(df[w_col], errors="coerce").to_numpy(dtype=float)
            m = np.isfinite(vals) & np.isfinite(w) & (w > 0)
            vals = vals[m]; w = w[m]
            mean_val = (np.sum(w * vals) / np.sum(w)) if vals.size else np.nan

        ratio = mean_val / center_value if (np.isfinite(mean_val) and center_value != 0) else np.nan
        rows.append({"label": lab, "ratio": float(ratio) if np.isfinite(ratio) else np.nan})

    return pd.DataFrame(rows)
    
def distance_cumulative_mean_ratio_to_center(df_dist, var_name, center_value, d_bins_km, w_col=None):
    """
    df_dist: output of build_distance_dataframe (has distance_km and var_name; optionally w_area)
    d_bins_km: list like [5, 10, 20, 50]
    Returns a df with labels D0, D1.. and ratios mean/center.
    """
    rows = [{"label": "D0", "ratio": 1.0}]

    for k, dmax in enumerate(d_bins_km, start=1):
        df_in = df_dist[df_dist["distance_km"] <= dmax]
        vals = pd.to_numeric(df_in[var_name], errors="coerce").to_numpy(dtype=float)

        if w_col is None:
            vals = vals[np.isfinite(vals)]
            mean_val = np.nanmean(vals) if vals.size else np.nan
        else:
            w = pd.to_numeric(df_in[w_col], errors="coerce").to_numpy(dtype=float)
            m = np.isfinite(vals) & np.isfinite(w) & (w > 0)
            vals = vals[m]; w = w[m]
            mean_val = (np.sum(w * vals) / np.sum(w)) if vals.size else np.nan

        ratio = mean_val / center_value if (np.isfinite(mean_val) and center_value != 0) else np.nan
        rows.append({"label": f"≤{dmax} ", "ratio": float(ratio) if np.isfinite(ratio) else np.nan})

    return pd.DataFrame(rows)