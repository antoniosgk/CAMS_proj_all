import numpy as np
def nearest_grid_index(st_lat, st_lon, lats, lons):
    """
    Return nearest grid indices (i,j).
    Works for 1D or 2D lats/lons arrays:
    - If lats, lons are 1D: find argmin per axis.
    - If 2D: compute haversine-ish distance (approx) and find argmin of flattened index.
    In our case lats/lons are 1D so the 2D case scenario may be removed
    """
    # Make arrays numpy
    lats = np.array(lats)
    lons = np.array(lons)

    # 1D case
    if lats.ndim == 1 and lons.ndim == 1:
        i = np.abs(lats - st_lat).argmin()
        j = np.abs(lons - st_lon).argmin()
        return int(i), int(j)