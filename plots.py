# plots.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from calculation import weighted_quantile, haversine_km
import matplotlib.ticker as ticker
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
import os
import re
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd



def _sanitize_filename(s: str) -> str:
    # keep it filesystem-safe
    s = re.sub(r"\s+", "_", str(s).strip())
    s = re.sub(r"[^A-Za-z0-9_.-]+", "", s)
    return s


def save_figure(fig, out_dir, filename_base, dpi=200):
    """
    Save a matplotlib figure as PNG into out_dir with a safe filename.
    """
    os.makedirs(out_dir, exist_ok=True)
    fname = _sanitize_filename(filename_base) + ".png"
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path

def build_meta_title(meta, kind=""):
    """
    kind: optional short label like 'Map', 'Profile P–T', etc.
    """
    if meta is None:
        return kind

    header = f"{meta['species']} ({meta['units']}) at {meta['time_str']}"
    if kind:
        header = f"{kind} | " + header

    line2 = (
        f"Station {meta['station_name']}: "
        f"({meta['station_lat']:.2f}, {meta['station_lon']:.2f}), alt={meta['station_alt']:.1f} m"
    )
    line3 = (
        f"Model: ({meta['model_lat']:.2f}, {meta['model_lon']:.2f}), "
        f"lev={meta['model_level']}, alt={meta['z_level_m']:.1f} m"
    )
    return header + "\n" + line2 + "\n" + line3

def plot_variable_on_map(
    lats_small,
    lons_small,
    data_arr,
    # station
    lon_s,
    lat_s,
    # labels/meta
    units="",
    species_name="var",
    time_str=None,
    meta=None,
    # map control
    d=0.4,
    proj=None,
    ax=None,
    # terrain background on a DIFFERENT grid (always 1D)
    lats_terrain=None,
    lons_terrain=None,
    z_orog_m=None,
    terrain_alpha=0.5,
    field_alpha=0.8,
    add_orog_contours=True,
    plot_species=True,plot_orography=False
    ):
    

    # --- Projection / axes (must be GeoAxes) ---
    if proj is None:
        proj = ccrs.PlateCarree()

    if ax is None:
        fig, ax = plt.subplots(subplot_kw={"projection": proj})
    else:
        fig = ax.figure

    # --- extent around station (lon/lat degrees => PlateCarree) ---
    lon_min, lon_max = lon_s - d, lon_s + d
    lat_min, lat_max = lat_s - d, lat_s + d
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    # --- ocean background first (so masked sea shows this color) ---
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue", zorder=-2)

    # --- TERRAIN underlay (optional) ---
    terrain_im = None
    if plot_orography and (z_orog_m is not None) and (lats_terrain is not None) and (lons_terrain is not None):
        lats_terrain = np.asarray(lats_terrain, dtype=float)
        lons_terrain = np.asarray(lons_terrain, dtype=float)
        z_orog_m = np.asarray(z_orog_m, dtype=float)

        LON_T, LAT_T = np.meshgrid(lons_terrain, lats_terrain)

        # Mask sea so it doesn't use the terrain colormap
        z_plot = np.ma.masked_where(~np.isfinite(z_orog_m) | (z_orog_m <= 0.0), z_orog_m)

        # Use land-only min/max for better contrast (avoid sea pulling vmin down)
        if np.ma.is_masked(z_plot):
            zmin = float(z_plot.min())
            zmax = float(z_plot.max())
        else:
            zmin = float(np.nanmin(z_orog_m))
            zmax = float(np.nanmax(z_orog_m))

        terrain_im = ax.pcolormesh(
            LON_T, LAT_T, z_plot,
            cmap="terrain",
            shading="auto",
            vmin=zmin, vmax=zmax,
            alpha=terrain_alpha,
            transform=ccrs.PlateCarree(),
            zorder=-1,
        )

        if add_orog_contours:
            step_m = 200.0
            levels = np.arange(np.floor(zmin / step_m) * step_m,
                               np.ceil(zmax / step_m) * step_m + step_m,
                               step_m)
            ax.contour(
                LON_T, LAT_T, z_orog_m,
                levels=levels,
                colors="k",
                linewidths=0.4,
                alpha=0.30,
                transform=ccrs.PlateCarree(),
                zorder=0,
            )

    # --- Species overlay (small domain) ---
    lats_small = np.asarray(lats_small, dtype=float)
    lons_small = np.asarray(lons_small, dtype=float)
    data_arr = np.asarray(data_arr, dtype=float)

    

    vmin = float(np.nanmin(data_arr))
    vmax = float(np.nanmax(data_arr))
    norm = Normalize(vmin=vmin, vmax=vmax)

    im = None
    if plot_species:
      LON_S, LAT_S = np.meshgrid(lons_small, lats_small)
      im = ax.pcolormesh(
        LON_S, LAT_S, data_arr,
        cmap="viridis",
        shading="auto",
        norm=norm,
        transform=ccrs.PlateCarree(),
        alpha=field_alpha,
        zorder=2,
    )


    # station marker
    ax.plot(
        lon_s, lat_s, "kx",
        markersize=12,
        transform=ccrs.PlateCarree(),
        zorder=4,
    )

    # coast/borders on top
    ax.coastlines(resolution="10m", linewidth=0.8)
    #ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, zorder=5)

    # --- Two colorbars (terrain left, species right) ---
    cb_w=0.02
    cb_h=0.60
    cb_y=0.18
    #left colorbar(terrain)
    if terrain_im is not None:
        cax_terr=fig.add_axes([0.01,cb_y,cb_w,cb_h])
        cb_terr=fig.colorbar(terrain_im,cax=cax_terr)
        cb_terr.set_label("Elevation (m)")

    #right colorbar (species)
    if plot_species:
      cax_sp=fig.add_axes([0.8,cb_y,cb_w,cb_h])
      cb_sp=fig.colorbar(im,cax=cax_sp)
      cb_sp.set_label(units)    
    
    if not plot_species and meta is not None:
      ax.set_title(
        f"Station {meta['station_name']} "
        f"({meta['station_lat']:.3f}°, {meta['station_lon']:.3f}°), "
        f"{meta['station_alt']:.0f} m ASL\n"
        "Topography map",
        pad=18,
    )


    # -----------------------------
    # Gridlines with DMS labels
    # -----------------------------
    step = 0.4  # degrees
    lon0 = np.floor(lon_min / step) * step
    lon1 = np.ceil(lon_max / step) * step
    lat0 = np.floor(lat_min / step) * step
    lat1 = np.ceil(lat_max / step) * step

    xticks = np.round(np.arange(lon0, lon1 + 0.5 * step, step), 6)
    yticks = np.round(np.arange(lat0, lat1 + 0.5 * step, step), 6)

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.6,
        alpha=0.35,
        linestyle="--",
        zorder=5,
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = ticker.FixedLocator(xticks)
    gl.ylocator = ticker.FixedLocator(yticks)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {"size": 9}
    gl.ylabel_style = {"size": 9}

    ax.set_title(build_meta_title(meta, kind="Map with Station"), pad=18)
    fig.subplots_adjust(left=0.1,right=0.9,top=0.82,bottom=0.1)  # keep title readable

    return fig, ax, im

def plot_rectangles(
    ax,
    lats_small,
    lons_small,
    ii,
    jj,
    im,
    units="",
    species_name="var",
    time_str=None,
    meta=None,
    radii=None
):
    import numpy as np
    import cartopy.crs as ccrs
    from matplotlib.patches import Rectangle

    # ---------- SAFETY ----------
    if radii is None:
        raise ValueError("plot_rectangles: 'radii' must be provided (e.g. [1,2,3]).")

    lats_small = np.asarray(lats_small, dtype=float)
    lons_small = np.asarray(lons_small, dtype=float)

    if not (0 <= ii < len(lats_small) and 0 <= jj < len(lons_small)):
        raise IndexError(f"(ii,jj)=({ii},{jj}) out of bounds.")

    # ---------- GRID SPACING ----------
    dlon = float(np.abs(lons_small[1] - lons_small[0]))
    dlat = float(np.abs(lats_small[1] - lats_small[0]))

    cx = float(lons_small[jj])
    cy = float(lats_small[ii])

    colors = ["black", "red", "blue", "orange", "purple", "brown"]

    # ---------- DRAW RECTANGLES ----------
    for k, r in enumerate(radii, start=1):
        color = colors[(k - 1) % len(colors)]

        width = (2 * r + 1) * dlon
        height = (2 * r + 1) * dlat
        left = cx - width / 2
        bottom = cy - height / 2

        rect = Rectangle(
            (left, bottom),
            width,
            height,
            facecolor="none",
            edgecolor=color,       
            linewidth=2.5,         
            transform=ccrs.PlateCarree(),
            zorder=20              # always above fields & terrain
        )
        ax.add_patch(rect)

    # ---------- TITLE ----------
    if meta is not None:
        ax.set_title(build_meta_title(meta, kind="Map with Sectors"), pad=22)
        ax.figure.subplots_adjust(top=0.84)

    return ax, im



def _sort_by_pressure_with_index(p_hPa, idx_level, *arrays):
    """
    Sort profiles from surface (max p) to top (min p),
    and return the new index of the selected level.

    Returns
    -------
    p_sorted, arrays_sorted..., idx_sorted
    """
    p_hPa = np.asarray(p_hPa)
    order = np.argsort(p_hPa)[::-1]  # descending: surface → top

    p_sorted = p_hPa[order]
    sorted_arrays = []
    for arr in arrays:
        if arr is None:
            sorted_arrays.append(None)
        else:
            arr = np.asarray(arr)
            sorted_arrays.append(arr[order])

    # find where the original idx_level moved to
    idx_sorted = int(np.where(order == idx_level)[0][0])

    return (p_sorted, *sorted_arrays, idx_sorted)