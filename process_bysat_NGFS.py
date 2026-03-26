#!/usr/bin/env python
# coding: utf-8
import os
import sys
import pandas as pd
import xarray as xr
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Any

# Functions:
def checkDir(path_check):
    if not os.path.isdir(path_check):
        os.makedirs(path_check, exist_ok=True)

def msg(text):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp}    {text}")

def checkInputDir(path_in):
    if not os.path.isdir(path_in):
        print(f" Error: Input directory not found: {path_in}")
        sys.exit(1)

def normalize_hour_utc_naive(hour_like):
    """Normalize hour-like value to timezone-naive UTC Timestamp."""
    hour = pd.Timestamp(hour_like)
    if hour.tzinfo is not None:
        hour = hour.tz_convert('UTC').tz_localize(None)
    return hour

def window_dates(start_time, end_time):
    """Return all UTC dates touched by the half-open interval [start_time, end_time)."""
    if end_time <= start_time:
        return []
    dates = []
    d0 = start_time.date()
    d1 = (end_time - timedelta(microseconds=1)).date()
    current = d0
    while current <= d1:
        dates.append(current)
        current += timedelta(days=1)
    return dates

def daily_ngfs_csv_path(data_dir, sat_platform, day_obj):
    ymd = day_obj.strftime("%Y_%m_%d")
    jjj = day_obj.strftime("%j")
    filename = f"NGFS_FIRE_DETECTIONS_{sat_platform}_ABI_CONUS_{ymd}_{jjj}.csv"
    return Path(data_dir) / filename

CORNER_COUNT = 4
LAT_CORNER_COLS = [f"latitude_c{i}" for i in range(1, CORNER_COUNT + 1)]
LON_CORNER_COLS = [f"longitude_c{i}" for i in range(1, CORNER_COUNT + 1)]

def get_version_tag() -> str:
    return str(globals().get("cove_version", code_version))

def build_output_paths(out_dir, start_like, R, sat_label="", integration_minutes=60):
    """
    Centralized output path builder for all NGFS NetCDF products.
    Update naming rules here only.
    """
    start_time = normalize_hour_utc_naive(start_like)
    ymdh = start_time.strftime('%Y%m%d%H%M%S')
    dt_minutes = int(integration_minutes)
    if dt_minutes <= 0:
        raise ValueError("integration_minutes must be > 0.")
    period_tag = "hourly" if dt_minutes == 60 else f"{dt_minutes}-min"
    file_tag = f"NGFS_{code_version}_{period_tag}"
    suffix = f"_{sat_label.strip()}" if sat_label.strip() else ""
    version_dir = Path(out_dir) / get_version_tag()
    checkDir(version_dir)
    return {
        "version_dir": version_dir,
        "start_time": start_time,
        "ymdh": ymdh,
        "integration_minutes": dt_minutes,
        "period_tag": period_tag,
        "suffix": suffix,
        "grid": version_dir / f'{file_tag}_{ymdh}{suffix}.nc',
        "point": version_dir / f'{file_tag}_pt_{ymdh}{suffix}.nc',
        "grid2d": version_dir / f'{file_tag}_2d_{ymdh}{suffix}.nc',
    }

def remove_file_if_exists(path_like):
    if path_like is None:
        return
    p = Path(path_like)
    if p.is_file():
        p.unlink()
        msg(f"Removed intermediate file: {p}")
    
def add_grid_cell_area(
    df,
    R,
    lat_col: str = 'latitude',
    area_col: str = 'grid_area_km2',
    radius_km: float = 6371.0):
    """
    Compute spherical grid-cell area (km^2) for a regular lat/lon grid
    of spacing R (degrees), using the *snapped* latitude as the cell center.
    Adds a column `area_col` to df.
    """

    # angular sizes in radians
    dlat = np.deg2rad(R)
    dlon = np.deg2rad(R)

    # cell-center latitude in radians
    lat_center_rad = np.deg2rad(df[lat_col].to_numpy())

    # edges
    lat1 = lat_center_rad - dlat / 2.0
    lat2 = lat_center_rad + dlat / 2.0

    area = (radius_km**2) * np.abs(np.sin(lat2) - np.sin(lat1)) * dlon
    df[area_col] = area
    return df

def snap2grid(df, bounding_box, R,
                 xcol='longitude', ycol='latitude',
                 inplace=True):
    """
    Snap (lon, lat) in df to nearest grid centers defined by
    min_lon/min_lat, spacing R, with centers at min+R/2.
    """
    min_lon, max_lon, min_lat, max_lat = bounding_box  # [lon_min, lon_max, lat_min, lat_max]
    start_lon = min_lon + R/2.0
    start_lat = min_lat + R/2.0

    # number of grid points (for clipping)
    nlon = int(np.floor((max_lon - min_lon)/R))
    nlat = int(np.floor((max_lat - min_lat)/R))

    x = df[xcol].to_numpy(dtype=float, copy=False)
    y = df[ycol].to_numpy(dtype=float, copy=False)

    # indices to nearest center
    i_lon = np.rint((x - start_lon) / R).astype(int)
    i_lat = np.rint((y - start_lat) / R).astype(int)

    # clip to domain
    i_lon = np.clip(i_lon, 0, nlon - 1)
    i_lat = np.clip(i_lat, 0, nlat - 1)

    snapped_lon = start_lon + i_lon * R
    snapped_lat = start_lat + i_lat * R
    
    # round to match grid resolution
    decimals = int(abs(np.log10(R))) + 1
    snapped_lon = np.round(snapped_lon, decimals)
    snapped_lat = np.round(snapped_lat, decimals)

    if inplace:
        df[xcol] = snapped_lon
        df[ycol] = snapped_lat
        return df
    else:
        out = df.copy()
        out[xcol] = snapped_lon
        out[ycol] = snapped_lat
        return out

def _prefer_one_zero_min(s: pd.Series):
    """Return 1 if any 1 present; else 0 if any 0 present; else min of remaining; NaN if empty."""
    s = s.dropna()
    if s.empty:
        return np.nan
    # coerce to numeric if it's object-y but numeric-like
    try:
        s = pd.to_numeric(s, errors='coerce').dropna()
    except Exception:
        pass
    if (s == 1).any():
        return 1
    if (s == 0).any():
        return 0
    return s.min() if not s.empty else np.nan

def _prefer_one_else_min(s: pd.Series):
    """For columns that should 'always keep 1 if present', otherwise min of the values."""
    s = pd.to_numeric(s, errors='coerce').dropna()
    if s.empty:
        return np.nan
    return 1 if (s == 1).any() else s.min()

def _mode_first(s: pd.Series):
    """Mode with tie-break to first encountered non-null."""
    s = s.dropna()
    if s.empty:
        return np.nan
    m = s.mode()
    return m.iloc[0] if len(m) else s.iloc[0]

def _first_valid(s: pd.Series):
    s = s.dropna()
    return s.iloc[0] if not s.empty else np.nan

def regrid_and_aggregate_metrics(
        df: pd.DataFrame,
        bounding_box,
        R,
        time_col='acq_date_time',
        lat_col='latitude',
        lon_col='longitude',
    ):
    """Snap to grid R, then aggregate duplicates on (time, lat, lon) with metrics."""
    # ensure time dtype
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')

    # 1 snap lon/lat to grid
    snap2grid(df, bounding_box, R, xcol=lon_col, ycol=lat_col)

    keys = [time_col, lat_col, lon_col]

    # 2 group sizes (nobs)
    sizes = (df.groupby(keys, as_index=False)
               .size()
               .rename(columns={'size': 'nobs'}))

    # 3 build named aggregations for metrics and flags
    named_aggs = {}

    if 'frp' in df.columns:
        # std with ddof=0 → 0 for single obs (instead of NaN)
        named_aggs['frp_total'] = ('frp', 'sum')
        named_aggs['frp_mean']  = ('frp', 'mean')
        named_aggs['frp_std']   = ('frp', lambda s: s.astype(float).std(ddof=0))
        named_aggs['frp_max']   = ('frp', 'max')

    if 'pixel_area' in df.columns:
        named_aggs['pixel_area_total'] = ('pixel_area', 'sum')
        named_aggs['pixel_area_mean']  = ('pixel_area', 'mean')

    if 'confidence' in df.columns:
        named_aggs['confidence'] = ('confidence', _prefer_one_else_min)

    if 'quality_flag' in df.columns:
        named_aggs['quality_flag'] = ('quality_flag', _prefer_one_zero_min)

    if 'type' in df.columns:
        named_aggs['type'] = ('type', _prefer_one_zero_min)

    if 'known_incident_type' in df.columns:
        named_aggs['known_incident_type'] = ('known_incident_type', _mode_first)

    # 4 aggregate with named aggregation
    agg_df = (df.groupby(keys, as_index=False)
                .agg(**named_aggs))

    # 5 join nobs and sort
    out = (agg_df.merge(sizes, on=keys, how='left')
                 .sort_values(keys, kind='mergesort')
                 .reset_index(drop=True))

    return out

def hourly_regrid_metrics(df, bounding_box, R):
    """
    Snap to grid, then aggregate per (hour, lat, lon).
    Hour window is (HH-1:00, HH:00] -> label at HH (right edge).
    """
    df = df.copy()
    df['acq_date_time'] = pd.to_datetime(df['acq_date_time'], utc=True, errors='coerce')
    
    # GAF this is wrong and makes no sense. Changing it to the left edge, so that
    # fires detected bewteen 2 and 3 pm are labeled as 2 pm, not 3 pm. This makes more
    # sense to the model since at the 2 pm integration it reads the averaged 
    # FRP/emissions for that hour. 
    # define hour key as the RIGHT edge of the window (HH)
    # df['hour'] = df['acq_date_time'].dt.floor('h') + pd.Timedelta(hours=1)
    # define hour key as the LOWER edge of the window (HH)
    df['hour'] = df['acq_date_time'].dt.floor('h').dt.tz_localize(None)

    # snap to grid (updates df['longitude'], df['latitude'] in place + rounding)
    snap2grid(df, bounding_box, R, xcol='longitude', ycol='latitude')
    
    # now add grid cell area
    add_grid_cell_area(df, R, lat_col='latitude', area_col='grid_area_km2')
    
    keys = ['hour', 'latitude', 'longitude']

    # group sizes
    sizes = (df.groupby(keys, as_index=False)
               .size()
               .rename(columns={'size': 'nobs'}))

    # metrics + flags (same rules you asked for)
    named_aggs = {
        'frp_total':        ('frp', 'sum'),
        'frp_mean':         ('frp', 'mean'),
        'frp_std':          ('frp', lambda s: s.astype(float).std(ddof=0)),
        'frp_max':          ('frp', 'max'),
        'pixel_area_total': ('pixel_area', 'sum'),
        'pixel_area_mean':  ('pixel_area', 'mean'),
        "grid_area_km2":    ("grid_area_km2", "first"),   # <- needed for scaling
        'confidence':       ('confidence', _prefer_one_else_min),
        'quality_flag':     ('quality_flag', _prefer_one_zero_min),
        'type':             ('type', _prefer_one_zero_min),
        'known_incident_type': ('known_incident_type', _mode_first),
    }
    for col in LAT_CORNER_COLS + LON_CORNER_COLS:
        if col in df.columns:
            named_aggs[col] = (col, _first_valid)

    agg_df = (df.groupby(keys, as_index=False)
                .agg(**{k:v for k,v in named_aggs.items() if v[0] in df.columns}))

    out = (agg_df.merge(sizes, on=keys, how='left')
                 .sort_values(keys, kind='mergesort')
                 .reset_index(drop=True))
    
    # ---- QFED-style mean FRP for the grid cell ----
    if {"frp_total", "pixel_area_total", "grid_area_km2"}.issubset(out.columns):
        denom = out["pixel_area_total"].astype(float).to_numpy()
        frp_total = out["frp_total"].astype(float).to_numpy()
        grid_area = out["grid_area_km2"].astype(float).to_numpy()

        # out["frp_mean"]    = np.where(denom > 0, (frp_total / denom) * grid_area, 0.0) # v0.1, v0.2
        out["frp_mean"]    = out["frp_mean"] # v0.3
        out["frp_density"] = np.where(denom > 0, frp_total / denom, 0.0)
        # FRE integration using nominal GOES cadence (5 min) times actual observation count.
        out["fre"]         = out["frp_mean"] * (300.0 * out["nobs"].astype(float))
        
    return out  # one row per (hour, lat, lon) with metrics and nobs

DEFAULT_FILL_VALUE = -999.0
_META_TABLE: pd.DataFrame | None = None
CORE_COORDS = {"lat", "lon", "time"}
OUTPUT_ALIASES = {
    "area": "GRID_AREA",
    "confidence": "FLAG_CONFIDENCE",
    "quality_flag": "FLAG_QUALITY",
    "type": "FLAG_TYPE",
    "known_incident_type": "FLAG_KNOWN_INCIDENT",
    "lat_corners": "LAT_CORNERS",
    "lon_corners": "LON_CORNERS",
}

def get_meta_table() -> pd.DataFrame:
    global _META_TABLE
    if _META_TABLE is None:
        meta_path = Path(__file__).with_name("ngfs_nc_variables.csv")
        _META_TABLE = pd.read_csv(meta_path).set_index("varname")
    return _META_TABLE

def _build_meta_name_lookup(meta: pd.DataFrame) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for name in meta.index.astype(str):
        lookup[name.lower()] = name
    return lookup

def output_var_name(var_name: str, meta: pd.DataFrame | None = None) -> str:
    """Convert internal variable names to output NetCDF variable names."""
    if var_name in CORE_COORDS:
        return var_name
    if meta is None:
        meta = get_meta_table()
    meta_lookup = _build_meta_name_lookup(meta)
    if var_name in meta.index:
        return var_name
    if var_name in OUTPUT_ALIASES:
        alias = OUTPUT_ALIASES[var_name]
        return alias if alias in meta.index else alias
    if var_name.upper() in meta.index:
        return var_name.upper()
    return meta_lookup.get(var_name.lower(), var_name.upper())

def _resolve_ds_var_name(ds: xr.Dataset, logical_name: str, meta: pd.DataFrame | None = None) -> str | None:
    """Find the actual dataset variable for a logical internal name."""
    if meta is None:
        meta = get_meta_table()
    candidates = [logical_name, output_var_name(logical_name, meta), logical_name.upper()]
    if logical_name in OUTPUT_ALIASES:
        alias = OUTPUT_ALIASES[logical_name]
        candidates.extend([alias, alias.lower()])
    for cand in candidates:
        if cand in ds:
            return cand
    return None

def rename_ds_for_output(ds: xr.Dataset, meta: pd.DataFrame | None = None) -> xr.Dataset:
    """Rename data variables to output names while preserving lat/lon/time."""
    if meta is None:
        meta = get_meta_table()
    rename_map: dict[str, str] = {}
    drop_vars: list[str] = []
    reserved_targets = set(ds.data_vars) | set(ds.coords)
    for var in ds.data_vars:
        target = output_var_name(var, meta)
        if target != var:
            # Avoid xarray rename collisions when an aliased target already exists.
            if target in reserved_targets:
                drop_vars.append(var)
            else:
                rename_map[var] = target
                reserved_targets.add(target)
    if drop_vars:
        ds = ds.drop_vars(drop_vars)
    if not rename_map:
        return ds
    return ds.rename(rename_map)

def lonlat_axes(bounding_box, R):
    lon_min, lon_max, lat_min, lat_max = bounding_box
    lon = np.arange(lon_min + R/2, lon_max + R/2, R)
    lat = np.arange(lat_min + R/2, lat_max + R/2, R)
    return lon.astype('float32'), lat.astype('float32')

def rasterize_hour_2d(df_hour, lon, lat, R, fields):
    start_lon, start_lat = lon[0], lat[0]
    i_lon = np.clip(np.rint((df_hour['longitude'].to_numpy()-start_lon)/R).astype(int), 0, len(lon)-1)
    i_lat = np.clip(np.rint((df_hour['latitude' ].to_numpy()-start_lat)/R).astype(int), 0, len(lat)-1)
    ny, nx = len(lat), len(lon)
    grids = {}
    for f in fields:
        grids[f] = np.full((ny, nx), DEFAULT_FILL_VALUE, dtype='float32')
    for f in fields:
        grids[f][i_lat, i_lon] = df_hour[f].to_numpy(dtype='float32')
    return grids

def point_dataset_from_hour_df(df_hour, R, bounding_box, description):
    if df_hour.empty:
        raise ValueError("Cannot build point dataset from empty aggregated dataframe.")

    if "hour" in df_hour.columns:
        point_time = pd.to_datetime(df_hour["hour"], errors="coerce")
    else:
        fallback_time = normalize_hour_utc_naive(df_hour['acq_date_time'].iloc[0])
        point_time = pd.Series(np.repeat(np.datetime64(fallback_time), len(df_hour)))

    fields = [c for c in [
            'frp_total', 'frp_mean', 'frp_std', 'frp_max',
            'pixel_area_total', 'pixel_area_mean', 'nobs',
            'frp_density', 'fre',
            'confidence', 'quality_flag', 'type', 'known_incident_type'
         ]
         if c in df_hour.columns]

    point_count = len(df_hour)
    coords: dict[str, Any] = {
        "point": np.arange(point_count, dtype="int32"),
        "lat": ("point", df_hour["latitude"].to_numpy(dtype="float32")),
        "lon": ("point", df_hour["longitude"].to_numpy(dtype="float32")),
        "time": ("point", point_time.to_numpy(dtype="datetime64[ns]")),
    }
    data_vars: dict[str, Any] = {
        f: (("point",), df_hour[f].to_numpy()) for f in fields
    }
    if "grid_area_km2" in df_hour.columns:
        data_vars["area"] = (("point",), df_hour["grid_area_km2"].to_numpy(dtype="float32"))
    has_lat_corners = all(c in df_hour.columns for c in LAT_CORNER_COLS)
    has_lon_corners = all(c in df_hour.columns for c in LON_CORNER_COLS)
    if has_lat_corners and has_lon_corners:
        coords["corner"] = np.arange(1, CORNER_COUNT + 1, dtype="int8")
        data_vars["lat_corners"] = (
            ("point", "corner"),
            df_hour[LAT_CORNER_COLS].to_numpy(dtype="float32"),
        )
        data_vars["lon_corners"] = (
            ("point", "corner"),
            df_hour[LON_CORNER_COLS].to_numpy(dtype="float32"),
        )

    ds = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={
            "grid_spacing_deg": float(R),
            "bounding_box": np.array(bounding_box, dtype='float32'),
            "description": description,
            "layout": "point",
        },
    )
    return ds

def point_dataset_to_grid_dataset(ds_points, R, bounding_box):
    lon, lat = lonlat_axes(bounding_box, R)
    point_lon = np.asarray(ds_points["lon"].to_numpy(), dtype=float)
    point_lat = np.asarray(ds_points["lat"].to_numpy(), dtype=float)
    point_time = ds_points["time"].to_numpy()
    if point_time.size == 0:
        raise ValueError("Point dataset has no time values.")
    hour = np.datetime64(pd.Timestamp(point_time[0]))

    start_lon, start_lat = float(lon[0]), float(lat[0])
    i_lon = np.clip(np.rint((point_lon - start_lon) / R).astype(int), 0, len(lon) - 1)
    i_lat = np.clip(np.rint((point_lat - start_lat) / R).astype(int), 0, len(lat) - 1)

    ny, nx = len(lat), len(lon)
    data_vars: dict[str, Any] = {}
    corner_values = None
    area_var_name = output_var_name("area")
    for var in ds_points.data_vars:
        if var == area_var_name:
            continue
        arr = ds_points[var]
        if "point" not in arr.dims:
            continue
        if arr.ndim == 1:
            grid = np.full((ny, nx), DEFAULT_FILL_VALUE, dtype='float32')
            grid[i_lat, i_lon] = np.asarray(arr.to_numpy(), dtype='float32')
            data_vars[var] = (("time", "lat", "lon"), grid[None, ...])
            continue
        if arr.ndim == 2 and "corner" in arr.dims:
            vals = np.asarray(arr.transpose("point", "corner").to_numpy(), dtype='float32')
            ncorner = vals.shape[1]
            grid = np.full((ncorner, ny, nx), DEFAULT_FILL_VALUE, dtype='float32')
            grid[:, i_lat, i_lon] = vals.T
            data_vars[var] = (("time", "corner", "lat", "lon"), grid[None, ...])
            if "corner" in ds_points.coords:
                corner_values = ds_points["corner"].to_numpy()
            elif corner_values is None:
                corner_values = np.arange(1, ncorner + 1, dtype="int8")
            continue
        raise ValueError(f"Unsupported point variable shape for gridding: {var} dims={arr.dims}")

    coords = {
        "time": [hour],
        "lat": lat,
        "lon": lon,
    }
    if corner_values is not None:
        coords["corner"] = corner_values
    ds_grid = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs=dict(ds_points.attrs),
    )
    ds_grid.attrs["layout"] = "grid"
    ds_grid[area_var_name] = (("lat", "lon"), eartharea(lon, lat).astype("float32"))
    return ds_grid

def load_static_emissions_lookup(static_file_path, R):
    static_path = Path(static_file_path)
    if not static_path.is_file():
        raise FileNotFoundError(f"Static emissions file not found: {static_path}")

    with xr.open_dataset(static_path) as ds_s:
        s_lats = np.asarray(ds_s["lat"].to_numpy(), dtype=np.float32)
        s_lons = np.asarray(ds_s["lon"].to_numpy(), dtype=np.float32)
        ef_std = np.asarray(ds_s["EFACTOR_PM25"].to_numpy(), dtype=np.float32).ravel()
        ef_flam = np.asarray(ds_s["EFACTOR_FLAMING_PM25"].to_numpy(), dtype=np.float32).ravel()
        beta_vfei = np.asarray(ds_s["BETA_VFEI"].to_numpy(), dtype=np.float32).ravel()
        lcf_raw = np.asarray(ds_s["land_cover_fraction"].to_numpy())

        # Normalize land_cover_fraction to shape (nclass, nlat*nlon).
        # Expected class count is 17; support any axis ordering containing (17, nlat, nlon).
        if lcf_raw.ndim != 3:
            raise ValueError("land_cover_fraction must be 3-D in static file.")
        lcf_shape = lcf_raw.shape
        if 17 not in lcf_shape:
            raise ValueError("land_cover_fraction must include a class dimension of size 17.")
        class_axis = int(np.where(np.array(lcf_shape) == 17)[0][0])
        lcf_moved = np.moveaxis(lcf_raw, class_axis, 0)  # -> (17, *, *)
        if lcf_moved.shape[1] != s_lats.size or lcf_moved.shape[2] != s_lons.size:
            raise ValueError(
                "land_cover_fraction spatial dimensions do not match lat/lon dimensions in static file."
            )
        lcf_flat = lcf_moved.reshape(17, -1)

    return {
        "R": float(R),
        "lat_min": float(np.min(s_lats)),
        "lat_max": float(np.max(s_lats)),
        "lon_min": float(np.min(s_lons)),
        "nlat": int(s_lats.size),
        "nlon": int(s_lons.size),
        "is_top_down": bool(s_lats[0] > s_lats[-1]),
        "ef_std_flat": ef_std,
        "ef_flam_flat": ef_flam,
        "beta_vfei_flat": beta_vfei,
        "land_cover_fraction_flat": lcf_flat,
    }

def append_pm25_emissions_to_points(ds_points, static_lookup, beta_val=0.38, debug=False):
    fre_name = _resolve_ds_var_name(ds_points, "fre")
    incident_name = _resolve_ds_var_name(ds_points, "known_incident_type")
    if fre_name is None or incident_name is None:
        raise ValueError("Merged dataset must contain FRE and FLAG_KNOWN_INCIDENT variables for emissions.")

    p_lat = np.asarray(ds_points["lat"].to_numpy(), dtype=np.float32)
    p_lon = np.asarray(ds_points["lon"].to_numpy(), dtype=np.float32)
    fre = np.asarray(ds_points[fre_name].to_numpy(), dtype=np.float32)
    incident_flag = np.asarray(ds_points[incident_name].to_numpy())

    R_static = float(static_lookup["R"])
    if static_lookup["is_top_down"]:
        lat_idx = np.rint((static_lookup["lat_max"] - p_lat) / R_static).astype(np.int64)
    else:
        lat_idx = np.rint((p_lat - static_lookup["lat_min"]) / R_static).astype(np.int64)
    lon_idx = np.rint((p_lon - static_lookup["lon_min"]) / R_static).astype(np.int64)

    lat_idx = np.clip(lat_idx, 0, static_lookup["nlat"] - 1)
    lon_idx = np.clip(lon_idx, 0, static_lookup["nlon"] - 1)
    flat_idx = lat_idx * static_lookup["nlon"] + lon_idx

    ef_std = np.take(static_lookup["ef_std_flat"], flat_idx)
    ef_flam = np.take(static_lookup["ef_flam_flat"], flat_idx)
    beta_vfei = np.take(static_lookup["beta_vfei_flat"], flat_idx)
    ef_to_use = np.where(incident_flag == 2, ef_flam, ef_std)

    emis_pm25 = fre * float(beta_val) * ef_to_use * 1e-3
    emis_pm25_vfei = fre * beta_vfei * ef_to_use * 1e-3
    invalid = (~np.isfinite(fre)) | (fre == float(DEFAULT_FILL_VALUE))
    emis_pm25 = np.where(invalid, float(DEFAULT_FILL_VALUE), emis_pm25).astype(np.float32, copy=False)
    emis_pm25_vfei = np.where(invalid, float(DEFAULT_FILL_VALUE), emis_pm25_vfei).astype(np.float32, copy=False)

    out = ds_points.copy()
    out["EMIS_PM25"] = (("point",), emis_pm25)
    out["EMIS_PM25"].attrs.update({
        "units": "kg",
        "long_name": f"PM2.5 emissions calculated from NGFS FRE (Beta={beta_val})",
        "coordinates": "lat lon time",
    })
    out["EMIS_PM25_VFEI"] = (("point",), emis_pm25_vfei)
    out["EMIS_PM25_VFEI"].attrs.update({
        "units": "kg",
        "long_name": "PM2.5 emissions calculated from NGFS FRE using BETA_VFEI",
        "coordinates": "lat lon time",
    })
    if debug:
        out["EFACTOR_PM25"] = (("point",), ef_std.astype(np.float32, copy=False))
        out["EFACTOR_PM25"].attrs.update({
            "units": "g MJ-1",
            "long_name": "Lookup EFACTOR_PM25 used for PM2.5 emissions diagnostics",
            "coordinates": "lat lon time",
        })
        out["EFACTOR_FLAMING_PM25"] = (("point",), ef_flam.astype(np.float32, copy=False))
        out["EFACTOR_FLAMING_PM25"].attrs.update({
            "units": "g MJ-1",
            "long_name": "Lookup EFACTOR_FLAMING_PM25 used for PM2.5 emissions diagnostics",
            "coordinates": "lat lon time",
        })
        out["EFACTOR_PM25_USED"] = (("point",), ef_to_use.astype(np.float32, copy=False))
        out["EFACTOR_PM25_USED"].attrs.update({
            "units": "g MJ-1",
            "long_name": "Emission factor selected by incident flag (flaming if flag==2, else standard)",
            "coordinates": "lat lon time",
        })
        if "land_cover_fraction_flat" in static_lookup:
            lcf = np.take(static_lookup["land_cover_fraction_flat"], flat_idx, axis=1)
            out = out.assign_coords(lc_class=np.arange(lcf.shape[0], dtype=np.int16))
            out["LAND_COVER_FRACTION"] = (("lc_class", "point"), lcf.astype(np.float32, copy=False))
            out["LAND_COVER_FRACTION"].attrs.update({
                "long_name": "Static land cover fraction profile used for diagnostics",
                "coordinates": "lat lon time",
            })
    return out

def eartharea(lon, lat):
    lon = np.asarray(lon)
    lat = np.asarray(lat)

    # Check 1-D
    if lon.ndim != 1 or lat.ndim != 1:
        raise ValueError("lon and lat must be 1-D.")

    # Uniform spacing assumption
    dlon = np.mean(np.diff(lon))
    dlat = np.mean(np.diff(lat))

    # R = 6371000.0  # meters
    R = 6371.0  # km

    # 2-D grid
    lon2d, lat2d = np.meshgrid(lon, lat, indexing="xy")

    # Convert spacing to radians
    dlat_rad = np.deg2rad(dlat)
    dlon_rad = np.deg2rad(dlon)

    # Latitude edges
    lat1 = np.deg2rad(lat2d - dlat / 2.0)
    lat2 = np.deg2rad(lat2d + dlat / 2.0)

    # Cell area
    garea = (R**2) * dlon_rad * (np.sin(lat2) - np.sin(lat1))
    
    return garea

def write_ngfs_netcdf(ds: xr.Dataset, out_path: Path | str, meta: pd.DataFrame | None = None):
    """
    Apply NGFS variable metadata, enforce dtypes/fill values, and write NetCDF.
    """
    if meta is None:
        meta = get_meta_table()
    ds = rename_ds_for_output(ds.copy(), meta)
    # Rebuild dataset so NetCDF variable declaration keeps all data variables
    # grouped before coordinate variables (stable, predictable order).
    ds = xr.Dataset(
        data_vars={name: ds[name] for name in ds.data_vars},
        coords={name: ds.coords[name] for name in ds.coords},
        attrs=dict(ds.attrs),
    )
    var_fill_values: dict[str, float] = {}

    # Coordinate attributes/dtypes (skip time to preserve CF handling)
    for coord in ds.coords:
        if coord == "time":
            continue
        if coord in meta.index:
            row = meta.loc[coord]
            dtype = str(row.get("type", "")).strip()
            if dtype:
                ds[coord] = ds[coord].astype(dtype)
            long_name = row.get("long_name")
            units = row.get("units")
            if isinstance(long_name, str) and long_name:
                ds[coord].attrs["long_name"] = long_name
            if isinstance(units, str) and units:
                ds[coord].attrs["units"] = units

    # Data variable metadata, dtype enforcement, and fill values
    for var in ds.data_vars:
        row = meta.loc[var] if var in meta.index else None
        target_dtype = ""
        if row is not None:
            target_dtype = str(row.get("type", "")).strip()
        if not target_dtype:
            target_dtype = str(ds[var].dtype)
        is_int = target_dtype.startswith("int")
        fill_value = int(DEFAULT_FILL_VALUE) if is_int else float(DEFAULT_FILL_VALUE)
        ds[var] = ds[var].fillna(fill_value).astype(target_dtype)
        var_fill_values[var] = fill_value

        if row is not None:
            long_name = row.get("long_name")
            units = row.get("units")
            if isinstance(long_name, str) and long_name:
                ds[var].attrs["long_name"] = long_name
            if isinstance(units, str) and units:
                ds[var].attrs["units"] = units

    # Encoding dictionary
    enc: dict[str, dict[str, Any]] = {}
    for var, da in ds.data_vars.items():
        if da.ndim == 3:
            base_chunks = (1, 512, 512)
            chunks = tuple(min(int(dim_len), int(base)) for dim_len, base in zip(da.shape, base_chunks))
        elif da.ndim == 2:
            base_chunks = (512, 512)
            chunks = tuple(min(int(dim_len), int(base)) for dim_len, base in zip(da.shape, base_chunks))
        else:
            chunks = None
        entry: dict[str, Any] = dict(zlib=True, complevel=7, dtype=str(da.dtype))
        if chunks:
            entry["chunksizes"] = chunks
        if var in var_fill_values:
            entry["_FillValue"] = var_fill_values[var]
        enc[var] = entry

    # Coordinate encodings (default float32 for lat/lon if present)
    if "lat" in ds.coords:
        enc["lat"] = {"dtype": str(ds["lat"].dtype)}
    if "lon" in ds.coords:
        enc["lon"] = {"dtype": str(ds["lon"].dtype)}

    out_path = Path(out_path)
    ds.to_netcdf(out_path, format="NETCDF4", encoding=enc)
    return out_path

def write_ngfs_point_netcdf(
    ds_grid: xr.Dataset,
    out_path: Path | str,
    meta: pd.DataFrame | None = None,
    fill_value: float = DEFAULT_FILL_VALUE,
):
    """
    Write compact 1-D point NetCDF from a gridded dataset.
    Keeps only active/valid cells to reduce file size.
    """
    if meta is None:
        meta = get_meta_table()
    ds = rename_ds_for_output(ds_grid.copy(), meta)

    # Stack regular grid into one point dimension.
    if {"time", "lat", "lon"}.issubset(ds.coords):
        stacked = ds.stack(point=("time", "lat", "lon"))
        point_time = stacked.indexes["point"].get_level_values("time").to_numpy()
        point_lat = stacked.indexes["point"].get_level_values("lat").to_numpy(dtype="float32")
        point_lon = stacked.indexes["point"].get_level_values("lon").to_numpy(dtype="float32")
    elif {"lat", "lon"}.issubset(ds.coords):
        stacked = ds.stack(point=("lat", "lon"))
        point_time = None
        point_lat = stacked.indexes["point"].get_level_values("lat").to_numpy(dtype="float32")
        point_lon = stacked.indexes["point"].get_level_values("lon").to_numpy(dtype="float32")
    else:
        raise ValueError("Point conversion expects coordinates including lat/lon.")

    # Keep only active fires (prefer FRE, fallback to FRP_MEAN).
    fre_var = _resolve_ds_var_name(stacked, "fre", meta)
    frp_var = _resolve_ds_var_name(stacked, "frp_mean", meta)
    if fre_var is not None:
        active = stacked[fre_var] > 0
    elif frp_var is not None:
        active = stacked[frp_var] > 0
    else:
        active = xr.ones_like(next(iter(stacked.data_vars.values())), dtype=bool)

    valid = active.fillna(False)
    for var_name, da in stacked.data_vars.items():
        if "point" not in da.dims:
            continue
        if np.issubdtype(da.dtype, np.floating):
            valid = valid & da.notnull() & np.isfinite(da) & (da != float(fill_value))
        elif np.issubdtype(da.dtype, np.integer):
            valid = valid & (da != int(fill_value))

    keep_idx = np.flatnonzero(valid.to_numpy())

    coords = {
        "point": np.arange(len(keep_idx), dtype="int32"),
        "lat": ("point", point_lat[keep_idx]),
        "lon": ("point", point_lon[keep_idx]),
    }
    if point_time is not None:
        coords["time"] = ("point", point_time[keep_idx])

    out_vars = {}
    for var_name, da in stacked.data_vars.items():
        if "point" in da.dims:
            out_vars[var_name] = (("point",), da.to_numpy()[keep_idx])

    ds_points = xr.Dataset(out_vars, coords=coords, attrs=ds.attrs)
    ds_points.attrs["layout"] = "point"
    ds_points.attrs["point_filter"] = "active fires only (FRP_MEAN or FRE > 0), excluding fill/NaN values"

    out_path = Path(out_path)
    write_ngfs_netcdf(ds_points, out_path, meta=meta)
    return out_path

def write_window_products_nc(
    df_hour,
    out_dir,
    R,
    bounding_box,
    start_time,
    valid_date_start,
    valid_date_end,
    integration_minutes=60,
    sat_label="",
):
    if df_hour.empty:
        return None

    paths = build_output_paths(
        out_dir,
        start_time,
        R,
        sat_label=sat_label,
        integration_minutes=integration_minutes,
    )
    fn_primary = paths["grid"]
    fn_points = paths["point"]
    dt_minutes = int(integration_minutes)
    period_desc = "hourly" if dt_minutes == 60 else f"{dt_minutes}-minute integration"

    ds_points = point_dataset_from_hour_df(
        df_hour=df_hour,
        R=R,
        bounding_box=bounding_box,
        description=f"{period_desc} metrics as point-source detections",
    )

    # Global attributes
    ds_points.attrs.update({
        "title": "NGFS integrated point-source FRP metrics",
        "author": "Gonzalo A. Ferrada (gonzalo.ferrada@noaa.gov)",
        "institution": "CIRES/CU Boulder, GSL/NOAA",
        "source": "NGFS point detections (https://cimss.ssec.wisc.edu/ngfs/)",
        "history": f"created {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "creation_date_utc": datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        "valid_date_start": valid_date_start,
        "valid_date_end": valid_date_end,
        "integration_minutes": dt_minutes,
        # "Conventions": "CF-1.8",
    })
    
    write_ngfs_netcdf(ds_points, fn_primary)
    if save_netcdf_points:
        write_ngfs_netcdf(ds_points, fn_points)
    return {
        "grid": fn_primary,
        "point": fn_points if save_netcdf_points else None,
        "grid2d": None,
    }

def _fire_presence_mask(ds: xr.Dataset):
    """Return boolean mask where any core fire metric is > 0."""
    candidate_vars = ['nobs', 'pixel_area_total', 'frp_total', 'pixel_area_mean', 'frp_mean', 'fre']
    mask = None
    for name in candidate_vars:
        ds_name = _resolve_ds_var_name(ds, name)
        if ds_name is not None:
            da = ds[ds_name]
            candidate = da > 0
            mask = candidate if mask is None else (mask | candidate)
    if mask is None:
        for name, da in ds.data_vars.items():
            if np.issubdtype(da.dtype, np.number):
                candidate = da != 0
                mask = candidate if mask is None else (mask | candidate)
    if mask is None:
        raise ValueError("Unable to determine fire mask; dataset lacks numeric variables.")
    return mask.fillna(False)

def _merge_variable(prefer_east, presence_mask, arr_east, arr_west, fill_value=DEFAULT_FILL_VALUE):
    if arr_east is None and arr_west is None:
        raise ValueError("Variable missing from both datasets during merge.")
    target = arr_east if arr_east is not None else arr_west
    if arr_east is None:
        arr_east = xr.zeros_like(arr_west)
    if arr_west is None:
        arr_west = xr.zeros_like(arr_east)
    arr_e = arr_east
    arr_w = arr_west
    if not np.issubdtype(arr_e.dtype, np.floating):
        arr_e = arr_e.astype('float32')
    if not np.issubdtype(arr_w.dtype, np.floating):
        arr_w = arr_w.astype('float32')
    merged = xr.where(prefer_east, arr_e, arr_w)
    merged = merged.where(presence_mask, fill_value).fillna(fill_value)
    if np.issubdtype(target.dtype, np.integer):
        merged = merged.round().astype(target.dtype)
    else:
        merged = merged.astype(target.dtype)
    return merged

def _point_ds_to_dataframe(ds: xr.Dataset) -> pd.DataFrame:
    if "point" not in ds.dims:
        raise ValueError("Point dataset missing 'point' dimension.")

    point_count = int(ds.sizes["point"])
    df = pd.DataFrame({
        "time": pd.to_datetime(ds["time"].to_numpy(), errors="coerce"),
        "lat": np.asarray(ds["lat"].to_numpy(), dtype="float32"),
        "lon": np.asarray(ds["lon"].to_numpy(), dtype="float32"),
    })
    if len(df) != point_count:
        raise ValueError("Point dataset coordinate lengths are inconsistent.")

    for var_name, da in ds.data_vars.items():
        if "point" not in da.dims:
            continue
        if da.ndim == 1:
            df[var_name] = da.to_numpy()
            continue
        if da.ndim == 2 and "corner" in da.dims:
            vals = np.asarray(da.transpose("point", "corner").to_numpy())
            if vals.shape[1] != CORNER_COUNT:
                raise ValueError(f"Expected {CORNER_COUNT} corners for variable '{var_name}'.")
            for i in range(1, CORNER_COUNT + 1):
                df[f"{var_name}_{i}"] = vals[:, i - 1]
            continue
        raise ValueError(f"Unsupported point variable shape in merge input: {var_name} dims={da.dims}")

    key_cols = ["time", "lat", "lon"]
    for col in key_cols:
        if col not in df.columns:
            raise ValueError(f"Point dataset missing '{col}' coordinate.")
    df = df.dropna(subset=key_cols)
    df = df.sort_values(key_cols, kind="mergesort").reset_index(drop=True)
    return df

def _merge_window_point_dataframes(df_e: pd.DataFrame, df_w: pd.DataFrame, pix_var_e: str, pix_var_w: str) -> pd.DataFrame:
    key_cols = ["time", "lat", "lon"]
    vars_e = [c for c in df_e.columns if c not in key_cols]
    vars_w = [c for c in df_w.columns if c not in key_cols]
    df_e_ren = df_e.rename(columns={c: f"{c}_e" for c in vars_e})
    df_w_ren = df_w.rename(columns={c: f"{c}_w" for c in vars_w})
    merged = df_e_ren.merge(df_w_ren, on=key_cols, how="outer", indicator=True)

    pix_e_col = f"{pix_var_e}_e"
    pix_w_col = f"{pix_var_w}_w"
    if pix_e_col not in merged.columns or pix_w_col not in merged.columns:
        raise ValueError("Both point datasets must contain pixel area mean variable for merge.")

    both = merged["_merge"] == "both"
    left_only = merged["_merge"] == "left_only"
    right_only = merged["_merge"] == "right_only"
    pix_e = merged[pix_e_col].where(merged[pix_e_col] > 0)
    pix_w = merged[pix_w_col].where(merged[pix_w_col] > 0)
    prefer_pixels = np.where(
        pix_e.notna() & pix_w.notna(),
        pix_e <= pix_w,
        np.where(pix_e.notna(), True, np.where(pix_w.notna(), False, True)),
    )
    prefer_e = np.where(left_only, True, np.where(right_only, False, prefer_pixels))
    prefer_e = np.where(both | left_only | right_only, prefer_e, False)

    out = merged[key_cols].copy()
    union_vars = sorted(set(vars_e) | set(vars_w))
    for var in union_vars:
        e_col = f"{var}_e"
        w_col = f"{var}_w"
        if e_col in merged.columns and w_col in merged.columns:
            out[var] = np.where(prefer_e, merged[e_col], merged[w_col])
        elif e_col in merged.columns:
            out[var] = merged[e_col]
        elif w_col in merged.columns:
            out[var] = merged[w_col]

    out["flag_data_source"] = np.where(prefer_e, 1, 2).astype("int16")
    out = out.sort_values(key_cols, kind="mergesort").reset_index(drop=True)
    return out

def _point_dataframe_to_dataset(df: pd.DataFrame, attrs: dict[str, Any] | None = None) -> xr.Dataset:
    key_cols = ["time", "lat", "lon"]
    vars_out = [c for c in df.columns if c not in key_cols]
    corner_bases = []
    for base in ("lat_corners", "lon_corners", "LAT_CORNERS", "LON_CORNERS"):
        needed = [f"{base}_{i}" for i in range(1, CORNER_COUNT + 1)]
        if all(col in vars_out for col in needed):
            corner_bases.append(base)
    corner_cols = {f"{base}_{i}" for base in corner_bases for i in range(1, CORNER_COUNT + 1)}

    coords: dict[str, Any] = {
        "point": np.arange(len(df), dtype="int32"),
        "lat": ("point", df["lat"].to_numpy(dtype="float32")),
        "lon": ("point", df["lon"].to_numpy(dtype="float32")),
        "time": ("point", pd.to_datetime(df["time"], errors="coerce").to_numpy()),
    }
    if corner_bases:
        coords["corner"] = np.arange(1, CORNER_COUNT + 1, dtype="int8")

    data_vars = {
        var: (("point",), df[var].to_numpy())
        for var in vars_out
        if var not in corner_cols
    }
    for base in corner_bases:
        cols = [f"{base}_{i}" for i in range(1, CORNER_COUNT + 1)]
        data_vars[base] = (("point", "corner"), df[cols].to_numpy(dtype="float32"))

    ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=(attrs or {}))
    ds.attrs["layout"] = "point"
    return ds

def merge_window_satellite_grids(
    start_time,
    integration_minutes,
    valid_date_start,
    valid_date_end,
    east_nc,
    west_nc,
    out_dir,
    R,
    bounding_box,
    estimate_emissions=False,
    emissions_lookup=None,
    emissions_beta=0.38,
    emissions_debug=False,
):
    paths = build_output_paths(
        out_dir,
        start_time,
        R,
        sat_label="",
        integration_minutes=integration_minutes,
    )
    ymdh = paths["ymdh"]
    combined_path = paths["grid"]
    combined_point_path = paths["point"]
    combined_grid2d_path = paths["grid2d"]
    msg(f"Merging GOES east/west grids for integration window starting {ymdh}Z")
    with xr.open_dataset(east_nc) as ds_east, xr.open_dataset(west_nc) as ds_west:
        pix_var_e = _resolve_ds_var_name(ds_east, 'pixel_area_mean')
        pix_var_w = _resolve_ds_var_name(ds_west, 'pixel_area_mean')
        if pix_var_e is None or pix_var_w is None:
            raise ValueError("Both satellite NetCDFs must contain pixel area mean variable")
        df_e = _point_ds_to_dataframe(ds_east)
        df_w = _point_ds_to_dataframe(ds_west)
        merged_df = _merge_window_point_dataframes(df_e, df_w, pix_var_e, pix_var_w)
        merged_attrs = dict(ds_west.attrs)
        merged_attrs['description'] = (
            ds_west.attrs.get('description', 'NGFS integrated metrics') +
            ' (east/west merged by minimum pixel area mean)'
        )
        merged_attrs['merge_note'] = (
            "Values pulled from satellite with smaller pixel area mean; "
            "flag_data_source indicates origin (1=east, 2=west)."
        )
        merged_attrs["valid_date_start"] = valid_date_start
        merged_attrs["valid_date_end"] = valid_date_end
        merged_attrs["integration_minutes"] = int(integration_minutes)
        combined = _point_dataframe_to_dataset(merged_df, attrs=merged_attrs)
        combined['flag_data_source'].attrs.update({
            "long_name": "Source satellite for point (1=east, 2=west)",
            "flag_values": [1, 2],
            "flag_meanings": "east west",
        })
        if estimate_emissions:
            if emissions_lookup is None:
                raise ValueError("estimate_emissions=True requires loaded emissions_lookup.")
            combined = append_pm25_emissions_to_points(
                combined,
                static_lookup=emissions_lookup,
                beta_val=emissions_beta,
                debug=emissions_debug,
            )
        write_ngfs_netcdf(combined, combined_path)
        if save_netcdf_points:
            write_ngfs_netcdf(combined, combined_point_path)
        if save_netcdf_2d:
            combined_grid = point_dataset_to_grid_dataset(combined, R, bounding_box)
            write_ngfs_netcdf(combined_grid, combined_grid2d_path)
    return {
        "grid": combined_path,
        "point": combined_point_path if save_netcdf_points else None,
        "grid2d": combined_grid2d_path if save_netcdf_2d else None,
    }
    
# ======================================================================
# User defined:
# PARSE ARGUMENTS FOR REAL TIME PROCESSING
if len(sys.argv) < 5:
    print("Usage: python process_bysat_NGFS.py   YYYY-MM-DD_HH:MM:SS   integration_minutes   path/to/goes_west_dir   path/to/goes_east_dir")
    sys.exit(1)

arg_timestamp = sys.argv[1]
arg_integration_minutes = sys.argv[2]
try:
    current_time = datetime.strptime(arg_timestamp, "%Y-%m-%d_%H:%M:%S")
    integration_minutes = int(arg_integration_minutes)
    if integration_minutes <= 0:
        raise ValueError
    start_time = normalize_hour_utc_naive(current_time)
    end_time = start_time + timedelta(minutes=integration_minutes)
    valid_date_start = start_time.strftime("%Y-%m-%d_%H:%M:%S")
    valid_date_end = end_time.strftime("%Y-%m-%d_%H:%M:%S")
except ValueError:
    print("Error: Timestamp must be YYYY-MM-DD_HH:MM:SS and integration_minutes must be a positive integer.")
    sys.exit(1)

msg(f"Processing integration window: {valid_date_start} -> {valid_date_end} ({integration_minutes} min)")

# Save options:
save_netcdf             = True
save_netcdf_points      = False
save_netcdf_2d          = False
remove_intermediate     = True

estimate_emissions      = True
emissions_beta          = 0.38
emissions_static_file   = None
emissions_debug         = False

# Paths:
path_main   = "/gpfs/f6/drsa-fire3/scratch/Gonzalo.Ferrada/FIRE/NGFS"
path_netcdf = path_main + "/output"
# path_netcdf = sys.argv[4]

# End user definitions
# No further modifications needed beyond this point
# ======================================================================
code_version = "v0.4"
if "cove_version" not in globals():
    cove_version = code_version
# output grid:
# Do not change bounding box, since this is the same used later when 
# generating emissions so we keep the same dimensions and lat/lons as
# the pre processed land use maps (IGBP)
bounding_box    = np.array([-135.0, -50.0, 15.0, 55.0])
# R               = 0.01 # Resolution in degrees
R               = 0.03

if emissions_static_file is None:
    emissions_static_file = f"{path_main}/static/NGFS_STATIC_A2024.061.CONUS.r{R}.nc"

emissions_lookup = None
if estimate_emissions:
    msg(f"Loading static emissions lookup: {emissions_static_file}")
    emissions_lookup = load_static_emissions_lookup(emissions_static_file, R)

# Create output directories:
checkDir(path_netcdf)

# Input directories containing daily NGFS CSV files:
data_dir_w = sys.argv[3]
data_dir_e = sys.argv[4]
checkInputDir(data_dir_w)
checkInputDir(data_dir_e)
    
sat_window_files = {}
try:
    sat_datasets = [
        ('w', 'GOES-18', data_dir_w),
        ('e', 'GOES-19', data_dir_e),
    ]
    window_day_list = window_dates(start_time, end_time)
    for sat_label, sat_platform, sat_dir in sat_datasets:
        msg(f"Processing {sat_platform} detections ({sat_label}) from directory: {sat_dir}")

        csv_files = []
        missing_files = []
        for d in window_day_list:
            fpath = daily_ngfs_csv_path(sat_dir, sat_platform, d)
            if fpath.is_file():
                csv_files.append(fpath)
            else:
                missing_files.append(fpath)

        if missing_files:
            msg(
                f"{sat_platform}: {len(missing_files)} daily file(s) missing for requested window. "
                f"Continuing with available files."
            )
            for mf in missing_files:
                msg(f"Missing: {mf}")

        if not csv_files:
            msg(f"No input CSV files found for {sat_platform} in requested window dates.")
            continue

        frames = []
        for fpath in csv_files:
            msg(f"Reading data from {fpath}")
            frames.append(pd.read_csv(fpath))
        df = pd.concat(frames, ignore_index=True)
        
        # Define numeric columns for pixel center point and pixel corners.
        cols_to_check = ['latitude', 'longitude', 'frp']
        numeric_cols = cols_to_check + ['pixel_area'] + LAT_CORNER_COLS + LON_CORNER_COLS
        
        # Remove rows with missing values
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=cols_to_check, inplace=True)
        
        # Keep only columns we use:
        base_cols = [
            "acq_date_time", "latitude", "longitude", "frp",
            "pixel_area", "confidence", "quality_flag", "type",
            "known_incident_type",
        ]
        missing_base = [c for c in base_cols if c not in df.columns]
        if missing_base:
            raise ValueError(f"Input CSV is missing required columns: {missing_base}")
        corner_cols = [c for c in (LAT_CORNER_COLS + LON_CORNER_COLS) if c in df.columns]
        df = df[base_cols + corner_cols]

        # Parse timestamps once, keep only the requested integration window before regridding.
        df["acq_date_time"] = pd.to_datetime(df["acq_date_time"], utc=True, errors="coerce")
        df.dropna(subset=["acq_date_time"], inplace=True)
        start_time_utc = pd.Timestamp(start_time, tz="UTC")
        end_time_utc = pd.Timestamp(end_time, tz="UTC")
        df = df[(df["acq_date_time"] >= start_time_utc) & (df["acq_date_time"] < end_time_utc)]
        if df.empty:
            msg(f"No detections found for {sat_platform} during requested window.")
            continue
        
        # Filter for bounding box
        df = df[(df['longitude'] > bounding_box[0]) & (df['longitude'] < bounding_box[1])]
        df = df[(df['latitude'] > bounding_box[2]) & (df['latitude'] < bounding_box[3])]
        
        # Sort data by time, lat and lon
        df.sort_values(
            by=['acq_date_time', 'latitude', 'longitude'],
            ascending=[True, True, True],
            inplace=True
        )
        
        # Replace values in known_incident_type
        mapping = {"WF": 1, "RX": 2}
        df['known_incident_type'] = (
            df['known_incident_type']
            .replace("", np.nan)          # treat empty string as NaN
            .map(mapping)                 # map WF->1, RX->2
            .fillna(-999)                 # replace NaN with -999
            .astype(int)                  # make sure column is integer
        )
        
        # Replace confidence values
        mapping = {"low": 0, "nominal": 1, "high": 2}
        df['confidence'] = (
            df['confidence']
            .replace("", np.nan)          # treat empty string as NaN
            .map(mapping)                 # map WF->1, RX->2
            .fillna(-999)                 # replace NaN with -999
            .astype(int)                  # make sure column is integer
        )
        
        # Remove type == 3 and 4:
        df.drop(df[df['type'].isin([3, 4])].index, inplace=True)
        
        # Regrid and aggregate by hour bins for all detections in the selected window.
        df_hourly = hourly_regrid_metrics(df, bounding_box, R)

        # Save
        if save_netcdf and not df_hourly.empty:
            nc_paths = write_window_products_nc(
                df_hourly,
                path_netcdf,
                R,
                bounding_box,
                start_time=start_time,
                valid_date_start=valid_date_start,
                valid_date_end=valid_date_end,
                integration_minutes=integration_minutes,
                sat_label=sat_label,
            )
            if nc_paths:
                sat_window_files[sat_label] = nc_paths
    
    if save_netcdf:
        # Merge if both satellites have data for the requested integration window.
        if "e" in sat_window_files and "w" in sat_window_files:
            merge_window_satellite_grids(
                start_time,
                integration_minutes,
                valid_date_start,
                valid_date_end,
                sat_window_files['e']["grid"],
                sat_window_files['w']["grid"],
                path_netcdf,
                R,
                bounding_box,
                estimate_emissions=estimate_emissions,
                emissions_lookup=emissions_lookup,
                emissions_beta=emissions_beta,
                emissions_debug=emissions_debug,
            )
            if remove_intermediate:
                for sat_label in ("e", "w"):
                    sat_paths = sat_window_files.get(sat_label)
                    if not sat_paths:
                        continue
                    remove_file_if_exists(sat_paths.get("grid"))
                    remove_file_if_exists(sat_paths.get("point"))
                    remove_file_if_exists(sat_paths.get("grid2d"))
        else:
            msg("Skipping merge: detections in the requested window are not available in both GOES-18 and GOES-19.")
        
    

except FileNotFoundError:
    print("Error: One or more required daily NGFS files could not be found.")
except ImportError:
    print("Error: This script requires pandas, matplotlib, and cartopy.")
    print("Please install them using: pip install pandas matplotlib cartopy")
except Exception as e:
    print(f"An error occurred: {e}")
    
msg("done!")
