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
    file_tag = "ngfs_processed"
    suffix = f"_{sat_label.strip()}" if sat_label.strip() else ""
    version_dir = Path(out_dir)
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

def window_regrid_metrics(df, bounding_box, R, window_start=None):
    """
    Aggregate over the requested processing window per (lat, lon).
    Uses a single time label (`hour`) equal to window_start when provided.
    """
    df = df.copy()
    df['acq_date_time'] = pd.to_datetime(df['acq_date_time'], utc=True, errors='coerce')

    if window_start is not None:
        hour_label = normalize_hour_utc_naive(window_start)
        df['hour'] = pd.Timestamp(hour_label)
    else:
        # Backward-compatible fallback (hourly buckets).
        df['hour'] = df['acq_date_time'].dt.floor('h').dt.tz_localize(None)

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
        'pixel_area':       ('pixel_area', 'mean'),
        # 'pixel_area_min':   ('pixel_area', 'min'),
        # 'pixel_area_max':   ('pixel_area', 'max'),
        'confidence':       ('confidence', _prefer_one_else_min),
        'quality_flag':     ('quality_flag', _prefer_one_zero_min),
        'type':             ('type', _prefer_one_zero_min),
        'known_incident_type': ('known_incident_type', _mode_first),
    }
    if 'flag_data_source' in df.columns:
        named_aggs['flag_data_source'] = ('flag_data_source', _mode_first)
    if 'feature_frp' in df.columns:
        named_aggs['frp_feature'] = ('feature_frp', 'mean')
    if 'feature_detection_duration' in df.columns:
        named_aggs['duration_max'] = ('feature_detection_duration', 'max')
    for col in LAT_CORNER_COLS + LON_CORNER_COLS:
        if col in df.columns:
            named_aggs[col] = (col, _first_valid)

    agg_df = (df.groupby(keys, as_index=False)
                .agg(**{k:v for k,v in named_aggs.items() if v[0] in df.columns}))

    out = (agg_df.merge(sizes, on=keys, how='left')
                 .sort_values(keys, kind='mergesort')
                 .reset_index(drop=True))
    
    if {"frp_total", "pixel_area_total"}.issubset(out.columns):
        denom = out["pixel_area_total"].astype(float).to_numpy()
        frp_total = out["frp_total"].astype(float).to_numpy()
        out["frp_density"] = np.where(denom > 0, frp_total / denom, 0.0)
    if {"frp_mean", "nobs"}.issubset(out.columns):
        # FRE integration using nominal GOES cadence (5 min) times actual observation count.
        out["fre"]         = out["frp_mean"] * (5.0 * out["nobs"].astype(float))
    if "pixel_area_total" in out.columns:
        out = out.drop(columns=["pixel_area_total"])
        
    return out  # one row per (hour, lat, lon) with metrics and nobs

DEFAULT_FILL_VALUE = -999.0
_META_TABLE: pd.DataFrame | None = None
CORE_COORDS = {"lat", "lon", "time"}
OUTPUT_ALIASES = {
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

def point_dataset_from_window_df(df_window, R, bounding_box, description):
    if df_window.empty:
        raise ValueError("Cannot build point dataset from empty aggregated dataframe.")

    if "hour" in df_window.columns:
        point_time = pd.to_datetime(df_window["hour"], errors="coerce")
    else:
        fallback_time = normalize_hour_utc_naive(df_window['acq_date_time'].iloc[0])
        point_time = pd.Series(np.repeat(np.datetime64(fallback_time), len(df_window)))

    fields = [c for c in [
            'frp_total', 'frp_mean', 'frp_std', 'frp_max',
            'pixel_area', 'nobs',
            'frp_density', 'fre',
            'frp_feature', 'duration_max',
            'confidence', 'quality_flag', 'type', 'known_incident_type',
            'flag_data_source',
         ]
         if c in df_window.columns]

    point_count = len(df_window)
    coords: dict[str, Any] = {
        "point": np.arange(point_count, dtype="int32"),
        "lat": ("point", df_window["latitude"].to_numpy(dtype="float32")),
        "lon": ("point", df_window["longitude"].to_numpy(dtype="float32")),
        "time": ("point", point_time.to_numpy(dtype="datetime64[ns]")),
    }
    data_vars: dict[str, Any] = {
        f: (("point",), df_window[f].to_numpy()) for f in fields
    }
    has_lat_corners = all(c in df_window.columns for c in LAT_CORNER_COLS)
    has_lon_corners = all(c in df_window.columns for c in LON_CORNER_COLS)
    if has_lat_corners and has_lon_corners:
        coords["corner"] = np.arange(1, CORNER_COUNT + 1, dtype="int8")
        data_vars["lat_corners"] = (
            ("point", "corner"),
            df_window[LAT_CORNER_COLS].to_numpy(dtype="float32"),
        )
        data_vars["lon_corners"] = (
            ("point", "corner"),
            df_window[LON_CORNER_COLS].to_numpy(dtype="float32"),
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

def load_static_emissions_lookup(static_file_path, R):
    static_path = Path(static_file_path)
    if not static_path.is_file():
        raise FileNotFoundError(f"Static emissions file not found: {static_path}")

    with xr.open_dataset(static_path) as ds_s:
        required_vars = [
            "lat",
            "lon",
            "MASK_GOES_SOURCE",
            "EFACTOR_PM25",
            "EFACTOR_FLAMING_PM25",
            "BETA_MAP",
            "BETA_VFEI",
            "land_cover_fraction",
        ]
        missing_vars = [var for var in required_vars if var not in ds_s]
        if missing_vars:
            raise ValueError(
                "Static emissions file is missing required variable(s): "
                + ", ".join(missing_vars)
            )

        s_lats = np.asarray(ds_s["lat"].to_numpy(), dtype=np.float32)
        s_lons = np.asarray(ds_s["lon"].to_numpy(), dtype=np.float32)
        if s_lats.size > 1 and s_lons.size > 1:
            r_lat = float(np.abs(np.diff(s_lats).mean()))
            r_lon = float(np.abs(np.diff(s_lons).mean()))
            R_static = float((r_lat + r_lon) / 2.0)
        else:
            R_static = float(R)
        mask_raw = np.asarray(ds_s["MASK_GOES_SOURCE"].to_numpy())
        # xarray may decode _FillValue to NaN; map non-finite values to -1 before int cast.
        mask_goes_source = np.where(np.isfinite(mask_raw), mask_raw, -1).astype(np.int16, copy=False).ravel()
        ef_std = np.asarray(ds_s["EFACTOR_PM25"].to_numpy(), dtype=np.float32).ravel()
        ef_flam = np.asarray(ds_s["EFACTOR_FLAMING_PM25"].to_numpy(), dtype=np.float32).ravel()
        beta_map = np.asarray(ds_s["BETA_MAP"].to_numpy(), dtype=np.float32).ravel()
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
        "R": R_static,
        "lat_min": float(np.min(s_lats)),
        "lat_max": float(np.max(s_lats)),
        "lon_min": float(np.min(s_lons)),
        "nlat": int(s_lats.size),
        "nlon": int(s_lons.size),
        "is_top_down": bool(s_lats[0] > s_lats[-1]),
        "mask_goes_source_flat": mask_goes_source,
        "ef_std_flat": ef_std,
        "ef_flam_flat": ef_flam,
        "beta_map_flat": beta_map,
        "beta_vfei_flat": beta_vfei,
        "land_cover_fraction_flat": lcf_flat,
    }

def select_satellite_rows_by_static_mask(df, static_lookup, sat_source_flag):
    """Filter detections using static MASK_GOES_SOURCE (1=east GOES-19, 2=west GOES-18)."""
    p_lat = np.asarray(df["latitude"].to_numpy(), dtype=np.float32)
    p_lon = np.asarray(df["longitude"].to_numpy(), dtype=np.float32)
    R_static = float(static_lookup["R"])

    if static_lookup["is_top_down"]:
        lat_idx = np.rint((static_lookup["lat_max"] - p_lat) / R_static).astype(np.int64)
    else:
        lat_idx = np.rint((p_lat - static_lookup["lat_min"]) / R_static).astype(np.int64)
    lon_idx = np.rint((p_lon - static_lookup["lon_min"]) / R_static).astype(np.int64)

    lat_idx = np.clip(lat_idx, 0, static_lookup["nlat"] - 1)
    lon_idx = np.clip(lon_idx, 0, static_lookup["nlon"] - 1)
    flat_idx = lat_idx * static_lookup["nlon"] + lon_idx

    mask_source = np.take(static_lookup["mask_goes_source_flat"], flat_idx).astype(np.int16, copy=False)
    out = df.copy()
    out["mask_goes_source"] = mask_source
    out = out[out["mask_goes_source"] == int(sat_source_flag)].copy()
    out["flag_data_source"] = int(sat_source_flag)
    return out

def append_pm25_emissions_to_points(ds_points, static_lookup, debug=False):
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
    beta_map = np.take(static_lookup["beta_map_flat"], flat_idx)
    beta_vfei = np.take(static_lookup["beta_vfei_flat"], flat_idx)
    ef_to_use = np.where(incident_flag == 2, ef_flam, ef_std)

    emis_pm25 = fre * beta_map * ef_to_use * 1e-3
    emis_pm25_vfei = fre * beta_vfei * ef_to_use * 1e-3
    invalid = (~np.isfinite(fre)) | (fre == float(DEFAULT_FILL_VALUE))
    emis_pm25 = np.where(invalid, float(DEFAULT_FILL_VALUE), emis_pm25).astype(np.float32, copy=False)
    emis_pm25_vfei = np.where(invalid, float(DEFAULT_FILL_VALUE), emis_pm25_vfei).astype(np.float32, copy=False)

    out = ds_points.copy()
    out["EMIS_PM25"] = (("point",), emis_pm25)
    out["EMIS_PM25"].attrs.update({
        "units": "kg",
        "long_name": "PM2.5 emissions calculated from NGFS FRE using BETA_MAP",
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
        out["BETA_MAP"] = (("point",), beta_map.astype(np.float32, copy=False))
        out["BETA_MAP"].attrs.update({
            "units": "kg[dry-matter]/MJ",
            "long_name": "Lookup BETA_MAP used for PM2.5 emissions",
            "coordinates": "lat lon time",
        })
        out["BETA_VFEI"] = (("point",), beta_vfei.astype(np.float32, copy=False))
        out["BETA_VFEI"].attrs.update({
            "units": "kg[dry-matter]/MJ",
            "long_name": "Lookup BETA_VFEI used for PM2.5 emissions diagnostics",
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

def write_window_products_nc(
    df_window,
    out_dir,
    R,
    bounding_box,
    start_time,
    valid_date_start,
    valid_date_end,
    integration_minutes=60,
    sat_label="",
    estimate_emissions=False,
    emissions_lookup=None,
    emissions_debug=False,
):
    if df_window.empty:
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

    ds_points = point_dataset_from_window_df(
        df_window=df_window,
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
        "source_selection": "MASK_GOES_SOURCE from static file (1=east GOES-19, 2=west GOES-18)",
        # "Conventions": "CF-1.8",
    })

    if estimate_emissions:
        if emissions_lookup is None:
            raise ValueError("estimate_emissions=True requires loaded emissions_lookup.")
        ds_points = append_pm25_emissions_to_points(
            ds_points,
            static_lookup=emissions_lookup,
            debug=emissions_debug,
        )
    
    write_ngfs_netcdf(ds_points, fn_primary)
    if save_netcdf_points:
        write_ngfs_netcdf(ds_points, fn_points)
    return {
        "grid": fn_primary,
        "point": fn_points if save_netcdf_points else None,
        "grid2d": None,
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

estimate_emissions      = True
emissions_static_file   = None
emissions_debug         = False

# End user definitions
# No further modifications needed beyond this point
# ======================================================================
code_version = "v0.6"
if "cove_version" not in globals():
    cove_version = code_version
# output grid:
# Do not change bounding box, since this is the same used later when 
# generating emissions so we keep the same dimensions and lat/lons as
# the pre processed land use maps (IGBP)
bounding_box    = np.array([-135.0, -50.0, 15.0, 55.0])
# R               = 0.01 # Resolution in degrees
R               = 0.01

# Paths:
REPO_DIR    = Path(__file__).resolve().parent
path_main   = Path(os.environ.get("NGFS_DIR", REPO_DIR)).resolve()
path_netcdf = Path(os.getenv("NGFS_OUTPUT", path_main / "output")).resolve()

if emissions_static_file is None:
    emissions_static_file = path_main / "static" / f"NGFS_STATIC_A2024.061.CONUS.r{R}.nc"

msg(f"Loading static lookup (emissions + MASK_GOES_SOURCE): {emissions_static_file}")
static_lookup = load_static_emissions_lookup(emissions_static_file, R)
emissions_lookup = static_lookup if estimate_emissions else None

# Create output directories:
checkDir(path_netcdf)

# Input directories containing daily NGFS CSV files:
data_dir_w = sys.argv[3]
data_dir_e = sys.argv[4]
checkInputDir(data_dir_w)
checkInputDir(data_dir_e)
    
selected_frames = []
try:
    sat_datasets = [
        ('w', 'GOES-18', data_dir_w, 2),  # MASK_GOES_SOURCE=2 -> keep GOES-18
        ('e', 'GOES-19', data_dir_e, 1),  # MASK_GOES_SOURCE=1 -> keep GOES-19
    ]
    window_day_list = window_dates(start_time, end_time)
    for sat_label, sat_platform, sat_dir, sat_source_flag in sat_datasets:
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
        numeric_cols = (
            cols_to_check
            + ['pixel_area', 'feature_frp', 'feature_detection_duration']
            + LAT_CORNER_COLS
            + LON_CORNER_COLS
        )
        
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
            "feature_frp", "feature_detection_duration",
        ]
        required_cols = [c for c in base_cols if c not in ("feature_frp", "feature_detection_duration")]
        missing_base = [c for c in required_cols if c not in df.columns]
        if missing_base:
            raise ValueError(f"Input CSV is missing required columns: {missing_base}")
        for optional_col in ("feature_frp", "feature_detection_duration"):
            if optional_col not in df.columns:
                df[optional_col] = np.nan
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

        # Select detections by static satellite-source mask using original lat/lon.
        df = select_satellite_rows_by_static_mask(df, static_lookup, sat_source_flag=sat_source_flag)
        if df.empty:
            msg(f"All {sat_platform} detections in window were masked out by MASK_GOES_SOURCE.")
            continue
        
        selected_frames.append(df)
    
    if selected_frames:
        detections = pd.concat(selected_frames, ignore_index=True)
        # Aggregate by hour and original pixel coordinates after static-mask filtering.
        df_window = window_regrid_metrics(detections, bounding_box, R, window_start=start_time)
        if save_netcdf and not df_window.empty:
            write_window_products_nc(
                df_window,
                path_netcdf,
                R,
                bounding_box,
                start_time=start_time,
                valid_date_start=valid_date_start,
                valid_date_end=valid_date_end,
                integration_minutes=integration_minutes,
                sat_label="",
                estimate_emissions=estimate_emissions,
                emissions_lookup=emissions_lookup,
                emissions_debug=emissions_debug,
            )
        else:
            msg("No aggregated detections available after processing window.")
    else:
        msg("No detections available after applying MASK_GOES_SOURCE selection.")
        
    

except FileNotFoundError:
    print("Error: One or more required daily NGFS files could not be found.")
except ImportError:
    print("Error: This script requires pandas, xarray, and numpy.")
    print("Please install them using: pip install pandas xarray numpy")
except Exception as e:
    print(f"An error occurred: {e}")
    
msg("done!")
