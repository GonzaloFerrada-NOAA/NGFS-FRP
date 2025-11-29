#!/usr/bin/env python
# coding: utf-8
import os
import sys
import glob
import pandas as pd
import xarray as xr
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone

# Functions:
def checkDir(path_check):
    if not os.path.isdir(path_check):
        os.makedirs(path_check, exist_ok=True)

def checkFile(file_in):
    # Check if a file exists. Exit with error if not found.
    if not os.path.isfile(file_in):
        print(f" Error: File not found: {file_in}")
        sys.exit(1)   # exit with error code

def msg(text):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp}    {text}")
    
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
    df['hour'] = df['acq_date_time'].dt.floor('h')

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

        out["frp_mean"]    = np.where(denom > 0, (frp_total / denom) * grid_area, 0.0)
        out["frp_density"] = np.where(denom > 0, frp_total / denom, 0.0)
        
    return out  # one row per (hour, lat, lon) with metrics and nobs

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
    # grids = {f: np.full((ny, nx), np.nan, dtype='float32') for f in fields}
    grids = {f: np.full((ny, nx), 0.0, dtype='float32') for f in fields}
    for f in fields:
        grids[f][i_lat, i_lon] = df_hour[f].to_numpy(dtype='float32')
    return grids

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

def write_hour_grid_nc(df_hour, out_dir, R, bounding_box, sat_label=""):
    if df_hour.empty:
        return None
    hour = df_hour['hour'].iloc[0]             # right-edge time label (HH)
    # ensure timezone-naive UTC before converting to numpy datetime64
    if hasattr(hour, 'tzinfo') and hour.tzinfo is not None:
        hour = hour.tz_convert('UTC').tz_localize(None)
    ymdh = hour.strftime('%Y%m%d_%H')
    Rout = f"{R}".replace('.', 'p')
    label = sat_label.strip()
    suffix = f"_{label}" if label else ""
    fn = Path(out_dir) / f'NGFS_{ymdh}Z_{Rout}{suffix}.nc'

    lon, lat = lonlat_axes(bounding_box, R)
    fields = [c for c in [
            'frp_total', 'frp_mean', 'frp_std', 'frp_max',
            'pixel_area_total', 'pixel_area_mean', 'nobs',
            'frp_density',
            'confidence', 'quality_flag', 'type', 'known_incident_type'
         ]
         if c in df_hour.columns]
    grids = rasterize_hour_2d(df_hour, lon, lat, R, fields)

    ds = xr.Dataset(
        {f: (('time','lat','lon'), grids[f][None, ...]) for f in grids},
        coords={'time': [np.datetime64(hour)], 'lat': lat, 'lon': lon},
        attrs={'grid_spacing_deg': float(R),
               'bounding_box': np.array(bounding_box, dtype='float32'),
               'description': 'Hourly metrics on regular lat/lon grid'}
    )
    
    # Add grid area:
    garea = eartharea(lon, lat)
    ds['area'] = (('lat', 'lon'), garea.astype('float32'))
    ds['area'].attrs['units'] = 'km2'
    ds['area'].attrs['long_name'] = 'grid cell area'
    
    # 1) Variable attributes
    var_attrs = {
        "frp_total":        {"long_name": "Total FRP in grid cell", "units": "MW"},
        "frp_mean":         {"long_name": "Grid-cell mean FRP (area-normalized then scaled to cell area)", "units": "MW"},
        "frp_std":          {"long_name": "Standard deviation of FRP detections in grid cell", "units": "MW"},
        "frp_max":          {"long_name": "Maximum FRP detection in grid cell", "units": "MW"},
        "frp_density":      {"long_name": "FRP density over detected pixel area", "units": "MW km-2"},
        "pixel_area_total": {"long_name": "Total detected pixel area in grid cell", "units": "km2"},
        "pixel_area_mean":  {"long_name": "Mean detected pixel area", "units": "km2"},
        "nobs":             {"long_name": "Number of detections in grid cell", "units": "1"},
        "area":             {"long_name": "Grid cell area", "units": "km2"},
        "confidence":        {"long_name": "NGFS confidence flag (min value within nobs)", "units": "1"},
        "quality_flag":      {"long_name": "NGFS quality flag (1 or min within nobs)",   "units": "1"},
        "type":              {"long_name": "NGFS detection type flag (1 or min within nobs)", "units": "1"},
        "known_incident_type": {"long_name": "NGFS known incident type (mode within nobs)", "units": "1"},
    }

    for v, a in var_attrs.items():
        if v in ds:
            ds[v].attrs.update(a)

    # Coordinate attrs
    ds["lat"].attrs.update({"long_name": "latitude",  "units": "degrees_north"})
    ds["lon"].attrs.update({"long_name": "longitude", "units": "degrees_east"})
    ds["time"].attrs.update({"long_name": "time"})

    # 2) Global attributes
    ds.attrs.update({
        "title": "NGFS hourly gridded FRP metrics",
        "author": "Gonzalo A. Ferrada (gonzalo.ferrada@noaa.gov)",
        "institution": "CIRES/CU Boulder, GSL/NOAA",
        "source": "NGFS point detections (https://cimss.ssec.wisc.edu/ngfs/)",
        "history": f"created {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "creation_date_utc": datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        # "Conventions": "CF-1.8",
    })
    
    
    # NetCDF4 + deflate level 7 + chunking
    # msg(f"Saving {fn}")
    enc = {f: dict(zlib=True, complevel=7, chunksizes=(1, 512, 512), dtype='float32') for f in grids}
    # enc = {f: dict(zlib=True, complevel=7, dtype='float32') for f in grids}
    # enc.update({'lat': {'dtype':'float32'}, 'lon': {'dtype':'float32'}})
    enc.update({
        'lat':  {'dtype': 'float32'},
        'lon':  {'dtype': 'float32'},
        'area': dict(zlib=True,
                    complevel=7,
                    chunksizes=(512, 512),
                    dtype='float32')
    }) # type: ignore
    ds.to_netcdf(fn, format='NETCDF4', encoding=enc)
    return fn

def _fire_presence_mask(ds: xr.Dataset):
    """Return boolean mask where any core fire metric is > 0."""
    candidate_vars = ['nobs', 'pixel_area_total', 'frp_total', 'pixel_area_mean', 'frp_mean']
    mask = None
    for name in candidate_vars:
        if name in ds:
            da = ds[name]
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

def _merge_variable(prefer_east, presence_mask, arr_east, arr_west):
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
    merged = merged.where(presence_mask, 0.0).fillna(0.0)
    if np.issubdtype(target.dtype, np.integer):
        merged = merged.round().astype(target.dtype)
    else:
        merged = merged.astype(target.dtype)
    return merged

def merge_hourly_satellite_grids(hour, east_nc, west_nc, out_dir, R):
    hour = pd.to_datetime(hour)
    ymdh = hour.strftime('%Y%m%d_%H')
    Rout = f"{R}".replace('.', 'p')
    combined_path = Path(out_dir) / f'NGFS_{ymdh}Z_{Rout}.nc'
    msg(f"Merging GOES east/west grids for {ymdh}Z")
    with xr.open_dataset(east_nc) as ds_east, xr.open_dataset(west_nc) as ds_west:
        if 'pixel_area_mean' not in ds_east or 'pixel_area_mean' not in ds_west:
            raise ValueError("Both satellite NetCDFs must contain 'pixel_area_mean'")
        ds_east, ds_west = xr.align(ds_east, ds_west, join='outer')
        mask_e = _fire_presence_mask(ds_east)
        mask_w = _fire_presence_mask(ds_west)
        combined_presence = (mask_e | mask_w)
        pix_e = ds_east['pixel_area_mean'].where(ds_east['pixel_area_mean'] > 0)
        pix_w = ds_west['pixel_area_mean'].where(ds_west['pixel_area_mean'] > 0)
        prefer_pixels = xr.where(
            pix_e.notnull() & pix_w.notnull(),
            pix_e <= pix_w,
            xr.where(pix_e.notnull(), True, xr.where(pix_w.notnull(), False, False))
        )
        prefer_e = xr.where(mask_e & ~mask_w, True,
                     xr.where(~mask_e & mask_w, False, prefer_pixels))
        prefer_e = prefer_e.where(combined_presence, False).fillna(False)
        combined_vars = {}
        for var in sorted(set(ds_east.data_vars) | set(ds_west.data_vars)):
            if var == 'area':
                continue
            arr_e = ds_east[var] if var in ds_east else None
            arr_w = ds_west[var] if var in ds_west else None
            combined_vars[var] = _merge_variable(prefer_e, combined_presence, arr_e, arr_w)
        if 'area' in ds_east:
            combined_vars['area'] = ds_east['area']
        elif 'area' in ds_west:
            combined_vars['area'] = ds_west['area']
        combined = xr.Dataset(combined_vars, coords=ds_east.coords)
        origin_flag = xr.zeros_like(prefer_e, dtype='int16')
        origin_flag = xr.where(mask_e & ~mask_w, 1, origin_flag)
        origin_flag = xr.where(~mask_e & mask_w, 2, origin_flag)
        origin_flag = xr.where(mask_e & mask_w, xr.where(prefer_e, 1, 2), origin_flag)
        origin_flag = origin_flag.where(combined_presence, 0)
        combined['data_source'] = origin_flag
        combined['data_source'].attrs.update({
            "long_name": "Source satellite for gridded cell (1=east, 2=west)",
            "flag_values": [1, 2],
            "flag_meanings": "east west",
        })
        combined.attrs.update(ds_west.attrs)
        combined.attrs['description'] = (
            ds_west.attrs.get('description', 'NGFS hourly metrics') +
            ' (east/west merged by minimum pixel_area_mean)'
        )
        combined.attrs['merge_note'] = (
            "Values pulled from satellite with smaller pixel_area_mean; "
            "data_source flag indicates origin (1=east, 2=west)."
        )
        enc = {}
        for var, da in combined.data_vars.items():
            if da.ndim == 3:
                chunks = (1, 512, 512)
            elif da.ndim == 2:
                chunks = (512, 512)
            else:
                chunks = None
            enc[var] = dict(zlib=True, complevel=7, dtype=str(da.dtype))
            if chunks:
                enc[var]['chunksizes'] = chunks
        enc.update({
            'lat': {'dtype': 'float32'},
            'lon': {'dtype': 'float32'}
        })
        combined.to_netcdf(combined_path, format='NETCDF4', encoding=enc)
    return combined_path
    
# ======================================================================
# User defined:
DATE1 = '2025-08-14'
DATE2 = '2025-08-14'
dates = pd.date_range(start=DATE1, end=DATE2, freq='D')

# Save options:
save_netcdf = True
save_csv    = True

# Paths:
path_main   = "/gpfs/f6/drsa-fire3/scratch/Gonzalo.Ferrada/FIRE/NGFS"
path_in     = path_main + "/data"
path_png    = path_main + "/png"
path_csv    = path_main + "/gridded/csv"
path_netcdf = path_main + "/gridded/netcdf"

# End user definitions
# No further modifications needed beyond this point
# ======================================================================
# output grid:
# bounding_box    = np.array([-170.0, -50.0, 15.0, 75.0])
bounding_box    = np.array([-138.0, -57.0, 20.0, 54.0])
# R               = 0.01 # Resolution in degrees
R               = 0.03

# Create output directories:
checkDir(path_csv)
checkDir(path_netcdf)

# Loop through dates
for d in dates:
    
    sdate   = d.strftime("%Y_%m_%d")    # "YYYY_MM_DD"
    sdoy    = d.strftime("%j")          # "JJJ"

    # construct full file path of ngfs:
    file_w = f"{path_in}/NGFS_FIRE_DETECTIONS_GOES-18_ABI_CONUS_{sdate}_{sdoy}.csv"
    file_e = f"{path_in}/NGFS_FIRE_DETECTIONS_GOES-19_ABI_CONUS_{sdate}_{sdoy}.csv"
    checkFile(file_w)
    checkFile(file_e)
    
    sat_hour_files = {'w': {}, 'e': {}}
    try:
        # Read the wildfire data from the CSV file
        msg(f"Reading data from {file_w}")
        dfw = pd.read_csv(file_w)
        
        msg(f"Reading data from {file_e}")
        dfe = pd.read_csv(file_e)
        
        sat_datasets = [
            ('w', dfw, 'GOES-18'),
            ('e', dfe, 'GOES-19'),
        ]
        for sat_label, df_raw, sat_name in sat_datasets:
            msg(f"Processing {sat_name} detections ({sat_label})")
            df = df_raw.copy()
            
            # Define columns for pixel center point and the four corners
            cols_to_check = ['latitude', 'longitude', 'frp']
            
            # Remove rows with missing values
            for col in cols_to_check:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df.dropna(subset=cols_to_check, inplace=True)
            
            # Keep only columns we use:
            df = df[[
                "acq_date_time", "latitude", "longitude", "frp",
                "pixel_area", "confidence", "quality_flag", "type",
                "known_incident_type"
            ]]
            
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
            
            # Regrid and aggregate by hour:
            df_hourly = hourly_regrid_metrics(df, bounding_box, R)
            
            # Save
            if save_csv:
                file_out = f"{path_csv}/ngfs_gridded_{R}_{sdate}_{sat_label}.csv"
                # msg(f"Saving {file_out}")
                df_hourly.to_csv(file_out, index=False);
            
            if save_netcdf:
                for hour, df_h in df_hourly.groupby('hour', sort=True):
                    nc_path = write_hour_grid_nc(df_h, path_netcdf, R, bounding_box, sat_label=sat_label)
                    if nc_path:
                        hour_key = pd.Timestamp(hour)
                        if hour_key.tzinfo is not None:
                            hour_key = hour_key.tz_convert('UTC').tz_localize(None)
                        sat_hour_files[sat_label][hour_key] = nc_path
                    # exit(0)
        
        if save_netcdf:
            shared_hours = sorted(set(sat_hour_files.get('w', {})) & set(sat_hour_files.get('e', {})))
            for hour_key in shared_hours:
                merge_hourly_satellite_grids(
                    hour_key,
                    sat_hour_files['e'][hour_key],
                    sat_hour_files['w'][hour_key],
                    path_netcdf,
                    R
                )
            
        
        # print(df.dtypes)
        
        

    except FileNotFoundError:
        print(f"Error: The file(s) '{file_e}' and/or '{file_w}' was/were not found.")
    except ImportError:
        print("Error: This script requires pandas, matplotlib, and cartopy.")
        print("Please install them using: pip install pandas matplotlib cartopy")
    except Exception as e:
        print(f"An error occurred: {e}")
        
msg("done!")
