import os
import shutil
import glob
import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset
from scipy.spatial import cKDTree
from datetime import timedelta

# --- Configuration ---
DATE1 = "2025-08-15 00:00:00"
DATE2 = "2025-08-15 23:00:00"

PATHIN = '/gpfs/f6/drsa-fire3/scratch/Gonzalo.Ferrada/FIRE/NGFS/gridded/netcdf'
PATHRAVE = '/gpfs/f6/drsa-fire3/world-shared/Gonzalo.Ferrada/input/rave/raw'
PATHOUT = '/gpfs/f6/drsa-fire3/scratch/Gonzalo.Ferrada/FIRE/NGFS/gridded/asrave'

SPECIES_IN = ["frp_mean", "frp_std"] # Lowercase in NGFS file
SPECIES_CALC = ["PM25", "NH3", "SO2", "CH4"]
SPECIES_OUT = ["FRP_MEAN", "FRP_SD", "FRE", "PM25", "NH3", "SO2", "CH4"]

# --- Helpers ---
def msg(s): print(f"{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}    {s}")

os.makedirs(PATHOUT, exist_ok=True)
# Updated 'H' to 'h' to fix FutureWarning
dates = pd.date_range(DATE1, DATE2, freq='h')

# --- Main Loop ---
for dt in dates:
    tag_ngfs = dt.strftime('%Y%m%d_%H')
    f_ngfs = os.path.join(PATHIN, f'NGFS_{tag_ngfs}Z_0p03.nc')
    
    msg(f_ngfs)
    if not os.path.exists(f_ngfs):
        print(f"  Warning: NGFS file not found: {f_ngfs}")
        continue

    # 1. Read NGFS Data
    with xr.open_dataset(f_ngfs) as ds_raw:
        # .squeeze() removes singleton dimensions (like Time=1) ensuring 2D data
        ds = ds_raw.squeeze()
        
        valid_mask = ds['frp_mean'].values > 0
        
        # Flatten and filter NGFS data
        ngfs_data = {}
        # Get coordinates for valid points (add 360 as per original script)
        ngfs_lon, ngfs_lat = np.meshgrid(ds['lon'].values + 360, ds['lat'].values)
        ngfs_data['lon'] = ngfs_lon[valid_mask]
        ngfs_data['lat'] = ngfs_lat[valid_mask]
        
        for v in SPECIES_IN:
            ngfs_data[v] = ds[v].values[valid_mask]

        if len(ngfs_data['lon']) == 0:
            print("  No active fires in NGFS file.")
            continue

    # 2. Find RAVE File (Lookback up to 3h)
    f_rave = None
    found_dt = dt
    for lookback in [0, 1, 2, 3]:
        search_dt = dt - timedelta(hours=lookback)
        tag_rave = search_dt.strftime('%Y%m%d%H')
        pattern = os.path.join(PATHRAVE, f'RAVE-HrlyEmiss-3km_v2r0_blend_s{tag_rave}00000_e*.nc')
        matches = glob.glob(pattern)
        if matches:
            f_rave = matches[0]
            found_dt = search_dt
            break
    
    if not f_rave:
        print(f"  Error: RAVE file not found for {dt} (checked 3h lookback)")
        continue

    # 3. Read RAVE Reference Data (for Ratios)
    with xr.open_dataset(f_rave) as ds_raw:
        # Squeeze RAVE data as well to ensure 2D grids
        ds = ds_raw.squeeze()
        
        # Load grid for later output mapping
        grid_lont = ds['grid_lont'].values
        grid_latt = ds['grid_latt'].values
        
        # Logic to extract valid RAVE points for ratio calculation
        r_lon = ds['grid_lont'].values - 360
        r_lat = ds['grid_latt'].values
        r_frp = ds['FRP_MEAN'].values
        
        # Create mask based on original logic
        mask_rave = (
            (r_lon >= -138) & (r_lon <= -57) & 
            (r_lat >= 20) & (r_lat <= 54) & 
            (r_frp > 0)
        )
        
        rave_subset = {}
        rave_subset['lon'] = r_lon[mask_rave] + 360
        rave_subset['lat'] = r_lat[mask_rave]
        rave_subset['FRP_MEAN'] = r_frp[mask_rave]
        
        for s in SPECIES_CALC:
            rave_subset[s] = ds[s].values[mask_rave]

    # 4. Processing (Vectorized)
    
    # A. Calculate Emissions based on closest RAVE fire
    # Build Tree for RAVE subset (valid fires)
    rave_points = np.column_stack((rave_subset['lon'], rave_subset['lat']))
    ngfs_points = np.column_stack((ngfs_data['lon'], ngfs_data['lat']))
    
    if len(rave_points) > 0:
        tree_phys = cKDTree(rave_points)
        # Find closest RAVE fire for every NGFS fire
        _, idx_phys = tree_phys.query(ngfs_points, k=1)
        
        # Calculate ratios and apply
        for s in SPECIES_CALC:
            # factor = rave_species / rave_frp_mean (at closest index)
            factor = rave_subset[s][idx_phys] / rave_subset['FRP_MEAN'][idx_phys]
            ngfs_data[s] = ngfs_data['frp_mean'] * factor
    else:
        # Fallback if no valid RAVE fires found in box
        for s in SPECIES_CALC:
            ngfs_data[s] = np.zeros_like(ngfs_data['frp_mean'])

    # Add derived fields
    ngfs_data['FRE'] = ngfs_data['frp_mean'] * 3600
    ngfs_data['FRP_MEAN'] = ngfs_data['frp_mean']
    ngfs_data['FRP_SD'] = ngfs_data['frp_std']

    # B. Map NGFS points to Output Grid
    # Build Tree for output grid (Full RAVE Grid)
    grid_points = np.column_stack((grid_lont.ravel(), grid_latt.ravel()))
    tree_grid = cKDTree(grid_points)
    
    # Find index in the flattened grid
    _, idx_grid = tree_grid.query(ngfs_points, k=1)
    
    # Convert flat indices back to 2D indices (lat, lon)
    y_idxs, x_idxs = np.unravel_index(idx_grid, grid_lont.shape)

    # 5. Write Output
    tag_out = dt.strftime('%Y%m%d%H')
    f_out = os.path.join(PATHOUT, f'RAVE-HrlyEmiss-3km_v2r0_blend_s{tag_out}00000.nc')
    
    # Copy template
    shutil.copy2(f_rave, f_out)
    
    # Open in append mode to overwrite
    with Dataset(f_out, 'r+') as dst:
        shape = grid_lont.shape
        
        for s in SPECIES_OUT:
            # Create empty grid
            data_grid = np.zeros(shape, dtype=np.float32)
            
            # Assign values to the mapped indices
            # If multiple NGFS points map to the same grid cell, this assignment
            # takes the last one (consistent with MATLAB behavior). 
            data_grid[y_idxs, x_idxs] = ngfs_data[s]
            
            # Overwrite variable
            dst.variables[s][:] = data_grid
            
    # Cleanup memory for next iteration
    del ngfs_data, rave_subset, tree_phys, tree_grid