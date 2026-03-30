import numpy as np
import netCDF4 as nc
from scipy.spatial import cKDTree
import os


def regrid_goes_to_target(goes_file, target_coords, target_shape, max_dist_degrees=0.05, trim_cols=1):
    """Reads GOES area, trims E/W edges, builds a KDTree, and finds nearest neighbor."""
    print(f"Reading {goes_file}...")
    with nc.Dataset(goes_file, 'r') as ds:
        # Slice [:, trim_cols:-trim_cols] to keep all rows (North/South),
        # but drop the outer 'trim_cols' columns (East/West)
        g_lat = ds.variables['latitude'][:, trim_cols:-trim_cols]
        g_lon = ds.variables['longitude'][:, trim_cols:-trim_cols]
        g_area = ds.variables['pixel_area'][:, trim_cols:-trim_cols]
        
    # 1. Filter out fill values
    valid_mask = (g_lat != -999.99) & (g_lon != -999.99) & (g_area != -999.99)
    valid_lats = g_lat[valid_mask]
    valid_lons = g_lon[valid_mask]
    valid_area = g_area[valid_mask]
    
    # 2. Build the KDTree
    print("  -> Building KDTree...")
    goes_coords = np.column_stack((valid_lats, valid_lons))
    tree = cKDTree(goes_coords)
    
    # 3. Query the tree 
    print("  -> Querying nearest neighbors...")
    distances, indices = tree.query(target_coords, k=1)
    
    # 4. Map the areas, but strictly flag out-of-bounds points as Infinity
    regridded_area = valid_area[indices]
    
    # Flag NGFS points that are too far from the nearest valid GOES center
    out_of_bounds = distances > max_dist_degrees
    regridded_area[out_of_bounds] = np.inf
    
    return regridded_area.reshape(target_shape)

if __name__ == "__main__":
    goes18_file = 'goes18_abi_conus_lat_lon.nc'
    goes19_file = 'goes19_abi_conus_lat_lon.nc'
    ngfs_file = 'NGFS_STATIC_A2024.061.CONUS.r0.01.nc'
    
    for f in [goes18_file, goes19_file, ngfs_file]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Missing file: {f}")

    print(f"Loading target grid from {ngfs_file}...")
    with nc.Dataset(ngfs_file, 'a') as ngfs_ds:
        t_lat = ngfs_ds.variables['lat'][:]
        t_lon = ngfs_ds.variables['lon'][:]
        
        t_lon_grid, t_lat_grid = np.meshgrid(t_lon, t_lat)
        target_shape = t_lat_grid.shape 
        target_coords = np.column_stack((t_lat_grid.ravel(), t_lon_grid.ravel()))

        # Regrid both arrays (out-of-bounds points will become np.inf)
        area_18 = regrid_goes_to_target(goes18_file, target_coords, target_shape)
        area_19 = regrid_goes_to_target(goes19_file, target_coords, target_shape)

        # --- NEW COMPARISON LOGIC ---
        # --- PART 3: COMPARE AND CREATE MASK (WITH 5% BUFFER) ---
        print("Comparing pixel areas...")
        
        # Initialize the mask with -1 (Undefined / Out of bounds for both)
        mask = np.full(target_shape, -1, dtype=np.int32)
        
        # 1. Base conditions (assign 1 or 2 based on strictly smaller area)
        mask[area_18 < area_19] = 2
        mask[area_19 < area_18] = 1
        
        # 2. Calculate relative difference
        # We use np.errstate to suppress warnings when dividing np.inf by np.inf
        # for points that are out of bounds for both satellites.
        with np.errstate(invalid='ignore', divide='ignore'):
            relative_diff = np.abs(area_18 - area_19) / np.minimum(area_18, area_19)
            
        # 3. Apply the buffer zone condition (<= 5% difference)
        # We must ensure both pixels are actually valid (not np.inf)
        valid_both = (area_18 != np.inf) & (area_19 != np.inf)
        # buffer_zone = valid_both & (relative_diff <= 0.05)
        # buffer_zone = valid_both & (relative_diff <= 0.0)
        
        # Overwrite the base conditions with 3 where the buffer applies
        # mask[buffer_zone] = 3
        
        # --- PART 4: WRITE TO NGFS NETCDF ---
        print(f"Writing mask to {ngfs_file}...")
        if 'MASK_GOES_SOURCE' in ngfs_ds.variables:
            print(" -> 'MASK_GOES_SOURCE' already exists. Overwriting.")
            mask_var = ngfs_ds.variables['MASK_GOES_SOURCE']
        else:
            print(" -> Creating new 'MASK_GOES_SOURCE' variable.")
            mask_var = ngfs_ds.createVariable('MASK_GOES_SOURCE', 'i4', ('lat', 'lon'), fill_value=-1)
            
        # Update attributes to reflect the new category
        mask_var.long_name = "Mask GOES source of grid points with smaller pixel area from GOES-18 and -19"
        mask_var.units = "1=GOES-19, 2=GOES-18, 3=Buffer (<=5% diff), -1=Undefined"
        
        # Dump the calculated mask array into the variable
        mask_var[:] = mask
        
    print("Process complete! The NGFS file has been updated with bounded domains.")