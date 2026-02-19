import numpy as np
import netCDF4 as nc
import os

def calculate_goes_pixel_area(nc_file_path):
    print(f"Calculating area for {nc_file_path}...")
    ds = nc.Dataset(nc_file_path, 'r')
    lat_center = ds.variables['latitude'][:]
    lon_center = ds.variables['longitude'][:]
    ds.close()
    
    fill_mask = (lat_center == -999.99) | (lon_center == -999.99)
    
    lat_pad = np.pad(lat_center, pad_width=1, mode='edge')
    lon_pad = np.pad(lon_center, pad_width=1, mode='edge')
    
    lat_corners = (lat_pad[:-1, :-1] + lat_pad[:-1, 1:] + 
                   lat_pad[1:, :-1] + lat_pad[1:, 1:]) / 4.0
    lon_corners = (lon_pad[:-1, :-1] + lon_pad[:-1, 1:] + 
                   lon_pad[1:, :-1] + lon_pad[1:, 1:]) / 4.0
                   
    lat_TL, lon_TL = lat_corners[:-1, :-1], lon_corners[:-1, :-1] 
    lat_TR, lon_TR = lat_corners[:-1, 1:], lon_corners[:-1, 1:]   
    lat_BL, lon_BL = lat_corners[1:, :-1], lon_corners[1:, :-1]   
    lat_BR, lon_BR = lat_corners[1:, 1:], lon_corners[1:, 1:]     
    
    a = 6378137.0         
    e2 = 0.006694380023   
    
    lat_rad = np.radians(lat_center)
    
    Rm = a * (1 - e2) / (1 - e2 * np.sin(lat_rad)**2)**1.5  
    Rn = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)           
    
    def get_local_xy(lat_v, lon_v):
        dy = Rm * np.radians(lat_v - lat_center)
        dx = Rn * np.cos(lat_rad) * np.radians(lon_v - lon_center)
        return dx, dy

    x_TL, y_TL = get_local_xy(lat_TL, lon_TL)
    x_TR, y_TR = get_local_xy(lat_TR, lon_TR)
    x_BR, y_BR = get_local_xy(lat_BR, lon_BR)
    x_BL, y_BL = get_local_xy(lat_BL, lon_BL)

    area_sq_meters = 0.5 * np.abs(
        (x_TL * y_TR - y_TL * x_TR) +
        (x_TR * y_BR - y_TR * x_BR) +
        (x_BR * y_BL - y_BR * x_BL) +
        (x_BL * y_TL - y_BL * x_TL)
    )
    
    area_sq_meters = area_sq_meters * 1e-6 # km2
    area_sq_meters[fill_mask] = -999.99
    
    return area_sq_meters

if __name__ == "__main__":
    # List your target datasets
    files_to_process = [
        'goes18_abi_conus_lat_lon.nc', 
        'goes19_abi_conus_lat_lon.nc',
        'goes19_abi_conus_interpolated_lat_lon.nc'
    ]
    
    for nc_file in files_to_process:
        # Check if the file actually exists before trying to open it
        if not os.path.exists(nc_file):
            print(f"File not found: {nc_file}. Skipping.")
            continue
            
        # 1. Get the calculated area array
        pixel_area_array = calculate_goes_pixel_area(nc_file)
        
        # 2. Open the NetCDF file in append/modify mode ('a')
        print(f"Writing data into {nc_file}...")
        with nc.Dataset(nc_file, 'a') as ds:
            
            # 3. Check if the variable already exists to avoid crash on re-runs
            if 'pixel_area' in ds.variables:
                print(" -> 'pixel_area' already exists. Overwriting with new data.")
                area_var = ds.variables['pixel_area']
            else:
                print(" -> Creating new 'pixel_area' variable.")
                # Create the variable using standard 32-bit float ('f4') and original dimensions
                area_var = ds.createVariable('pixel_area', 'f4', ('rows', 'columns'), fill_value=-999.99)
                
                # 4. Add CF-compliant metadata attributes
                area_var.long_name = "Pixel area calculated from vertices"
                area_var.units = "km2"
                area_var.valid_range = np.array([0., 1e10], dtype='f4')
                area_var.comment = "Calculated using Shoelace formula on GRS80 ellipsoid local metric distances"
            
            # 5. Dump the array into the NetCDF variable
            area_var[:] = pixel_area_array
            
        print(f"Successfully updated {nc_file}!\n")