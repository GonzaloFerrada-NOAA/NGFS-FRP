#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import os
import matplotlib.dates as mdates
import glob
from netCDF4 import Dataset, MFDataset
from datetime import date, datetime, timedelta
import fnmatch
import netCDF4
import xarray as xr

def checkDir(plot_dir):
    if os.path.isdir(plot_dir):
        print('Plot dir already exists!')
    else:
        print('Plot dir does not exist, creating it!')
        os.mkdir(plot_dir);
        
currdir   = os.getcwd()
plot_dir  = currdir
checkDir(plot_dir)

# NGFS File
file_path = 'NGFS_FIRE_DETECTIONS_GOES-18_ABI_CONUS_2025_09_01_244.csv'

try:
    # Read the wildfire data from the CSV file
    print(f"Reading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Define columns for pixel center point and the four corners
    cols_to_check = [
        'latitude', 'longitude', 'frp',
        'latitude_c1', 'longitude_c1',
        'latitude_c2', 'longitude_c2',
        'latitude_c3', 'longitude_c3',
        'latitude_c4', 'longitude_c4'
    ]
    
    # Remove rows with missing values
    for col in cols_to_check:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=cols_to_check, inplace=True)
    
    # Convert the 'pixel_date_time' column to datetime objects (UTC is inferred from 'Z')
    df['acq_date_time'] = pd.to_datetime(df['acq_date_time'])
    plot_timestamp = df['acq_date_time'].iloc[0]
    title_timestamp_str = plot_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')
    
    # Filter for North American bounding box
    df = df[(df['longitude'] > -170) & (df['longitude'] < -50)]
    df = df[(df['latitude'] > 15) & (df['latitude'] < 75)]

    # Set up the map projection
    print("Setting up the map projection...")
    # projection = ccrs.LambertConformal(central_longitude=-98.0,central_latitude=39.5,standard_parallels=(33, 45))
    projection = ccrs.PlateCarree()
    
    # Create a figure and axes for the plot
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(1, 1, 1, projection=projection)

    # Set the geographic extent of the map
    ax.set_extent([-130, -65, 20, 55], crs=ccrs.PlateCarree()) # NorthAmerica

    # Add geographical features to the map
    ax.add_feature(cfeature.COASTLINE, zorder=2)
    ax.add_feature(cfeature.BORDERS, linestyle=':', zorder=2)
    ax.add_feature(cfeature.STATES, linestyle='-', edgecolor='gray', zorder=1)
    # ax.add_feature(cfeature.OCEAN, color='lightblue', zorder=1)
    # ax.add_feature(cfeature.LAND, color='lightgray', zorder=1)
    # ax.add_feature(cfeature.LAKES, color='lightblue', zorder=1)

    # Create and plot a polygon for each fire detection pixel
    patches = []
    for index, row in df.iterrows():
        # Get the corner coordinates for the current pixel
        # Assuming a standard c1 -> c2 -> c3 -> c4 order
        corners_lon = [row['longitude_c1'], row['longitude_c2'], row['longitude_c3'], row['longitude_c4']]
        corners_lat = [row['latitude_c1'], row['latitude_c2'], row['latitude_c3'], row['latitude_c4']]
        
        # Create a list of (lon, lat)
        polygon_coords = list(zip(corners_lon, corners_lat))        
        polygon = mpatches.Polygon(polygon_coords, closed=True)
        patches.append(polygon)

    collection = PatchCollection(patches,
                                 cmap='hot_r',
                                 alpha=1,
                                 transform=ccrs.PlateCarree(),
                                 zorder=3)

    # Set the colors of the polygons based on the Fire Radiative Power (FRP)
    # collection.set_array(np.log10(df['frp']))
    collection.set_array(df['frp'])
    collection.set_clim(vmin=0, vmax=500)
    
    # Add the collection of polygons to the map
    ax.add_collection(collection)
    ax.set_title(f'{title_timestamp_str}', fontsize=12)

    cbar = plt.colorbar(collection, ax=ax, orientation='horizontal', pad=0.1, shrink=0.7)
    cbar.set_label('NGFS Pixel FRP [MW]', fontsize=12)
    
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, rotate_labels=False,
                  linewidth=1, color='none', alpha=0.1, linestyle='--', x_inline=False, y_inline=False)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    gl.top_labels = False
    gl.right_labels = False
    
    # Save the plot
    fname = plot_dir+'/'+'NGFS_Mapping_'+plot_timestamp.strftime('%Y%m%d%H%M%S')+'_UTC_CONUS.png'
    plt.savefig(fname, bbox_inches='tight', dpi=300)

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except ImportError:
    print("Error: This script requires pandas, matplotlib, and cartopy.")
    print("Please install them using: pip install pandas matplotlib cartopy")
except Exception as e:
    print(f"An error occurred: {e}")


# GAF: now it is reading all csv files in the directory to construct a time series
# in the subset NW region_of_interest
region_of_interest = [-133., -108., 37., 55.] #NW
file_pattern = './*.csv'

lon_min, lon_max, lat_min, lat_max = region_of_interest

# Find all files matching the specified pattern
file_list = glob.glob(file_pattern)

if not file_list:
    print(f"Error: No files found matching the pattern '{file_pattern}'.")

print(f"Found {len(file_list)} files to analyze.")

# List to store the results from each file
frp_data = []

# Loop through each file
for file_path in file_list:
    try:
        print(f"Processing {os.path.basename(file_path)}...")

        # Read the data file
        df = pd.read_csv(file_path)

        # Convert relevant columns to numeric, coercing errors
        for col in ['latitude', 'longitude', 'frp']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Convert timestamp column to datetime objects
        df['acq_date_time'] = pd.to_datetime(df['acq_date_time'], errors='coerce')

        # Drop any rows where essential data is missing
        df.dropna(subset=['latitude', 'longitude', 'frp', 'acq_date_time'], inplace=True)

        # Filter Data by Region
        mask = (
            (df['longitude'] >= lon_min) & (df['longitude'] <= lon_max) &
            (df['latitude'] >= lat_min) & (df['latitude'] <= lat_max)
        )
        regional_fires = df[mask]

        # Calculate Total FRP
        # GAF this is the total FRP in the whole Western region (?)
        if not regional_fires.empty:
            total_frp = regional_fires['frp'].sum()
            total_pixel_area = regional_fires['pixel_area'].sum()
            # Get the timestamp for this observation from the 'acq_date_time' column
            timestamp = regional_fires['acq_date_time'].iloc[0]
            
            # Find the row with the maximum FRP
            max_frp_row = regional_fires.loc[regional_fires['frp'].idxmax()]
            max_frp     = max_frp_row['frp']
            pixel_area_at_max = max_frp_row['pixel_area']

            # Store the result
            frp_data.append({
                    'time': timestamp, 
                    'total_frp': total_frp,
                    'max_frp': max_frp,
                    'pixel_area_at_max': pixel_area_at_max,
                    'total_pixel_area': total_pixel_area
                })

    except Exception as e:
        print(f"Could not process file {file_path}. Error: {e}")

# --- Plotting the Timeseries ---
if not frp_data:
    print("\nNo fire radiative power was detected in the specified region across all files.")

# Convert results to a DataFrame for easy sorting and plotting
frp_df = pd.DataFrame(frp_data)
frp_df.sort_values(by='time', inplace=True)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True) 

# Total FRP (with color showing total pixel area)
scatter1 = ax1.scatter(frp_df['time'], frp_df['total_frp'],
                       s=60, 
                       c=frp_df['total_pixel_area'], # Color points by total pixel area
                       cmap='magma',
                       alpha=0.8,
                       edgecolor='k',
                       linewidth=0.5)
ax1.set_title(f'Bounding box: [{lon_min}, {lon_max}, {lat_min}, {lat_max}]', fontsize=16)
ax1.set_ylabel('Total FRP [MW]', fontsize=12)
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
cbar1 = fig.colorbar(scatter1, ax=ax1, orientation='vertical', pad=0.02)
cbar1.set_label('Total Pixel Area [sq km]', fontsize=12)

# Max FRP (with marker color showing pixel area at max)
scatter2 = ax2.scatter(frp_df['time'], frp_df['max_frp'], 
                       s=60, 
                       c=frp_df['pixel_area_at_max'], # Color points by pixel area
                       cmap='rainbow',
                       alpha=0.8,
                       edgecolor='k',
                       linewidth=0.5)
ax2.set_ylabel('Maximum Single-Pixel FRP [MW]', fontsize=12)
ax2.set_xlabel('UTC', fontsize=12)
ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

cbar2 = fig.colorbar(scatter2, ax=ax2, orientation='vertical', pad=0.02)
cbar2.set_label('Pixel Area at Max FRP [sq km]', fontsize=12)

# date formatting on the x-axis
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M'))
fig.autofmt_xdate()

plt.tight_layout(rect=[0, 0, 1, 0.96])

fname = plot_dir+'/'+'NGFS_Mapping_Timeseries_for_RegionSelected.png'
plt.savefig(fname, bbox_inches='tight', dpi=300)
#plt.show()

# GAF: it seems that this will create the individual csv and png files for each named fire
# Find all files matching the specified pattern
file_list = glob.glob(file_pattern)

if not file_list:
    print(f"Error: No files found matching the pattern '{file_pattern}'.")
    print("Please make sure your CSV files are in the same directory as this script.")

print(f"Found {len(file_list)} files. Combining them into a single dataset...")

# Read and combine all files into a single DataFrame
all_data_df = pd.concat([pd.read_csv(f) for f in file_list], ignore_index=True)
print("All files have been combined.")
# Convert relevant columns to numeric, coercing errors
for col in ['frp', 'pixel_area']:
    all_data_df[col] = pd.to_numeric(all_data_df[col], errors='coerce')

# Convert timestamp column to datetime objects
all_data_df['acq_date_time'] = pd.to_datetime(all_data_df['acq_date_time'], errors='coerce')

# Clean up the incident name column
all_data_df['known_incident_name'] = all_data_df['known_incident_name'].astype(str).fillna('Unknown')

# Drop any rows where essential data is missing
all_data_df.dropna(subset=['frp', 'pixel_area', 'acq_date_time'], inplace=True)

nIncidents = len(all_data_df['known_incident_name'].unique())

print("Found..."+str(nIncidents)+"...incidents")

for incident_name_query in all_data_df['known_incident_name'].unique():
    if incident_name_query != 'nan':
        mask = all_data_df['known_incident_name'].str.contains(incident_name_query, case=False, na=False)
        incident_df = all_data_df[mask]
        
        if incident_df.empty:
            print(f"\nNo incidents found matching '{incident_name_query}'. Please check the name and try again.")
            available_names = all_data_df['known_incident_name'].unique()
            print("\nAvailable incident names in the dataset (top 20):")
            print(available_names[:20])
        
        found_name = incident_df['known_incident_name'].unique()[0]
        incident_lat = np.nanmin(incident_df['latitude'])   # GAF why min?
        incident_lon = np.nanmin(incident_df['longitude'])  # GAF why min?
        
        print(f"Found data for incident: '{found_name}'. Aggregating data by timestamp...")
        
        grouped = incident_df.groupby('acq_date_time').agg(
            total_frp=('frp', 'sum'),
            total_pixel_area=('pixel_area', 'sum')
        ).reset_index()
        
        # Find the max FRP and its corresponding area for each timestamp
        max_frp_data = incident_df.loc[incident_df.groupby('acq_date_time')['frp'].idxmax()]
        
        # Merge the aggregated data with the max FRP data
        final_df = pd.merge(grouped, max_frp_data[['acq_date_time', 'frp', 'pixel_area']], on='acq_date_time')
        final_df.rename(columns={'frp': 'max_frp', 'pixel_area': 'pixel_area_at_max'}, inplace=True)
        final_df.sort_values(by='acq_date_time', inplace=True)
        
        data = {
            "Latitude": incident_lat,
            "Lonitude": incident_lon,        
            "UTC_Time": final_df['acq_date_time'],
            "Sum_FRP": final_df['total_frp'],
            "Max_FRP": final_df['max_frp'],
            "Sum_PixelArea": final_df['total_pixel_area'],
            "PixelArea_at_Max_FRP": final_df['pixel_area_at_max']}
        
        df = pd.DataFrame(data)
        
        df.to_csv('./'+found_name+'_NGFS_GOES18.csv', index=False)
        
        # Plotting the Timeseries
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # Total FRP
        scatter1 = ax1.scatter(final_df['acq_date_time'], final_df['total_frp'],
                               s=60, c=final_df['total_pixel_area'], cmap='magma',
                               alpha=0.8, edgecolor='k', linewidth=0.5)
        ax1.set_title(f"Known Incident: '{found_name}'", fontsize=16)
        ax1.set_ylabel('Total FRP [MW]', fontsize=12)
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
        cbar1 = fig.colorbar(scatter1, ax=ax1, orientation='vertical', pad=0.02)
        cbar1.set_label('Total Pixel Area [sq km]', fontsize=12)
        
        # Max FRP
        scatter2 = ax2.scatter(final_df['acq_date_time'], final_df['max_frp'], 
                               s=60, c=final_df['pixel_area_at_max'], cmap='rainbow',
                               alpha=0.8, edgecolor='k', linewidth=0.5)
        ax2.set_ylabel('Maximum Single-Pixel FRP [MW]', fontsize=12)
        ax2.set_xlabel('UTC', fontsize=12)
        ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
        cbar2 = fig.colorbar(scatter2, ax=ax2, orientation='vertical', pad=0.02)
        cbar2.set_label('Pixel Area at Max FRP [sq km]', fontsize=12)
        
        # Formatting
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M'))
        fig.autofmt_xdate()
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fname = plot_dir+'/'+'NGFS_Mapping_G18_Timeseries_for_'+found_name+'.png'
        plt.savefig(fname, bbox_inches='tight', dpi=300)
        #plt.show()
        plt.close()
    





