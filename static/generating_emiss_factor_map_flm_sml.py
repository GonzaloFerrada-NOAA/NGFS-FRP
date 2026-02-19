#########################################################################
#                                                                       #
# Python script for fire emissions preprocessing from RAVE FRP and FRE  #
# (Li et al.,2022). Written by Johana Romero-Alvarez and Haiqin Li      #
# based on Kai Wang and Jianping Huang prototype                        #
# Feb 2025: Modified by Gonzalo A. Ferrada (GAF) for NGFS data.         #
#                                                                       #
#########################################################################
import xarray as xr
import numpy as np
import os
import datetime as dt

# MODIS IGBP land use map for NGFS domain: 17 categories in NGFS
Year = 2024
Res  = 0.01
veg_map = f"/gpfs/f6/drsa-fire3/scratch/Gonzalo.Ferrada/FIRE/NGFS/MCD12Q1.A{Year}.061.CONUS.r{Res}.nc"
# print(veg_map)

ds = xr.open_dataset(veg_map)

# (GAF) NGFS land use map is in regular grid, thus lon/lat are 1-D.
# Let's create their 2-D analogous:
geolon, geolat = xr.broadcast(ds.lon, ds.lat)
ds["geolon"] = geolon
ds["geolat"] = geolat

#constants emissions estimation
beta= 0.38 # (kg[dry-matter] / MJ) based on Wooster et al. 2005

#Emission factors based on SERA US Forest Service
EF_FLM = dict({'frst':           19.0,
               'hwd':             9.4,
               'mxd':            14.6,
               'shrb':            9.3,
               'shrb_grs':       10.7,
               'grs':            13.3,
               'org_soil_mix':   10.8,
               'org_soil_woody': 11.8, 
               'peats':          18.9 })

EF_SML = dict({'frst':           28.0,
               'hwd':            37.7,
               'mxd':            17.6,
               'shrb':           36.6,
               'shrb_grs':       36.7,
               'grs':            38.4,
               'org_soil_mix':   20.7,
               'org_soil_woody': 23.3 })

EF_east=dict({'one':    18.15,
              'two':    12.63,
              'three':  18.50,
              'four':   11.98,
              'five':   15.67,
              'six':      6.7,
              'seven':    6.7,
              'eight':   18.5,
              'nine':     6.7,
              'ten':     9.29,
              'eleven': 16.31,
              '12':     11.91,
              '13':      8.20, 
              '14':      8.20, 
              '16':      8.20 })

# Land categories present in fractional map
land_cats=["Evergreen_Needleleaf_Forests", 
           "Evergreen_Broadleaf_Forests", 
           "Deciduous_Needleleaf_Forests",
           "Deciduous_Broadleaf_Forests",
           "Mixed_Forests",
           "Closed_Shrublands",
           "Open_Shrublands",
           "Woody_Savannas",
           "Savannas",
           "Grasslands",
           "Permanent_Wetlands",
           "Croplands",
           "Urban_and_Builtup_Lands",
           "Cropland_Natural_Vegetation_Mosaics",
           "Permanent_Snow_and_Ice",
           "Barren",
           "Water_Bodies" ]

# (GAF) we do not have these. NGFS' land use map is from MODIS (17 cat), not MODIS modified by WRF/WPS (21 cat).
#    "Wooded_Tundra",
#    "Mixed_Tundra",
#    "Barren_Tundra"

# Beta factors from VFEI (derived from GFAS)
# betas = np.array([1.55, 0.96, 1.55, 1.55, 1.55, 0.78, 1.55, 1.55, 0.78, 0.78, 1.55, 0.29, 0.29, 0.78, 0.49, 0.78])

"""
mask_northeast  = (ds['geolon'] >  -77.0) & (ds['geolon'] < -70.0) & (ds['geolat'] < 45.0) & (ds['geolat'] > 39.0)
mask_southeast  = (ds['geolon'] >  -89.0) & (ds['geolon'] < -75.0) & (ds['geolat'] < 39.0) & (ds['geolat'] > 24.5)
mask_midwest    = (ds['geolon'] > -102.0) & (ds['geolon'] < -89.0) & (ds['geolat'] < 49.0) & (ds['geolat'] > 36.0)
"""
northern_us_limit = 49  # Approximate northern border with Canada

# Define the lower latitude limit for the U.S. (avoiding Mexico)
southern_us_limit = 25  # Approximate southern border with Mexico

# Mask definitions
# Northeast mask: Avoiding Canada
mask_northeast = ((ds['geolon'] > -102.0) & (ds['geolon'] < -77.0) & (ds['geolat'] < northern_us_limit) & (ds['geolat'] > 39.0)).fillna(False)

# Southeast mask: Avoiding both Canada and Mexico
mask_southeast = ((ds['geolon'] > -102.0) & (ds['geolon'] < -77.0) & (ds['geolat'] < 39.0) & (ds['geolat'] > southern_us_limit)).fillna(False)

# Midwest mask: Central areas starting from longitude -102, not too far north to avoid Canada
mask_midwest   = ((ds['geolon'] > -102.0) & (ds['geolon'] < -89.0) & (ds['geolat'] < northern_us_limit) & (ds['geolat'] > 36)).fillna(False)


#Open LU map and extract land categories
def generate_EFs(veg_map, EF_FLM, EF_SML, land_cats, EF_east):
   
    nc_land      = xr.open_dataset(veg_map)
    tosave       = xr.open_dataset(veg_map)
    vtype_val    = nc_land['land_cover_fraction']
    vtype        = vtype_val
    
    # Create a truly clean grid of zeros
    total_ef = xr.zeros_like(ds.land_cover_fraction[0, :, :])
    total_fl = xr.zeros_like(ds.land_cover_fraction[0, :, :])
    total_sm = xr.zeros_like(ds.land_cover_fraction[0, :, :])
    total_beta = xr.zeros_like(ds.land_cover_fraction[0, :, :])

    for lc,i in zip(land_cats, range(len(land_cats))):

        # vtype_ind = vtype_val[i,:,:]
        vtype_ind = vtype_val[i, :, :].fillna(0) # Fill NaNs with 0 for calculation

        if   lc == "Evergreen_Needleleaf_Forests":
            layer_flaming       = vtype_ind * EF_FLM['frst']
            layer_smoldering    = vtype_ind * EF_SML['frst']
            layer_ef            = vtype_ind * (( 0.75 * EF_FLM['frst'] ) + ( 0.25 * EF_SML['frst'] ))
            beta_map            = vtype_ind * 1.55

        elif lc ==  "Evergreen_Broadleaf_Forests":
            layer_flaming       = vtype_ind * EF_FLM['frst']
            layer_smoldering    = vtype_ind * EF_SML['frst']
            layer_ef            = vtype_ind * (( 0.75 * EF_FLM['frst'] ) + ( 0.25 * EF_SML['frst'] ))
            beta_map            = vtype_ind * 0.96

        elif lc == "Deciduous_Needleleaf_Forests":
            layer_flaming       = vtype_ind * EF_FLM['hwd']
            layer_smoldering    = vtype_ind * EF_SML['hwd']
            layer_ef            = vtype_ind * (( 0.80 * EF_FLM['hwd'] ) + ( 0.20 * EF_SML['hwd'] ))
            beta_map            = vtype_ind * 1.55

        elif lc == "Deciduous_Broadleaf_Forests":
            layer_flaming       = vtype_ind * EF_FLM['hwd']
            layer_smoldering    = vtype_ind * EF_SML['hwd']
            layer_ef            = vtype_ind * (( 0.80 * EF_FLM['hwd'] ) + ( 0.20 * EF_SML['hwd'] ))
            beta_map            = vtype_ind * 1.55

        elif lc == "Mixed_Forests":
            layer_flaming       = vtype_ind * EF_FLM['mxd']
            layer_smoldering    = vtype_ind * EF_SML['mxd']
            layer_ef            = vtype_ind * (( 0.85 * EF_FLM['mxd'] ) + ( 0.15 * EF_SML['mxd'] ))
            beta_map            = vtype_ind * 1.55

        elif lc == "Closed_Shrublands":
            layer_flaming       = vtype_ind * EF_FLM['shrb']
            layer_smoldering    = vtype_ind * EF_SML['shrb']
            layer_ef            = vtype_ind * (( 0.95 * EF_FLM['shrb'] ) + ( 0.05 * EF_SML['shrb'] ))
            beta_map            = vtype_ind * 0.78

        elif lc == "Open_Shrublands":
            layer_flaming       = vtype_ind * EF_FLM['shrb']
            layer_smoldering    = vtype_ind * EF_SML['shrb']
            layer_ef            = vtype_ind * (( 0.95 * EF_FLM['shrb'] ) + ( 0.05 * EF_SML['shrb'] ))
            beta_map            = vtype_ind * 1.55

        elif lc == "Woody_Savannas":
            layer_flaming = xr.where(mask_northeast | mask_southeast | mask_midwest, 
                                vtype_ind * EF_east['eight'], 
                                vtype_ind * EF_FLM['shrb_grs'] )
            layer_smoldering = xr.where(mask_northeast | mask_southeast | mask_midwest, 
                                vtype_ind * EF_east['eight'], 
                                vtype_ind * EF_SML['shrb_grs'] )
            layer_ef = xr.where(mask_northeast | mask_southeast | mask_midwest, 
                        vtype_ind * EF_east['eight'], 
                        vtype_ind * (( 0.95 * EF_FLM['shrb_grs'] ) + ( 0.05 * EF_SML['shrb_grs'] )))
            beta_map            = vtype_ind * 1.55

        elif lc == "Savannas":
            layer_flaming= xr.where(mask_northeast | mask_southeast | mask_midwest,
                                    vtype_ind * EF_east['nine'], 
                                    vtype_ind * EF_FLM['grs'] )
            layer_smoldering= xr.where(mask_northeast | mask_southeast | mask_midwest,
                                    vtype_ind * EF_east['nine'], 
                                    vtype_ind * EF_SML['grs'] )
            layer_ef= xr.where(mask_northeast | mask_southeast | mask_midwest,
                            vtype_ind * EF_east['nine'], 
                            vtype_ind * (( 0.95 * EF_FLM['grs'] ) + ( 0.05 * EF_SML['grs'] )))
            beta_map            = vtype_ind * 0.78

        elif lc == "Grasslands":
            layer_ef= xr.where(mask_northeast | mask_southeast | mask_midwest,
                        vtype_ind * EF_east['ten'],
                        vtype_ind * EF_east['ten'])
            layer_flaming = layer_ef
            layer_smoldering = layer_ef
            beta_map            = vtype_ind * 0.78
            
        elif lc == "Permanent_Wetlands":
            layer_ef = vtype_ind * (EF_FLM['peats'])
            layer_flaming = layer_ef
            layer_smoldering = layer_ef
            beta_map            = vtype_ind * 1.55

        elif lc == "Croplands":
            layer_ef=  xr.where(mask_northeast | mask_southeast | mask_midwest,
                        vtype_ind * EF_east['12'],
                        vtype_ind * (8.2))
            layer_flaming = layer_ef
            layer_smoldering = layer_ef
            beta_map            = vtype_ind * 0.29
            
        elif lc == "Urban_and_Builtup_Lands":
            # (GAF) using grassland as urban fires
            layer_ef = vtype_ind * EF_east['ten']
            layer_flaming = layer_ef
            layer_smoldering = layer_ef
            beta_map            = vtype_ind * 0.29
            
        elif lc == "Cropland_Natural_Vegetation_Mosaics":
            layer_ef=  xr.where(mask_northeast | mask_southeast | mask_midwest,
                                vtype_ind * EF_east['14'],
                                vtype_ind * 8.2 )
            layer_flaming = layer_ef
            layer_smoldering = layer_ef
            beta_map            = vtype_ind * 0.78
            
        elif lc == "Barren":
            # (GAF) using 25% of grassland as urban fires
            layer_ef = vtype_ind * EF_east['ten'] * 0.25
            layer_flaming = layer_ef
            layer_smoldering = layer_ef
            beta_map            = vtype_ind * 0.49
            
        # (GAF) we don't have these in NGFS:
        # elif lc == "Wooded_Tundra":
        #     layer_ef = vtype_ind * (( 0.7 * EF_FLM['org_soil_woody'] ) + ( 0.3 * EF_SML['org_soil_woody'] )) 

        # elif lc == "Mixed_Tundra":  
        #     layer_ef = vtype_ind * (( 0.7 * EF_FLM['org_soil_mix'] ) + ( 0.3 * EF_SML['org_soil_mix']))

        else: # Permanent_Snow_and_Ice, Water_Bodies
            layer_ef = xr.zeros_like(vtype_val[0, :, :])
            layer_flaming = xr.zeros_like(vtype_val[0, :, :])
            layer_smoldering = xr.zeros_like(vtype_val[0, :, :])
            beta_map            = vtype_ind * 0.0
        
        print(f"Category {lc}: Max EF is {layer_ef.max().values}") # Debug
        
        total_ef    += layer_ef
        total_fl    += layer_flaming
        total_sm    += layer_smoldering
        total_beta  += beta_map
        
        
    tosave['EFACTOR_PM25']              = total_ef #* beta
    tosave['EFACTOR_FLAMING_PM25']      = total_fl #* beta
    tosave['EFACTOR_SMOLDERING_PM25']   = total_sm #* beta
    tosave['BETA_VFEI']                 = total_beta
    
    var_attrs = {
        'EFACTOR_PM25':             {'long_name': 'Emission factor of PM2.5', 'units': 'g[PM2.5]/kg[dry-matter]'},
        'EFACTOR_FLAMING_PM25':     {'long_name': 'Flaming emission factor of PM2.5', 'units': 'g[PM2.5]/kg[dry-matter]'},
        'EFACTOR_SMOLDERING_PM25':  {'long_name': 'Smoldering emission factor of PM2.5', 'units': 'g[PM2.5]/kg[dry-matter]'},
        'BETA_VFEI':                {'long_name': 'Beta parameter based on GFAS and VFEI', 'units': 'kg[dry-matter]/MJ'},
    }

    for v, attrs in var_attrs.items():
        tosave[v].attrs.update(attrs)


    # Create a dictionary for ALL variables in the dataset
    all_vars = list(tosave.data_vars) + list(tosave.coords)
    encoding_settings = {v: {'zlib': True, 'complevel': 9, 'shuffle': True} for v in all_vars}

    # Save the file
    fout = f"/gpfs/f6/drsa-fire3/scratch/Gonzalo.Ferrada/FIRE/NGFS/NGFS_EF_A{Year}.061.CONUS.r{Res}.nc"
    print(fout)
    tosave.to_netcdf(fout, encoding=encoding_settings)
    
    return total_ef,vtype

generate_EFs(veg_map, EF_FLM, EF_SML, land_cats, EF_east)
