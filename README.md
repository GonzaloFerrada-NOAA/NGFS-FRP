# NGFS-FRP processing tool

## Packages needed:

```conda install -c conda-forge pandas xarray numpy netcdf4 dask```

or...

```pip install pandas xarray numpy netcdf4 dask```


## process_bysat_NGFS.py
Main tool that reads NGFS inputs in CSV format and grids them into a target grid of user-specified resolution. It applied multiple filters (QA flags), aggregates the data hourly, calculates metrics and merge data from fires observed by both GOES-East and -West based on pixel area. The resulting outputs are saved in gridded/ in both netcdf (2-D) and csv (1-D) formats. The aggregation method follows Darmenov and da Silva (2015), i.e., the QFED dataset, and it geolocates the input coordinates using the nearest neighbor approach.

## convert2rave.py
(Temporary tool) Produces emissions of PM2.5 to be used as input to the cheMPAS-Fire model. It scales the emissions based on RAVE inputs (needed for processing). Note that for this tool to work properly, the NGFS data needs to be at a resolution of 0.03 degrees. An analogous tool in MATLAB is included.

## Sample data
#### data/
NGFS input files from both GOES-18 and -19 for 2025-08-14.
#### gridded/
Outputs in csv and netcdf formats from the main tool.


# Change log
#### v0.31 (12 Feb 2026)
Improved processing speed. Now default product is always 1-D, i.e., as "point-source" instead of 2-D arrays, since when processing at R=0.01 degrees the code became too slow for real-time applications. Added the option save_netcdf_2d in case the user wants to save the final merged product in 2-D (not recommended)

#### v0.3 (11 Feb 2026)
Fixed a bug when calculating FRP_MEAN. Now NGFS FRP is much more comparable to RAVE, especially at higher latitudes.

#### v0.2 (9 Feb 2026)
Overhauled the output product: variable names, attributes. Added FRE to the list of output variables.

#### v0.1 (December 2025)
Fixed a bug when calculating FRP density.

#### v0 (September 2025)
First version of the code