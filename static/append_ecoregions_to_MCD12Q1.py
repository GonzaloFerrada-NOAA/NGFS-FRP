#!/gpfs/f6/drsa-fire3/scratch/Gonzalo.Ferrada/miniconda3/bin/python

import numpy as np
from netCDF4 import Dataset
from scipy.spatial import cKDTree


# ============================================================
# User settings
# ============================================================

SRC_FILE = "/gpfs/f6/drsa-fire3/world-shared/Gonzalo.Ferrada/input/aux/ecoregion/raw/NA_RRFS_Ecoregions_and_EFsoriginal.nc"
TGT_FILE = "/gpfs/f6/drsa-fire3/scratch/Gonzalo.Ferrada/FIRE/NGFS/static/MCD12Q1.A2024.061.CONUS.r0.03.nc"

SRC_VAR = "ecoregion_ID"
TGT_WATER_VAR = "land_cover_fraction"
OUT_VAR = "ecoregion_ID"

CHUNK_ROWS = 200          # target rows processed at a time
TREE_LEAFSIZE = 64

# Target pixels with no land-cover support should stay missing. In this
# MCD12Q1 file, no-data pixels can appear as zeros across every class because
# land_cover_fraction has _FillValue=0.0.
MIN_TOTAL_LCF = 1.0e-6

# Only pixels that are effectively all water are masked. Mixed coastal/lake
# pixels still receive the nearest land ecoregion.
FULL_WATER_FRACTION = 0.999

# If True, source water class (ecoregion_ID == 0) is excluded from donors.
# This is usually what you want here.
EXCLUDE_SOURCE_WATER = True


# ============================================================
# Helpers
# ============================================================

def wrap_to_180(lon):
    """Convert longitude to [-180, 180)."""
    return ((lon + 180.0) % 360.0) - 180.0


def latlon_to_unitxyz(lat_deg, lon_deg):
    """
    Convert lat/lon in degrees to 3-D unit-sphere Cartesian coordinates.
    This makes nearest-neighbor search more geographically meaningful.
    """
    lat_rad = np.deg2rad(lat_deg)
    lon_rad = np.deg2rad(lon_deg)

    coslat = np.cos(lat_rad)
    x = coslat * np.cos(lon_rad)
    y = coslat * np.sin(lon_rad)
    z = np.sin(lat_rad)

    return np.column_stack((x, y, z))


def get_water_category_index(ds):
    """
    Find the index corresponding to category value 17.
    Falls back to Python index 16 if category variable is unavailable or unusual.
    """
    if "category" in ds.variables:
        cats = ds.variables["category"][:]
        cats = np.asarray(cats)
        hits = np.where(cats == 17)[0]
        if hits.size > 0:
            return int(hits[0])

    return 16


def filled_array(data, fill_value=np.nan):
    """Return a plain ndarray, replacing masked values when present."""
    if np.ma.isMaskedArray(data):
        return data.filled(fill_value)
    return np.asarray(data)


# ============================================================
# Read source grid and build donor tree
# ============================================================

print("Reading source file...")
with Dataset(SRC_FILE, "r") as ds_src:
    src_lat = filled_array(ds_src.variables["geolat"][:, :]).astype(np.float64)
    src_lon = filled_array(ds_src.variables["geolon"][:, :]).astype(np.float64)
    src_lon = wrap_to_180(src_lon)

    src_data = filled_array(ds_src.variables[SRC_VAR][0, :, :]).astype(np.float64)

# Valid source points
src_valid = np.isfinite(src_lat) & np.isfinite(src_lon) & np.isfinite(src_data)

# Optionally exclude source water donors
if EXCLUDE_SOURCE_WATER:
    src_valid &= (src_data != 0)

src_lat_1d = src_lat[src_valid]
src_lon_1d = src_lon[src_valid]
src_val_1d = src_data[src_valid]

print(f"Number of valid source donor points: {src_val_1d.size:,}")

if src_val_1d.size == 0:
    raise RuntimeError("No valid source donor points found.")

print("Building KD-tree...")
src_xyz = latlon_to_unitxyz(src_lat_1d, src_lon_1d)
tree = cKDTree(src_xyz, leafsize=TREE_LEAFSIZE)


# ============================================================
# Read target grid coordinates
# ============================================================

print("Reading target file...")
with Dataset(TGT_FILE, "r") as ds_tgt:
    tgt_lat_1d = ds_tgt.variables["lat"][:].astype(np.float64)
    tgt_lon_1d = ds_tgt.variables["lon"][:].astype(np.float64)
    tgt_lon_1d = wrap_to_180(tgt_lon_1d)

    water_idx = get_water_category_index(ds_tgt)

nlat = tgt_lat_1d.size
nlon = tgt_lon_1d.size

print(f"Target grid shape: ({nlat}, {nlon})")
print(f"Water category index used: {water_idx}")


# ============================================================
# Create / write output variable chunk by chunk
# ============================================================

print("Opening target file for append...")
with Dataset(TGT_FILE, "a") as ds_out:

    if OUT_VAR in ds_out.variables:
        outvar = ds_out.variables[OUT_VAR]
        if outvar.dimensions != ("lat", "lon"):
            raise RuntimeError(
                f"Variable '{OUT_VAR}' already exists but does not have dimensions ('lat','lon')."
            )
    else:
        outvar = ds_out.createVariable(
            OUT_VAR,
            "f4",
            ("lat", "lon"),
            zlib=True,
            complevel=1,
            fill_value=np.nan,
            chunksizes=(min(CHUNK_ROWS, nlat), min(1000, nlon)),
        )

    outvar.long_name = "EPA Level 1 Ecoregion ID regridded by nearest neighbor from curvilinear source grid"
    outvar.description = (
        "Nearest-neighbor regridding of source ecoregion_ID using geolat/geolon source centers. "
        "Output is NaN where target land_cover_fraction has no support or category=17 is full water."
    )
    outvar.units = "unitless"
    outvar.source_file = SRC_FILE
    outvar.note = (
        "Source donor points with ecoregion_ID == 0 were excluded. "
        f"Target cells require sum(land_cover_fraction) > {MIN_TOTAL_LCF:g} "
        f"and water fraction < {FULL_WATER_FRACTION:g}."
    )

    lcf_var = ds_out.variables[TGT_WATER_VAR]
    old_auto_mask = lcf_var.mask
    lcf_var.set_auto_mask(False)

    # Initialize entire output to NaN
    outvar[:, :] = np.float32(np.nan)

    print("Regridding in chunks...")
    total_valid = 0
    total_full_water = 0
    total_no_lcf_data = 0

    for j0 in range(0, nlat, CHUNK_ROWS):
        j1 = min(j0 + CHUNK_ROWS, nlat)

        lat_chunk_1d = tgt_lat_1d[j0:j1]

        lcf_chunk = lcf_var[:, j0:j1, :].astype(np.float32)
        water_chunk = lcf_chunk[water_idx, :, :]
        lcf_total = np.nansum(lcf_chunk, axis=0)

        target_has_lcf_data = np.isfinite(lcf_total) & (lcf_total > MIN_TOTAL_LCF)
        target_is_full_water = np.isfinite(water_chunk) & (water_chunk >= FULL_WATER_FRACTION)

        # Build 2-D target coordinates for this chunk
        lon2d, lat2d = np.meshgrid(tgt_lon_1d, lat_chunk_1d)

        valid_chunk = target_has_lcf_data & ~target_is_full_water
        out_chunk = np.full((j1 - j0, nlon), np.nan, dtype=np.float32)

        nvalid = np.count_nonzero(valid_chunk)
        nfull_water = np.count_nonzero(target_is_full_water)
        nno_lcf_data = np.count_nonzero(~target_has_lcf_data)
        total_valid += nvalid
        total_full_water += nfull_water
        total_no_lcf_data += nno_lcf_data
        print(
            f"  Rows {j0:4d}:{j1:4d}  valid = {nvalid:,}  "
            f"full water = {nfull_water:,}  no lcf data = {nno_lcf_data:,}"
        )

        if nvalid > 0:
            q_lat = lat2d[valid_chunk]
            q_lon = lon2d[valid_chunk]

            q_xyz = latlon_to_unitxyz(q_lat, q_lon)

            # workers=-1 uses all available CPUs if supported by scipy version
            _, idx = tree.query(q_xyz, k=1, workers=-1)

            out_chunk[valid_chunk] = src_val_1d[idx].astype(np.float32)

        outvar[j0:j1, :] = out_chunk

    lcf_var.set_auto_mask(old_auto_mask)

print(f"Total valid target points written: {total_valid:,}")
print(f"Total full-water target points masked: {total_full_water:,}")
print(f"Total no-land-cover-data target points masked: {total_no_lcf_data:,}")
print("Done.")
print(f"Variable '{OUT_VAR}' written into:")
print(TGT_FILE)
