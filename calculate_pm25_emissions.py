import argparse
import os
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

# --- Configuration ---
R = 0.01
BETA_VAL = 0.38
FILL = -999.0
STATIC_FILE_PATH = f"static/NGFS_EF_A2024.061.CONUS.r{R}.nc"


def load_static_data(static_path: str):
    """Load static rasters once; all heavy lookups are done in-memory afterward."""
    with Dataset(static_path, "r") as ds_s:
        s_lats = np.asarray(ds_s.variables["lat"][:], dtype=np.float32)
        s_lons = np.asarray(ds_s.variables["lon"][:], dtype=np.float32)

        static = {
            "lat_min": float(np.min(s_lats)),
            "lat_max": float(np.max(s_lats)),
            "lon_min": float(np.min(s_lons)),
            "nlat": int(s_lats.size),
            "nlon": int(s_lons.size),
            "is_top_down": bool(s_lats[0] > s_lats[-1]),
            # Flatten for fast 1-D point lookup using np.take.
            "ef_std_flat": np.asarray(ds_s.variables["EFACTOR_PM25"][:], dtype=np.float32).ravel(),
            "ef_flam_flat": np.asarray(ds_s.variables["EFACTOR_FLAMING_PM25"][:], dtype=np.float32).ravel(),
            "beta_vfei_flat": np.asarray(ds_s.variables["BETA_VFEI"][:], dtype=np.float32).ravel(),
        }
    return static


def _get_or_create_var(nc: Dataset, var_name: str, long_name: str):
    if var_name in nc.variables:
        return nc.variables[var_name]
    v = nc.createVariable(var_name, "f4", ("point",), fill_value=FILL)
    v.units = "kg"
    v.long_name = long_name
    v.coordinates = "lat lon time"
    return v


def process_single_file(fpath: str, static: dict):
    if not os.path.exists(fpath):
        print(f"Error: File {fpath} not found.")
        return False

    print(f"Processing: {os.path.basename(fpath)}")
    with Dataset(fpath, "a") as nc:
        p_lat = np.asarray(nc.variables["lat"][:], dtype=np.float32)
        p_lon = np.asarray(nc.variables["lon"][:], dtype=np.float32)
        fre = np.asarray(nc.variables["FRE"][:], dtype=np.float32)
        incident_flag = np.asarray(nc.variables["FLAG_KNOWN_INCIDENT"][:], dtype=np.int16)

        if static["is_top_down"]:
            lat_idx = np.rint((static["lat_max"] - p_lat) / R).astype(np.int64)
        else:
            lat_idx = np.rint((p_lat - static["lat_min"]) / R).astype(np.int64)
        lon_idx = np.rint((p_lon - static["lon_min"]) / R).astype(np.int64)

        lat_idx = np.clip(lat_idx, 0, static["nlat"] - 1)
        lon_idx = np.clip(lon_idx, 0, static["nlon"] - 1)
        flat_idx = lat_idx * static["nlon"] + lon_idx

        s_ef_std = np.take(static["ef_std_flat"], flat_idx)
        s_ef_flam = np.take(static["ef_flam_flat"], flat_idx)
        s_beta_vfei = np.take(static["beta_vfei_flat"], flat_idx)

        ef_to_use = np.where(incident_flag == 2, s_ef_flam, s_ef_std)
        emis_pm25 = fre * BETA_VAL * ef_to_use * 1e-3
        emis_pm25_vfei = fre * s_beta_vfei * ef_to_use * 1e-3

        invalid = (~np.isfinite(fre)) | (fre == FILL)
        emis_pm25 = np.where(invalid, FILL, emis_pm25).astype(np.float32, copy=False)
        emis_pm25_vfei = np.where(invalid, FILL, emis_pm25_vfei).astype(np.float32, copy=False)

        var1 = _get_or_create_var(
            nc,
            "EMIS_PM25",
            f"PM2.5 emissions calculated from NGFS FRE (Beta={BETA_VAL})",
        )
        var2 = _get_or_create_var(
            nc,
            "EMIS_PM25_VFEI",
            "PM2.5 emissions calculated from NGFS FRE using BETA_VFEI",
        )
        var1[:] = emis_pm25
        var2[:] = emis_pm25_vfei

    print(f"Successfully updated {fpath}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Append PM2.5 emissions variables to NGFS point-source NetCDF files."
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="One or more NetCDF files (shell wildcards are supported by bash).",
    )
    parser.add_argument(
        "--static-file",
        default=STATIC_FILE_PATH,
        help=f"Path to static EF NetCDF (default: {STATIC_FILE_PATH})",
    )
    args = parser.parse_args()

    if not os.path.exists(args.static_file):
        raise FileNotFoundError(f"Static file not found: {args.static_file}")

    print(f"Loading static data once: {args.static_file}")
    static = load_static_data(args.static_file)

    ok = 0
    for fpath in args.files:
        if process_single_file(fpath, static):
            ok += 1
    print(f"Done. Updated {ok}/{len(args.files)} files.")


if __name__ == "__main__":
    main()
