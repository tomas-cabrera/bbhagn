import os.path as pa

import numpy as np
import pandas as pd

from .paths import PROJDIR, DATADIR
from .io import get_gwtc_skymap_path

TABLEDIR = pa.join(DATADIR, "graham23_tables")

# Get GW catalog information
DF_GW_G23 = pd.read_csv(f"{TABLEDIR}/graham23_table1.plus.dat", sep="\s+")

# Get GW-flare association information
DF_ASSOC = pd.read_csv(f"{TABLEDIR}/graham23_table3.plus.dat", sep="\s+")

# Extract ra, dec from flare names
flareras = []
flaredecs = []
for f in DF_ASSOC["flarename"]:
    # RA
    rastr = f[1:10]
    ra = float(rastr[0:2]) + float(rastr[2:4]) / 60.0 + float(rastr[4:]) / 3600.0
    ra *= 360 / 24  # Convert to degrees
    flareras.append(ra)

    # Dec
    decstr = f[10:]
    dec = float(decstr[0:3]) + float(decstr[3:5]) / 60.0 + float(decstr[5:]) / 3600.0
    flaredecs.append(dec)
DF_ASSOC["flare_ra"] = flareras
DF_ASSOC["flare_dec"] = flaredecs

# Get flare information
# Columns not identical across GW events are dropped
DF_FLARE = DF_ASSOC.drop_duplicates(subset=["flarename"]).drop(
    columns=["gweventname", "ConfLimit", "vk_max"]
)
DF_FLARE.rename(columns={"flare_ra": "ra", "flare_dec": "dec"}, inplace=True)
# Add fit parameters
df_fitparams = pd.read_csv(f"{PROJDIR}/fit_lightcurves_prev/fitparams.csv")
df_temp = []
# Iterate over flares
for fn in np.unique(df_fitparams["flarename"]):
    dict_temp = {"flarename": fn}
    # Get fit parameters for this flare
    df_fpfn = df_fitparams[df_fitparams["flarename"] == fn]
    # Iterate over filters
    for f in np.unique(df_fitparams["filter"]):
        # Iterate over columns, skipping flarename, filter
        for col in df_fitparams.columns:
            if col not in ["flarename", "filter"]:
                # If filter is present for this flare, get value
                if f in df_fpfn["filter"].values:
                    dict_temp[f"{col}_{f}"] = df_fpfn[df_fpfn["filter"] == f][
                        col
                    ].values[0]
                # Otherwise, set to nan
                else:
                    dict_temp[f"{col}_{f}"] = np.nan
    df_temp.append(dict_temp)
df_temp = pd.DataFrame(df_temp)
DF_FLARE = DF_FLARE.merge(df_temp, on="flarename", how="left")
DF_FLARE["mjd"] = DF_FLARE["t_peak_g"] - 3 * DF_FLARE["t_rise_g"]
df_flare_path = pa.join(DATADIR, "graham23_tables", "flare.csv")
DF_FLARE.to_csv(df_flare_path, index=False)

# Get bright? GW information
DF_GWBRIGHT = pd.read_csv(f"{TABLEDIR}/graham23_table4.plus.dat", sep="\s+")

# Get background flare information
DF_ASSOCPARAMS = pd.read_csv(f"{TABLEDIR}/graham23_table5.plus.dat", sep="\s+")

### DF_GW
# Get gweventname, f_cover from table 1
DF_GW = DF_GW_G23[["gweventname", "f_cover"]].copy()

# Remove events
skip_events = [
    "GW190425",  # BNS?
    "GW190426_152155",  # NSBH?
    "GW191219_163120",  # NSBH?
    "GW200105_162426",  # NSBH
    "GW200115_042309",  # NSBH
    "GW190424_180648",  # Lowered significance in GWTC2.1
    "GW190909_114149",  # Lowered significance in GWTC2.1
]
mask = [e.strip("*") not in skip_events for e in DF_GW["gweventname"]]
DF_GW = DF_GW[mask].reset_index(drop=True)

# Add skymap paths
MAPDIR = "/hildafs/projects/phy220048p/share/skymaps"
DF_GW["skymap_path"] = DF_GW["gweventname"].apply(
    lambda x: get_gwtc_skymap_path(MAPDIR, x)
)

# Remove * from event names (required to select waveform)
DF_GW["gweventname"] = DF_GW["gweventname"].str.strip("*")

# Add gwtc data
DF_GWTC = pd.read_csv(f"{DATADIR}/gwtc/events.csv")
matchrows = []
for i, row in DF_GW.iterrows():
    if row["gweventname"] in skip_events:
        print(f"Skipping {row['gweventname']}")
        continue
    # GW200105_162426 is not in the table
    if row["gweventname"] == "GW200105_162426":
        print("Copying custom data for GW200105_162426")
        dummyrow = dict(zip(DF_GWTC.columns, [None] * DF_GWTC.shape[1]))
        # Data to add (source: updated parameters in https://gwosc.org/eventapi/html/GWTC-3-marginal/GW200105_162426/v2/)
        adddata = {
            "name": "GW200105_162426",
            "mass_1_source": 9.1,
            "mass_1_source_lower": -1.7,
            "mass_1_source_upper": 1.7,
            "mass_2_source": 1.91,
            "mass_2_source_lower": -0.24,
            "mass_2_source_upper": 0.33,
            "luminosity_distance": 270,
            "luminosity_distance_lower": -110,
            "luminosity_distance_upper": 120,
            "chi_eff": 0.00,
            "chi_eff_lower": -0.18,
            "chi_eff_upper": 0.13,
            "total_mass_source": 11.0,
            "total_mass_source_lower": -1.4,
            "total_mass_source_upper": 1.5,
            "chirp_mass_source": 3.42,
            "chirp_mass_source_lower": -0.08,
            "chirp_mass_source_upper": 0.08,
            "chirp_mass": None,
            "chirp_mass_lower": None,
            "chirp_mass_upper": None,
            "final_mass_source": 10.8,
            "final_mass_source_lower": -1.4,
            "final_mass_source_upper": 1.5,
        }
        dummyrow.update(adddata)
        matchrows.append(dummyrow)
    elif row["gweventname"].strip("*") in DF_GWTC["name"].values:
        matchrow = DF_GWTC.loc[
            DF_GWTC["name"] == row["gweventname"].strip("*")
        ].to_dict(orient="records")[0]
        matchrows.append(matchrow)
    else:
        print(f"Did not find {row['gweventname']} in gwtc")
df_match = pd.DataFrame(matchrows).reset_index(drop=True)
DF_GW = pd.concat(
    [
        DF_GW,
        df_match,
    ],
    axis=1,
)
df_gw_path = pa.join(DATADIR, "graham23_tables", "gw.csv")
DF_GW.to_csv(df_gw_path, index=False)
