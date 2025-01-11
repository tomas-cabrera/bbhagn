import os.path as pa

import pandas as pd

from .paths import DATADIR

TABLEDIR = pa.join(DATADIR, "graham23_tables")

# Get GW catalog information
DF_GW = pd.read_csv(f"{TABLEDIR}/graham23_table1.plus.dat", sep="\s+")

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

# Get bright? GW information
DF_GWBRIGHT = pd.read_csv(f"{TABLEDIR}/graham23_table4.plus.dat", sep="\s+")

# Get background flare information
DF_ASSOCPARAMS = pd.read_csv(f"{TABLEDIR}/graham23_table5.plus.dat", sep="\s+")
