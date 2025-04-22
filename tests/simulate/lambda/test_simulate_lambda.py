import os.path as pa
import sys
from glob import glob

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from myagn import distributions as myagndistributions

GDIR = pa.dirname(pa.dirname(pa.dirname(pa.dirname(pa.abspath(__file__)))))
sys.path.insert(1, GDIR)
from utils.simulate import simulate_flares

################################################################################

# Setup
lamb_0 = 0.3
H0_0 = 70
Om0_0 = 0.3
cosmo_0 = FlatLambdaCDM(H0=H0_0, Om0=Om0_0)
rng_0 = np.random.default_rng(12345)
f_cover_0 = 0.9
# Generate df_gw_input
n_gw = 200
GWDIR = "/hildafs/projects/phy220048p/share/skymap_fits_kunnumkai/O5_bbh_fits"
skymap_paths = glob(f"{GWDIR}/*.fits")
skymap_paths = rng_0.choice(skymap_paths, n_gw, replace=False)
df_gw_input = []
print("Generating df_gw_input...")
for smp in skymap_paths:
    with fits.open(smp) as hdul:
        t = hdul[1].header["MJD-OBS"]
        df_gw_input.append(
            {
                "skymap_path": smp,
                "t": t,
                "f_cover": f_cover_0,
            }
        )
df_gw_input = pd.DataFrame(df_gw_input)
# \end Generate df_gw_input
agn_dist_0 = myagndistributions.ConstantPhysicalDensity(
    10**-4.75 * u.Mpc**-3,
)
dt_followup = 200
nproc = 32

# Simulate flares
print("Simulating flares...")
df_flares = simulate_flares(
    lamb_0,
    cosmo_0,
    df_gw_input,
    agn_dist_0,
    dt_followup,
    z_grid=np.linspace(0, 2, 1000),
    rng=rng_0,
    nproc=nproc,
)
print(df_flares)
