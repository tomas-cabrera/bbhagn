# The Graham+23 flares per AGN per time rate reported in Table 5 of the text is the rate in the observer's frame.
# Because the myagn code more theoretical, it is more useful to have the rate in the rest frame of the AGN.
# This script calculates the conversion factors, assuming that the rest frame rate does not change as a function of redshift.
# Specifically, the conversion rate is r_rest / r_obs = int(dz * dn/dz) / int(dz * dn/dz / (1+z)),
#   where "int" is an integral over the redshift domain of interest.

import os.path as pa
import sys

import astropy.units as u
import numpy as np
import yaml
from myagn import distributions as myagndistributions

# Local imports
sys.path.append(pa.dirname(pa.dirname(pa.dirname(__file__))))
from utils.paths import PROJDIR

################################################################################


def calc_rest_frame_rate_conversion(directory):
    # Config
    config = yaml.safe_load(open(pa.join(directory, "config.yaml")))
    if config["agn_distribution"]["model"] == "ConstantPhysicalDensity":
        config["agn_distribution"]["args"] = (
            config["agn_distribution"]["args"] * u.Mpc**-3
        )
    if "brightness_limits" in config["agn_distribution"]["density_kwargs"]:
        config["agn_distribution"]["density_kwargs"]["brightness_limits"] = [
            float(bl)
            for bl in config["agn_distribution"]["density_kwargs"]["brightness_limits"]
        ] * u.ABmag
    agndist_config = config["agn_distribution"]
    print(agndist_config)

    # Get AGN distribution
    agndist = getattr(
        myagndistributions,
        agndist_config["model"],
    )(
        *agndist_config["args"],
        **agndist_config["kwargs"],
    )
    # If Milliquas, add Graham+23 cuts
    # z <= 1.2 cut not included, because Graham+23 already does the 3D crossmatch and the z_grid we use here extends to 2.0
    if agndist_config["model"] == "Milliquas":
        mask = np.array(
            ["q" not in t for t in agndist._catalog["Type"]]
        )  # & (agndist._catalog["z"] <= 1.2)
        agndist._catalog = agndist._catalog[mask]

    # Calculate the conversion factor
    dn_dOmega_dz = agndist.dn_dOmega_dz
    zs = np.linspace(0, 1.2, 100)[1:]
    cfact = np.trapezoid(dn_dOmega_dz(zs), zs) / np.trapezoid(
        dn_dOmega_dz(zs) / (1 + zs), zs
    )
    print(f"Conversion factor: {cfact}")


################################################################################

# Get the directory path from the command line
if len(sys.argv) == 1:
    _default_array_jobs = [1, 2, 3, 4, 5, 7, 9, 10]
    print("Usage: python graham23_flarerate_obs_to_rest.py <path_to_directory>...")
    print(f"Defaulting to array jobs {_default_array_jobs}.")
    paths = [
        pa.join(PROJDIR, f"Posterior_sims_lambda_O4/array/{i}")
        for i in _default_array_jobs
    ]
else:
    paths = sys.argv[1:]

# Calculate the rest frame rates
for p in paths:
    calc_rest_frame_rate_conversion(p)
