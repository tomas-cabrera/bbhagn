import glob
import os.path as pa
import sys

import matplotlib.pyplot as plt
import numpy as np
import yaml
from astropy.cosmology import FlatLambdaCDM
from ligo.skymap.postprocess.crossmatch import crossmatch
from myagn import distributions as myagndistributions
from numpy.polynomial.polynomial import Polynomial

# Local imports
sys.path.append(pa.dirname(pa.dirname(pa.dirname(__file__))))
import utils.graham23_tables as g23
from Posterior_sims_H0_O4 import get_gwtc_skymap
from utils.paths import DATADIR, PROJDIR

if __name__ == "__main__":
    # Initialze plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Iterate over config files
    globstr = pa.join(PROJDIR, "Posterior_sims_H0_O4", "array", "*", "config.yaml")
    configs = glob.glob(globstr)
    for config_path in configs:

        # Load yaml file
        config_file = sys.argv[
            1
        ]  # "/hildafs/home/tcabrera/HIPAL/bbhagn/bbhagn/config.yaml"
        config_dir = pa.dirname(config_file)
        config = yaml.safe_load(open(config_file))

        ### Set AGN count model (polynomial as function of H0)
        # Get AGN distribution
        agndist = getattr(
            myagndistributions,
            config["agn_distribution"]["model"],
        )(
            *config["agn_distribution"]["args"],
            **config["agn_distribution"]["kwargs"],
        )

        # Set density_kwargs as attributes
        for k, v in config["agn_distribution"]["density_kwargs"].items():
            setattr(agndist, k, v)

        # Model number of AGNs
        # Load skymap
        sm = get_gwtc_skymap(config["gwmapdir"], g23.DF_GW["gweventname"][i])
        # Iterate over a sample of H0 values
        n_agns = []
        hs = np.linspace(20, 120, num=10)
        for h in hs:
            tempcosmo = FlatLambdaCDM(H0=h, Om0=config["Om0"])
            n_agn = crossmatch(
                sm,
                contours=[0.9],
                cosmology=True,
                cosmo=tempcosmo,
                integrand=agndist.dn_d3Mpc_at_dL,
            ).contour_vols[0]
            n_agns.append(n_agn)
        # Calculate polynomial fit to model n_agn as a function of H0
        fit = Polynomial.fit(hs, n_agns, 4)
        coeffs = fit.convert().coef
        del sm

        # Plot volumes
        ax.scatter(hs, n_agns, label=pa.basename(pa.dirname(config)))

        # Plot polynomial fit
        hs_fit = np.linspace(20, 120, num=101)
        n_agns_fit = fit(hs_fit)
        ax.plot(
            hs_fit, n_agns_fit, label=pa.basename(pa.dirname(config)), linestyle="--"
        )

    plt.xlabel("H0")
    plt.ylabel("N(AGN)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(__file__.replace(".py", ".png"))
    plt.close()
