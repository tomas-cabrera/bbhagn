"""! Script for the calculations described in https://arxiv.org/abs/2103.16069 """

###############################################################################

import os.path as pa
import sys
from multiprocessing import Pool

import astropy.units as u
import astropy_healpix as ah
import corner
import emcee
import numpy as np
import pandas as pd
import yaml
from astropy.cosmology import FlatLambdaCDM
from ligo.skymap.postprocess.crossmatch import crossmatch
from myagn import distributions as myagndistributions
from myagn.flares import models as myflaremodels
from numpy.polynomial.polynomial import Polynomial

# Local imports
sys.path.append(pa.dirname(pa.dirname(__file__)))
import utils.graham23_tables as g23
import utils.inference as inference
import utils.io as io
from utils.paths import DATADIR, PROJDIR

###############################################################################

##############################
###        Functions       ###
##############################


def lnprior(theta):
    """! Prior function for the lambda and H0 parameters.
    Currently defined as a uniform distibution in the initial domain.
    Returns 0.0 if theta is in the initial domain; -np.inf otherwise.

    @param  theta       The argument of the function.
                        Should be a tuple[-like]: (lambda, H0)

    """
    # Extract values
    lam_pr = theta
    # If the values are in the uniform domain
    if 0.0 < lam_pr < 1.0:
        return 0.0
    # Otherwise
    return -np.inf


def lnprob(
    theta,
    s_arrs,
    b_arrs,
    frac_det,
):
    """! Calculates the log posterior probability, given some signal and background.

    @param  theta               Tuple of hypothesis lambda and H0
    @param  cand_hpixs_arr      Array of healpix index arrays for each followup
                                (shape=(N_GW_followup)(# of candidates in followup))
    @param  cand_zs_arr         Array of redshift arrays for each followup
                                (shape=(N_GW_followup)(# of candidates in followup))
    @param  pb                  Array of probability arrays for each candidate's healpixes
                                (shape=(N_GW_followup)(# of healpix in idx_sort_cut region))
    @param  distnorm            Normalization constant on distance estimate (takes into account d_L^2 weighting)
    @param  distmu              Mean of distance estimate for event, in Mpc
    @param  distsigma           Uncertainty on distance estimate, in Mpc
    @param  f                   Fraction of detected AGN flares to total AGN flares
    @param  N_GW_followups      Number of GW followups

    @return lp+lnlikesum        (log) posterior probability

    """
    # Calculate prior; if the prior is not finite, return -np.inf (event outside of prescribed domain)
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf

    # Extract lambda and H0 values
    lam_po = theta

    # Calculate lnlike
    lnlike = inference.lnlike_all(lam_po, s_arrs, b_arrs, frac_det)

    return lp + lnlike


###############################################################################

if __name__ == "__main__":

    ##############################
    ###       Constants        ###
    ##############################

    # Load yaml file
    config_file = sys.argv[
        1
    ]  # "/hildafs/home/tcabrera/HIPAL/bbhagn/bbhagn/config.yaml"
    config_dir = pa.dirname(config_file)
    config = yaml.safe_load(open(config_file))

    # Parse AGN distribution config
    if config["agn_distribution"]["model"] == "ConstantPhysicalDensity":
        config["agn_distribution"]["args"] = (
            config["agn_distribution"]["args"] * u.Mpc**-3
        )
    if "brightness_limits" in config["agn_distribution"]["density_kwargs"]:
        config["agn_distribution"]["density_kwargs"]["brightness_limits"] = [
            float(bl)
            for bl in config["agn_distribution"]["density_kwargs"]["brightness_limits"]
        ] * u.ABmag

    # Define constants
    N_GW_followups = g23.DF_GW.shape[0]  # [50,10]
    lam_arr = np.linspace(0, 1.0, num=100)

    # Load lightcurve fit parameters
    df_fitparams = pd.read_csv(f"{PROJDIR}/fit_lightcurves/fitparams.csv")

    # Calculate maximum distance for background events
    cosmo = FlatLambdaCDM(H0=config["H00"], Om0=config["Om0"])
    maxdist = cosmo.luminosity_distance(config["z_max_b"])
    ds_arr_norm = np.linspace(0, maxdist.value, num=1000)

    ##############################
    ###         Science        ###
    ##############################
    print("*" * 20, "Preparation", "*" * 20)

    lnprob_args = inference.setup(config, df_fitparams)

    ##############################
    ###       Calc. arrs       ###
    ##############################
    print("Calculating signal and background arrays...")

    s_arrs, b_arrs = inference.calc_arrs(
        config["H00"],
        config["Om0"],
        config["followup_prob"],
        *lnprob_args,
        config["agn_distribution"],
        config["z_min_b"],
        config["z_max_b"],
    )

    ##############################
    ###          MCMC          ###
    ##############################
    print("*" * 20, "MCMC", "*" * 20)

    # Initialize some random lambdas and H0s for MCMC (w/ gaussian perturbations)
    # np.random.randn(nwalkers, ndim=nparam)
    pos = [config["lambda0"]] + 1e-4 * np.random.randn(config["nwalkers"], 1)
    nwalkers, ndim = pos.shape
    # Define sampler args
    sampler_args = (nwalkers, ndim, lnprob)
    # Define sampler kwargs
    sampler_kwargs = {
        "args": (
            s_arrs,
            b_arrs,
            config["frac_det"],
        ),
    }
    # Define run_mcmc args
    mcmc_args = (pos, config["nsteps"])
    # Define run_mcmc kwargs
    mcmc_kwargs = {
        "progress": True,
    }
    # Run the MCMC, in parallel or serial
    if config["nproc"] != 1:
        with Pool(config["nproc"]) as pool:
            sampler_kwargs["pool"] = pool
            # Initialize sampler + run MCMC
            sampler = emcee.EnsembleSampler(*sampler_args, **sampler_kwargs)
            sampler.run_mcmc(*mcmc_args, **mcmc_kwargs)
    else:
        # Initialize sampler + run MCMC
        sampler = emcee.EnsembleSampler(*sampler_args, **sampler_kwargs)
        sampler.run_mcmc(*mcmc_args, **mcmc_kwargs)

    ##############################
    ###         Output         ###
    ##############################
    print("*" * 20, "Output", "*" * 20)

    # Reduce and retreive MCMC samples
    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
    print(flat_samples.shape)

    # Save data
    print("Saving data...")
    np.savetxt(f"{config_dir}/O4_samples_graham23.dat", flat_samples)

    # Make+save corner plot
    print("Making corner plot...")
    labels = [r"$\lambda$", r"$H_0$ [km/s/Mpc]"]
    fig = corner.corner(
        flat_samples,
        labels=labels,
        # truths=[lamb, H0],
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 12},
    )
    fig.savefig(f"{config_dir}/corner_O4_graham23.png", dpi=200)

    print("Done.")
