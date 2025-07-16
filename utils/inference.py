import os.path as pa
import sys
from math import pi

import astropy.units as u
import astropy_healpix as ah
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from ligo.skymap.postprocess.crossmatch import crossmatch
import ligo.skymap.moc as lsm_moc
from numpy.polynomial.polynomial import Polynomial
from scipy.stats import norm
from multiprocessing import current_process

# Local imports
PROJDIR = pa.dirname(pa.dirname(__file__))
sys.path.append(PROJDIR)
from myagn import distributions as myagndistributions
from myagn.flares import models as myflaremodels
import utils.graham23_tables as g23
import utils.io as io
from utils.gwselection import selection_beta

######################################################################


def gauss(mu_gauss, std_gauss, x_value):
    """! Calculate value of gaussian(mu_gauss, std_gauss) at x_value."""
    return np.exp(-((x_value - mu_gauss) ** 2.0) / (2 * std_gauss**2.0)) / (
        std_gauss * (2.0 * pi) ** 0.5
    )


def lnlike(theta, s_arr, b_arr, frac_det, n_flares_bg):
    """! Calculates the log likelihood on the given lambda for a single GW event.
    c.f. eq. 7 in the paper.

    @param  lam_li      lambda-like; the parameter to evaluate the likelihood of.
                        lambda = fraction of BBH mergers that produce AGN flares
    @param  s_arr       Probabilities for each of len(s_arr) BBH AGN flares
    @param  b_arr       Probabilities for each of len(b_arr) background AGN flares
    @param  f           Fraction of detected to real flares

    """
    # Extract values
    lam_li, H0_li, omegam_li = theta

    return (
        np.sum(np.log(lam_li * s_arr + b_arr))
        - np.log(selection_beta(H0_li))
        - frac_det * (lam_li + n_flares_bg)
    )


def lnlike_all(
    theta,
    s_arrs,
    b_arrs,
    frac_det,
    n_agns,
    flares_per_agn_average,
    flare_rate_distribution,
):
    # Extract values
    lam_po, H0_po, omega_po = theta

    # Calculate maximum in n_flares_bgs
    max_n_flares_bg = np.nanmax([np.nanmax(f) for f in n_agns if len(f) > 0])

    # Sample flare rate
    # Currently using process id as seed
    # Note that the pid does not index the processes spawned by emcee but appears to be the python process id;
    # in this case the numbers are ~70-80
    # Not sure if this ensures that the rng is different for each process
    rng = np.random.default_rng(int(current_process().name.split("-")[-1]))
    dist = getattr(rng, flare_rate_distribution["type"])
    flare_rate = np.nan * np.ones_like(flares_per_agn_average)
    for i, f in enumerate(flares_per_agn_average):
        if np.isnan(f):
            continue
        if ~np.isnan(flare_rate[i]):
            continue
        fr = -1
        while fr < 0:
            fr = dist(f, **flare_rate_distribution["kwargs"])
        flare_rate[flares_per_agn_average == f] = fr

    # Iterate over GW followups
    lnlike_arr = np.zeros(len(s_arrs))
    for i in range(len(s_arrs)):
        # Get arrays
        s_arr_i = s_arrs[i]
        b_arr_i = b_arrs[i]
        n_agns_i = n_agns[i]

        # Get non-filled values
        s_arr_i = s_arr_i[s_arr_i != 0]
        b_arr_i = b_arr_i[b_arr_i != 1]
        ####################
        ###  Likelihood  ###
        ####################
        # If there are any candidates, calculate the log likelihood
        if (s_arr_i.shape[0]) > 0:
            # n_agns_i = n_agns_i[~np.isnan(n_agns_i)]
            n_flares_bgs_i = n_agns_i * flare_rate
            n_flares_bgs_i = n_flares_bgs_i[~np.isnan(n_flares_bgs_i)]

            # lnlike_arr.append(lnlike(s_arr,b_arr,lam_arr,f))
            lnlike_arr[i] = lnlike(
                theta,
                s_arr_i,
                b_arr_i * n_flares_bgs_i,
                frac_det,
                np.nanmax(n_flares_bgs_i),
            )

        else:
            # I think we need to do something different here, maybe s_arr needs to be set to 0
            # lnlike_arr[i] = -lam_po
            lnlike_arr[i] = -np.log(selection_beta(H0_po)) - frac_det * (
                lam_po + max_n_flares_bg
            )

    # Sum log likelihoods and return with prior
    lnlikesum = np.sum(lnlike_arr, axis=0)
    # fixer=lnlikesum[0]
    # like=np.exp(lnlikesum-fixer)
    # normalization=np.trapezoid(like,lam_arr)

    return lnlikesum


def calc_s_arr(
    pbden,
    distnorm,
    distmu,
    distsigma,
    z_flares,
    cosmo,
):
    ncands = z_flares.shape[0]
    dL_flares = cosmo.luminosity_distance(z_flares)
    # ddL/dz, to transform dprob/dd_L to dprob/dz
    jacobian = (
        cosmo.comoving_distance(z_flares).value
        + (1.0 + z_flares) / cosmo.H(z_flares).value
    )
    # Calculate signal probability densities, multiplying HEALPix and redshift contributions
    s_arr = (
        pbden  # HEALPix probability density
        * distnorm  # redshift probability density
        * norm(distmu, distsigma).pdf(dL_flares)
        * dL_flares**2
        * jacobian
    )
    return s_arr


def calc_b_arr(
    z_flares,
    cosmo,
    agndist_config,
):
    # NOTE: Since implementing the flares_per_agn sampling, b_arr is the number density of AGN,
    # i.e. it still needs to be multiplied by [n_flares_per_agn] to get the background AGN flare number density

    # Get AGN distribution
    # Hardcoded values use QLFHopkins for all
    agndist = getattr(
        myagndistributions,
        # agndist_config["model"],
        "QLFHopkins",
    )(
        # *agndist_config["args"],
        *[] * u.Mpc**-3,
        # **agndist_config["kwargs"],
        **{},
    )

    # If Milliquas, add Graham+23 cuts
    # z <= 1.2 cut not included, because Graham+23 already does the 3D crossmatch and the z_grid we use here extends to 2.0
    # if agndist_config["model"] == "Milliquas":
    if False:
        mask = np.array(
            ["q" not in t for t in agndist._catalog["Type"]]
        )  # & (agndist._catalog["z"] <= 1.2)
        agndist._catalog = agndist._catalog[mask]

    # Calculate background probability densities
    b_arr = agndist.dn_dOmega_dz(
        z_flares,
        # *agndist_config["density_args"],
        *[],
        cosmo=cosmo,
        # **agndist_config["density_kwargs"],
        **{
            "brightness_limits": [float(b) for b in [20.5, "-inf"]] * u.ABmag,
            "band": "g",
        },
    )

    return b_arr


def calc_arrs(
    H0,
    omegam,
    f_covers,
    distnorm,
    distmu,
    distsigma,
    cand_hpxprobden_arr,
    cand_zs_arr,
    n_agn_coeffs,
    agndist_config,
    z_min_b,
    z_max_b,
):
    # Initialize cosmology
    thiscosmo = FlatLambdaCDM(H0=H0, Om0=omegam)

    # Iterate through true associations
    # NOTE: np.<>_like(u.Quantity) returns a Quantity with the same unit as the input
    s_arrs = np.zeros_like(cand_hpxprobden_arr)
    b_arrs = np.ones_like(cand_hpxprobden_arr)
    n_agns = np.ones_like(cand_hpxprobden_arr).value * np.nan
    # Iterate over GW followups
    for gwi in range(cand_hpxprobden_arr.shape[0]):
        # Get flare indices
        fis = list(np.where(~np.isnan(cand_hpxprobden_arr[gwi]))[0])

        # Skip remainder if no candidates
        if len(fis) == 0:
            continue

        # Get/calculate candidate coordinates, number of candidates for this followup
        followup_zs = cand_zs_arr[fis]
        ncands = followup_zs.shape[0]

        ####################
        ###Expected flares##
        ####################

        # Use coeffs to interpolate number of AGN in volume
        n_agn = np.polynomial.Polynomial(n_agn_coeffs[gwi])(H0)

        # Add to array
        n_agns[gwi, fis] = n_agn

        ####################
        ###    Signal    ###
        ####################

        # Calculate GW counterpart probabilities
        s_arr = calc_s_arr(
            cand_hpxprobden_arr[gwi][fis],
            distnorm[gwi][fis],
            distmu[gwi][fis],
            distsigma[gwi][fis],
            followup_zs,
            thiscosmo,
        )

        ####################
        ###  Background  ###
        ####################

        # Calculate background AGN probabilities
        b_arr = calc_b_arr(
            followup_zs,
            thiscosmo,
            agndist_config,
        )

        ####################
        ###    Add to arrs   ###
        ####################

        # Add to arrays
        s_arrs[gwi, fis] = s_arr
        b_arrs[gwi, fis] = b_arr

    return s_arrs.to(1 / u.sr).value, b_arrs.to(1 / u.sr).value, n_agns


def _setup_task(i, config, df_fitparams):
    print("*" * 30)
    print(f"Index {i}")
    return_dict = {}

    ##############################
    ###  Signals (BBH flares)  ###
    ##############################

    # Get eventname, strip asterisk if needed
    gweventname = g23.DF_GW["gweventname"][i]
    if gweventname.endswith("*"):
        gweventname = gweventname[:-1]
    print(gweventname)

    ##############################
    ###       GW skymap        ###
    ##############################
    print("Loading skymap...")

    # Load skymap
    sm = io.get_gwtc_skymap(config["gwmapdir"], g23.DF_GW["gweventname"][i])

    # Get data from skymap
    pbden = sm["PROBDENSITY"]
    distnorm = sm["DISTNORM"]
    distmu = sm["DISTMU"]
    distsigma = sm["DISTSIGMA"]

    # # Make idx_sort_up (array of pb indices, from highest to lowest pb)
    # sm["AREA"] = lsm_moc.uniq2pixarea(sm["UNIQ"]) * u.sr
    # sm["PROB"] = sm["PROBDENSITY"] * sm["AREA"]
    # pb = sm["PROB"]
    # idx_sort = np.argsort(pb)
    # idx_sort_up = list(reversed(idx_sort))
    # # Add pbs until pb_frac is reached
    # sum = 0.0
    # id = 0
    # while sum < config["followup_prob"]:
    #     this_idx = idx_sort_up[id]
    #     sum = sum + pb[this_idx]
    #     id = id + 1
    # # Cut indices to <pb_frac> credible region
    # idx_sort_cut = idx_sort_up[:id]
    # return_dict["n_idx_sort_cut"] = len(idx_sort_cut)

    ##############################
    ###         Flares         ###
    ##############################
    print("Loading flares...")

    # Get flares for this followup
    assoc_mask = g23.DF_ASSOC["gweventname"] == gweventname
    followup_flares = g23.DF_ASSOC["flarename"][assoc_mask]

    # Iterate over flares
    pbdens = []
    distnorms = []
    distmus = []
    distsigmas = []
    for _, fr in g23.DF_FLARE.iterrows():
        # Check if flare in followup
        if fr["flarename"] not in followup_flares.values:
            pbdens.append(np.nan)
            distnorms.append(np.nan)
            distmus.append(np.nan)
            distsigmas.append(np.nan)
            continue

        # Get uniqs for the flare
        lon = fr["flare_ra"] * u.deg
        lat = fr["flare_dec"] * u.deg
        uniq = io.lonlat_to_uniq(
            lon,
            lat,
            sm["UNIQ"],
        )
        ind_sm = np.where(sm["UNIQ"] == uniq)[0][0]

        # Get data for flare
        pbdens.append(pbden[ind_sm])
        distnorms.append(distnorm[ind_sm])
        distmus.append(distmu[ind_sm])
        distsigmas.append(distsigma[ind_sm])

    # Add to return_dict
    return_dict["pbdens"] = pbdens * pbden.unit
    return_dict["distnorms"] = distnorms * distnorm.unit
    return_dict["distmus"] = distmus * distmu.unit
    return_dict["distsigmas"] = distsigmas * distsigma.unit

    ##############################
    ###    Background rate     ###
    ##############################

    print("Calculating background rate...")

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
    # Define volume integrand
    def volume_integrand(
        dL,
        cosmo=FlatLambdaCDM(H0=config["H00"], Om0=config["Om0"]),
    ):
        # Includes unit conversions to cohere with myagn convention
        return (
            agndist._dn_d3Mpc_at_dL(
                dL * u.Mpc,
                cosmo=cosmo,
                **config["agn_distribution"]["density_kwargs"],
            )
            .to(u.Mpc**-3)
            .value
        )

    # Load skymap
    sm = io.get_gwtc_skymap(config["gwmapdir"], g23.DF_GW["gweventname"][i])
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
            volume_integrand=volume_integrand,
        ).contour_vols[0]
        n_agns.append(n_agn)
    # Calculate polynomial fit to model n_agn as a function of H0
    fit = Polynomial.fit(hs, n_agns, 4)
    coeffs = fit.convert().coef
    return_dict["n_agn_coeffs"] = coeffs
    del sm

    # Append f_cover for GW event
    return_dict["f_covers"] = g23.DF_GW["f_cover"][i]

    return return_dict


def setup(config, df_fitparams, nproc=1):
    ##############################
    ###    Flares  ###
    ##############################

    # Set cand_zs
    cand_zs = g23.DF_FLARE["Redshift"].values

    # flares_per_agn rates
    # Set flare model
    flaremodel = getattr(
        myflaremodels,
        config["flare_rate"]["model"],
    )(
        *config["flare_rate"]["args"],
        **config["flare_rate"]["kwargs"],
    )

    # Iterate over flares
    flares_per_agn_average = []
    for _, fr in g23.DF_FLARE.iterrows():
        # Get fitparams
        df_fitparams_flare = df_fitparams[df_fitparams["flarename"] == fr["flarename"]]

        # Iterate over filters
        rates = []
        print("\n", fr["flarename"])
        for f in df_fitparams_flare["filter"]:
            # Use only g and r band data
            if f not in ["g", "r"]:
                continue

            # Get row
            params = df_fitparams_flare[df_fitparams_flare["filter"] == f]

            # Calculate structure function prob
            # 3σ = 3 * t_rise is a conservative estimate for the Δt of the Δm
            # / (1+z) converts to rest frame timescale
            delta_t = 3 * params["t_rise"].values[0] / (1 + fr["Redshift"])
            prob = flaremodel.flare_rate(
                f,
                delta_t,
                -params["f_peak"].values[0],
                fr["Redshift"],
            )

            # Scale to follow-up window
            rate = prob * config["dt_followup"]
            print(f, rate)

            # Append to list
            rates.append(rate)

        # Save lowest rate
        flares_per_agn_average.append(np.nanmin(rates))

    ##############################
    ###    GWs + flares        ###
    ##############################

    # Iterate over followups
    if nproc > 1:
        from multiprocessing import Pool

        with Pool(nproc) as p:
            results = p.starmap(
                _setup_task,
                [(i, config, df_fitparams) for i in g23.DF_GW.index],
            )
    else:
        results = [_setup_task(i, config, df_fitparams) for i in g23.DF_GW.index]

    # Unpack results
    f_covers = np.array([r["f_covers"] for r in results])
    distnorms = u.Quantity([r["distnorms"] for r in results])
    distmus = u.Quantity([r["distmus"] for r in results])
    distsigmas = u.Quantity([r["distsigmas"] for r in results])
    pbdens = u.Quantity([r["pbdens"] for r in results])
    n_agn_coeffs = [r["n_agn_coeffs"] for r in results]
    flares_per_agn_average = np.array(flares_per_agn_average)

    return (
        f_covers,
        distnorms,
        distmus,
        distsigmas,
        pbdens,
        cand_zs,
        n_agn_coeffs,
        flares_per_agn_average,
    )
