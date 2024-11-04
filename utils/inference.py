from math import pi
import os.path as pa
import sys

from myagn.flares import models as myflaremodels
import numpy as np
import astropy.units as u
import astropy_healpix as ah
from astropy.cosmology import FlatLambdaCDM
from myagn import distributions as myagndistributions
from scipy.stats import norm
from ligo.skymap.postprocess.crossmatch import crossmatch
from numpy.polynomial.polynomial import Polynomial

# Local imports
sys.path.append(pa.dirname(pa.dirname(__file__)))
import utils.graham23_tables as g23
import utils.io as io

######################################################################


def gauss(mu_gauss, std_gauss, x_value):
    """! Calculate value of gaussian(mu_gauss, std_gauss) at x_value."""
    return np.exp(-((x_value - mu_gauss) ** 2.0) / (2 * std_gauss**2.0)) / (
        std_gauss * (2.0 * pi) ** 0.5
    )


def lnlike(lam_li, s_arr, b_arr, frac_det):
    """! Calculates the log likelihood on the given lambda for a single GW event.
    c.f. eq. 7 in the paper.

    @param  lam_li      lambda-like; the parameter to evaluate the likelihood of.
                        lambda = fraction of BBH mergers that produce AGN flares
    @param  s_arr       Probabilities for each of len(s_arr) BBH AGN flares
    @param  b_arr       Probabilities for each of len(b_arr) background AGN flares
    @param  f           Fraction of detected to real flares

    """
    return np.sum(np.log(lam_li * s_arr + b_arr)) - frac_det * lam_li


def lnlike_all(lam_po, s_arrs, b_arrs, frac_det):

    # Iterate over followups
    lnlike_arr = np.zeros(len(s_arrs))
    for i in range(len(s_arrs)):
        # Get arrays
        s_arr = s_arrs[i]
        b_arr = b_arrs[i]

        ####################
        ###  Likelihood  ###
        ####################
        # If there are any candidates, calculate the log likelihood
        if (s_arr.shape[0]) > 0:
            # lnlike_arr.append(lnlike(s_arr,b_arr,lam_arr,f))
            lnlike_arr[i] = lnlike(lam_po, s_arr.value, b_arr, frac_det)

        else:
            # I think we need to do something different here, maybe s_arr needs to be set to 0
            lnlike_arr[i] = -lam_po

    # Sum log likelihoods and return with prior
    lnlikesum = np.sum(lnlike_arr, axis=0)
    # fixer=lnlikesum[0]
    # like=np.exp(lnlikesum-fixer)
    # normalization=np.trapezoid(like,lam_arr)

    return lnlikesum


def calc_s_arr(
    pb,
    pb_frac,
    distnorm,
    distmu,
    distsigma,
    z_flares,
    cosmo,
    z_grid,
):
    ncands = z_flares.shape[0]
    dL_flares = cosmo.luminosity_distance(z_flares)
    # ddL/dz, to transform dprob/dd_L to dprob/dz
    jacobian = (
        cosmo.comoving_distance(z_flares).value
        + (1.0 + z_flares) / cosmo.H(z_flares).value
    )
    # Renormalize candidate posteriors along los to same z range as the background
    # (the original skymaps normalize the posteriors over z=[0,inf))
    new_normaliz = np.ones(ncands)
    ds_arr = cosmo.luminosity_distance(z_grid).value
    jacobian_arr = (
        cosmo.comoving_distance(z_grid).value + (1.0 + z_grid) / cosmo.H(z_grid).value
    )
    for l in range(ncands):
        this_post = (
            gauss(
                distmu[l],
                distsigma[l],
                ds_arr,
            )
            * ds_arr**2
            * jacobian_arr
        )
        # Calculate normalization constant by ~integrating posterior
        new_normaliz[l] = distnorm[l] * np.trapezoid(this_post, z_grid)
    # Calculate signal probabilities, multiplying HEALPix and redshift probabilities
    s_arr = (
        pb  # HEALPix probability
        / pb_frac  # This isn't in the function's namespace; needed to normalize over credible region
        * distnorm  # redshift probability
        * norm(distmu, distsigma).pdf(dL_flares)
        * dL_flares**2
        / new_normaliz
        * jacobian
    )
    return s_arr


def calc_b_arr(
    z_flares,
    f_cover,
    n_idx_sort_cut,
    B_expected_n,
    cosmo,
    z_grid,
    agndist_config,
):
    # Get AGN distribution
    agndist = getattr(
        myagndistributions,
        agndist_config["model"],
    )(
        *agndist_config["args"],
        **agndist_config["kwargs"],
    )

    # Calculate normalization for background probabilities
    pb_Vunif = agndist.dn_dOmega_dz(
        z_grid,
        *agndist_config["density_args"],
        cosmo=cosmo,
        **agndist_config["density_kwargs"],
    )
    normaliz = np.trapezoid(pb_Vunif, z_grid)

    # Calculate background probabilities
    b_arr = (
        agndist.dn_dOmega_dz(
            z_flares,
            *agndist_config["density_args"],
            cosmo=cosmo,
            **agndist_config["density_kwargs"],
        )
        / normaliz
        / n_idx_sort_cut
        * B_expected_n
        * f_cover
    )

    return b_arr


def calc_arrs(
    H0,
    omegam,
    pb_frac,
    f_covers,
    distnorm,
    distmu,
    distsigma,
    cand_hpxprob_arr,
    cand_hpixs_arr,
    cand_zs_arr,
    n_idx_sort_cut,
    n_agn_coeffs,
    flares_per_agn,
    agndist_config,
    z_min_b,
    z_max_b,
):
    # Initialize cosmology
    thiscosmo = FlatLambdaCDM(H0=H0, Om0=omegam)

    # Iterate through followups
    zs_arr = np.linspace(z_min_b, z_max_b, num=1000)
    s_arrs = []
    b_arrs = []
    for i in range(len(cand_hpixs_arr)):
        # Get/calculate candidate coordinates, number of candidates for this followup
        followup_hpixs = cand_hpixs_arr[i]
        followup_zs = cand_zs_arr[i]
        ncands = followup_zs.shape[0]

        # Skip if no candidates
        if ncands == 0:
            s_arrs.append(np.array([]))
            b_arrs.append(np.array([]))
            continue

        ####################
        ###    Signal    ###
        ####################

        # Calculate GW counterpart probabilities
        s_arr = calc_s_arr(
            cand_hpxprob_arr[i],
            pb_frac,
            distnorm[i],
            distmu[i],
            distsigma[i],
            followup_zs,
            thiscosmo,
            zs_arr,
        )

        ####################
        ###  Background  ###
        ####################
        # normaliz = np.trapezoid(ds_arr_norm**2,ds_arr_norm)
        # b_arr = followup_ds**2/normaliz/len(idx_sort_cut)*B_expected_n

        # Use coeffs to interpolate number of AGN in volume
        n_agn = np.polynomial.Polynomial(n_agn_coeffs[i])(H0)

        # Multiply by flare rate to get expected number of background flares
        B_expected_n = n_agn * flares_per_agn[i]

        # Calculate background probabilities
        b_arr = calc_b_arr(
            followup_zs,
            f_covers[i],
            n_idx_sort_cut[i],
            B_expected_n,
            thiscosmo,
            zs_arr,
            agndist_config,
        )

        ####################
        ###    Append    ###
        ####################

        s_arrs.append(s_arr)
        b_arrs.append(b_arr)

    return s_arrs, b_arrs


def setup(config, df_fitparams):
    # Iterate over followups
    cand_hpixs = []
    cand_zs = []
    pbs, distnorms, distmus, distsigmas = [], [], [], []
    flares_per_agn = []
    n_agn_coeffs = []
    f_covers = []
    n_idx_sort_cut = []
    for i in g23.DF_GW.index:
        print("*" * 30)
        print(f"Index {i}")

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
        hs_flat = io.get_flattened_skymap(
            config["gwmapdir"], g23.DF_GW["gweventname"][i]
        )

        # Get data from skymap
        pb = np.array(hs_flat["PROB"])
        distnorm = np.array(hs_flat["DISTNORM"])
        distmu = np.array(hs_flat["DISTMU"])
        distsigma = np.array(hs_flat["DISTSIGMA"])
        NSIDE = ah.npix_to_nside(len(hs_flat))

        # Make idx_sort_up (array of pb indices, from highest to lowest pb)
        idx_sort = np.argsort(pb)
        idx_sort_up = list(reversed(idx_sort))
        # Add pbs until pb_frac is reached
        sum = 0.0
        id = 0
        while sum < config["followup_prob"]:
            this_idx = idx_sort_up[id]
            sum = sum + pb[this_idx]
            id = id + 1
        # Cut indices to <pb_frac> credible region
        idx_sort_cut = idx_sort_up[:id]
        n_idx_sort_cut.append(len(idx_sort_cut))

        ##############################
        ###         Flares         ###
        ##############################
        print("Loading flares...")

        # Get flares for this followup
        assoc_mask = g23.DF_ASSOC["gweventname"] == gweventname
        df_assoc_event = g23.DF_ASSOC[assoc_mask]

        # Get hpxs for the flares
        hpixs = np.array(
            ah.lonlat_to_healpix(
                df_assoc_event["flare_ra"].values * u.deg,
                df_assoc_event["flare_dec"].values * u.deg,
                NSIDE,
                order="nested",
            )
        )

        # Get redshifts for the flares
        zs = df_assoc_event["Redshift"].values

        # Compile positions and z for all candidates in this follow up
        cand_hpixs.append(np.array(hpixs))
        cand_zs.append(np.array(zs))
        if len(hpixs) == 0:
            pbs.append(np.array([]))
            distnorms.append(np.array([]))
            distmus.append(np.array([]))
            distsigmas.append(np.array([]))
        else:
            pbs.append(pb[hpixs])
            distnorms.append(distnorm[hpixs])
            distmus.append(distmu[hpixs])
            distsigmas.append(distsigma[hpixs])
        # cand_ds=np.concatenate((s_ds,b_ds))
        # ncands=cand_hpixs.shape[0]
        # cand_ds=np.zeros(ncands)
        # for l in range(ncands):
        #    cand_ds[l]=cosmo.luminosity_distance(cand_zs[l]).value

        ##############################
        ###    Background rate     ###
        ##############################

        # Skip volume calculation if no candidates
        if len(hpixs) == 0:
            flares_per_agn.append([np.nan] * df_assoc_event.shape[0])
            n_agn_coeffs.append([np.nan])
            f_covers.append(np.nan)
            print("No candidates; continuing.")
            continue

        print("Calculating background rate...")

        ### Set flares/AGN rate

        # Set flare model
        flaremodel = getattr(
            myflaremodels,
            config["flare_rate"]["model"],
        )(
            *config["flare_rate"]["args"],
            **config["flare_rate"]["kwargs"],
        )

        # Iterate over flares
        flares_per_agn_temp = []
        for ri, r in df_assoc_event.iterrows():
            # Get fitparams
            df_fitparams_flare = df_fitparams[
                df_fitparams["flarename"] == r["flarename"]
            ]

            # Iterate over filters
            rates = []
            print("\n", r["flarename"])
            for f in df_fitparams_flare["filter"]:
                # Use only g and r band data
                if f not in ["g", "r"]:
                    continue

                # Get row
                params = df_fitparams_flare[df_fitparams_flare["filter"] == f]

                # Calculate structure function prob
                # 3σ = 3 * t_rise is a conservative estimate for the Δt of the Δm
                # / (1+z) converts to rest frame timescale
                delta_t = 3 * params["t_rise"].values[0] / (1 + r["Redshift"])
                prob = flaremodel.flare_rate(
                    f,
                    delta_t,
                    -params["f_peak"].values[0],
                    r["Redshift"],
                )

                # Scale to follow-up window
                rate = prob * config["dt_followup"]
                print(f, rate)

                # Append to list
                rates.append(rate)

            # Save maximum rate
            flares_per_agn_temp.append(np.nanmin(rates))

        # Cast as array
        flares_per_agn_temp = np.array(flares_per_agn_temp)
        flares_per_agn.append(flares_per_agn_temp)

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
        n_agn_coeffs.append(coeffs)
        del sm

        # Append f_cover for GW event
        f_covers.append(g23.DF_GW["f_cover"][i])

    return (
        f_covers,
        distnorms,
        distmus,
        distsigmas,
        pbs,
        cand_hpixs,
        cand_zs,
        n_idx_sort_cut,
        n_agn_coeffs,
        flares_per_agn,
    )
