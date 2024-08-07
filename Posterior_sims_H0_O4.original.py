"""! Script for the calculations described in https://arxiv.org/abs/2103.16069 """

###############################################################################

import random
from math import pi
from multiprocessing import Pool

import astropy.units as u
import corner
import emcee
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM, z_at_value
from astropy.io import fits
from joblib import Parallel, delayed
from scipy import interpolate, optimize
from scipy.integrate import simps
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.stats import norm

###############################################################################

##############################
###         Colors         ###
##############################

# These are the "Tableau 20" colors as RGB.
tableau20 = [
    (31, 119, 180),
    (174, 199, 232),
    (255, 127, 14),
    (255, 187, 120),
    (44, 160, 44),
    (152, 223, 138),
    (214, 39, 40),
    (255, 152, 150),
    (148, 103, 189),
    (197, 176, 213),
    (140, 86, 75),
    (196, 156, 148),
    (227, 119, 194),
    (247, 182, 210),
    (127, 127, 127),
    (199, 199, 199),
    (188, 189, 34),
    (219, 219, 141),
    (23, 190, 207),
    (158, 218, 229),
]

# Tableau Color Blind 10
tableau20blind = [
    (0, 107, 164),
    (255, 128, 14),
    (171, 171, 171),
    (89, 89, 89),
    (95, 158, 209),
    (200, 82, 0),
    (137, 137, 137),
    (163, 200, 236),
    (255, 188, 121),
    (207, 207, 207),
]

# Rescale to values between 0 and 1
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255.0, g / 255.0, b / 255.0)
for i in range(len(tableau20blind)):
    r, g, b = tableau20blind[i]
    tableau20blind[i] = (r / 255.0, g / 255.0, b / 255.0)

###############################################################################

##############################
###        Functions       ###
##############################


def plt_style():
    plt.rcParams.update(
        {
            "lines.linewidth": 1.0,
            "lines.linestyle": "-",
            "lines.color": "black",
            "font.family": "serif",
            "font.weight": "normal",
            "font.size": 16.0,
            "text.color": "black",
            "text.usetex": True,
            "axes.edgecolor": "black",
            "axes.linewidth": 1.0,
            "axes.grid": False,
            "axes.titlesize": "large",
            "axes.labelsize": "large",
            "axes.labelweight": "normal",
            "axes.labelcolor": "black",
            "axes.formatter.limits": [-4, 4],
            "xtick.major.size": 7,
            "xtick.minor.size": 4,
            "xtick.major.pad": 8,
            "xtick.minor.pad": 8,
            "xtick.labelsize": "large",
            "xtick.minor.width": 1.0,
            "xtick.major.width": 1.0,
            "ytick.major.size": 7,
            "ytick.minor.size": 4,
            "ytick.major.pad": 8,
            "ytick.minor.pad": 8,
            "ytick.labelsize": "large",
            "ytick.minor.width": 1.0,
            "ytick.major.width": 1.0,
            "legend.numpoints": 1,
            "legend.fontsize": "large",
            "legend.shadow": False,
            "legend.frameon": False,
        }
    )


def lnprior(theta):
    """! Prior function for the lambda and H0 parameters.
    Currently defined as a uniform distibution in the initial domain.
    Returns 0.0 if theta is in the initial domain; -np.inf otherwise.

    @param  theta       The argument of the function.
                        Should be a tuple[-like]: (lambda, H0)

    """
    # Extract values
    lam_pr, H0_pr = theta
    # If the values are in the uniform domain
    if 0.0 < lam_pr < 1.0 and 20.0 < H0_pr < 120.0:
        return 0.0
    # Otherwise
    return -np.inf


def lnlike(lam_li, s_arr, b_arr, f):
    """! Calculates the log likelihood on the given lambda.
    c.f. eq. 7 in the paper.

    @param  lam_li      lambda-like; the parameter to evaluate the likelihood of.
                        lambda = fraction of BBH mergers that produce AGN flares
    @param  s_arr       Probabilities for each of len(s_arr) BBH AGN flares
    @param  b_arr       Probabilities for each of len(b_arr) background AGN flares
    @param  f           Fraction of detected to real flares

    """
    return np.sum(np.log(lam_li * s_arr + b_arr)) - f * lam_li


def lnprob(
    theta,
    cand_hpixs_arr,
    cand_zs_arr,
    pb,
    distnorm,
    distmu,
    distsigma,
    f,
    N_GW_followups,
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
    lam_po, H0_po = theta

    # Cosmo. params
    omegam = 0.3
    thiscosmo = FlatLambdaCDM(H0=H0_po, Om0=omegam)

    # Iterate through followups
    zs_arr = np.linspace(z_min_b, z_max_b, num=1000)
    lnlike_arr = np.zeros(N_GW_followups)
    for i in range(N_GW_followups):
        ####################
        ###    Signal    ###
        ####################
        # Get/calculate candidate coordinates, number of candidates for this followup
        followup_hpixs = cand_hpixs_arr[i]
        followup_zs = cand_zs_arr[i]
        followup_ds = thiscosmo.luminosity_distance(followup_zs)
        ncands = followup_zs.shape[0]
        # ddL/dz, to transform dprob/dd_L to dprob/dz
        jacobian = (
            thiscosmo.comoving_distance(followup_zs).value
            + (1.0 + followup_zs) / thiscosmo.H(followup_zs).value
        )
        # Renormalize candidate posteriors along los to same z range as the background
        # (the original skymaps normalize the posteriors over z=[0,inf))
        new_normaliz = np.ones(ncands)
        ds_arr = thiscosmo.luminosity_distance(zs_arr).value
        jacobian_arr = (
            thiscosmo.comoving_distance(zs_arr).value
            + (1.0 + zs_arr) / thiscosmo.H(zs_arr).value
        )
        for l in range(ncands):
            this_post = (
                gauss(
                    distmu[i][followup_hpixs[l]],
                    distsigma[i][followup_hpixs[l]],
                    ds_arr,
                )
                * ds_arr**2
                * jacobian_arr
            )
            # Calculate normalization constant by ~integrating posterior
            new_normaliz[l] = distnorm[i][followup_hpixs[l]] * np.trapz(
                this_post, zs_arr
            )
        # Calculate signal probabilities, multiplying HEALPix and redshift probabilities
        s_arr = (
            pb[i][followup_hpixs]  # HEALPix probability
            / pb_frac  # This isn't in the function's namespace; needed to normalize over credible region
            * distnorm[i][followup_hpixs]  # redshift probability
            * norm(distmu[i][followup_hpixs], distsigma[i][followup_hpixs]).pdf(
                followup_ds
            )
            * followup_ds**2
            / new_normaliz
            * jacobian
        )

        ####################
        ###  Background  ###
        ####################
        # normaliz = np.trapz(ds_arr_norm**2,ds_arr_norm)
        # b_arr = followup_ds**2/normaliz/len(idx_sort_cut)*B_expected_n

        normaliz = np.trapz(
            pb_Vunif, zs_arr
        )  # pb_Vunit isn't in the function's namespace, along with some things below
        b_arr = (
            thiscosmo.comoving_distance(followup_zs).value ** 2
            / thiscosmo.H(followup_zs).value
            / normaliz
            / len(idx_sort_cut)
            * B_expected_n
        )

        ####################
        ###  Likelihood  ###
        ####################
        # If there are any candidates, calculate the log likelihood
        if (s_arr.shape[0]) > 0:
            # lnlike_arr.append(lnlike(s_arr,b_arr,lam_arr,f))
            lnlike_arr[i] = lnlike(lam_po, s_arr.value, b_arr, f)

        else:
            # I think we need to do something different here, maybe s_arr needs to be set to 0
            lnlike_arr[i] = -lam_po

    # Sum log likelihoods and return with prior
    lnlikesum = np.sum(lnlike_arr, axis=0)
    # fixer=lnlikesum[0]
    # like=np.exp(lnlikesum-fixer)
    # normalization=np.trapz(like,lam_arr)
    return lp + lnlikesum


def gauss(mu_gauss, std_gauss, x_value):
    """! Calculate value of gaussian(mu_gauss, std_gauss) at x_value."""
    return np.exp(-((x_value - mu_gauss) ** 2.0) / (2 * std_gauss**2.0)) / (
        std_gauss * (2.0 * pi) ** 0.5
    )


def cl_around_mode(edg, myprob):

    peak = edg[np.argmax(myprob)]
    idx_sort_up = np.argsort(myprob)[::-1]

    i = 0
    bins = []
    integr = 0.0
    bmax = idx_sort_up[0]
    bmin = bmax
    bmaxbound = edg.shape[0] - 1
    bminbound = 0

    while integr < 0.68:
        if bmax == bmaxbound:
            bmin = bmin - 1
        elif bmin == bminbound:
            bmax = bmax + 1
        elif myprob[bmax + 1] > myprob[bmin - 1]:
            # print("Adding ",bmax_lo+1)
            bmax = bmax + 1
            bmin = bmin
            # bins_now_good = np.append(bins_now_good,
            bins.append(bmax + 1)
        else:
            # print("Adding ",bmin-1)
            bmin = bmin - 1
            bmax = bmax
            bins.append(bmin - 1)
        integr = simps(myprob[bmin:bmax], edg[bmin:bmax])
    print(integr, edg[bmin], edg[bmax])

    return peak, edg[bmin], edg[bmax]


###############################################################################

##############################
###       Constants        ###
##############################

f = 1.0  # f=1 in the limit where we detect all AGNs flares
lamb = 0.2  # da_array=[0.2,0.9]
N_GW_followups = 50  # [50,10]
# lamba_array=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.]
# N_GW_followups=[500,200,100,100,100,100,100,50,50,50,50]
events_file = "pareto_id_Bs_top3percent.dat"
event_name = "O4_top3"
H0 = 70.0
omegam = 0.3
mapdir = "/data/des70.b/data/palmese/bayestar-sims/HLVK_O4/bbh/mass_gt_50_flatten_fits/"
steps = 5000
z_min_b = 0.01  # Minimum z for background events
z_max_b = 1.0  # Maximum z for background events
mpi = False  # Compute in parallel or not
lam_arr = np.linspace(0, 1.0, num=100)
events_data = np.genfromtxt(mapdir + events_file)

cosmo = FlatLambdaCDM(H0=H0, Om0=omegam)
out_name = event_name

##############################
###         Science        ###
##############################

# Randomly assign 0 or 1 signal events to each followup (frac0:frac1::lamb:(1-lamb))
randnum = np.random.uniform(0, 1, size=N_GW_followups)
S_cands = np.ones(N_GW_followups, dtype=int)
mask = randnum > lamb
S_cands[mask] = 0

# Some kind of pareto calculation
sim_pareto = np.genfromtxt(mapdir + events_file)
simid = sim_pareto[:, 0]
Bn = sim_pareto[:, 1]

maxdist = cosmo.luminosity_distance(z_max_b)
ds_arr_norm = np.linspace(0, maxdist.value, num=1000)

# Iterate over followups
cand_hpixs = []
cand_zs = []
pbs, distnorms, distmus, distsigmas = [], [], [], []
for i in range(N_GW_followups):

    ##############################
    ###  Signals (BBH flares)  ###
    ##############################

    # Load skymap
    hs = fits.open(mapdir + str(int(simid[i])) + ".fits.fits.gz")[1]

    # Get data from skymap
    pb = hs.data["PROB"]
    distnorm = hs.data["DISTNORM"]
    distmu = hs.data["DISTMU"]
    distsigma = hs.data["DISTSIGMA"]
    NSIDE = hs.header["NSIDE"]

    # Calculate <pb_frac> credible region; outputs healpix indices that compose region
    pb_frac = 0.90
    # Make idx_sort_up (array of pb indices, from highest to lowest pb)
    idx_sort = np.argsort(pb)
    idx_sort_up = list(reversed(idx_sort))
    # Add pbs until pb_frac is reached
    sum = 0.0
    id = 0
    while sum < pb_frac:
        this_idx = idx_sort_up[id]
        sum = sum + pb[this_idx]
        id = id + 1
    # Cut indices to <pb_frac> credible region
    idx_sort_cut = idx_sort_up[:id]

    # Draw a random HEALpix out of the <pb_frac> region for each of the #=S_cands[i] signals in a followup,
    #   weighting by their probabilities
    s_hpixs = np.random.choice(
        idx_sort_cut, p=pb[idx_sort_cut] / pb[idx_sort_cut].sum(), size=S_cands[i]
    )
    # Given the drawn HEALPixes, draw a random redshift for each candidate
    s_zs = []
    for j in range(S_cands[i]):
        # Get the LIGO/Virgo posterior probability for each of the test distances
        ln_los = gauss(distmu[s_hpixs[j]], distsigma[s_hpixs[j]], ds_arr_norm)
        # Weight the probability by the square of the distance (assumes constant density of events)
        post_los = ln_los * ds_arr_norm**2
        # Draw a distance from the test array, weighting by the calculated probability
        s_ds = np.random.choice(ds_arr_norm, p=post_los / post_los.sum())
        # Transform to redshift and append to output array
        s_zs.append(
            z_at_value(cosmo.luminosity_distance, s_ds * u.Mpc, zmin=0.00, zmax=5.0)
        )
    s_zs = np.array(s_zs)

    ##############################
    ### Background AGN flares  ###
    ##############################

    # Draw a number of background AGN flare candidates (Poisson distribution)
    B_expected_n = Bn[i]
    B_cands = np.random.poisson(Bn[i])
    # print("Background events: ",B_cands)
    # Sample healpixes for background flares (uniform probability over 90% cred.reg.)
    b_hpixs = np.random.choice(idx_sort_cut, size=B_cands)
    # Sample redshifts for background flares
    # Let's assume they are just uniform in comoving volume between z_min and z_max
    zs_arr = np.linspace(z_min_b, z_max_b, num=1000)
    # Define a probability that is uniform in comoving volume
    # in redshift that is ~D_comoving^2/H
    pb_Vunif = cosmo.comoving_distance(zs_arr).value ** 2 / cosmo.H(zs_arr).value
    b_zs = np.random.choice(zs_arr, p=pb_Vunif / pb_Vunif.sum(), size=B_cands)
    # print(s_zs,b_zs)

    ##############################
    ###       All flares       ###
    ##############################

    # Compile positions and z for all candidates in this follow up
    cand_hpixs.append(np.concatenate((s_hpixs, b_hpixs)))
    cand_zs.append(np.concatenate((s_zs, b_zs)))
    pbs.append(pb)
    distnorms.append(distnorm)
    distmus.append(distmu)
    distsigmas.append(distsigma)
    # cand_ds=np.concatenate((s_ds,b_ds))
    # ncands=cand_hpixs.shape[0]
    # cand_ds=np.zeros(ncands)
    # for l in range(ncands):
    #    cand_ds[l]=cosmo.luminosity_distance(cand_zs[l]).value

##############################
###          MCMC          ###
##############################

# Initialize some random lambdas and H0s for MCMC (w/ gaussian perturbations)
# np.random.randn(nwalkers, ndim=nparam)
pos = [lamb, H0] + 1e-4 * np.random.randn(32, 2)
nwalkers, ndim = pos.shape
# Run the MCMC, in parallel or serial
if mpi:
    with Pool() as pool:
        # Initialize sampler + run MCMC
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            lnprob,
            args=(
                cand_hpixs,
                cand_zs,
                pbs,
                distnorm,
                distmu,
                distsigma,
                f,
                N_GW_followups,
            ),
            pool=pool,
        )
        sampler.run_mcmc(pos, steps, progress=True)
else:
    # Initialize sampler + run MCMC
    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        lnprob,
        args=(
            cand_hpixs,
            cand_zs,
            pbs,
            distnorms,
            distmus,
            distsigmas,
            f,
            N_GW_followups,
        ),
    )
    sampler.run_mcmc(pos, steps, progress=True)

# Reduce and retreive MCMC samples
flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
print(flat_samples.shape)

# Make+save corner plot
labels = [r"$\lambda$", r"$H_0$ [km/s/Mpc]"]
fig = corner.corner(
    flat_samples,
    labels=labels,
    truths=[lamb, H0],
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_kwargs={"fontsize": 12},
)
fig.savefig("corner_O4_lam0.2_50.png", dpi=200)

# Save data
np.savetxt("out/O4_samples_lam0.2_50.dat", flat_samples)
