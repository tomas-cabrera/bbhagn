### This module handles the simulation of a follow-up suite.

### Inputs:
#     - df_fitparams (flarename, filter, t_peak, f_peak, t_rise, t_decay, f_base)
#     - DF_GW (GWname --> skymap, f_cover)
#     - sim config (lambda, H0, Om0, AGN distribution, flare model)

### Outputs:
# - GW events
# - Flares

### Steps:
# Select a number of GW events; assign times if needed
# Draw background flares from AGN distribution in GW volumes
# - Define easily sampleable volume (i.e. cosmo sphere)
# - Determine average number of flares in volume; draw from Poisson
# - Sample times for flares
# - Sample locations for flares from whole volume
# - Keep flares within GW volume (if f_cover available, randomly drop flares appropriately)
# Select GW events producing AGN flares
# - Draw flare time
# - Draw flare location from GW skymap

# Do a version where only massive events produce flares as well, and see if it is distinguishable from the other version

import multiprocessing as mp

import astropy.units as u
import astropy_healpix as ah
import ligo.skymap.distance as lsm_dist
import ligo.skymap.moc as lsm_moc
import numpy as np
import pandas as pd
import parmap
from astropy.coordinates import SkyCoord
from astropy.cosmology import z_at_value
from ligo.skymap.io import read_sky_map
from ligo.skymap.postprocess.crossmatch import crossmatch


def draw_continuous_from_discrete(
    x_grid,
    x_pdf,
    rng=np.random.default_rng(12345),
    n_draws=1,
):
    """Draws continuous values from a discrete probability distribution via Von Neumann.
    Uses linear interpolation to determine probability at arbitrary points.
    """
    draws = []
    max_p = np.max(x_pdf) * 1.1
    while len(draws) < n_draws:
        a = rng.uniform(np.min(x_grid), np.max(x_grid))
        b = rng.uniform(0, max_p)
        if b <= np.interp(a, x_grid, x_pdf):
            draws.append(a)
    if n_draws == 1:
        return draws[0]
    else:
        return draws


def _crossmatch_skymap(
    gw_row,
    scs,
    cosmo,
):
    """Generates mask for flares in GW volume.
    Uses the df_gw_input and sc_flares variables defined above.
    Assumes that the f_cover for each skymap corresponds to the highest probability interval;
    in practice this is not always the case due to observational constraints.

    Parameters
    ----------
    sm_index : int
        Integer indexing into df_gw_input.

    Returns
    -------
    _type_
        Boolean mask on df_gw_input; True if flare is in GW volume.
    """
    # gw_row = df_gw_input.iloc[gw_idx]
    sm = read_sky_map(gw_row["skymap_path"], moc=True)
    cm = crossmatch(sm, scs, cosmology=True, cosmo=cosmo)
    mask = cm.searched_prob_vol <= gw_row["f_cover"]
    return mask


def _draw_gw_flare_coords(
    gw_row,
    dl_grid,
    dt_followup,
):
    """Draws flare coordinates from GW skymap.
    Uses the df_gw_input and sc_flares variables defined above.

    Parameters
    ----------
    gw_idx : int
        Integer indexing into df_gw_input.

    Returns
    -------
    _type_
        SkyCoord object for flare.
    """
    rng = np.random.default_rng(gw_row["rng_seed"])
    # Draw time
    t = gw_row["t"] + rng.uniform(0, dt_followup)
    # Draw location
    # Draw pixel
    sm = read_sky_map(gw_row["skymap_path"], moc=True)
    sm["PIXAREA"] = lsm_moc.uniq2pixarea(sm["UNIQ"])
    sm["PROB"] = sm["PROBDENSITY"] * sm["PIXAREA"]
    hpx_idx = rng.choice(
        np.arange(len(sm)),
        p=sm["PROB"] / np.sum(sm["PROB"]),
    )
    hpx_row = sm[hpx_idx]
    uniq = hpx_row["UNIQ"]
    distmu = hpx_row["DISTMU"]
    distsigma = hpx_row["DISTSIGMA"]
    distnorm = hpx_row["DISTNORM"]
    del sm
    # Draw RA, dec
    level, ipix = lsm_moc.uniq2nest(uniq)
    nside = ah.level_to_nside(level)
    ra, dec = ah.healpix_to_lonlat(
        ipix,
        nside,
        dx=rng.uniform(0, 1),
        dy=rng.uniform(0, 1),
        order="nested",
    )
    ra = ra.to(u.deg).value
    dec = dec.to(u.deg).value
    # Draw redshift
    dl_pdf = lsm_dist.conditional_pdf(
        dl_grid,
        [distmu] * len(dl_grid),
        [distsigma] * len(dl_grid),
        [distnorm] * len(dl_grid),
    )
    dl = draw_continuous_from_discrete(
        dl_grid,
        dl_pdf,
        rng=rng,
    )
    z = z_at_value(cosmo.luminosity_distance, dl)
    # Return
    return {
        "t": t,
        "ra": ra,
        "dec": dec,
        "z": z,
    }


def simulate_flares(
    lamb,
    cosmo,
    df_gw_input,  # Should include skymap_path, t, f_cover
    agn_dist,
    dt_followup,
    z_grid=np.linspace(0, 2, 1000),
    rng=np.random.default_rng(12345),
    nproc=16,
):
    # Setup
    dl_grid = cosmo.luminosity_distance(z_grid)
    dcm_grid = cosmo.comoving_distance(z_grid)

    print("Generating background flares...")
    # Background flares
    # Determine number of flares
    dn_dz = 4 * np.pi * u.sr * agn_dist.dn_dOmega_dz(z_grid)
    n_flares_bg_avg = np.trapezoid(dn_dz, z_grid)
    n_flares_bg_avg = 100
    n_flares_bg = rng.poisson(n_flares_bg_avg)
    print(f"Average number of background flares: {n_flares_bg_avg}")
    # Draw times
    t_max_bg = np.nanmax(df_gw_input["t"])
    t_flares_bg = rng.uniform(0, t_max_bg + dt_followup, n_flares_bg) * u.day
    # Draw RAs, decs
    ra_flares_bg = rng.uniform(0, 360, n_flares_bg)
    dec_flares_bg = 180 / np.pi * np.arcsin(rng.uniform(-1, 1, n_flares_bg))
    # Draw redshifts
    z_flares_bg = draw_continuous_from_discrete(
        z_grid,
        dn_dz,
        rng=rng,
        n_draws=n_flares_bg,
    )
    dl_flares_bg = cosmo.luminosity_distance(z_flares_bg)
    sc_flares_bg = SkyCoord(
        ra_flares_bg,
        dec_flares_bg,
        unit=(u.deg, u.deg, dl_flares_bg.unit),
        distance=dl_flares_bg.value,
    )

    # Keep flares in GW volumes (using f_cover)
    print("Crossmatching flares with GW skymaps...")
    df_rows = [row for _, row in df_gw_input.iterrows()]
    cm_masks = parmap.map(
        _crossmatch_skymap,
        df_rows,
        sc_flares_bg,
        cosmo,
        pm_processes=nproc,
    )
    cm_mask = np.any(cm_masks, axis=0)
    df_flares_bg = pd.DataFrame(
        {
            "t": t_flares_bg,
            "ra": ra_flares_bg,
            "dec": dec_flares_bg,
            "z": z_flares_bg,
        }
    )[cm_mask]
    df_flares_bg["gw"] = False

    print("Generating GW flares...")
    # GW flares
    # Determine number of flares
    n_flares_gw_avg = lamb * df_gw_input.shape[0]
    print(f"Average number of GW flares: {n_flares_gw_avg}")
    n_flares_gw = rng.poisson(n_flares_gw_avg)
    # Select GW events
    gw_em_idxs = rng.choice(
        np.arange(df_gw_input.shape[0]),
        n_flares_gw,
        replace=False,
    )

    print("Drawing GW flare coordinates...")
    # Draw coordinates
    df_temp = df_gw_input.copy()
    df_temp["rng_seed"] = rng.integers(0, 2**32 - 1, df_temp.shape[0])
    gw_rows = [df_temp.iloc[i] for i in gw_em_idxs]
    gw_flare_coords = parmap.map(
        _draw_gw_flare_coords,
        gw_rows,
        dl_grid,
        dt_followup,
        pm_processes=nproc,
    )
    df_flares_gw = pd.DataFrame(gw_flare_coords)
    df_flares_gw["gw"] = True

    # Return
    df_flares = pd.concat([df_flares_bg, df_flares_gw], ignore_index=True)
    return df_flares


if __name__ == "__main__":
    from astropy.cosmology import FlatLambdaCDM
    from myagn import distributions as myagndistributions

    # Setup
    lamb = 0.3
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    df_gw_input = pd.read_csv("data/df_gw_input.csv")
    agn_dist = getattr(
        myagndistributions,
        # agndist_config["model"],
        "ConstantPhysicalDensity",
    )(
        # *agndist_config["args"],
        # **agndist_config["kwargs"],
        10**-4.75,
        {},
    )
    dt_followup = 200

    # Simulate
    df_flares = simulate_flares(
        lamb,
        cosmo,
        df_gw_input,
        agn_dist,
        dt_followup,
    )
    df_flares.to_csv("data/df_flares.csv", index=False)
