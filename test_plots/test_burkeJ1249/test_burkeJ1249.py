"""Reproduce AGN components of Figure 1 from Palmese+21.
The original figure includes information from the skymap for GW190521,
so the resulting figure from this script will not be identical."""

import os
import os.path as pa

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from numpy.polynomial.polynomial import Polynomial

from myagn.distributions import QLFHopkins
from myagn.qlfhopkins import qlfhopkins

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)


def test_palmese21():
    """Reproduce Figure 1 from Palmese+21."""

    # Initialize distribution
    qlf = QLFHopkins()

    # Initialize figure, redshifts
    fig, ax = plt.subplots(figsize=(6*2/3, 5*2/3))
    zs = np.linspace(0, 6, 500)
    zs = zs[1:]

    ##############################
    ###       10**-4.75 * dL^2       ###
    ##############################

    # Get number density
    dn_dOmega_dz = 10**-4.75 * cosmo.differential_comoving_volume(zs)
    
    # Multiply by GW190521 area in sr
    dn_dz = dn_dOmega_dz * ((936 * u.deg**2).to(u.sr))

    # Plot
    ax.plot(
        zs,
        np.log10(dn_dz.value),
        color="xkcd:blue",
        linestyle="-.",
        label=r"$10^{-4.75}*d_L^2$",
    )

    ##############################
    ###       i<19 mag       ###
    ##############################

    # Get number density
    dn_dOmega_dz = qlf.dn_dOmega_dz(
        zs=zs,
        cosmo=cosmo,
        brightness_limits=(19, -np.inf) * u.ABmag,
        band="i",
    )

    # Multiply by GW190521 area in sr
    dn_dz = dn_dOmega_dz * ((936 * u.deg**2).to(u.sr))

    # Plot
    ax.plot(
        zs,
        np.log10(dn_dz.value),
        color="xkcd:red",
        linestyle=":",
        label="i<19",
    )

    ##############################
    ###        Milliquas       ###
    ##############################

    # Define paths
    milliquas_paths = {
        "MILLIQUAS-GW190521": "/home/tomas/academia/projects/decam_followup_O4/crossmatch/GW190521/GW190521_cr90_2D_milliquas.fits",
        "MILLIQUAS*gwarea/skyarea": "/home/tomas/academia/data/milliquas/milliquas.fits",
    }
    milliquas_ls = {
        "MILLIQUAS-GW190521": "--",
        "MILLIQUAS*gwarea/skyarea": ":",
    }

    # Iterate over Milliquas files
    for k, v in milliquas_paths.items():
        # Load
        hdul = fits.open(v)
        data = hdul[1].data

        # Mask Rmag > 5
        mask = data["Rmag"] > 5
        data = data[mask]

        # Calculate histogram
        counts, bins = np.histogram(
            data["Z"],
            bins=25,
            range=(0,6),
        )

        # Scale if needed 
        if k == "MILLIQUAS*(936deg²).to(sr)":
            counts = counts * (936 * u.deg**2).to(u.sr).value
        if k == "MILLIQUAS*gwarea/skyarea":
            counts = counts * (936 * u.deg**2).to(u.sr).value / (4 * np.pi)

        # Interpolate as in J1249 notebook
        counts_interp = np.interp(zs, bins[:-1], counts)

        # Plot
        ax.plot(
            zs,
            np.log10(counts_interp),
            label=k,
            color="xkcd:gray",
            ls=milliquas_ls[k],
        )

    ##############################
    ###     Clean and save     ###
    ##############################

    # Set bounds, add labels
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 7.5)
    ax.set_xlabel("Redshift")
    ax.set_ylabel(r"$\log_{10} \left( dn / dz \right)$")

    # Save figure
    plt.legend(loc="upper right")
    plt.tight_layout()
    figpath = __file__.replace('.py', '.png')
    plt.savefig(figpath, dpi=300)
    plt.close()
    assert True


if __name__ == "__main__":
    test_palmese21()
