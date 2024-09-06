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

from myagn.distributions import QLFHopkins
from myagn.qlfhopkins import qlfhopkins

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)


def test_palmese21():
    """Reproduce Figure 1 from Palmese+21."""

    # Initialize distribution
    qlf = QLFHopkins()

    # Initialize figure, redshifts
    fig, ax = plt.subplots(figsize=(6 * 2/3, 5 * 2/3))
    zs = np.linspace(0, 6, 500)
    zs = zs[1:]

    ##############################
    ###    Luminosity bins     ###
    ##############################

    # Define luminosity bins
    color2Lbin = {
        "b": (1e44, 1e45) * u.erg / u.s,
        "teal": (1e45, 1e46) * u.erg / u.s,
        "g": (1e46, 1e47) * u.erg / u.s,
        "orange": (1e47, 1e48) * u.erg / u.s,
        "r": (1e48, 1e49) * u.erg / u.s,
    }

    # Iterate over redshifts
    dn_d3Mpc = []
    for color, Lbin in color2Lbin.items():
        # Get number density
        dn_d3Mpc_temp = qlf.dn_d3Mpc(
            zs=zs,
            cosmo=cosmo,
            brightness_limits=Lbin,
        )

        # Append to list
        dn_d3Mpc.append(dn_d3Mpc_temp)

    # Cast to array
    dn_d3Mpc = u.Quantity(dn_d3Mpc)

    # Plot the number density as a function of redshift
    for ci, color in enumerate(color2Lbin.keys()):
        ax.plot(
            zs,
            np.log10(dn_d3Mpc[ci, :].value),
            color=color,
            lw=2,
        )

    ##############################
    ###    Luminosity interp   ###
    ##############################

    # Iterate over redshifts
    dn_d3Mpc = []
    for color, Lbin in color2Lbin.items():
        # Set brightness limits
        Lbin_lo = Lbin[0].value
        Lbin = (Lbin_lo, Lbin_lo * 10**0.1) * u.erg / u.s

        # Get number density at edge
        dn_d3Mpc_temp = qlf.dn_d3Mpc(
            zs=zs,
            cosmo=cosmo,
            brightness_limits=Lbin,
        )

        # Multiply by 10 to cover bin width
        dn_d3Mpc_temp *= 10

        # Append to list
        dn_d3Mpc.append(dn_d3Mpc_temp)

    # Cast to array
    dn_d3Mpc = u.Quantity(dn_d3Mpc)

    # Plot the number density as a function of redshift
    for ci, color in enumerate(color2Lbin.keys()):
        ax.plot(
            zs,
            np.log10(dn_d3Mpc[ci, :].value),
            color=color,
            lw=2,
            ls="--",
        )

    ##############################
    ###       i<19 mag       ###
    ##############################

    # Get number density
    dn_d3Mpc = qlf.dn_d3Mpc(
        zs=zs,
        cosmo=cosmo,
        brightness_limits=(19, -np.inf) * u.ABmag,
        band="i",
    )

    # Plot
    ax.plot(
        zs,
        np.log10(dn_d3Mpc.value),
        color="r",
        linestyle=":",
        label="i<19",
    )

    ##############################
    ###     Clean and save     ###
    ##############################

    # Set bounds, add labels
    ax.set_xlim(0, 6)
    ax.set_ylim(-10, -3.2)
    ax.set_xlabel("Redshift")
    ax.set_ylabel(r"$\log_{10} \left( \phi(L) / dz~[Mpc^{-3}] \right)$")

    # Save figure
    plt.legend(loc="upper right")
    plt.tight_layout()
    figpath = __file__.replace('.py', '.png')
    plt.savefig(figpath, dpi=300)
    plt.close()
    assert True


if __name__ == "__main__":
    test_palmese21()
