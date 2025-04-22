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

    # Convert to dn/dOmega/dz
    dn_dOmega_dz = dn_d3Mpc * cosmo.differential_comoving_volume(zs)

    # Multiply by GW190521 area in sr
    dn_dz = dn_dOmega_dz * ((936 * u.deg**2).to(u.sr))

    # Plot the number density as a function of redshift
    for ci, color in enumerate(color2Lbin.keys()):
        ax.plot(
            zs,
            np.log10(dn_dz[ci, :].value),
            color=color,
            lw=2,
        )

    # Check min and max values
    # assert dn_dOmega_dz.min().to(u.sr**-1).value == pytest.approx(
    #     0.0009601991928953038, rel=1e-7
    # )
    # assert dn_dOmega_dz.max().to(u.sr**-1).value == pytest.approx(
    #     5113430.529626767, rel=1000
    # )

    ##############################
    ###    Luminosity interp     ###
    ##############################

    # Iterate over redshifts
    dn_d3Mpc = []
    for color, Lbin in color2Lbin.items():
        # Set brightness limits
        Lbin_lo = Lbin[0].value
        Lbin = (Lbin_lo, Lbin_lo * 10**0.1) * u.erg / u.s

        # Get number density
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

    # Convert to dn/dOmega/dz
    dn_dOmega_dz = dn_d3Mpc * cosmo.differential_comoving_volume(zs)

    # Multiply by GW190521 area in sr
    dn_dz = dn_dOmega_dz * ((936 * u.deg**2).to(u.sr))

    # Plot the number density as a function of redshift
    for ci, color in enumerate(color2Lbin.keys()):
        ax.plot(
            zs,
            np.log10(dn_dz[ci, :].value),
            color=color,
            lw=2,
            ls="--",
        )

    ##############################
    ###       g<20.5 mag       ###
    ##############################

    # Get number density
    dn_dOmega_dz = qlf.dn_dOmega_dz(
        zs=zs,
        cosmo=cosmo,
        brightness_limits=(20.5, -np.inf) * u.ABmag,
        band="g",
    )

    # Multiply by GW190521 area in sr
    dn_dz = dn_dOmega_dz * ((936 * u.deg**2).to(u.sr))

    # Plot
    ax.plot(
        zs,
        np.log10(dn_dz.value),
        color="xkcd:black",
        linestyle="--",
        label="g<20.5",
    )

    # Check min and max values
    # assert dn_dOmega_dz.min().to(u.sr**-1).value == pytest.approx(
    #     10.043361526747251, rel=1e-2
    # )
    # assert dn_dOmega_dz.max().to(u.sr**-1).value == pytest.approx(
    #     190765.70313124405, rel=100
    # )

    ##############################
    ###        Milliquas       ###
    ##############################

    # Define paths
    milliquas_paths = {
        "MILLIQUAS-GW190521": "/home/tomas/academia/projects/decam_followup_O4/crossmatch/GW190521/GW190521_cr90_2D_milliquas.fits",
        "MILLIQUAS*gwarea/skyarea": "/home/tomas/academia/data/milliquas/milliquas.fits",
    }
    milliquas_ls = {
        "MILLIQUAS-GW190521": "-.",
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

        # Divide by bin width
        counts = counts / np.diff(bins)

        # Scale if needed 
        if k == "MILLIQUAS*gwarea/skyarea":
            counts = counts * (936 * u.deg**2).to(u.sr).value / (4 * np.pi)

        # Plot
        ax.plot(
            (bins[:-1] + bins[1:]) / 2,
            np.log10(counts),
            label=k,
            color="xkcd:gray",
            ls=milliquas_ls[k],
        )

    ##############################
    ###     Clean and save     ###
    ##############################

    # Set bounds, add labels
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 7.6)
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
