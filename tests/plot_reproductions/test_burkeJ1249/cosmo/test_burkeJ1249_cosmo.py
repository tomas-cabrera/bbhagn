"""Reproduce AGN components of Figure 1 from Palmese+21.
The original figure includes information from the skymap for GW190521,
so the resulting figure from this script will not be identical."""

import os
import os.path as pa

import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from numpy.polynomial.polynomial import Polynomial
from scipy.optimize import curve_fit

from myagn.distributions import QLFHopkins
from myagn.qlfhopkins import qlfhopkins

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

def powerlaw(x, c, k):
    return c * x**k

def gauss(x, m, s):
    return np.exp(-0.5 * ((x - m)/s)**2) / (s * 2**0.5)

def pp(x, c, k, m, s):
    return powerlaw(x, c, k) * gauss(x, m, s)

def test_palmese21():
    """Reproduce Figure 1 from Palmese+21."""

    # Initialize distribution
    qlf = QLFHopkins()

    # Initialize figure, redshifts
    fig, ax = plt.subplots(
        #figsize=(6*2/3, 5*2/3),
    )
    zs = np.linspace(0, 6, 500)
    zs = zs[1:]

    ##############################
    ###       i<19 mag       ###
    ##############################

    hs = np.linspace(0.20,1.20,51)
    popts = []
    cmap = matplotlib.cm.get_cmap("inferno")
    for hi,h in enumerate(hs):
        # Set cosmo
        tempcosmo = FlatLambdaCDM(H0=h*100, Om0=0.3)

        # Get number density
        dn_dOmega_dz = qlf.dn_dOmega_dz(
            zs=zs,
            cosmo=tempcosmo,
            brightness_limits=(19, -np.inf) * u.ABmag,
            band="i",
        )

        # Multiply by GW190521 area in sr
        dn_dz = dn_dOmega_dz * ((936 * u.deg**2).to(u.sr))

        # Set x/y arrays
        x = zs
        y = dn_dz.value

        # Plot
        color = cmap(hi/len(hs))
        ax.plot(
            x,
            y-hi*5e3,
            #color="xkcd:red",
            #linestyle=":",
            lw=0.5,
            #label=f"i<19,h={h}",
            #label=f"h={h:.1f}",
            color=color,
        )

        p0 = (10000, 0.5, h, h)
        try:
        # Fit pp
            popt, pcov = curve_fit(pp, x, y, p0=p0, bounds=((0, 0, -np.inf, 0), (np.inf, np.inf, np.inf, np.inf)))
            plt.plot(
                x,
                pp(x, *popt)-hi*5e3,
                lw=1,
                color=color,
            )
            popts.append(popt)
        except:
            popts.append([np.nan]*len(p0))
            continue
    
    # Add colorbar
    plt.colorbar(
        matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(
                vmin=min(hs),
                vmax=max(hs),
            ),
            cmap=cmap,
        ),
        ax=ax,
        label=r"$h$",
    )


    ##############################
    ###        Milliquas       ###
    ##############################

#    # Define paths
#    milliquas_paths = {
#        "MILLIQUAS-GW190521": "/home/tomas/academia/projects/decam_followup_O4/crossmatch/GW190521/GW190521_cr90_2D_milliquas.fits",
#        "MILLIQUAS*gwarea/skyarea": "/home/tomas/academia/data/milliquas/milliquas.fits",
#    }
#    milliquas_ls = {
#        "MILLIQUAS-GW190521": "--",
#        "MILLIQUAS*gwarea/skyarea": ":",
#    }
#
#    # Iterate over Milliquas files
#    for k, v in milliquas_paths.items():
#        # Load
#        hdul = fits.open(v)
#        data = hdul[1].data
#
#        # Mask Rmag > 5
#        mask = data["Rmag"] > 5
#        data = data[mask]
#
#        # Calculate histogram
#        counts, bins = np.histogram(
#            data["Z"],
#            bins=25,
#            range=(0,6),
#        )
#
#        # Scale if needed 
#        if k == "MILLIQUAS*(936deg²).to(sr)":
#            counts = counts * (936 * u.deg**2).to(u.sr).value
#        if k == "MILLIQUAS*gwarea/skyarea":
#            counts = counts * (936 * u.deg**2).to(u.sr).value / (4 * np.pi)
#
#        # Interpolate as in J1249 notebook
#        counts_interp = np.interp(zs, bins[:-1], counts)
#
#        # Plot
#        ax.plot(
#            zs,
#            np.log10(counts_interp),
#            label=k,
#            color="xkcd:gray",
#            ls=milliquas_ls[k],
#        )

    ##############################
    ###     Clean and save     ###
    ##############################

    # Set bounds, add labels
    ax.set_xlim(0, 6)
    #ax.set_ylim(0, 7.5)
    ax.set_xlabel("Redshift")
    ax.set_ylabel("dn / dz + Offset")

    # Save figure
    #plt.legend(loc="upper right")
    plt.tight_layout()
    figpath = __file__.replace('.py', '.png')
    plt.savefig(figpath, dpi=300)
    plt.close()
    assert True

    # Plot optimized parameters
    popts = np.array(popts)
    print(popts)
    i2c = dict(zip(np.arange(popts.shape[1]), ["c", "k", "m", "s"]))
    for i in np.arange(popts.shape[1]):
        x = hs
        y = popts[:,i]
        plt.plot(
            x,
            y,
            label=i2c[i],
        )
    plt.legend(title=r"$\frac{dn}{dz}(z)=c \cdot z^k \cdot \text{norm}(z,\text{mu}=m,\text{sigma}=s)$")
    plt.xlabel(r"$h$")
    plt.ylabel("Parameter")
    plt.yscale("symlog")
    plt.tight_layout()
    figpath = __file__.replace('.py', '_params.png')
    plt.savefig(figpath, dpi=300)
    plt.close()

    # Plot optimized parameters
    popts = np.array(popts)
    print(popts)
    i2c = dict(zip(np.arange(popts.shape[1]), ["c", "k", "m", "s"]))
    for i in np.arange(popts.shape[1]):
        fig, ax = plt.subplots(
            figsize=(4, 3),
        )
        x = hs
        y = popts[:,i]
        plt.plot(
            x,
            y,
            label=i2c[i],
        )
        #plt.legend(
        #    title=r"$\frac{dn}{dz}(z)=c \cdot z^k \cdot \text{norm}(z,\text{mu}=m,\text{sigma}=s)$",
        #)
        plt.xlabel(r"$h$")
        plt.ylabel(i2c[i])
        if i2c[i] in ["c"]:
            plt.yscale("log")
        plt.tight_layout()
        figpath = __file__.replace('.py', f'_params_{i2c[i]}.png')
        plt.savefig(figpath, dpi=300)
        plt.close()

if __name__ == "__main__":
    test_palmese21()
