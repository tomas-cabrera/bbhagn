"""Module handling AGN distributions"""

import os
import shutil
import os.path as pa
import astropy.constants as const
import astropy.cosmology.units as cu
import astropy.units as u
from astropy.table import Table
import numpy as np
from astropy.cosmology import FlatLambdaCDM, z_at_value
from urllib.request import urlretrieve
import subprocess

from myagn.qlfhopkins import qlfhopkins


class AGNDistribution:
    """A class to handle AGN distributions"""

    def __init__(self) -> None:
        pass

    def dn_d3Mpc_at_dL(
        self,
        dLs,
    ):
        """_summary_

        Parameters
        ----------
        dLs : _type_
            _description_
        cosmology : _type_, optional
            _description_, by default FlatLambdaCDM(H0=70, Om0=0.3)
        brightness_limits : _type_, optional
            _description_, by default None
        band : str, optional
            _description_, by default "g"
        """
        # Calculate redshifts
        zs = z_at_value(self.cosmo.luminosity_distance, dLs)

        # Get density of AGNs in Mpc^-3
        dn_d3Mpc = self.dn_d3Mpc(
            zs.value,
            cosmo=self.cosmo,
            brightness_limits=self.brightness_limits,
            band=self.band,
        )

        return dn_d3Mpc

    def _dn_d3Mpc_at_dL(
        self,
        dLs,
        cosmo=FlatLambdaCDM(H0=70, Om0=0.3),
        brightness_limits=None,
        band="g",
    ):
        """_summary_

        Parameters
        ----------
        dLs : _type_
            _description_
        cosmology : _type_, optional
            _description_, by default FlatLambdaCDM(H0=70, Om0=0.3)
        brightness_limits : _type_, optional
            _description_, by default None
        band : str, optional
            _description_, by default "g"
        """
        # Calculate redshifts
        zs = z_at_value(cosmo.luminosity_distance, dLs)

        # Get density of AGNs in Mpc^-3
        dn_d3Mpc = self.dn_d3Mpc(
            zs.value,
            cosmo=cosmo,
            brightness_limits=brightness_limits,
            band=band,
        )

        return dn_d3Mpc

    def dn_dOmega_dz(
        self,
        zs,
        cosmo=FlatLambdaCDM(H0=70, Om0=0.3),
        brightness_limits=None,
        band="g",
        mask=None,
    ):
        """Calculate number density of visible AGNs at given redshifts.

        Parameters
        ----------
        zs : _type_, optional
            _description_, by default np.linspace(0, 1, 50)
        cosmo : _type_, optional
            _description_, by default FlatLambdaCDM(H0=70, Om0=0.3)
        brightness_limits : _type_, optional
            Tuple-like of lower and upper brightness limits.
            If None (default), then no brightness cut is applied.
            The units of the argument are used to determine usage:
                - if magnitude or flux, then assumed to be the limits for ZTF g-band
                - if luminosity, then assumed to be bolometric luminosity

        Returns
        -------
        _type_
            _description_
        """

        # Get density of AGNs in Mpc^-3
        if mask is not None:
            dn_d3Mpc = self.dn_d3Mpc(
                zs,
                mask=mask,
            )
        else:
            dn_d3Mpc = self.dn_d3Mpc(
                zs,
                cosmo=cosmo,
                brightness_limits=brightness_limits,
                band=band,
            )

        # Convert to dn/dOmega/dz
        dn_dOmega_dz = dn_d3Mpc * cosmo.differential_comoving_volume(zs)

        return dn_dOmega_dz

    def dp_dOmega_dz(
        self,
        z_grid,
        z_evaluate=None,
        DOmega=4 * np.pi * u.sr,
        cosmo=FlatLambdaCDM(H0=70, Om0=0.3),
        brightness_limits=None,
    ):
        """Convert dn_dOmega_dz to probability density.
        This is done by integrating over redshift and solid angle.

        Parameters
        ----------
        z_grid : _type_, optional
            The redshift domain for the probability density, by default np.linspace(0, 1, 50)*cu.redshift
        z_evaluate : _type_, optional
            The redshift array to evaluate the probability density at.
            By default None; if None, then z_evaluate=z_grid (the probability density is evaluated at every point in z_grid)
        DOmega : _type_, optional
            _description_, by default 4*np.pi*u.sr
        cosmo : _type_, optional
            _description_, by default FlatLambdaCDM(H0=70, Om0=0.3)
        brightness_limits : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_
        """
        # If no z_evaluate is given, use z_grid
        if z_evaluate is None:
            z_evaluate = z_grid

        # Get number distribution over values
        dn_dOmega_dz_evaluate = self.dn_dOmega_dz(
            zs=z_evaluate,
            cosmo=cosmo,
            brightness_limits=brightness_limits,
        )

        # Get number distribution over grid
        dn_dOmega_dz_grid = self.dn_dOmega_dz(
            zs=z_grid,
            cosmo=cosmo,
            brightness_limits=brightness_limits,
        )

        # Normalize evaluate values to grid, solid angle
        dp_dOmega_dz_evaluate = dn_dOmega_dz_evaluate / (
            np.trapz(dn_dOmega_dz_grid, z_grid) * DOmega.to(u.sr)
        )

        return dp_dOmega_dz_evaluate

    def dp_dz(
        self,
        z_grid,
        z_evaluate=None,
        cosmo=FlatLambdaCDM(H0=70, Om0=0.3),
        brightness_limits=None,
    ):
        """Convert dn_dOmega_dz to probability density.
        This is done by integrating over redshift and solid angle.

        Parameters
        ----------
        z_grid : _type_, optional
            The redshift domain for the probability density, by default np.linspace(0, 1, 50)*cu.redshift
        z_evaluate : _type_, optional
            The redshift array to evaluate the probability density at.
            By default None; if None, then z_evaluate=z_grid (the probability density is evaluated at every point in z_grid)
        DOmega : _type_, optional
            _description_, by default 4*np.pi*u.sr
        cosmo : _type_, optional
            _description_, by default FlatLambdaCDM(H0=70, Om0=0.3)
        brightness_limits : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_
        """
        # If no z_evaluate is given, use z_grid
        if z_evaluate is None:
            z_evaluate = z_grid

        # Get number distribution
        dn_dOmega_dz_evaluate = self.dn_dOmega_dz(
            zs=z_evaluate,
            cosmo=cosmo,
            brightness_limits=brightness_limits,
        )

        # Get number in grid
        dn_dOmega_dz_grid = self.dn_dOmega_dz(
            zs=z_grid,
            cosmo=cosmo,
            brightness_limits=brightness_limits,
        )

        # Normalize
        dp_dz_evaluate = dn_dOmega_dz_evaluate / dn_dOmega_dz_grid.sum()

        return dp_dz_evaluate

    def sample_z(
        self,
        n_samples,
        z_grid,
        cosmo=FlatLambdaCDM(H0=70, Om0=0.3),
        brightness_limits=None,
        rng_np=np.random.default_rng(12345),
    ):
        """Sample redshifts from the probability density function.

        Parameters
        ----------
        n_samples : _type_
            _description_
        z_grid : _type_
            _description_
        DOmega : _type_, optional
            _description_, by default 4*np.pi*u.sr
        cosmo : _type_, optional
            _description_, by default FlatLambdaCDM(H0=70, Om0=0.3)
        brightness_limits : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_
        """
        # Get probability density
        dp_dz = self.dp_dz(
            z_grid=z_grid,
            cosmo=cosmo,
            brightness_limits=brightness_limits,
        )

        # Sample redshifts
        z_samples = rng_np.choice(z_grid, size=n_samples, p=dp_dz)

        return z_samples


class ConstantPhysicalDensity(AGNDistribution):
    """AGN distribution with constant physical number density"""

    def __init__(self, n_per_Mpc3) -> None:
        """_summary_

        Parameters
        ----------
        n_per_Mpc3 : _type_
            The number density of AGNs to use
        """
        super().__init__()
        self.n_per_Mpc3 = n_per_Mpc3

    def dn_d3Mpc(
        self,
        *args,
        **kwargs,
    ):
        zs = args[0]
        return u.Quantity(self.n_per_Mpc3.to(u.Mpc**-3) * np.ones_like(zs))


class QLFHopkins(AGNDistribution):
    """AGN distribution modeled after Hopkins+06

    Parameters
    ----------
    AGNDistribution : _type_
        _description_
    """

    def dn_d3Mpc(
        self,
        zs,
        cosmo=FlatLambdaCDM(H0=70, Om0=0.3),
        brightness_limits=None,
        band="g",
    ):
        """Calculate number density of visible AGNs at given redshifts.

        Parameters
        ----------
        zs : _type_, optional
            _description_, by default np.linspace(0, 1, 50)
        cosmo : _type_, optional
            _description_, by default FlatLambdaCDM(H0=70, Om0=0.3)
        brightness_limits : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_
        """
        # Do initial limit calculations
        luminosity_limits = None
        flux_limits = None
        if brightness_limits is None:
            luminosity_limits = (0, np.inf) * u.erg / u.s
        # If a magnitude or flux is used
        elif brightness_limits.unit.is_equivalent(u.ABmag):
            wl = {"g": 475, "i": 806}[band]
            # Assume ZTF g-band magnitude/flux for now
            f_nu_det = (brightness_limits).to(u.erg / u.s / u.cm**2 / u.Hz)
            nu = (const.c / (wl * u.nm)).to(u.Hz)
            flux_limits = f_nu_det * nu
        # If a luminosity is used
        elif brightness_limits.unit.is_equivalent(u.erg / u.s):
            luminosity_limits = brightness_limits
        else:
            raise NotImplementedError(
                "brightness_limits must have units of magnitude, flux, or luminosity"
            )

        # Iterate over redshifts
        dn_d3Mpc = []
        for z in zs:
            # Calculate luminosity limits if needed;
            # calculation is triggered if flux_limits is not None
            if flux_limits is not None:
                # Calculate limiting luminosity, assuming bolometric luminosity is 10x g-band
                luminosity_limits = (
                    10 * flux_limits * 4 * np.pi * cosmo.luminosity_distance(z) ** 2
                ).to(u.erg / u.s)

            # Calculate log luminosity
            # This section is largely here to avoid a bunch of RuntimeWarnings due to talking logs of zero
            log_luminosity_limits = [
                -np.inf if lim.value == 0 else np.log10(lim.value) for lim in luminosity_limits
            ]

            # Get qlf info
            df_qlf = qlfhopkins.compute(0, z)

            # Sum all number densities for luminosities larger than limit
            dn_dlog10L_d3Mpc = df_qlf[
                (
                    df_qlf["bolometric_luminosity"]
                    >= log_luminosity_limits[0]
                )
                & (
                    df_qlf["bolometric_luminosity"]
                    < log_luminosity_limits[1]
                )
            ]["comoving_number_density"].sum()

            # Append to list
            # (0.1 is the luminosity bin width used in the qlfhopkins calculator,
            #   so we multiply to integrate over log10L)
            dn_d3Mpc.append(dn_dlog10L_d3Mpc * 0.1)

        # Convert to numpy array, add units
        dn_d3Mpc = np.array(dn_d3Mpc) * u.Mpc**-3

        return dn_d3Mpc


class Milliquas(AGNDistribution):

    def __init__(
        self,
        cachedir=f"{pa.dirname(__file__)}/.cache",
        # catalog_url="https://quasars.org/milliquas.fits.zip",
        catalog_url="https://cdsarc.cds.unistra.fr/viz-bin/nph-Cat/fits.gz?VII/290/catalog.dat.gz", # MQv7.2, compressed version
        # catalog_url="https://cdsarc.cds.unistra.fr/viz-bin/nph-Cat/fits?VII/290/catalog.dat.gz", # MQv7.2
        z_grid=np.linspace(0, 6, 100),
    ) -> None:
        """_summary_

        Parameters
        ----------
        n_per_Mpc3 : _type_
            The number density of AGNs to use
        """
        super().__init__()

        # Make Milliquas cache
        self._cachedir = cachedir

        # Get catalog
        self.get_catalog(catalog_url)
        # self.unzip_catalog()

        # Load catalog
        self.load_catalog()

        # Set z_grid
        self.z_grid = z_grid

    def get_catalog(self, catalog_url=None):
        # Update catalog URL if needed
        if catalog_url is not None:
            self._catalog_url = catalog_url

        # Generate catalog path
        os.makedirs(self._cachedir, exist_ok=True)
        self._catalog_path = pa.join(self._cachedir, pa.basename(self._catalog_url))
        if pa.exists(self._catalog_path):
            return

        # Download catalog
        # urlretrieve(self._catalog_url, self._catalog_path)
        subprocess.run(
            [
                "wget",
                f"--output-document={self._catalog_path}",
                # f"--directory-prefix={self._cachedir}",
                f"{self._catalog_url}",
            ],
        )

    def unzip_catalog(self):
        shutil.unpack_archive(self._catalog_path, self._cachedir, format="gztar")
        # subprocess.run(
        #     [
        #         "tar",
        #         "-xzf",
        #         f"{self._catalog_url}",
        #         f"{self._catalog_path}",
        #     ],
        # )

    def load_catalog(self):
        self._catalog = Table.read(self._catalog_path, format="fits")

    def _dn_dOmega_dz(
        self,
        *args,
        **kwargs
    ):
        return self._dn_dz(*args,**kwargs) / (4 * np.pi * u.sr)

    def _dn_dz(
        self,
        zs,
        z_grid=None,
        **kwargs,
    ):
        # z_grid
        if z_grid is None:
            z_grid = self.z_grid

        # Make histogram of redshifts
        hist, _ = np.histogram(self._catalog["z"], bins=self.z_grid)

        # Get number density
        dn_dz_grid = hist / np.diff(self.z_grid)
        z_grid_temp = self.z_grid[:-1] + np.diff(self.z_grid) / 2

        # Get number density at zs
        dn_dz = np.interp(zs, z_grid_temp, dn_dz_grid)

        return dn_dz

    def dn_d3Mpc(self, zs, cosmo=FlatLambdaCDM(H0=70, Om0=0.3), **kwargs):
        dn_d3Mpc = self._dn_dOmega_dz(zs) / cosmo.differential_comoving_volume(zs)

        return dn_d3Mpc