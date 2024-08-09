import glob
import os
import os.path as pa

import numpy as np
from astropy.io.votable import parse
import pandas as pd

from .paths import DATADIR


class Lightcurve:
    pass

    def apply_sigma_cut(self, field, sigma):
        # Calculate mean and standard deviation
        mean = np.mean(self.data[field])
        std = np.std(self.data[field])

        # Apply cut
        mask = np.abs(self.data[field] - mean) < sigma * std
        self.data = self.data[mask]

    def calculate_fluxes(self, maglabel="mag"):
        mflux = 10 ** (-0.4 * (self.data[maglabel] - self.data[f"{maglabel}_zp"]))
        mflux_err_hi = (1 - 10 ** (-0.4 * self.data[f"{maglabel}_err"])) * mflux
        mflux_err_lo = -(1 - 10 ** (0.4 * self.data[f"{maglabel}_err"])) * mflux

        return mflux, mflux_err_hi, mflux_err_lo


class AlerceLightcurve(Lightcurve):
    lcdir = pa.join(DATADIR, "xml_lightcurves")

    def __init__(self, objid):
        self.objid = objid
        self.data = self._get_alerce_data(objid)
        self.data.rename(
            columns={"magerr": "mag_err", "magzp": "mag_zp", "magzprms": "mag_zp_rms"},
            inplace=True,
        )
        self.data["filter"] = self.data["filtercode"].map(
            {"zg": "g", "zr": "r", "zi": "i"}
        )

    def _get_alerce_data(self, objid):
        # Find data file
        globstr = pa.join(self.lcdir, f"{objid}.xml")
        datafile = glob.glob(globstr)[0]

        # Load data
        data = parse(datafile).get_first_table().to_table().to_pandas()

        return data
