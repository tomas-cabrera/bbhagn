import os.path as pa
import sys

import numpy as np
import pandas as pd

df = pd.read_csv(
    pa.join(
        pa.dirname(pa.dirname(pa.dirname(__file__))),
        "data",
        "graham23_tables",
        "flare20.csv",
    )
).sort_values("flarename")

# Header
tabstr = (
    r"""
\begin{deluxetable}{ccccc}
    \label{tab:flares}
    \tablecaption{
        Parameters for the """
    + str(len(df))
    + r""" AGN flares used in this study.
        Host redshifts are reproduced from \citealt{veronesi_agn-flares_2025}.
        Onset dates ${\rm MJD}_0$ are calculated as ${\rm MJD}_{\rm peak} - 3 \sigma_{\rm rise}$, where the two parameters are taken from fitting a Gaussian rise-exponential decay model to the ZTF $g$-band lightcurve \citep{graham_light_2023}: ${\rm MJD}_{\rm peak}$ is the fit time of flare maximum, and $\sigma_{\rm rise}$ is the fit standard deviation of the Gaussian rise.
    }
    \tablehead{
        \colhead{Event ID} & \colhead{Redshift} & \colhead{${\rm MJD}_{\rm peak}$} & \colhead{$\sigma_{\rm rise}$} & \colhead{${\rm MJD}_0$}
    }
    \startdata
"""
)

# Data
for i, row in df.iterrows():
    rowstr = "\t\t"
    rowstr += " & ".join(
        [
            row["flarename"],
            f"{row['Redshift']:.3f}",
            f"{row['mjd_peak_fit']:.1f}",
            f"{row['tg_rise_fit']:.1f}",
            f"{row['mjd_peak_fit'] - 3 * row['tg_rise_fit']:.1f}",
        ]
    )
    rowstr += r" \\"
    tabstr += rowstr + "\n"
"""
    J124942.30+344928.9 & 0.438 & 2019-06-08 \\
    J181719.94+541910.0 & 0.234 & 2019-08-08 \\
    J224333.95+760619.2 & 0.353 & 2019-07-15 \\
    J120437.98+500024.0 & 0.389 & 2019-12-15 \\
"""

# Footer
tabstr += r"""\enddata
\end{deluxetable}
"""

# Save
with open(__file__.replace(".py", ".tex"), "w") as f:
    f.write(tabstr)
