import os.path as pa
import sys

import numpy as np
import pandas as pd
from ligo.skymap.postprocess.crossmatch import crossmatch
from ligo.skymap.io import read_sky_map
import parmap
from multiprocessing import Pool

# Local imports
sys.path.append(pa.dirname(pa.dirname(pa.dirname(__file__))))
import utils.graham23_tables as g23

# Load data
df = g23.DF_GW.sort_values("gweventname")


# Calculate areas
def calc_area90(i):
    row = df.iloc[i]
    skymap = read_sky_map(row["skymap_path"], moc=True)
    result = crossmatch(skymap, contours=(0.9,))
    return result.contour_areas[0]


area90s_path = "gw_table_area90s.npy"
if not pa.exists(area90s_path):
    with Pool(32) as pool:
        area90s = parmap.map(
            calc_area90,
            range(len(df)),
            pm_pool=pool,
            pm_pbar=True,
        )
    np.save("gw_table_area90s.npy", area90s)
area90s = np.load("gw_table_area90s.npy")
df["area90"] = area90s

# Header
tabstr = (
    r"""\startlongtable
\begin{deluxetable*}{ccccccc}
    \label{tab:gws}
    \tablecaption{
        Parameters for the """
    + str(len(df))
    + r""" gravitational wave events used in this study, reproduced from \citealt{abbott_gwtc-2_2021, abbott_gwtc-3_2023}.
        For all events, the 90\% areas were calculated with the \texttt{ligo.skymap.postprocess.crossmatch.crossmatch} function, using IMRPhenomXPHM waveform skymaps.
    }
    \tablehead{
        \colhead{Event ID} & \colhead{90\% Area} & \colhead{$d_L$} & \colhead{$m_1$} & \colhead{$m_2$} & \colhead{$m_{\rm fin}$} & \colhead{$\chi_{\rm eff}$} \\
        & deg$^2$ & Mpc & $M_\odot$ & $M_\odot$ & $M_\odot$ & 
    }
    \startdata
"""
)

# Data
for i, row in df.iterrows():
    rowstr = "\t\t"
    rowstr += " & ".join(
        [
            row["gweventname"].replace("_", r"\_"),
            f"{row['area90']:.0f}",
            f"${int(row['luminosity_distance'])}_{{{int(row['luminosity_distance_lower'])}}}^{{+{int(row['luminosity_distance_upper'])}}}$",
            f"${row['mass_1_source']}_{{{row['mass_1_source_lower']}}}^{{+{row['mass_1_source_upper']}}}$",
            f"${row['mass_2_source']}_{{{row['mass_2_source_lower']}}}^{{+{row['mass_2_source_upper']}}}$",
            f"${row['final_mass_source']}_{{{row['final_mass_source_lower']}}}^{{+{row['final_mass_source_upper']}}}$",
            f"${row['chi_eff']}_{{{row['chi_eff_lower']}}}^{{+{row['chi_eff_upper']}}}$",
        ]
    )
    rowstr += " \\\\ \n"
    tabstr += rowstr
"""
    GW190403\_051519 & IMRPhenomXPHM & 3900  & $8280_{-4290}^{+6720}$ & $85_{-33}^{+27.8}$ & $20_{-8.4}^{+26.3}$ & $102.2_{-24.3}^{+26.3}$ & $0.68_{-0.43}^{+0.16}$ \\
    GW190424\_180648 & NRSur7dq4 & 28000 & $2200_{-1160}^{+1580}$ & $40.5_{-7.3}^{+11.1}$ & $31.8_{-7.7}^{+7.6}$ & $68.9_{-9.7}^{+12.5}$ & $0.13_{-0.22}^{+0.22}$ \\
    GW190514\_065416 & NRSur7dq4 & 3000  & $3890_{-2070}^{+2610}$ & $40.9_{-9.3}^{+17.3}$ & $28.4_{-10.1}^{+10}$ & $66.4_{-11.5}^{+19}$ & $-0.08_{-0.35}^{+0.29}$ \\
    GW190521         & NRSur7dq4 & 1000  & $3310_{-1800}^{+2790}$ & $98.4_{-21.7}^{+33.6}$ & $57.2_{-30.1}^{+27.1}$ & $147.4_{-16}^{+40}$ & $-0.14_{-0.45}^{+0.5}$ \\
    GW190803\_022701 & SEOBNRv4PHM & 1500  & $3190_{-1470}^{+1630}$ & $37.7_{-6.7}^{+9.8}$ & $27.6_{-8.5}^{+7.6}$ & $62.1_{-7.6}^{+11.2}$ & $-0.01_{-0.28}^{+0.23}$ \\
    GW190909\_114149 & SEOBNRv4PHM & 4700  & $3770_{-2220}^{+3270}$ & $45.8_{-13.3}^{+52.7}$ & $28.3_{-12.7}^{+13.4}$ & $72.0_{-16.8}^{+54.9}$ & $-0.06_{-0.36}^{+0.37}$
"""

# Footer
tabstr += r"""\enddata
\end{deluxetable*}
"""

# Save
with open(__file__.replace(".py", ".tex"), "w") as f:
    f.write(tabstr)
