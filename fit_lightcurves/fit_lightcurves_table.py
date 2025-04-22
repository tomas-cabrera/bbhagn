import os.path as pa

import numpy as np
import pandas as pd
from myagn.flares import models as myflaremodels

import utils.graham23_tables as g23

################################################################################

# Load fitparams
df = pd.read_csv(f"{pa.dirname(__file__)}/fitparams.csv")
df.sort_values(["flarename", "filter"], inplace=True)

# Load flare params
df_flares = g23.DF_ASSOC

# Load flare model
flaremodel = getattr(
    myflaremodels,
    "Kimura20",
)(
    *(),
    **{},
)

##############################
###      Generate tex      ###
##############################

# Iterate over rows
tablestr = f"""\\begin{{deluxetable*}}{{cccccccc}}
    \\label{{tab:fitparams}}
    \\tablecaption{{
        Fit parameters for the flares, according to the model described by Eq. \\ref{{fig:gred}}.
        The SF flare rates as calculated with Eq. \\ref{{eq:pflareSF}} are also included.
    }}
    \\tablehead{{
        \\colhead{{Flare}} & \\colhead{{Filter}} & \\colhead{{$m_0$}} & \\colhead{{$\\Delta m_{{\\rm flare}}$}} & \\colhead{{$t_0$}} & \\colhead{{$t_g$}} & \\colhead{{$t_e$}} & \\colhead{{$p_{{\\rm flare,SF}}$}} \\\\
        &  &  &  & [MJD] & [days] & [days] & $ \\left[ \\frac{{\\rm flares}}{{\\rm AGN \\cdot day}} \\right]$
    }}
    \\startdata
"""

# Add candidates
tablecols = {
    "m_0": "f_base",
    "Delta_m_flare": "f_peak",
    "t_0": "t_peak",
    "t_g": "t_rise",
    "t_e": "t_decay",
}
fils = ["g", "r"]
for fn in np.sort(df["flarename"].unique()):
    # Get flare redshift
    z_flare = df_flares[df_flares["flarename"] == fn]["Redshift"].values[0]
    for fil in fils:
        # Get row
        row = df[(df["flarename"] == fn) & (df["filter"] == fil)]
        # Add flare name if first row
        tempstr = "    "
        if fil == fils[0]:
            tempstr += f"{fn}"
        else:
            tempstr += " " * len(fn)
        # Add data
        tempstr += f" & {fil}"
        tempstr += f" & {row['f_base'].values[0]:.2f}"
        tempstr += f" & {row['f_peak'].values[0]:.2f}"
        tempstr += f" & {row['t_peak'].values[0]:.0f}"
        tempstr += f" & {np.abs(row['t_rise'].values[0]):4.1f}"
        tempstr += f" & {row['t_decay'].values[0]:5.1f}"
        # Calculate flare SF
        p_flare = flaremodel.flare_rate(
            row["filter"].values[0],
            3 * np.abs(row["t_rise"].values[0]) / (1 + z_flare),
            -row["f_peak"].values[0],
            z_flare,
        )
        tempstr += f" & {p_flare:.2e}"
        # Add newline
        tempstr += r" \\" + "\n"
        # Add to tablestr
        tablestr += tempstr

tablestr += f"""\enddata
\end{{deluxetable*}}"""

# Write to file
texpath = __file__.replace(".py", ".tex")
with open(texpath, "w") as f:
    f.write(tablestr)
