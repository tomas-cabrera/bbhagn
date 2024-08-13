import math

import pandas as pd

from .paths import DATADIR

DF_KIMURA20 = pd.read_csv(f"{DATADIR}/kimura20_SFparams.dat", sep="\s+")
print(DF_KIMURA20)


def calc_kimura20_sf(filter, delta_time):
    # Get parameters
    params = DF_KIMURA20[DF_KIMURA20["band"] == filter]

    # Calculate structure function
    sf = (
        params["SF0"].values[0]
        * (delta_time / params["dt0"].values[0]) ** params["bt"].values[0]
    )

    return sf


def calc_kimura20_sf_prob(filter, delta_time, delta_mag):
    # Get SF
    sf = calc_kimura20_sf(filter, delta_time)

    # Calculate probability
    prob = 0.5 * math.erfc(delta_mag / sf / (2**0.5))

    return prob
