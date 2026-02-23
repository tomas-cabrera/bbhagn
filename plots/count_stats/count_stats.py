import os.path as pa

import numpy as np
import pandas as pd

# List of directories to analyze
dirs = [
    11,
    18,
    12,
    13,
    14,
    15,
    16,
    17,
    "11_planck",
    "11_shoes",
]
dir_labels = dict(
    zip(
        dirs,
        [
            "GWTC-3.0 (76 BBHs, 8 flares)",
            "BBHs with flares (8 BBHs, 8 flares)",
            r"$m_1 < 40~M_\odot$ (47 BBHs, 3 flares)",
            r"$m_1 \geq 40~M_\odot$ (29 BBHs, 6 flares)",
            r"$m_{\rm fin} < 40~M_\odot$ (24 BBHs, 1 flare)",
            r"$m_{\rm fin} \geq 40~M_\odot$ (52 BBHs, 7 flares)",
            r"$L_{\rm bol} \geq$ 3e42 erg/s",
            r"$L_{\rm bol} \geq$ 5e41 erg/s",
            "Planck cosmology",
            "SHoES cosmology",
        ],
    )
)

# Iterate over directories
for d in dirs:
    print("*" * 50)
    print(d, dir_labels[d])
    # Compose path
    assoc_path = pa.join(
        pa.dirname(pa.dirname(pa.dirname(__file__))),
        f"Posterior_inference_lambda_O3/array/{d}/assoc.csv",
    )

    # Load data
    df = pd.read_csv(assoc_path, index_col=0)

    # Calc stats
    gws = df.index[np.any(df.values, axis=1)]
    flares = df.columns[np.any(df.values, axis=0)]
    n_assoc = np.sum(df.values)

    # Print stats
    print(f"GWs with associations: {len(gws)}/{df.shape[0]}")
    for gw in gws:
        print(f"  {gw}")
    print(f"Flares with associations: {len(flares)}/{df.shape[1]}")
    for flare in flares:
        print(f"  {flare}")
    print(f"Number of associations: {n_assoc}")
