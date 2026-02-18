import os.path as pa
import sys

import ligo.skymap.moc as lsm_moc
from astropy.coordinates import SkyCoord
import ligo.skymap.plot
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from ligo.skymap import postprocess
from matplotlib.colors import LinearSegmentedColormap

sys.path.append(pa.dirname(pa.dirname(pa.dirname(__file__))))
import utils.io as io
from utils.paths import PROJDIR

# Style file
plt.style.use(f"{PROJDIR}/plots/matplotlibrc.mplstyle")

################################################################################


def parse_skymap_args(skymap_filename=None, lvk_eventname=None):
    if skymap_filename is None and lvk_eventname is None:
        raise ValueError("Either skymap_filename or lvk_eventname must be provided.")
    if skymap_filename is not None and lvk_eventname is not None:
        raise ValueError(
            "Only one of skymap_filename or lvk_eventname should be provided."
        )
    if skymap_filename is not None:
        return Table.read(skymap_filename)
    if lvk_eventname is not None:
        # TODO: download from GraceDB + flatten
        pass


def _get_probs_for_skymap(skymap):
    try:
        areas = lsm_moc.uniq2pixarea(skymap["UNIQ"])
    except ValueError:
        areas = 4 * np.pi / skymap.shape[0]
    probs = skymap["PROBDENSITY"] * areas
    return probs


def calc_contours_for_skymap(skymap_flat, contours):
    # Get probs
    probs = _get_probs_for_skymap(skymap_flat)

    # Find credible levels
    i = np.flipud(np.argsort(probs))
    cumsum = np.cumsum(probs[i])
    cls = np.empty_like(probs)
    cls[i] = cumsum * 100

    # Generate contours
    # Indexing scheme is paths[CI%][mode][vertex][ra,dec]
    paths = list(postprocess.contour(cls, contours, nest=True, degrees=True))

    return paths


def plot_skymap_with_contours(skymap_flat, contours, ax=None, label=None, cmap="cylon"):
    # Initialize defaults if needed
    if ax is None:
        ax = plt.axes(projection="astro hours mollweide")
        ax.grid()
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    # Plot contours
    cs = calc_contours_for_skymap(skymap_flat, contours)
    plot_kwargs = {
        "transform": ax.get_transform("world"),
        "rasterized": True,
    }
    for c, cv in zip(cs, contours):
        color = cmap(cv / 100)
        plot_kwargs["color"] = color
        # plot_kwargs["lw"] = 2 * (1 - cv / 100) + 1
        for mi, m in enumerate(c):
            # Get coordinates
            ra = np.array([v[0] for v in m])
            dec = np.array([v[1] for v in m])
            # Find breaks in the contour (where ra wraps around)
            dra = np.abs(np.diff(ra))
            wrap_indices = np.where(dra > 180)[0] + 1
            # Plot each segment separately
            if len(wrap_indices) > 0:
                indices = np.concatenate(([0], wrap_indices, [len(ra)]))
                for i in range(len(indices) - 1):
                    start = indices[i]
                    end = indices[i + 1]
                    ax.plot(
                        ra[start:end],
                        dec[start:end],
                        **plot_kwargs,
                    )
                    # Shading
                    # ax.fill(
                    #     ra[start:end],
                    #     dec[start:end],
                    #     alpha=0.5 * (1 - cv / 100) + 0.5,
                    #     **plot_kwargs,
                    # )
            # Or just plot the whole thing
            else:
                ax.plot(
                    ra,
                    dec,
                    **plot_kwargs,
                )
                # Shading
                # ax.fill(
                #     ra,
                #     dec,
                #     alpha=0.5 * (1 - cv / 100) + 0.5,
                #     **plot_kwargs,
                # )
    # Add label if provided
    if label is not None:
        ax.plot([], [], color=cmap(1), label=label)
    return ax


################################################################################

# Groups
gwlists = [
    ["GW190403_051519", "GW190514_065416", "GW190521"],
    ["GW190620_030421", "GW190803_022701"],
    ["GW190426_190642"],
    ["GW190708_232457*"],
    ["GW190719_215514"],
]
flarelists = [
    ["J124942.30+344928.9", "J143157.51+451544.0", "J224333.95+760619.2"],
    ["J120437.98+500024.0", "J160822.16+401217.8"],
    ["J154806.31+291216.3"],
    ["J233746.08-013116.3"],
    ["J181719.95+541910.0"],
]
# Create colormaps from matplotlib color cycle
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = [color for color in prop_cycle.by_key()["color"]]
cmaps = []
for color in colors:
    # Create a colormap that goes from the color to white
    tempcmap = LinearSegmentedColormap.from_list(
        f"custom_{color}",
        [color, (1, 1, 1)],
    )
    # Make a colormap as a subset of that colormap
    cmaps.append(
        LinearSegmentedColormap.from_list(
            f"custom_{color}",
            [color, tempcmap(0.5)],
        )
    )

# Initialize figure
mosaic = """0000
0000
1122
3344
"""
fig, axd = plt.subplot_mosaic(
    mosaic,
    figsize=(10, 10),
    subplot_kw={"projection": "astro hours mollweide"},
)

for axi, ax in axd.items():
    # Plot GW skymaps
    for gwi, gw in enumerate(gwlists[int(axi)]):
        # Load skymap
        skymap_path = io.get_gwtc_skymap_path(
            "/hildafs/projects/phy220048p/share/skymaps", gw
        )
        sm = parse_skymap_args(skymap_filename=skymap_path)
        sm_flat = lsm_moc.rasterize(sm)

        # Plot skymap with contours
        plot_skymap_with_contours(
            sm_flat,
            [50, 90],
            ax=ax,
            cmap=cmaps[int(gwi)],
            label=gw.strip("*"),
        )
    # Plot flares
    for fi, f in enumerate(flarelists[int(axi)]):
        sc = SkyCoord.from_name(f)
        ax.plot(
            sc.ra,
            sc.dec,
            transform=ax.get_transform("world"),
            marker="*",
            markersize=12,
            markerfacecolor="goldenrod",
            markeredgecolor="black",
            rasterized=True,
        )
    # Finalize axis
    ax.legend(loc="lower right")
    ax.grid()

plt.tight_layout()
plt.savefig(__file__.replace(".py", ".png"), dpi=300)
plt.savefig(__file__.replace(".py", ".pdf"), dpi=300)
plt.close()
