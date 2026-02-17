import matplotlib.pyplot as plt
import sys
import os.path as pa
from matplotlib.lines import Line2D
import numpy as np
from scipy.optimize import curve_fit

sys.path.append(pa.dirname(pa.dirname(pa.dirname(__file__))))
from utils.paths import PROJDIR

plt.style.use(f"{PROJDIR}/plots/matplotlibrc.mplstyle")


def inverse_power(x, k):
    return k / x


# Data from lambda_posteriors_violin.py
nBBH = np.array([76, 7, 47, 29, 24, 52])
lambda1sigma = np.array([0.017, 0.156, 0.025, 0.040, 0.047, 0.023])
lambda90 = np.array([0.030, 0.325, 0.049, 0.077, 0.094, 0.043])
mask_fit = np.array(
    [
        True,
        False,
        True,
        True,
        True,
        True,
    ]
)

# Initialize plot
fig = plt.figure()
x = np.linspace(
    min(nBBH),
    max(nBBH),
    100,
)

# 1sigma
popt, pcov = curve_fit(
    inverse_power,
    nBBH[mask_fit],
    lambda1sigma[mask_fit],
    p0=[10],
)
line = plt.plot(
    nBBH[mask_fit],
    lambda1sigma[mask_fit],
    ls="",
    marker="o",
    label=r"$\lambda_{1 \sigma}$",
)
plt.plot(
    nBBH[~mask_fit],
    lambda1sigma[~mask_fit],
    ls="",
    marker="o",
    color=line[0].get_color(),
    fillstyle="none",
)
plt.plot(
    x,
    inverse_power(x, *popt),
    ls="--",
    color=line[0].get_color(),
)

# 90
popt, pcov = curve_fit(
    inverse_power,
    nBBH[mask_fit],
    lambda90[mask_fit],
    p0=[10],
)
line = plt.plot(
    nBBH[mask_fit],
    lambda90[mask_fit],
    ls="",
    marker="o",
    label=r"$\lambda_{90\%}$",
)
plt.plot(
    nBBH[~mask_fit],
    lambda90[~mask_fit],
    ls="",
    marker="o",
    color=line[0].get_color(),
    fillstyle="none",
)
plt.plot(
    x,
    inverse_power(x, *popt),
    ls="--",
    color=line[0].get_color(),
)

# Finish
plt.xlabel(r"$n_{\rm BBH}$ used for inference")
plt.ylabel(r"$\lambda$ upper limit")
plt.plot(
    [],
    [],
    ls="--",
    color="k",
    label=r"$k * x^{-1}$ fit",
)
plt.plot(
    [],
    [],
    ls="",
    color="k",
    marker="o",
    fillstyle="none",
    label=r"Not used in fit",
)
plt.legend()
plt.tight_layout()
plt.savefig(__file__.replace(".py", ".png"))
plt.close()
