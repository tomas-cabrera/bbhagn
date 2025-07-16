import numpy as np
import matplotlib.pyplot as plt

dists = {
    "normal": {
        "loc": 0,
        "scale": 1,
    },
    "uniform": {
        "low": 0,
        "high": 1,
    },
    "exponential": {
        "scale": 1,
    },
    "lognormal": {
        "mean": 0,
        "sigma": 1,
    },
    "gamma": {
        "shape": 1,
        "scale": 1,
    },
    # "poisson": {
    #     "lam": 1,
    # },
    "beta": {
        "a": 1,
        "b": 1,
    },
    "chisquare": {
        "df": 1,
    },
    "weibull": {
        "a": 1,
    },
    "rayleigh": {
        "scale": 1,
    },
    "pareto": {
        "a": 1,
    },
    "laplace": {
        "loc": 0,
        "scale": 1,
    },
    # "geometric": {
    #     "p": 0.5,
    # },
    # "negative_binomial": {
    #     "n": 1,
    #     "p": 0.5,
    # },
}

bins = np.linspace(-5, 5, 25)
rng = np.random.default_rng(123456)
for d, p in dists.items():
    print(f"Generating {d} distribution with parameters {p}")
    print(f"\t{getattr(rng, d)(**p, size=10)}")
    plt.hist(
        getattr(rng, d)(**p, size=1000),
        bins=bins,
        label=d,
        histtype="step",
    )
plt.legend()
plt.savefig(__file__.replace(".py", ".pdf"))
plt.close()
