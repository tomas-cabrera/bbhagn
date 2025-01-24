import numpy as np
from scipy.integrate import simpson
from scipy.stats import norm

###############################################################################


def cl_around_mode(edg, myprob):
    """_summary_

    Parameters
    ----------
    edg : _type_
        x-values that the function is evaluated at
    myprob : _type_
        values of the function at the x-values

    Returns
    -------
    _type_
        _description_
    """

    peak = edg[np.argmax(myprob)]
    idx_sort_up = np.argsort(myprob)[::-1]

    i = 0
    bins = []
    integr = 0.0
    bmax = idx_sort_up[0]
    bmin = bmax
    bmaxbound = edg.shape[0] - 1
    bminbound = 0

    while integr < 0.68:
        if bmax == bmaxbound:
            bmin = bmin - 1
        elif bmin == bminbound:
            bmax = bmax + 1
        elif myprob[bmax + 1] > myprob[bmin - 1]:
            # print("Adding ",bmax_lo+1)
            bmax = bmax + 1
            bmin = bmin
            # bins_now_good = np.append(bins_now_good,
            bins.append(bmax + 1)
        else:
            # print("Adding ",bmin-1)
            bmin = bmin - 1
            bmax = bmax
            bins.append(bmin - 1)
        integr = simpson(y=myprob[bmin:bmax], x=edg[bmin:bmax])
    # print("cl_around_mode cl, min, max:", integr, edg[bmin], edg[bmax])

    return peak, edg[bmin], edg[bmax]


def calc_zero_cl(edg, myprob):
    """_summary_

    Parameters
    ----------
    edg : _type_
        x-values that the function is evaluated at
    myprob : _type_
        values of the function at the x-values

    Returns
    -------
    _type_
        _description_
    """

    peak = edg[np.argmax(myprob)]
    idx_sort_up = np.argsort(myprob)[::-1]

    i = 0
    bins = []
    integr = 0.0
    bmax = idx_sort_up[0]
    bmin = bmax
    bmaxbound = edg.shape[0] - 1
    bminbound = 0

    while bmin != bminbound:
        if bmax == bmaxbound:
            bmin = bmin - 1
        elif bmin == bminbound:
            bmax = bmax + 1
        elif myprob[bmax + 1] > myprob[bmin - 1]:
            # print("Adding ",bmax_lo+1)
            bmax = bmax + 1
            bmin = bmin
            # bins_now_good = np.append(bins_now_good,
            bins.append(bmax + 1)
        else:
            # print("Adding ",bmin-1)
            bmin = bmin - 1
            bmax = bmax
            bins.append(bmin - 1)
    integr = simpson(y=myprob[bmin:bmax], x=edg[bmin:bmax])
    cl_sigma = norm.ppf(0.5 + integr / 2.0)
    print(
        "calc_zero_cl cl, cl_sigma, peak, min, max:",
        integr,
        cl_sigma,
        peak,
        edg[bmin],
        edg[bmax],
    )

    return peak, edg[bmin], edg[bmax]
