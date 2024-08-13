import numpy as np

from .paths import DATADIR


def _graham23_flare_rise(t, t_peak, t_rise, f_peak, f_base):
    """Model for flare rise

    Parameters
    ----------
    t : _type_
        _description_
    t_peak : _type_
        _description_
    t_rise : _type_
        _description_
    f_peak : _type_
        _description_
    f_base : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    return f_base + f_peak * np.exp(-((t - t_peak) ** 2) / (2 * t_rise**2))


def _graham23_flare_decay(t, t_peak, t_decay, f_peak, f_base):
    """Model for flare decay

    Parameters
    ----------
    t : _type_
        _description_
    t_peak : _type_
        _description_
    t_decay : _type_
        _description_
    f_peak : _type_
        _description_
    f_base : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    return f_base + f_peak * np.exp(-(t - t_peak) / t_decay)


def graham23_flare_model(t, t_peak, f_peak, t_rise, t_decay, f_base):
    """Graham+23 model for AGN flares.

    Parameters
    ----------
    t : _type_
        _description_
    t_peak : _type_
        _description_
    f_peak : _type_
        _description_
    t_rise : _type_
        _description_
    t_decay : _type_
        _description_
    f_base : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    # Identify scalar t's by trying to use in if statement
    try:
        if t <= t_peak:
            f_return = _graham23_flare_rise(t, t_peak, t_rise, f_peak, f_base)
        else:
            f_return = _graham23_flare_decay(t, t_peak, t_decay, f_peak, f_base)

    # If ValueError, assume t is an array
    except ValueError:
        # Determine whether t is/are before/after peak
        mask_rise = t <= t_peak

        # Initialize flux return array
        f_return = np.zeros_like(t)

        # Calculate flux values
        f_return[mask_rise] = _graham23_flare_rise(
            t[mask_rise], t_peak, t_rise, f_peak, f_base
        )
        f_return[~mask_rise] = _graham23_flare_decay(
            t[~mask_rise], t_peak, t_decay, f_peak, f_base
        )

    return f_return
