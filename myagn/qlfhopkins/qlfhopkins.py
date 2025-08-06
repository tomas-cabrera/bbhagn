import os
import os.path as pa
import subprocess

import pandas as pd
import requests

###############################################################################

QLF_CALCULATOR_URL = "https://www.tapir.caltech.edu/~phopkins/Site/qlf_calculator.c"
QLF_CALCULATOR_CPATH = f"{pa.dirname(__file__)}/qlf_calculator.c"
QLF_CALCULATOR_PATH = QLF_CALCULATOR_CPATH.replace(".c", "")
assert QLF_CALCULATOR_CPATH != QLF_CALCULATOR_PATH

###############################################################################


def fetch_qlf_calculator(url=QLF_CALCULATOR_URL, path=QLF_CALCULATOR_CPATH):
    """Downloads Hopkin's qlf calculator script.

    Parameters
    ----------
    url : _type_, optional
        _description_, by default QLF_CALCULATOR_URL
    path : _type_, optional
        _description_, by default QLF_CALCULATOR_CPATH

    Returns
    -------
    _type_
        _description_
    """
    # Get the script
    try:
        r = requests.get(url, timeout=60)
    except requests.exceptions.SSLError:
        print(
            "NOTE FROM qlfhopkins.fetch_qlf_calculator: "
            + "Sometimes the requests package doesn't like the security of the qlf_calculator.c webpage.  "
            + f"Download manually from [{url}] to [{path}] and you should be good."
        )
        raise
    # Write to file
    with open(path, "wb") as f:
        f.write(r.content)
    # Return the path
    return path


def compile_qlf_calculator(
    compiler="gcc", path=QLF_CALCULATOR_CPATH, outpath=QLF_CALCULATOR_PATH
):
    """Compiles the qlf calculator

    Parameters
    ----------
    compiler : str, optional
        _description_, by default "gcc"
    path : _type_, optional
        _description_, by default QLF_CALCULATOR_CPATH
    outpath : _type_, optional
        _description_, by default QLF_CALCULATOR_PATH

    Returns
    -------
    _type_
        _description_
    """
    # Compile the script
    subprocess.run([compiler, path, "-o", outpath, "-lm"], check=True)
    # Return the new path
    return outpath


def setup_qlf_calculator(
    url=QLF_CALCULATOR_URL,
    path=QLF_CALCULATOR_CPATH,
    compiler="gcc",
    outpath=QLF_CALCULATOR_PATH,
):
    """Sets up Hopkin's qlf calculator for local use.

    Parameters
    ----------
    url : _type_, optional
        _description_, by default QLF_CALCULATOR_URL
    path : _type_, optional
        _description_, by default QLF_CALCULATOR_CPATH
    compiler : str, optional
        _description_, by default "gcc"
    outpath : _type_, optional
        _description_, by default QLF_CALCULATOR_PATH

    Returns
    -------
    _type_
        _description_
    """
    do_compile = False
    # Fetch qlf calculator script
    if not pa.exists(path):
        path = fetch_qlf_calculator(url=url, path=path)
        do_compile = True
    # Compile qlf calculator script
    if not pa.exists(outpath) or do_compile:
        outpath = compile_qlf_calculator(compiler=compiler, path=path, outpath=outpath)
    return outpath


def _compute(
    nu,
    redshift,
    fit_key=0,
    qlf_command=QLF_CALCULATOR_PATH,
    cachedir=f"{pa.dirname(__file__)}/.cache",
    force=False,
):
    """
    TC: Adapted from https://github.com/burke86/J1249/blob/main/J1249.ipynb

    Computes the qusar luminosity function at given frequencies and redshifts

    This python function is a wrapper for Phil Hopkins' C program:
    http://www.tapir.caltech.edu/~phopkins/Site/qlf.html

    Input:
        1 : nu -- the *rest frame* frequency of interest. for a specific band computed in
            HRH06, enter :
                0.0 = bolometric, -1.0 = B-band, -2.0 = mid-IR (15 microns)
               -3.0 = soft X-ray (0.5-2 keV), -4.0 = hard X-ray (2-10 keV)
            otherwise, give nu in Hz, which is taken to be the effective
            frequency of the band (for which attenuation, etc. is calculated)

        2 : redshift -- the redshift of the luminosity function to be returned

        3

    Returns:
        1 : observed luminosity in the band, in (log_{10}(L [erg/s]))
                (luminosities output are nu*L_nu, unless the bolometric,
                 soft X-ray, or hard X-ray flags are set, in which case the luminosities
                 are integrated over the appropriate frequency range)

        2 : corresponding absolute monochromatic AB magnitude at the given frequency nu.
                (for bolometric, soft and hard X-ray bands which are integrated
                 over some frequency range, this adopts effective frequencies of
                 2500 Angstrom, 1 keV and 5 keV, respectively -- these are totally
                 arbitrary, of course, but the AB magnitude is not well defined
                 in any case for these examples).

        3 : corresponding observed monochromatic flux in (log_{10}(S_nu [milliJanskys]))
                (i.e. log_{10}( S_nu [10^-26 erg/s/cm^2/Hz] ))
                (This calculates the flux with the luminosity distance appropriate for the
                 adopted cosmology in HRH06: Omega_Matter=0.3, Omega_Lambda=0.7, h=0.7.
                 Attentuation from observed column density distributions is included,
                 as described in the paper, but intergalactic attentuation appropriate
                 near Lyman-alpha, for example, is NOT included in this calculation.
                 Be careful, as this also does NOT include the K-correction, defined as :
                      m = M + (distance modulus) + K
                      K = -2.5 * log[ (1+z) * L_{(1+z)*nu_obs} / L_{nu_obs} ]
                 For the bolometric, soft and hard X-rays, this returns the
                 integrated flux over the
                 appropriate frequency ranges, in CGS units (erg/s/cm^2).)

        4 : corresponding bolometric luminosity (given the median bolometric
                corrections as a function of luminosity), in (log_{10}(L [erg/s]))

        5 : comoving number density per unit log_{10}(luminosity) :
                dphi/dlog_{10}(L)  [ Mpc^{-3} log_{10}(L)^{-1} ]
                (make sure to correct by the appropriate factor to convert to e.g.
                 the number density per unit magnitude)
    """

    # Compose the path to the output file
    qlf_fileout = f"{cachedir}/nu-{nu}_z-{redshift}_fit-key-{fit_key}.txt"

    # Try to read the file if it exists
    if pa.exists(qlf_fileout) and not force:
        try:
            qlf_out = pd.read_csv(qlf_fileout, sep="\s+")
            return qlf_out
        except pd.errors.EmptyDataError:
            pass

    # If things have progressed past the above section, run the calculator
    # Make the outdir path
    if not pa.exists(cachedir):
        os.makedirs(cachedir, exist_ok=True)

    # Generate the file
    with open(qlf_fileout, "w") as f:
        # Header
        f.write(
            "observed_luminosity ABmag S_nu bolometric_luminosity comoving_number_density\n"
        )
    with open(qlf_fileout, "a") as f:
        # Run the calculator, write output to file
        subprocess.run(
            [qlf_command, str(nu), str(redshift), str(fit_key)],
            stdout=f,
            check=True,
        )

    # Read the file and return
    qlf_out = pd.read_csv(qlf_fileout, sep="\s+")
    return qlf_out


def compute(
    nu,
    redshift,
    fit_key=0,
    qlf_command=QLF_CALCULATOR_PATH,
    cachedir=f"{pa.dirname(__file__)}/.cache",
    force=False,
):
    """
    TC: A wrapper for _compute that interpolates the values from a grid resolved to 0.01 in redshift.
    """
    # Get the nearest redshifts
    redshift_lo = int(redshift * 100) * 0.01
    redshift_hi = redshift_lo + 0.01

    # Get the values at the nearest redshifts
    qlf_lo = _compute(
        nu,
        redshift_lo,
        fit_key=fit_key,
        qlf_command=qlf_command,
        cachedir=cachedir,
        force=force,
    )
    qlf_hi = _compute(
        nu,
        redshift_hi,
        fit_key=fit_key,
        qlf_command=qlf_command,
        cachedir=cachedir,
        force=force,
    )

    # Interpolate the values
    qlf = qlf_lo + (qlf_hi - qlf_lo) * (redshift - redshift_lo) / 0.01

    return qlf


###############################################################################

# Setup calculator on import
# setup_qlf_calculator()
