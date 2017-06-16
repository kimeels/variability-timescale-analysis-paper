from __future__ import print_function

import logging

import numpy as np
from scipy.optimize import curve_fit

import flarefits.ingest as ingest
from astropy.stats import sigma_clip
from attr import attrib, attrs
from attr.validators import instance_of
from flarefits.ingest import DataCols

logger = logging.getLogger(__name__)


@attrs
class Flare(object):
    """
    Stores information about a single detected flare.

    Holds array-indices marking flare boundaries.
    Stores results from fitting rise and decay slopes of a flare.
    """
    trigger = attrib(instance_of(int))
    rise = attrib(instance_of(int))
    peak = attrib(instance_of(int))
    fall = attrib(instance_of(int))
    rise_slope = attrib(default=None)
    rise_slope_err = attrib(default=None)
    decay_slope = attrib(default=None)
    decay_slope_err = attrib(default=None)
    background_estimate = attrib(default=None)
    rise_fit_pars = attrib(default=None)
    decay_fit_pars = attrib(default=None)


@attrs
class DatasetProperties(object):
    """
    Store estimated background-level, noise-level, etc.
    """
    median = attrib(instance_of(float))
    clipped_std_dev = attrib(instance_of(float))
    clipped_median = attrib(instance_of(float))
    percentile10 = attrib(instance_of(float))


def calculate_dataset_properties(dataset):
    fluxes = dataset[DataCols.flux]
    clipped_fluxes = get_sigma_clipped_fluxes(fluxes)
    return DatasetProperties(
        median=np.median(fluxes),
        percentile10=np.percentile(fluxes, 10.),
        clipped_median=np.ma.median(clipped_fluxes),
        clipped_std_dev=np.ma.std(clipped_fluxes),
    )


def find_and_fit_flares(dataset, background, noise_level):
    flares = find_flares(dataset,
                         background=background,
                         noise_level=noise_level)
    for flare_marker in flares:
        fit_flare(dataset, flare_marker)
    logger.info("Found {} flares".format(len(flares)))

    return flares


def get_sigma_clipped_fluxes(raw_fluxes):
    """
    Uses the astropy sigma_clip function to try and get rid of outliers.

    ( https://astropy.readthedocs.org/en/stable/api/astropy.stats.sigma_clip.html )

    Returns a masked array where all outlier values are masked.
    """

    # First we try to run sigma clip with the defaults - hopefully this will
    # iterate until it converges:
    clipped_fluxes = sigma_clip(raw_fluxes, iters=None)
    # If it fails (number of unmasked values <3),
    # then we just accept the result from a single iteration:
    if len(clipped_fluxes.compressed()) < 3:
        logger.warning("Sigma clipping did not converge, "
                       "using single iteration")
        clipped_fluxes = sigma_clip(raw_fluxes, iters=1)
    return clipped_fluxes


def find_flares(dataset, background, noise_level):
    """
    Find flares in ``dataset`` that peak more than 5*sigma above the background

    Returns a list of ``Flare`` namedtuple objects, containing the array indices
    marking a flare.

    Args:
        dataset (dict): Dictionary of numpy arrays
        background (float): Estimated background flux level
        noise_level (float): Estimated typical noise level for the dataset

    Returns:
        list: List of Flare objects

    """
    rise_threshold = background + 5 * noise_level
    fall_threshold = background + 1 * noise_level
    fluxes = dataset[DataCols.flux]
    flux_errs = dataset[DataCols.flux_err]
    assert flux_errs.shape == fluxes.shape
    logger.debug("Flux / errorbar series lengths: {}, {}".format(
        len(fluxes), len(flux_errs)))

    def find_next_trigger(start):
        for idx, flux in enumerate(fluxes[start:], start):
            if flux > rise_threshold:
                return idx
        return None

    def find_next_fall(start):
        for idx, flux in enumerate(fluxes[start:], start):
            err = flux_errs[idx]
            if (flux - err) < fall_threshold:
                return idx
        return None

    def find_rise_start(trigger):
        # step backwards until sunk
        for idx, flux in list(enumerate(fluxes[:trigger]))[::-1]:
            err = flux_errs[idx]
            if (flux - err) < fall_threshold:
                return idx
        return None

    flare_list = []
    idx = 0
    start_pos = find_next_fall(idx)

    match = 0
    while start_pos is not None:
        match = match + 1
        trigger = find_next_trigger(start_pos)
        if trigger is not None:
            fall = find_next_fall(trigger + 1)
        else:
            fall = None
        start_pos = fall

        if fall is not None:
            # Generate a mask leaving only this trigger-->fall interval, to find peak
            mask = np.zeros_like(fluxes)
            mask[:trigger] = True
            mask[fall:] = True
            masked = np.ma.MaskedArray(fluxes)
            masked.mask = mask
            peak_val = masked.max()
            peak_idx = np.where(masked == peak_val)[0]
            rise = find_rise_start(trigger)
            flare = Flare(trigger=trigger,
                          rise=rise,
                          peak=peak_idx[0],
                          fall=fall)
            flare_list.append(flare)
    return flare_list


def straight_line(x, gradient, intercept):
    return gradient * x + intercept


def fit_simple_exponential(timestamps, fluxes, flux_errors):
    """
    Fit a simple exponential of the form

        y = A*exp(x/tau)


    (Done by fitting a straight line in log-linear space.)
    """

    param, cov = curve_fit(f=straight_line,
                           xdata=timestamps,
                           ydata=np.log(fluxes),
                           sigma=flux_errors / fluxes,
                           )

    param_err = np.sqrt(np.diag(cov))

    return param, param_err


def fit_flare_section(dataset, start_idx, end_idx, background_flux):
    """
    Extract the relevant slice of data, subtract background flux, fit.

    Args:
        dataset (dict):
        start_idx (int):
        end_idx (int):
        background_flux (float):

    Returns:

    """

    sl = slice(start_idx, end_idx + 1)
    flux_section = dataset[DataCols.flux][sl].copy()
    flux_err_section = dataset[DataCols.flux_err][sl].copy()
    timestamp_section = dataset[DataCols.time][sl].copy()

    flux_section -= background_flux
    # Trim out any datapoints below 0, since we're about to take the log:
    good_data_idx = flux_section > 0
    flux_trimmed = flux_section[good_data_idx]
    err_trimmed = flux_err_section[good_data_idx]
    timestamp_trimmed = timestamp_section[good_data_idx]
    return fit_simple_exponential(timestamps=timestamp_trimmed,
                                  fluxes=flux_trimmed,
                                  flux_errors=err_trimmed)


def fit_flare(dataset, flare):
    """
    Attempt to fit exponential rise / decay models to data marked as a flare.

    Args:
        dataset (dict): Dictionary representing dataset
        flare (Flare): Flare boundaries.

    Returns:
        Flare: Class containing fitted flare parameters
            (or `None` values in case of failed fit ).
    """

    # Estimate the background as the minimum value in the dataset
    flare.background_estimate = np.min(dataset[DataCols.flux])

    # Initialise empty data-struct, in case one / both fits fail

    # Fit rise section
    try:
        rise_par, rise_par_err = fit_flare_section(
            dataset,
            start_idx=flare.rise,
            end_idx=flare.peak,
            background_flux=flare.background_estimate)
        flare.rise_slope = rise_par[0]
        flare.rise_slope_err = rise_par_err[0]
        flare.rise_fit_pars = rise_par.tolist()

    except Exception as e:
        print("Rise fit failed:")
        print(e)
        pass

    # Fit decline section
    try:
        fall_par, fall_par_err = fit_flare_section(
            dataset,
            start_idx=flare.peak,
            end_idx=flare.fall,
            background_flux=flare.background_estimate)
        flare.decay_slope = fall_par[0]
        flare.decay_slope_err = fall_par_err[0]
        flare.decay_fit_pars = fall_par.tolist()
    except Exception as e:
        print("fall fit failed:")
        print(e)
        pass
    return flare


def smooth_with_window(x, window_len=24, window='flat'):
    """
    Smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    # Concatenate the input with half-window length reflections at either end
    s = np.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    # Return convolution after trimming end-reflections:
    return y[(window_len / 2 - 1):-(window_len / 2)]
