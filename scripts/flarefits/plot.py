import logging

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from matplotlib import gridspec
from astropy.stats import median_absolute_deviation

from .fitting import get_sigma_clipped_fluxes, straight_line
from .ingest import DataCols

logger = logging.getLogger(__name__)


def plot_lightcurve_with_flares(dataset, flares,  ax=None):
    if ax is None:
        ax = plt.gca()
    plot_lightcurve(dataset, ax)
    for flr in flares:
        plot_flare_markers(flr,
                           timestamps=dataset[DataCols.time],
                           fluxes=dataset[DataCols.flux],
                           ax=ax)
    return ax


def plot_lightcurve(dataset, ax):
    """
    Add a basic lightcurve errorbar-plot to a matplotlib axis
    """
    ax.errorbar(dataset[DataCols.time], dataset[DataCols.flux],
                dataset[DataCols.flux_err],
                fmt='o',
                markersize=2)
    ax.set_xlabel(dataset[DataCols.time_units])
    ax.set_ylabel(dataset[DataCols.flux_units])
    return ax


def plot_sigma_clipping_hist(dataset, ax):
    """
    Plot bi-color histogram highlighting data-range masked by sigma-clipping.

    Overplot gaussian PDF matched to median, std. dev of sigma-clipped data.
    """

    fluxes = dataset[DataCols.flux]
    hist, bin_edges = np.histogram(fluxes,
                                   bins=max(len(fluxes) / 20, 15),
                                   normed=True)

    clipped_fluxes = get_sigma_clipped_fluxes(fluxes)

    bin_centres = []
    for bin_idx in range(len(bin_edges) - 1):
        bin_centres.append(0.5 * (bin_edges[bin_idx] + bin_edges[bin_idx + 1]))
    bin_centres = np.asarray(bin_centres)

    bin_width = bin_edges[1] - bin_edges[0]
    # Get a mask for the histogram that only shows flux values that are inside
    # the sigma-clipped range
    # (nb len(bin_edges)==len(bins)+1)
    clip_mask = ((bin_centres > clipped_fluxes.min()) *
                 (bin_centres < clipped_fluxes.max()))
    #
    # Plot the hist of all flux values, in red:
    ax.bar(bin_centres, hist, width=bin_width,
           color='r',
           label="All flux values")
    # The overplot the flux values that are inside the sigma-clip range,
    # using the mask
    ax.bar(bin_centres[clip_mask],
           hist[clip_mask],
           width=bin_width,
           label="Fluxes after sigma-clipping")

    xlim = np.percentile(fluxes, 0.5), np.percentile(fluxes, 99.5)
    ax.set_xlim(xlim)

    # Overplot a gaussian curve with same median and std dev as clipped data,
    # for comparison. (In yellow)
    x = np.linspace(xlim[0], xlim[1], 1000)
    clip_pars_norm = scipy.stats.norm(
        np.ma.median(clipped_fluxes),
        #np.ma.std(clipped_fluxes),
        median_absolute_deviation(clipped_fluxes)
    )
    ax.plot(x, clip_pars_norm.pdf(x), color='y',
            label="Normal dist. for comparison")

    ax.set_xlabel(dataset[DataCols.flux_units])
    ax.set_ylabel("Relative prob")
    ax.legend(loc='best')
    return ax


def plot_single_flare_lightcurve(dataset, flare, ax=None):
    if ax is None:
        ax = plt.gca()
    fluxes = dataset[DataCols.flux]
    flux_errs = dataset[DataCols.flux_err]
    timestamps = dataset[DataCols.time]
    fluxes_minus_bg = fluxes - flare.background_estimate
    log_fluxes_minus_bg = np.log(fluxes_minus_bg)

    ax.set_xlabel('Time [days]', fontsize=15)
    ax.set_ylabel('Log(Flux [Jy])', fontsize=15)

    flare_duration_idx = slice(flare.rise, flare.fall + 1)
    flare_rise_idx = slice(flare.rise, flare.peak + 1)
    flare_decay_idx = slice(flare.peak, flare.fall + 1)
    # Plot errorbars
    ax.errorbar(timestamps[flare_duration_idx],
                log_fluxes_minus_bg[flare_duration_idx],
                yerr=(flux_errs / fluxes)[flare_duration_idx],
                linestyle='None',
                )

    # And rise/fall datapoints in different colours
    ax.scatter(timestamps[flare_rise_idx],
               log_fluxes_minus_bg[flare_rise_idx],
               marker='o', color='Gold',
               )
    ax.scatter(timestamps[flare_decay_idx],
               log_fluxes_minus_bg[flare_decay_idx],
               marker='o', color='Lime',
               )

    plot_flare_markers(flare, timestamps, log_fluxes_minus_bg, ax)

    # Now fitted slopes
    if flare.rise_slope is not None:
        ax.plot(timestamps[flare_rise_idx],
                straight_line(timestamps[flare_rise_idx], *flare.rise_fit_pars),
                'b-', )
    if flare.decay_slope is not None:
        ax.plot(timestamps[flare_decay_idx],
                straight_line(timestamps[flare_decay_idx], *flare.decay_fit_pars),
                'r-', )
    # plt.ylim([np.log(np.percentile(fluxes, 0.5)), np.log(np.max(fluxes))])
    return ax


def plot_flare_markers(flare, timestamps, fluxes, ax):
    # Mark the boundary points:
    marker_colour = 'Black'
    flare_markers = {
        flare.rise: '^',
        flare.trigger: 'p',
        flare.peak: '*',
        flare.fall: 'v'
    }
    for idx, marker_shape in flare_markers.items():
        ax.plot(timestamps[idx], fluxes[idx],
                marker=marker_shape,
                color=marker_colour,
                zorder=10)


def plot_thresholds(background, low_threshold, high_threshold, ax):
    ax.axhline(y=background,
               label='background',
               ls=':', c='g', lw=2)
    ax.axhline(
        y=low_threshold,
        label='$b+1\sigma$',
        ls='--', c='b', lw=2)

    ax.axhline(
        y=high_threshold,
        label='$b+5\sigma$',
        ls='--', c='r', lw=2)
