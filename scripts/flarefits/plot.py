import logging

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from matplotlib import gridspec

from .fitting import (get_sigma_clipped_fluxes, straight_line)
from .ingest import DataCols

logger = logging.getLogger(__name__)


def plot_dataset_with_histogram(dataset, dataset_properties):
    fig = plt.gcf()
    # Set up a 3 row plot-`.
    gs = gridspec.GridSpec(3, 1)
    # Use the top 2/3rds for the lightcurve axis
    lightcurve_ax = plt.subplot(gs[:2])
    # Use the bottom 1/3rd for the histogram axis
    hist_ax = plt.subplot(gs[2])
    fig.suptitle(dataset[DataCols.id])

    plot_lightcurve(dataset, lightcurve_ax)
    plot_flux_vals_hist(dataset, dataset_properties, hist_ax)
    return fig


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


def plot_flux_vals_hist(dataset, dataset_properties, ax):
    """
    Plot bi-color histogram highlighting data-range masked by sigma-clipping.

    Overplot gaussian PDF matched to mean, std. dev of sigma-clipped data.
    """

    fluxes = dataset[DataCols.flux]
    hist, bin_edges = np.histogram(fluxes,
                                   bins=max(len(fluxes) / 20, 15),
                                   normed=True)

    clipped_fluxes = get_sigma_clipped_fluxes(fluxes)


    # Get a mask for the histogram that only shows flux values that are inside
    # the sigma-clipped range
    # (nb len(bin_edges)==len(bins)+1)
    clip_mask = ((bin_edges > clipped_fluxes.min()) *
                 (bin_edges < clipped_fluxes.max()))
    #
    # Plot the hist of all flux values, in red:
    ax.bar(bin_edges[:-1], hist, width=bin_edges[1] - bin_edges[0],
           color='r',
           label="All flux values")
    # The overplot the flux values that are inside the sigma-clip range,
    # using the mask
    ax.bar(bin_edges[clip_mask],
           hist[clip_mask],
           width=bin_edges[1] - bin_edges[0],
           label="Fluxes after sigma-clipping")

    xlim = np.percentile(fluxes, 0.5), np.percentile(fluxes,99.5)
    ax.set_xlim(xlim)

    # Overplot a gaussian curve with same median and std dev as clipped data,
    # for comparison. (In yellow)
    x=np.linspace(xlim[0], xlim[1],1000)
    clip_pars_norm = scipy.stats.norm(
        dataset_properties.clipped_median,
        dataset_properties.clipped_std_dev)
    ax.plot(x, clip_pars_norm.pdf(x), color='y',
            label = "Normal dist. for comparison")

    ax.set_xlabel(dataset[DataCols.flux_units])
    ax.set_ylabel("Relative prob")
    ax.legend(loc='best')
    return ax


def plot_flare(dataset, flare, ax=None):
    if ax is None:
        ax = plt.gca()
    fluxes = dataset[DataCols.flux]
    flux_errs = dataset[DataCols.flux_err]
    timestamps = dataset[DataCols.time]
    fluxes_minus_bg = fluxes - flare.background_estimate

    flare_duration_idx = slice(flare.rise, flare.fall+1)
    flare_rise_idx = slice(flare.rise, flare.peak+1)
    flare_decay_idx = slice(flare.peak, flare.fall + 1)
    #Plot errorbars
    ax.errorbar(timestamps[flare_duration_idx],
                np.log(fluxes_minus_bg[flare_duration_idx]),
                yerr = (flux_errs/fluxes)[flare_duration_idx],
                linestyle='None',
                )

    #And rise/fall datapoints in different colours
    ax.scatter(timestamps[flare_rise_idx],
               np.log(fluxes_minus_bg[flare_rise_idx]),
               marker='o', color='Gold',
               )
    ax.scatter(timestamps[flare_decay_idx],
               np.log(fluxes_minus_bg[flare_decay_idx]),
               marker='o', color='Lime',
               )

    #Now fitted slopes
    ax.plot(timestamps[flare_rise_idx],
            straight_line(timestamps[flare_rise_idx], *flare.rise_fit_pars),
            'b-',)
    ax.plot(timestamps[flare_decay_idx],
            straight_line(timestamps[flare_decay_idx], *flare.decay_fit_pars),
            'r-', )

    # plt.ylim([np.log(np.percentile(fluxes, 0.5)), np.log(np.max(fluxes))])
    ax.set_xlabel('Time [days]', fontsize=15)
    ax.set_ylabel('Flux [Jy]', fontsize=15)
    return ax
