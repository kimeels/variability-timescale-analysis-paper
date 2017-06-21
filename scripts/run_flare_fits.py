#!/usr/bin/env python
from __future__ import print_function

import json
import logging
import os
import pprint
import sys
from collections import defaultdict

import attr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from astropy.stats import median_absolute_deviation

import flarefits.ingest as ingest
from flarefits.fitting import (
    Flare, find_and_fit_flares, fit_flare, get_sigma_clipped_fluxes,
    smooth_with_window
)
from flarefits.ingest import (
    DataCols, FitMethods, IndexCols, trim_outliers_below_percentile
)
from flarefits.plot import (
    plot_lightcurve_with_flares, plot_sigma_clipping_hist,
    plot_single_flare_lightcurve, plot_thresholds
)

logging.basicConfig(
    # level=logging.DEBUG,
    level=logging.INFO,
)
logger = logging.getLogger()

PROJECT_ROOT = os.path.abspath('..')
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')
INDEX_PATH = os.path.join(PROJECT_ROOT, "target-distances-and-class.txt")
DEFAULT_OUTPUT_DIR = './results'
PLOT_FORMAT = 'png'


def main():
    pp = pprint.PrettyPrinter()

    # Find the datafiles
    logger.info("DATA_ROOT: {}".format(DATA_ROOT))
    data_files = ingest.recursive_glob(DATA_ROOT)
    logger.info("Found {} datasets to analyze".format(len(data_files)))

    # sys.exit(0)
    # Load the index telling us which analysis to run for each datafile
    data_index = ingest.load_datafiles_index(INDEX_PATH)
    logger.info("Found {} entries in index".format(len(data_index)))

    # pp.pprint(data_index)
    # pp.pprint(data_files)

    # Check the index / datafiles look sensible
    check_data_files_and_index(data_files, data_index)

    # Go do science...
    for fpath in data_files:
        dataset_name = os.path.basename(fpath)
        if dataset_name.endswith('.txt'):
            dataset_name = dataset_name[:-4]
        analyze_dataset(dataset_name, fpath, data_index)

    return 0


def check_data_files_and_index(data_files, data_index):
    """
    Check that the data-files and index look sensible.

    - Ensure there is only one data-file of each name (so we know that we aren't
      mixing up datasets in different directories)
    - Ensure that every data-file has a corresponding entry in the index
    - List any index entries for which we don't have a data-file (this is OK but
      worth knowing about).
    """
    basenames = [os.path.basename(fpath) for fpath in data_files]
    id_to_paths_map = defaultdict(list)
    for bname in basenames:
        if bname.endswith('.txt'):
            id_to_paths_map[bname[:-4]].append(fpath)
        else:
            id_to_paths_map[bname].append(fpath)

    for dataset_id, fpaths in id_to_paths_map.items():
        if dataset_id not in data_index:
            raise RuntimeError("No entry in index for dataset:{}".format(
                dataset_id
            ))
        if len(fpaths) != 1:
            raise RuntimeError("Multiple files for id {}: {}".format(
                dataset_id, fpaths))

    for dataset_id in data_index.keys():
        if dataset_id not in id_to_paths_map:
            logger.debug("Not found: {}".format(dataset_id))


def analyze_dataset(dataset_id, dataset_filepath, data_index):
    """
    For each data file, check the index and run the corresponding analysis.
    """
    fit_method = data_index[dataset_id][IndexCols.fit_method]
    logger.info("Analyzing dataset {}, fit method {}, from path {}".format(
        dataset_id, fit_method, dataset_filepath))
    logger.debug("(Path: {}".format(dataset_filepath))
    # logger.debug("Method: {}".format(fit_method))
    dataset = ingest.load_dataset(dataset_filepath, dataset_id, fit_method)
    fluxes = dataset[DataCols.flux]

    if fit_method == FitMethods.gbi:
        clipped_fluxes = get_sigma_clipped_fluxes(dataset[DataCols.flux])
        background_estimate = np.ma.median(clipped_fluxes)
        noise_estimate = median_absolute_deviation(clipped_fluxes)
        flares = find_and_fit_flares(
            dataset,
            background=background_estimate,
            noise_level=noise_estimate)
    elif fit_method == FitMethods.gbi_single_flare:
        dataset[DataCols.flux] = smooth_with_window(dataset[DataCols.flux])
        background_estimate = np.percentile(fluxes, 15.)
        noise_estimate = np.mean(dataset[DataCols.flux_err])
        flares = find_and_fit_flares(
            dataset,
            background=background_estimate,
            noise_level=noise_estimate)
    elif fit_method == FitMethods.paper:
        background_estimate = np.percentile(fluxes, 15.)
        noise_estimate = np.mean(dataset[DataCols.flux_err])
        flares = find_and_fit_flares(
            dataset,
            background=background_estimate,
            noise_level=noise_estimate
        )

    elif fit_method == FitMethods.paper_single_flare:
        # Single visually identified flare
        single_flare = Flare(rise=0, trigger=0,
                             peak=np.argmax(fluxes),
                             fall=len(fluxes) - 1
                             )
        flares = [single_flare, ]
        fit_flare(dataset, single_flare)
        # Dummy values, passed to plotting routines:
        background_estimate = None
        noise_estimate = None
    else:
        raise ValueError("Unknown fit method: {}".format(fit_method))

    save_results(dataset_id, dataset, flares,
                 background_estimate=background_estimate,
                 noise_estimate=noise_estimate,
                 fit_method=fit_method)


def ensure_dir(dirname):
    """Ensure directory exists, or raise exception if file of same name exists."""
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    elif not os.path.isdir(dirname):
        raise RuntimeError("Path exists but is not directory: \n" + dirname)


def save_results(dataset_id, dataset, flares,
                 background_estimate, noise_estimate,
                 fit_method,
                 output_dir=DEFAULT_OUTPUT_DIR):
    """
    Write flares to JSON and output plots
    """

    method_dir = os.path.join(output_dir, fit_method)
    dataset_dir = os.path.join(method_dir, dataset_id)
    ensure_dir(dataset_dir)
    write_flares_to_json(
        os.path.join(method_dir, dataset_id + '_flares.json'),
        flares)

    fig = plt.gcf()
    fig.suptitle(dataset[DataCols.id])
    # Set up a 3 row plot-`.
    gs = gridspec.GridSpec(3, 1)
    # Use the top 2/3rds for the lightcurve axis
    lightcurve_ax = plt.subplot(gs[:2])
    # Use the bottom 1/3rd for the histogram axis
    hist_ax = plt.subplot(gs[2])

    if fit_method == FitMethods.gbi:
        plot_sigma_clipping_hist(dataset, ax=hist_ax)
    elif fit_method in (FitMethods.gbi_single_flare, FitMethods.paper):
        hist_ax.hist(dataset[DataCols.flux], normed=True)
        hist_ax.set_xlabel(dataset[DataCols.flux_units])
        hist_ax.set_ylabel("Relative prob")
    elif fit_method == FitMethods.paper_single_flare:
        # Don't bother with a histogram plot if we're looking at a preselected
        # flare.
        lightcurve_ax = plt.subplot(gs[:])

    plot_lightcurve_with_flares(dataset, flares, ax=lightcurve_ax)
    if fit_method != FitMethods.paper_single_flare:
        plot_thresholds(
            background=background_estimate,
            low_threshold=background_estimate + 1 * noise_estimate,
            high_threshold=background_estimate + 5 * noise_estimate,
            ax=lightcurve_ax,
        )
    lightcurve_ax.legend(loc='best')

    overview_plot_filename = dataset_id + '_overview.' + PLOT_FORMAT
    # Save 2 copies of overview plot: one in main 'method directory'
    fig.savefig(os.path.join(method_dir, overview_plot_filename))
    # Another in dataset directory, with the single-flare plots
    fig.savefig(os.path.join(dataset_dir, overview_plot_filename))
    plt.close(fig)
    timestamps = dataset[DataCols.time]
    for flr_count, flr in enumerate(flares):
        ax = plot_single_flare_lightcurve(dataset, flr)
        flare_filename = ("{}_flare_{}_t{}_t{}." + PLOT_FORMAT).format(
            dataset_id,
            flr_count,
            timestamps[flr.rise],
            timestamps[flr.fall]
        )
        plt.savefig(os.path.join(dataset_dir, flare_filename))
        plt.clf()


def write_flares_to_json(output_path, flares):
    flare_fits_dicts = [attr.asdict(flr) for flr in flares]
    with open(output_path, 'wb') as fp:
        json.dump(flare_fits_dicts, fp)


if __name__ == '__main__':
    sys.exit(main())
