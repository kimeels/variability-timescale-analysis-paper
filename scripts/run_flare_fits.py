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

import flarefits.ingest as ingest
from flarefits.fitting import (
    calculate_dataset_properties, find_and_fit_flares,
    smooth_with_window
)
from flarefits.ingest import DataCols, FitMethods, IndexCols
from flarefits.plot import (
    plot_dataset_with_histogram, plot_single_flare_lightcurve
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
    if fit_method == FitMethods.gbi:
        dataset = ingest.load_dataset(dataset_filepath, dataset_id, fit_method)
        data_props = calculate_dataset_properties(dataset)
        flares = find_and_fit_flares(
            dataset,
            background=data_props.clipped_median,
            noise_level=data_props.clipped_std_dev)
    elif fit_method == FitMethods.gbi_smoothed:
        dataset = ingest.load_dataset(dataset_filepath, dataset_id, fit_method,
                                      trim_outliers_below_percentile=3.)
        dataset[DataCols.flux] = smooth_with_window(dataset[DataCols.flux])
        data_props = calculate_dataset_properties(dataset)
        flares = find_and_fit_flares(
            dataset,
            background=data_props.percentile10,
            noise_level=data_props.clipped_std_dev)
    elif fit_method == FitMethods.paper:
        dataset = ingest.load_dataset(dataset_filepath, dataset_id, fit_method)
        data_props = calculate_dataset_properties(dataset)
        return None
    elif fit_method == FitMethods.paper_smoothed:
        dataset = ingest.load_dataset(dataset_filepath, dataset_id, fit_method)
        data_props = calculate_dataset_properties(dataset)
        return None
    else:
        raise ValueError("Unknown fit method: {}".format(fit_method))
    save_results(dataset_id, dataset, data_props, flares, fit_method)


def ensure_dir(dirname):
    """Ensure directory exists, or raise exception if file of same name exists."""
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    elif not os.path.isdir(dirname):
        raise RuntimeError("Path exists but is not directory: \n" + dirname)


def save_results(dataset_id, dataset, dataset_properties, flares, fit_method,
                 output_dir=DEFAULT_OUTPUT_DIR):
    method_dir = os.path.join(output_dir, fit_method)
    dataset_dir = os.path.join(method_dir, dataset_id)
    ensure_dir(dataset_dir)
    write_flares_to_json(
        os.path.join(method_dir, dataset_id + '_flares.json'),
        flares)
    dataset_fig = plot_dataset_with_histogram(
        dataset, dataset_properties, flares, fit_method)
    overview_plot_filename = dataset_id + '_overview.' + PLOT_FORMAT
    #Save 2 copies of overview plot: one in main 'method directory'
    dataset_fig.savefig(os.path.join(method_dir,overview_plot_filename))
    #Another in dataset directory, with the single-flare plots
    dataset_fig.savefig(os.path.join(dataset_dir,overview_plot_filename))
    plt.close(dataset_fig)
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
