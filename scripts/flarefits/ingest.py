import csv
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


class FitMethods:
    gbi = "GBI-m"
    gbi_smoothed = "GBI-s"
    paper = "Papers-m"
    paper_smoothed = "Papers-s"


class IndexCols:
    """
    Columns found in the index file (tsv-format)
    """
    name = 'name'
    dist_mpc = 'distance_mpc'
    target_class = 'target_class'
    freq_ghz = 'frequency_ghz'
    fit_method = 'fit_method'


class DataCols:
    """
    Standardized columns for all datasets
    """
    id = 'id'
    time = 'time'
    time_units = 'time_units'
    flux = 'flux'
    flux_err = 'flux_err'
    flux_units = 'flux_units'
    deltas = 'deltas'
    slopes = 'slopes'


class GbiCols:
    """
    Columns found in GBI datasets
    """
    mjd = 'MJD'
    lst = 'LST'
    flux_2ghz = 'Flux @ 2.25GHz'
    flux_2ghz_err = '1-sigma flux error @ 2.25GHz '
    flux_8ghz = 'Flux @ 8.3GHz'
    flux_8ghz_err = '1-sigma flux error @ 8.3GHz '
    alpha = 'Spectral index'

    col_hdrs = (mjd, lst,
                flux_2ghz,
                flux_8ghz,
                alpha,
                flux_2ghz_err,
                flux_8ghz_err)


def recursive_glob(root_dir):
    """
    Recursively get a list of abspaths for all files under one of ``data_dirs``.

    Args:
        root_dir (str): The top-level directory to search under

    Returns:
        list: A list of absolute-paths to datafiles.

    """

    matches = []
    for parentpath, dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            matches.append(os.path.abspath(os.path.join(parentpath, fname)))

    return sorted(matches)


def load_datafiles_index(path_to_index_file):
    """
    Load a nested dict indexing the datafiles by class, fitting method etc.

    Args:
        path_to_index_file (str): Where to find the index file
            (This should be a tab-separated values file)

    Returns:
        dict: Nested dict mapping dataset-name -> dict of name, distance, class,
        frequency, and fitting method.

    (See also ``IndexCols``).
    """
    with open(path_to_index_file) as f:
        rdr = csv.reader(f, delimiter='\t', skipinitialspace=True)
        datarows = [row for row in rdr
                    if len(row) and not row[0].startswith('#')
                    ]
    index = {}
    for row in datarows:
        index[row[0]] = {
            IndexCols.name: row[0],
            IndexCols.dist_mpc: row[1],
            IndexCols.target_class: row[2],
            IndexCols.freq_ghz: row[3],
            IndexCols.fit_method: row[4]
        }

    return index


def read_gbi_datafile(path):
    """
    Loads GBI data from file at ``path``.

    Converts it to a dictionary of numpy arrays - one entry for each column.
    The dictionary keys are defined in class GbiCols (see above).
    """
    logger.debug("Reading: " + path)
    with open(path) as f:
        rdr = csv.reader(f, delimiter=' ', skipinitialspace=True)
        datarows = [row for row in rdr if len(row)]

    for i, p in enumerate(datarows):
        datarows[i] = [float(j) for j in p]

    # Some bad observations are flagged by value == -1.0000
    def is_bad_data(r):
        if r[2] == -1.0 or r[3] == -1.0:
            return True
        else:
            return False

    datarows = [r for r in datarows if not is_bad_data(r)]
    # Sort the data by x value:
    datarows.sort(key=lambda pair: pair[0])
    data_vecs = zip(*datarows)
    data_vecs = [np.asarray(v) for v in data_vecs]

    if not (len(data_vecs) == len(GbiCols.col_hdrs)):
        logging.warning("Datafile " + path + " has header / col mismatch")
    dataset = dict(zip(GbiCols.col_hdrs, data_vecs))
    return dataset


def standardize_gbi_dataset(raw_dataset):
    """
    Change the keys of a dataset dictionary, to match what we've previously
    found in non-GBI files.
    Add 'time_units' and 'flux_units' entries, set these to 'MJD', 'Jy'
    then replace the keys 'MJD' -> `time`, 'Jy'->'flux'.
    """
    # NB, have chosen to use the 8GHz flux values.
    flux, err = GbiCols.flux_8ghz, GbiCols.flux_8ghz_err
    raw_dataset[DataCols.time_units] = GbiCols.mjd
    raw_dataset[DataCols.time] = raw_dataset.pop(GbiCols.mjd)
    raw_dataset[DataCols.flux_units] = 'Jy'
    raw_dataset[DataCols.flux] = raw_dataset.pop(flux)
    raw_dataset[DataCols.flux_err] = raw_dataset.pop(err)
    return raw_dataset


def load_gbi_dataset(datafile_abspath, dataset_id,
                     trim_outliers_below_percentile=None):
    """
    Load a GBI dataset.

    Load from GBI format files, and then convert to a dictionary of numpy arrays
    etc with standard keys as listed in the ``DataCols`` class.

    GBI data often has weird 'drop out' fluxes that look much lower than
    rest of the timeseries, we assume these are due to some instrumental or
    data-processing error. We trim them by simply dropping everything below
    a certain percentile.

    Args:
        datafile_abspath (str): Path to dataset file (tab-separated values ascii)
        dataset_id (str): Unique identifier
        trim_outliers_below_percentile (float or None): Drop all flux values
            (and corresponding timestamps, errors) below this percentile.

    Returns:
        dict: See ``DataCols`` class.
    """
    raw_dataset = read_gbi_datafile(datafile_abspath)
    dataset = standardize_gbi_dataset(raw_dataset)
    if trim_outliers_below_percentile is not None:
        low_val_outlier_threshold= np.percentile(dataset[DataCols.flux],
                                         trim_outliers_below_percentile)
        good_data_idx = dataset[DataCols.flux] > low_val_outlier_threshold
        for col in (DataCols.time, DataCols.flux, DataCols.flux_err, ):
            dataset[col] = dataset[col][good_data_idx]

    dataset[DataCols.id] = dataset_id
    return dataset
