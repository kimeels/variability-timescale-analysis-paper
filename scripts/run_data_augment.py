#!/usr/bin/env python3
import numpy as np
import pylab as pl
import gpflow
import csv
import os
import seaborn as sns
from astropy.stats import median_absolute_deviation, sigma_clip
import gc

def get_sigma_clipped_fluxes(raw_fluxes):
    """
    Uses the astropy sigma_clip function to try and get rid of outliers.

    ( https://astropy.readthedocs.org/en/stable/api/astropy.stats.sigma_clip.html )

    Returns a masked array where all outlier values are masked.
    """

    # First we try to run sigma clip with the defaults - hopefully this will
    # iterate until it converges:
    clipped_fluxes = sigma_clip(raw_fluxes, iters=None,
                                stdfunc=median_absolute_deviation)
    # If it fails (number of unmasked values <3),
    # then we just accept the result from a single iteration:
    if len(clipped_fluxes.compressed()) < 3:
        logger.warning("Sigma clipping did not converge, "
                       "using single iteration")
        clipped_fluxes = sigma_clip(raw_fluxes, iters=1,
                                    stdfunc=median_absolute_deviation)
    return clipped_fluxes


def get_ls(x,y,err):
    clipped_fluxes = get_sigma_clipped_fluxes(y)
    background = np.ma.median(clipped_fluxes)
    noise = median_absolute_deviation(clipped_fluxes)

    rise_threshold = background + 5 * noise
    fall_threshold = background + 1 * noise
    flux_plus_err = y + err

    trigger = np.where(flux_plus_err > rise_threshold)[0]
    if len(trigger) == 0:
        trigger = np.where(flux_plus_err == np.max(flux_plus_err))[0][0]
    else:
        trigger = trigger[0]

    indexes = np.where(y < fall_threshold)[0]
    if len(indexes) == 0:
        indexes = np.where(y < background)[0]

    fall_indexes = np.array([indexes[i] for i in range(len(indexes)-1) if y[indexes[i]] > y[indexes[i+1]]])

    fall = np.where(fall_indexes > trigger)[0]
    if len(fall) == 0:
        fall_idx = len(y) - 1
    else:
        fall_idx = fall_indexes[np.where(fall_indexes > trigger)][0]

    rise = np.where(indexes <= trigger)[0]
    if len(rise) == 0:
        rise_idx = trigger
    else:
        rise_idx = indexes[np.where(indexes < trigger)][-1]

    return  ((x[fall_idx] - x[rise_idx]))/(2*np.sqrt(2*np.log(2)))

def loadfile(file):
    if "GBI" in file:
        data = np.loadtxt(file)
        x,y,err = data[:,0],data[:,3],data[:,6]
        y_eq_minus1 = np.where(y == -1)
        y = np.delete(y,y_eq_minus1)
        x = np.delete(x,y_eq_minus1)
        err =  np.delete(err,y_eq_minus1)

    else:
        data = np.loadtxt(file)
        x,y,err = data[:,0],data[:,1],data[:,2]

    return x,y,err


def sample_curve(x,y,err,xsample_range,num_of_sample_curves,filename):

    clipped_fluxes = get_sigma_clipped_fluxes(y)
    background = np.ma.median(clipped_fluxes)
    noise = median_absolute_deviation(clipped_fluxes)

    ls = get_ls(x,y,err)
    var = noise

    # print('Background = ',background)
    # print('Noise = ', noise)
    # print('ls = ',ls)

    k = gpflow.kernels.RBF(1,lengthscales=ls,variance=var)
    m = gpflow.gpr.GPR(x.reshape(len(x),1), y.reshape(len(y),1), kern=k)
    m.kern.lengthscales.prior = gpflow.priors.Gaussian(ls, ls/5)
    m.kern.variance.prior = gpflow.priors.Gaussian(5*var, var)
    m.optimize()

    xx = np.linspace(min(x), max(x), 1000)[:,None]
    mean, var = m.predict_y(xx)

    pl.figure(figsize=(12, 6))
    pl.plot(x, y, 'kx', mew=2)
    pl.plot(xx, mean, 'b', lw=2)
    pl.fill_between(xx[:,0], mean[:,0] - 2*np.sqrt(var[:,0]), mean[:,0] + 2*np.sqrt(var[:,0]), color='blue', alpha=0.2)
    pl.xlabel('Days [JD]')
    pl.ylabel('Flux  [Jy]')
    pl.savefig(filename+'.png')
    pl.close()

    xnew = xsample_range[:,None]
    ynew = m.predict_f_samples(xnew,num_of_sample_curves)

    return xnew,ynew





print("#######################################################################")
print("#                    Starting Data Augmentation                       #")
print("#######################################################################")
print("")

root = "../data/"
data_augment_root = "../data.augment/"

allfiles = []
for path, subdirs, files in os.walk(root):
    subdirs[:] = [d for d in subdirs if d not in ['GBI']]
    for filename in files:
        f = os.path.join(path, filename)
        allfiles.append(f)

not_good = ["../data/TDE/ASASSN-14li.txt",
               "../data/DwarfNova/SSCyg.txt",
               "../data/XRB/CygX-2p.txt",
               "../data/XRB/M31ULX.txt",
               "../data/GRB/GRB110709B.txt",
               "../data/GRB/GRB060418.txt",
               "../data/RSCVn/CFOct.txt",
               "../data/Flare-Star/AUMic.txt",
               "../data/Flare-Star/ADLeo",
               "../data/Magnetic-CV/V834Cen.txt"]

number_of_curves = 10

for j in range(len(allfiles)):
    datafile = allfiles[j]
    if datafile not in not_good:
        x,y,err = loadfile(datafile)
        xsample_range = np.arange(np.min(x),np.max(x),1/4)
        print('Sampling file: ', datafile)
        print('---------------------------------------------------------------')
        #try:
        if datafile.endswith('.txt'):
            datafile = datafile[:-4]
        filename = data_augment_root + datafile[len(root):]
        xnew,ynew = sample_curve(x,y,err,xsample_range,number_of_curves,filename)

        for i in range(number_of_curves):
            with open(filename +'_'+str(i),'w+') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerows(zip(xnew[:,0],ynew[i][:,0]))

        gc.collect()
        # except Exception as e:
        #     print(datafile, e)
