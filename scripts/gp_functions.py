import numpy as np
import pylab as pl
import os
from astropy.stats import median_absolute_deviation, sigma_clip
import george
from george import kernels
import scipy.optimize as op

def loadfile(file):
    """
        Loads the 2 different types of datasets

        Param
        ------

        file: path to dataset

    """

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
    """
        Returns a length scale of a peak in the dataset

        Params
        ------

        x   : time axis
        y   : Flux axis
        err : error on the flux measurements
    """
    clipped_fluxes = get_sigma_clipped_fluxes(y)
    background = np.ma.median(clipped_fluxes)
    noise = median_absolute_deviation(clipped_fluxes)

    rise_threshold = background + 5 * noise
    fall_threshold = background #+ 1 * noise
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


##--------------------------------------------------------------------##
#                            Main Functions                            #
##--------------------------------------------------------------------##

def get_gp(file):
    """
        Returns a Gaussian Process (george) object marginalised on the data in file.

        Param
        ------

        file : path to dataset
    """
    x,y,err = loadfile(file)

    ls = get_ls(x,y,err)
    kernel =  np.var(y)* kernels.ExpSquaredKernel(ls**2)

    gp = george.GP(kernel,fit_mean=True, white_noise=np.max(err)**2,fit_white_noise=True)
    gp.compute(x,err)

    # Define the objective function (negative log-likelihood in this case).
    def neg_ln_like(p):
        gp.set_parameter_vector(p)
        return -gp.log_likelihood(y)

    def grad_neg_ln_like(p):
        gp.set_parameter_vector(p)
        return -gp.grad_log_likelihood(y)

    results = op.minimize(neg_ln_like, gp.get_parameter_vector(), jac=grad_neg_ln_like,method="L-BFGS-B", tol = 1e-3)

    # Run the optimization routine.
    p0 = gp.get_parameter_vector()

    # Update the kernel and print the final log-likelihood.
    gp.set_parameter_vector(results.x)

    return gp

def plot_gp(file, save = False):
    """
       Plots a dataset with overlaid GP

       Params
       -------
       
       file : path to dataset
       save : Option to save plot - default is false
    """
    x,y,err = loadfile(file)
    gp = get_gp(file)

    t = np.linspace(np.min(x), np.max(x), 500)
    mu, cov = gp.predict(y, t)
    xnew = np.linspace(np.min(x),np.max(x),20)
    ynew = gp.sample_conditional(y,xnew,2)

    std = np.sqrt(np.diag(cov))

    f, axarr = pl.subplots(2,figsize=(15,12), sharex=True)

    axarr[0].errorbar(x,y,err,fmt='kx')
    axarr[0].set_title('Raw Data -' + file)
    axarr[0].set_xlabel('Time  [days]')
    axarr[0].set_ylabel('Flux  [Jy]')


    axarr[1].errorbar(x,y,err,fmt='kx')
    axarr[1].plot(t,mu)
    axarr[1].fill_between(t, mu - 2*np.sqrt(std**2), mu + 2*np.sqrt(std**2), color='blue', alpha=0.2)
    axarr[1].set_title('Data with GPs overlaid')
    axarr[1].set_xlabel('Time  [days]')
    axarr[1].set_ylabel('Flux  [Jy]')


    if save == True:
        pl.savefig(file+'.png')
    pl.show()
