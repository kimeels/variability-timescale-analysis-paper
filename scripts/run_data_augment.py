#!/usr/bin/env python3
import numpy as np
import pylab as pl
import gpflow
import csv
import os
import seaborn as sns
from astropy.stats import median_absolute_deviation, sigma_clip
import george
from george import kernels
import scipy.optimize as op
import gc
from multiprocessing import Process
import time
import glob
from gp_functions import *



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


def sample_curve_v2(x,y,err,xsample_range,num_of_curves,filename):
    ls = get_ls(x,y,err)

    k1 = np.var(y)* kernels.ExpSquaredKernel(ls**2)
    k2 = 1 * kernels.RationalQuadraticKernel(log_alpha=1, metric=ls**2)
    kernel = k1 + k2

    gp = george.GP(kernel, fit_mean=True)
    gp.compute(x,err)

    # Define the objective function (negative log-likelihood in this case).
    def nll(p):
        gp.set_parameter_vector(p)
        ll = gp.lnlikelihood(y, quiet=True)
        return -ll if np.isfinite(ll) else 1e25

    # And the gradient of the objective function.
    def grad_nll(p):
        gp.set_parameter_vector(p)
        return -gp.grad_lnlikelihood(y, quiet=True)

    p0 = gp.get_parameter_vector()
    results = op.minimize(nll, p0, jac=grad_nll, method="L-BFGS-B")
    gp.set_parameter_vector(results.x)

    xnew = xsample_range
    ynew = gp.sample_conditional(y,xnew,number_of_curves)

    t = np.linspace(np.min(x), np.max(x), 100)
    mu, cov = gp.predict(y, t)

    pl.errorbar(x,y,err,fmt='kx')
    pl.plot(t,mu)
    pl.fill_between(t, mu - 2*np.sqrt(std**2), mu + 2*np.sqrt(std**2), color='blue', alpha=0.2)
    pl.savefig(filename+'.pdf')

    # mu, cov = gp.predict(y, t)
    # std = np.sqrt(np.diag(cov))


    return xnew,ynew

def augment(datafile,number_of_curves):
    x,y,err = loadfile(datafile)

    sample_rate = 2/(24*60*60)


    xmax = np.max(x)
    i = 0
    xbeg = np.min(x)
    xend = np.min(x)
    while xend < xmax:

    xsample_range = np.linspace(np.min(x),np.max(x),1000)
    print('Sampling file: ', datafile)
    print('---------------------------------------------------------------')
    #try:
    if datafile.endswith('.txt'):
        datafile = datafile[:-4]
    filename = data_augment_root + datafile[len(root):]
    #xnew,ynew = sample_curve(x,y,err,xsample_range,number_of_curves,filename)
    #mu,std = sample_curve_v2(x,y,err,xsample_range,filename)

    xnew,ynew = sample_curve_v2(x,y,err,xsample_range,number_of_curves,filename)

    if "GBI" in file:
        z = np.zeros(len(xnew))
        for i in range(number_of_curves):
            with open(filename +'_'+str(i),'w+') as f:
                    writer = csv.writer(f, delimiter='\t')
                    writer.writerows(zip(xnew,z,z,ynew[i],z,z,z))
    else:
        for i in range(number_of_curves):
            with open(filename +'_'+str(i),'w+') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerows(zip(xnew,ynew[i],np.zeros(len(xnew))))


print("#######################################################################")
print("#                    Starting Data Augmentation                       #")
print("#######################################################################")
print("")

root = "../data/"
data_augment_root = "../data.augment/"

all_files = glob.glob(os.path.join(root, "RSCVn/GBI/*"),recursive=True)

allfiles = []
for path, subdirs, files in os.walk(root):
    subdirs[:] = [d for d in subdirs if d not in ['GBI']]
    for filename in files:
        f = os.path.join(path, filename)
        allfiles.append(f)

not_good = ["../data/DwarfNova/SSCyg.txt",
               "../data/XRB/MAXIJ1836-194"]

number_of_curves = 20

if __name__ == '__main__':
    for j in range(len(all_files)):
        datafile = all_files[j]
        if datafile not in not_good:
            p = Process(target=augment, args=(datafile,number_of_curves,))
            p.start()
            p.join()
            gc.collect()


        # for i in range(number_of_curves):
        #     with open(filename +'_'+str(i),'w+') as f:
        #         writer = csv.writer(f, delimiter='\t')
        #         writer.writerows(zip(xnew[:,0],ynew[i][:,0]))

        # except Exception as e:
        #     print(datafile, e)
