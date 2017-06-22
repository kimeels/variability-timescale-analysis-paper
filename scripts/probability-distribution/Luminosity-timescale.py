import numpy as np
import pylab
import math
import matplotlib
import glob, os
import csv
import argparse
from scipy import *
from pylab import *
import scipy.optimize
from scipy.optimize import curve_fit
import shutil
import matplotlib.gridspec as gridspec
from scipy.odr import *

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--complete_measurements', action='store', type=str, help='Directory in which *complete_rise/decline* files are stored.', default='')
    args = parser.parse_args()

## Upload slope, slope err and luminosity as 'data' from complete_rise/decline files
## Upload all classes of object in those files as 'cls'
matplotlib.style.use('classic')
matplotlib.rcParams['patch.force_edgecolor'] = True
matplotlib.rcParams['patch.facecolor'] = 'b'
for phase in ('rise', 'decline' ):
    data = np.loadtxt(args.complete_measurements+'complete_{}'.format(phase), dtype = float, usecols = (1,2,7), unpack=True)
    cls = np.loadtxt(args.complete_measurements+"complete_{}".format(phase), dtype = str, usecols = (8,), unpack = True)

    markertype = ('o', (5,0,0), (5,1,180), (11,1,30), (4,1,0), '^', 'v', (4,0,180), (4,1,45), 'd', 's', (5,1,0), (4,0,120), 'H', (4,1,45), (7,1,0), (5,1,0), 'D')
    colors = ('Red', 'Red', 'Red', 'Turquoise', 'Gold', 'LimeGreen', 'Green', 'DodgerBlue', 'DodgerBlue', 'DodgerBlue', 'DodgerBlue', '#A631FF', 'Maroon', 'Yellow', 'OrangeRed', 'Cyan', '#FE9A2E', 'Magenta')
    types = ('Blazar', 'QSO', 'AGN', 'TDE', 'GRB', 'SN', 'Nova', 'XRB', 'BHXRB', 'NSXRB', 'Binary pulsar', 'Magnetar', 'ULX', 'RSCVn', 'DN', 'Algol', 'Magnetic CV', 'Flare stars')
    markersizes = (11, 16, 15, 17, 16, 15, 15, 13, 16, 13, 9, 22, 13, 13, 18, 15, 15, 13) 

    ## Make a plot.
    fig = pylab.figure(figsize=(16,10.2))
    ax  = fig.add_subplot(111)
    ax.yaxis.grid(color='gray', linestyle=':')
    ax.xaxis.grid(color='gray', linestyle=':')
    for i in range(0, cls.shape[0]):
        if cls[i] == 'Flare_Stars':
            cls[i] = cls[i].replace('Flare_Stars', 'Flare stars')
        if cls[i] == 'BLAZAR':
            cls[i] = cls[i].replace('BLAZAR', 'Blazar')
        if cls[i] == 'magCV':
            cls[i] = cls[i].replace('magCV', 'Magnetic CV')
        if cls[i] == 'BinaryPulsars':
            cls[i] = cls[i].replace('BinaryPulsars', 'Binary pulsar')
                           
    for j in range(0, len(types)):
        c = 0
        for i in range(0, cls.shape[0]):
            if cls[i] == types[j]:
                c = c+1
                ax.errorbar(abs(data[0,i]), data[2,i], xerr = abs(data[1,i]), yerr=None, marker = None, linestyle='None', ecolor="Black", elinewidth=0.5)
                obj = ax.plot(abs(data[0,i]), data[2,i], marker = markertype[j], markerfacecolor =str(colors[j]), markeredgewidth = 1.2, markeredgecolor = 'black', markersize = markersizes[j],  linestyle='None', label=str(types[j]) if c == 1 else "")

    ax.legend(loc = 0, numpoints=1)

    # Make a fit to the luminosity - time-scale data in log log space
    fit = np.loadtxt(args.complete_measurements+'complete_{}'.format(phase), dtype=float, usecols=(9,10,11), unpack=True)

    logx = fit[2]
    logy = fit[0]
    logyerr = fit[1]
    

    def lin(x, a, b):
        return a * x + b

    fit_pars, fit_cov = curve_fit(lin, logx, logy, sigma = logyerr, maxfev=800)
    fit_pars_err = np.sqrt(np.diag(fit_cov))  

    aa = 1.0/fit_pars[0]
    bb = -fit_pars[1]/fit_pars[0]
    aa_err = abs(fit_pars_err[0]/(fit_pars[0]*fit_pars[0]))
    bb_err = abs(fit_pars[1]*fit_pars_err[0]/(fit_pars[0]*fit_pars[0])) + abs(fit_pars_err[1]/fit_pars[0])

    b = 10**(bb)
    t = np.arange(0.0001, 20000.0, 100.)
    s = b * t**(aa)
    ax.plot(t, s, lw=2, linestyle='--', color='Black')
    ax.text(0.0005,6*10e46,'$L_{{\mathrm{{peak}}}} = 10^{{{:.2f} \pm {:.2f}}} \\tau_{{\mathrm{{{}}}}}^{{{:.2f} \pm {:.2f}}} $'.format(bb, bb_err, phase[0].title(), aa, aa_err), fontsize=22)


    for i in range(0, 18, 2):
        L0 = t*t * 10**i * 5.12e24
        ax.plot(t, L0, lw=0.9, linestyle=':', color='maroon')
    
        ax.text(600,6.7*10e35,'$T_{B,min}=10^{6}K$', rotation=16.8, fontsize=17,color='maroon')
        ax.text(600,0.7*10e38,'$T_{B,min}=10^{8}K$', rotation=16.8, fontsize=17,color='maroon')
        ax.text(0.06,7.2*10e33,'$T_{B,min}=10^{12}K$', rotation=16.8,fontsize=17,color='maroon')
        ax.text(0.06,7.2*10e35,'$T_{B,min}=10^{14}K$', rotation=16.8,fontsize=17,color='maroon')

    ax.set_yscale('log', nonposy="clip")
    ax.set_xscale('log', nonposx="clip")
    ax.set_yticks((10**24,10**26, 10**28, 10**30, 10**32,10**34,10**36, 10**38, 10**40,10**42, 10**44, 10**46))

    ax.set_xlabel('$\\tau_{{\mathrm{{{}}}}}$ $(\mathrm{{day}})$'.format(phase[0].title()), fontsize=20)
    ax.set_ylabel('$L_{\mathrm{peak}}$ $(\mathrm{erg}$ $\mathrm{s}^{-1})$',fontsize=20)

    pylab.xticks(size=20,)
    pylab.yticks(size=20,)

    ax6 = ax.twinx()
    ax6.set_ylabel('$L_{\mathrm{peak}}$ $(\mathrm{Jy}$ $\mathrm{kpc}^{2}$ $\mathrm{GHz})$', fontsize=20)
    ax6.set_ylim(5*10e-8,10e17)
    ax6.set_yscale('log')
    ax6.set_yticks((10**(-5),10**0,10**5,10**10,10**15))
    pylab.yticks(size=20,)
    ax.set_xlim([0.0005,20000])
    ax.set_ylim([5*10e20,10e46])
    pylab.subplots_adjust(left=0.08, bottom=0.12, right=0.92, top=0.915,
                wspace=0.2, hspace=0.2)

    pylab.savefig('Luminosity-timescale-{}.pdf'.format(phase))
    pylab.close()
    


