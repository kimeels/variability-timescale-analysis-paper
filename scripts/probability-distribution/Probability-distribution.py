import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib
import glob, os
import csv
import argparse
from scipy import *
import scipy.optimize
from scipy.optimize import curve_fit
import shutil
import matplotlib.gridspec as gridspec
from scipy.odr import *


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--complete_measurements', action='store', type=str, help='Directory in which *complete_rise/decline* files are stored.', default='')
    parser.add_argument('-r', '--sky_densities', action='store', type=str, help='Directory in which sky-densities.txt file is stored.', default='')
    parser.add_argument('-s', '--step', action='store', type=float, help='Logarithmic timestep over which to average the probability tables.', default=0.5)
    args = parser.parse_args()


matplotlib.style.use('classic')
matplotlib.rcParams['patch.force_edgecolor'] = True
matplotlib.rcParams['patch.facecolor'] = 'b'

for phase in ('rise', 'decline'):
    types = np.loadtxt(args.sky_densities+'sky-densities.txt', dtype=str, skiprows=2, usecols=(0,), unpack=True)
    rates = np.loadtxt(args.sky_densities+'sky-densities.txt', dtype=float, skiprows=2, usecols=(1,), unpack=True)	

    classes = np.loadtxt(args.complete_measurements+'complete_{}'.format(phase), dtype=str, skiprows=1, usecols=(8,), unpack=True)
    data = np.loadtxt(args.complete_measurements+'complete_{}'.format(phase), dtype=float, skiprows=1, usecols = (1,2), unpack=True)
    slope, err = abs(data)

    pars = open('parameters_v1_{}'.format(phase), 'w')
    pars.write('#Type Slope_mean(day) Slope_std(day) Type_rate(deg-2)'+'\n') 
    
    for tp, rate in zip(types, rates):
        # Group different subclasses into one general class, such as Blazar, Qso, AGN -> AGN.
        for i in range(0, classes.shape[0]):
            if classes[i] == 'BLAZAR':
                classes[i] = classes[i].replace('BLAZAR', 'AGN')
            if classes[i] == 'QSO':
                classes[i] = classes[i].replace('QSO', 'AGN')
            if classes[i] == 'NSXRB':
                classes[i] = classes[i].replace('NSXRB', 'XRB')
            if classes[i] == 'ULX':
                classes[i] = classes[i].replace('ULX', 'XRB')
            if classes[i] == 'BHXRB':
                classes[i] = classes[i].replace('BHXRB', 'XRB')
            if classes[i] == 'BinaryPulsars':
                classes[i] = classes[i].replace('BinaryPulsars', 'XRB')
                           
        ## Group measured rise times by classes
        collected_slopes = slope[np.where(classes==tp)]
        
        ## Skip cases where there are no measurements for a given class
        if collected_slopes.shape[0] == 0:
            continue

        ## Calculate log10 of the slopes, then mean and std
        collected_slopes_log = np.log10(collected_slopes)
        parameters = []
        parameters.append(str(tp))
        mean = np.mean(collected_slopes_log)
        parameters.append(str(mean))
        sigma = np.std(collected_slopes_log)
        parameters.append(str(sigma))
        parameters.append(str(rate))
        pars.write('\t'.join(parameters) + '\n')

    
    pars.close()    
        
    ## For classes where sigma==0, assign an estimated sigma value equal to the mean value of the std calculated for other classes.
    data = np.loadtxt('parameters_v1_{}'.format(phase), dtype=float, usecols=(1,2,3), unpack=True)
    mean, sigma, rate = data
    mean_sigma = np.mean(sigma[np.where(sigma>0)])
    cls = np.loadtxt('parameters_v1_{}'.format(phase), dtype = str, usecols=(0,), unpack=True)
    f = open('gaussian-model-parameters_{}'.format(phase), 'w')
    f.write('#Type Slope_mean(day) Slope_std(day) Type_rate(deg-2)'+'\n') 
    sigma[sigma==0]=mean_sigma 
    for c, m, s, r in zip(cls, mean, sigma, rate):
        writelist=[]
        writelist.append(c)
        writelist.append(str(m))
        writelist.append(str(s))
        writelist.append(str(r))
        f.write('\t'.join(writelist) + '\n')
    f.close()

   ## Remove parameters_v1_rise/decline file which is no longer needed
    os.remove('parameters_v1_{}'.format(phase))
       
    ## To each class assign gaussian distribution based on the parameters from 'parameters_rise/decline' file, and, correct for the estimtaed sky densities of objects.

    Exp = math.exp
    Pi = math.pi
    sq = math.sqrt

    cls = np.loadtxt('gaussian-model-parameters_{}'.format(phase), dtype = str, usecols = (0,), unpack = True)
    mean = np.loadtxt('gaussian-model-parameters_{}'.format(phase), usecols = (1,), dtype=float, unpack = True)
    std = np.loadtxt('gaussian-model-parameters_{}'.format(phase), usecols = (2,), dtype=float, unpack = True)
    rate = np.loadtxt('gaussian-model-parameters_{}'.format(phase), usecols = (3,), dtype=float, unpack = True)
 
    f = open('probabilities_{}.txt'.format(phase), 'w')
    f.write('x' + '\t')
    for i in range(0, cls.shape[0]):
        f.write('{}'.format(cls[i]) + '\t')
    f.write('sum')
   
    y = np.arange(-4.0,4.,0.005)
          
    for x in y:
        writelist=[]
        x = np.float(x)
        writelist.append(str(x))
        sum = 0
        for c, m, s, r in zip(cls, mean, std, rate):
            prob  = (r/(sq(2*Pi)*s))*(Exp((-(x-m)**2)/(2*(s**2))))
            prob   = np.float(prob)
            sum += prob
            writelist.append(str(prob))
        writelist.append(str(sum))
        
        f.write( '\n'+'\t'.join(writelist) + '\n')

    f.close()


    # Normalise probability distributions to 100%
    
    probabilities = np.loadtxt('probabilities_{}.txt'.format(phase), dtype = float, skiprows=1, unpack=True)
    cls = np.loadtxt('gaussian-model-parameters_{}'.format(phase), dtype = str, usecols = (0,), unpack = True)

    x = probabilities[0] 
    sum = probabilities[probabilities.shape[0]-1]
        
    p = open('probability_distribution_table_{}'.format(phase), 'w')
    p.write('x' + '\t')
    for i in range(0, cls.shape[0]):
        p.write('{}'.format(cls[i]) + '\t')
    p.write('\n')

    for i in range(0, x.shape[0]):
        prob_new = []
        prob_new.append(str(x[i]))
        for cls in range(1, probabilities.shape[0]-1):
            prob_cls = probabilities[cls]
            cls_new = prob_cls[i] * 100. / sum[i]
            prob_new.append(str(cls_new))
        p.write( '\t'.join(prob_new)+'\n')
    p.close()        


   ## Remove probabilities_/decline file with non-normalised Gaussian distributions, which is no longer needed
    os.remove('probabilities_{}.txt'.format(phase))

## Make plots of the probability distribution for rise timescales corrected for sky rates and normalized to 100%

for phase in ('rise', 'decline'):
    prob = np.loadtxt('probability_distribution_table_{}'.format(phase), skiprows=1, unpack=True)
    cls = np.loadtxt('gaussian-model-parameters_{}'.format(phase), dtype = str, usecols = (0,), unpack = True)


    colors = ('#FA1919', '#19FB52', '#1930FA', '#A631FF', '#FAEF19', 'Maroon', '#0B6B0E', '#6B4C0B', '#FD3F95','#0F9DC9','Orange', 'Turquoise', 'OrangeRed')
    types = ('AGN', 'SN', 'XRB', 'Magnetar', 'GRB', 'ULX',  'Nova', 'RSCVn', 'Flare star', 'Algol', 'Mag CV', 'TDE', 'DN')
    msize = (8., 9., 10., 8., 9., 9., 9., 10., 7., 10., 9., 9., 9.)
    line = ('-', '-', '-', '--', '-', '--', '-', '-', '-', '-', '-', '-', '--') 
    markertype = ('o', '^', (4,0,180), (5,1,0), (4,1,0), (4,0,120), 'v', 'H', 'D', (7,1,0), (5,1,0), (11,1,30), (4,1,45))

    fig = plt.figure(figsize=(12.5,8))
    ax  = fig.add_subplot(111)
    for i in range(0, cls.shape[0]):
        if cls[i] == 'Flare_Stars':
            cls[i] = cls[i].replace('Flare_Stars', 'Flare star')
        if cls[i] == 'magCV':
            cls[i] = cls[i].replace('magCV', 'Mag CV')
                           
    for i in range(0, cls.shape[0]):
        for j in range(0, len(types)):
            if cls[i] == types[j]:
                plot  = ax.plot(prob[0],prob[i+1], color=str(colors[j]), linestyle=line[j], linewidth=2., marker=markertype[j], markevery=100, markersize = msize[j], label=str(types[j]))

    ax.legend(numpoints=1, loc=0)
    ax.set_xlabel('{} rate of the event $\mathrm{{log}}\, \\tau_{{\mathrm{{{}}}}}\,(\mathrm{{day}})$'.format(phase.title(), phase[0].title()), fontsize=16)
    ax.set_ylabel('Probability distribution (%)',fontsize=16)


    ax.set_xlim([-3.1,3.4])
    ax.set_ylim([0,105])
    plt.xticks(size=16,)
    plt.yticks(size=16,)
    ax.yaxis.grid(color='gray', linestyle=':')
    ax.xaxis.grid(color='gray', linestyle=':')

    plt.tight_layout()
    plt.savefig('Probability-distribution-{}.pdf'.format(phase))    
    plt.close()


def Round_To_n(x, n):
    return round(x, -int(np.floor(np.sign(x) * np.log10(abs(x)))) + n)


for id in ('rise', 'decline'):
    prob = np.loadtxt('probability_distribution_table_{}'.format(id), dtype=float, skiprows=1, unpack=True)
    cls = np.loadtxt('gaussian-model-parameters_{}'.format(id), dtype = str, usecols = (0,), unpack = True)
            
    p = open('prob_distribution_table_averaged_{}_{}'.format(args.step, id), 'w')
    p.write('x1  x2' + '\t')
    for i in range(0, cls.shape[0]):
        p.write('{}'.format(cls[i]) + '\t')
    p.write('\n')
    
    x = prob[0]
    for i in range(0,int(x.shape[0]), int(args.step*200.)):
        prob_av = []
        prob_av.append(str(x[i]))
        prob_av.append(str(x[i]+args.step))
        for cls in range(1, prob.shape[0]):
            prob_cls = prob[cls]
            x_range_idx = np.where(np.logical_and(x>=x[i], x<= x[i]+args.step)) 
            prob_cls_new = prob_cls[x_range_idx]  
            mean = np.mean(prob_cls_new)
            if mean==0:
                mean = 0.0
            else:
                mean = Round_To_n(mean, 1)
            prob_av.append(str(mean))
        
        p.write( '\t \t'.join(prob_av)+ '\n ')
    p.close()
