#!/usr/bin/env python
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pylab
from scipy.optimize import curve_fit


def process_datafile(filename):
    R = open('results/measurements/{}_rise_parameters.txt'.format(id.split('.txt')[0]), 'w')
    D = open('results/measurements/{}_decline_parameters.txt'.format(id.split('.txt')[0]), 'w')
    D.write('#Name Slope(1/day) Slope_err(1/day) Chi2 Fmax(Jy) NegativePts CountR(NR) Match'+'\n')   
    R.write('#Name Slope(1/day) Slope_err(1/day) Chi2 Fmax(Jy) NegativePts CountD(ND) Match'+'\n')  
    data = np.loadtxt(filename, usecols=(0, 1, 2), dtype=float, unpack=True)
    t_all, f_all, f_all_err = data
   
    countR = 0
    countD = 0
    match = 1
    #Find index of the maximum flux (fmax_ind)
    fmax_ind = np.argmax(f_all)
    
    # Shift time values so that fmax is at t=0.
    t_all_shifted = t_all - t_all[fmax_ind]

    # set range of times for rise phase (tr) (to peak flux time) and
    # decline (td) (from peak flux time)
    rise_idx = t_all_shifted <= 0
    t_rise = t_all_shifted[rise_idx]
    f_rise = f_all[rise_idx]
    f_rise_err = f_all_err[rise_idx]

    decline_idx = t_all_shifted >= 0
    t_dec = t_all_shifted[decline_idx]
    f_dec = f_all[decline_idx]
    f_dec_err = f_all_err[decline_idx]

    fmax = f_all[fmax_ind]

    fig = plt.figure()
    ax =  fig.add_subplot(111)
    fig.set_size_inches(7.,4.)
    ax.errorbar(t_dec, np.log(f_dec), xerr=None, yerr=f_dec_err/f_dec,  linestyle='None', ecolor="Grey", elinewidth=1)
    ax.errorbar(t_rise, np.log(f_rise), xerr=None, yerr=f_rise_err/f_rise,  linestyle='None', ecolor="Grey", elinewidth=1)
    ax.plot(t_all_shifted, np.log(f_all), 'ko')
    ax.plot(t_dec, np.log(f_dec), marker='o', markerfacecolor= 'Grey', markeredgecolor='None', linestyle='None')
    ax.plot(t_rise, np.log(f_rise), marker='o', markerfacecolor= 'Grey', markeredgecolor='None', linestyle='None')

    values_rise = []
    values_decline = []
       
    try:
        t_min_idx = np.argmin(t_rise)
        t_min = t_rise[t_min_idx]
        f_t_min = f_rise[t_min_idx]

        pos = f_rise>0
        neg = f_rise<=0
        f_neg = f_rise[neg] 
        negnb = f_neg.shape[0]
        f_rise = f_rise[pos]
        f_rise_err = f_rise_err[pos]
        t_rise = t_rise[pos]
           

        rise_fit_pars, rise_fit_cov = curve_fit(log_lin, t_rise, np.log(f_rise), sigma=f_rise_err/f_rise, maxfev=800)
        rise_fit_pars_err = np.sqrt(np.diag(rise_fit_cov)) 

        
#calc reduced chi squared chi2r for rise phase:
        chi2 = 0
        for i in range(0, t_rise.shape[0]):
            R_i = log_lin(t_rise[i], rise_fit_pars[0], rise_fit_pars[1])
            d_i = np.log(f_rise[i])
            var_i = (f_rise_err[i]*f_rise_err[i])/(f_rise[i] * f_rise[i])
            chi2_i = ((d_i - R_i)*(d_i - R_i))/var_i
            chi2 += chi2_i
        N = t_rise.shape[0]-2.0-1.0    
        chi2r = chi2/N    
        
        countR = countR + 1
        ax.plot(t_rise, log_lin(t_rise, *rise_fit_pars), 'b-', 
        label='$\\tau^{{-1}}_{{\mathrm{{R}}}}={:.3f}\pm{:.3f}$ \n $\chi^{{2}}_{{\mathrm{{red}}}}={:.1f}$'.format(rise_fit_pars[0], rise_fit_pars_err[0], chi2r))
        plt.legend(numpoints=2, loc=0, prop={'size':18})


        slopeR = rise_fit_pars[0]
        bR  = rise_fit_pars[1]
        slopeerrR = rise_fit_pars_err[0]
        slopeR = np.float(slopeR)
        values_rise.append(str(slopeR))
        slopeerrR = np.float(slopeerrR)
        values_rise.append(str(slopeerrR)) 
        chi2r = np.float(chi2r)
        values_rise.append(str(chi2r))
        fmax = np.float(fmax)
        values_rise.append(str(fmax))
        negnb = np.float(negnb)
        values_rise.append(str(negnb))
        values_rise.append(str(countR))
        values_rise.append(str(match))

        print "Rise params fit:", rise_fit_pars, 'Rise params fit err:', rise_fit_pars_err, 'Peak', fmax        
        R.write(id.split('.txt')[0]+'\t'+'\t'.join(values_rise) + '\n')
                     



    except Exception as e:
        print "Rise fit failed:"
        print e
        pass

    try:
        t_max_idx = np.argmax(t_dec)
        t_max = t_dec[t_max_idx]
        f_t_max = f_dec[t_max_idx]
        pos = f_dec>0
        neg = f_dec<=0
        f_neg = f_dec[neg] 
        negnb = f_neg.shape[0]
        f_dec = f_dec[pos]
        f_dec_err = f_dec_err[pos]
        t_dec = t_dec[pos]
        print 'dec t shape', t_dec.shape[0], 'dec f shape', np.log(f_dec).shape[0], 'dec f err shape', f_dec_err.shape[0], f_dec.shape[0]
        decline_fit_pars, decline_fit_cov = curve_fit(log_lin, t_dec, np.log(f_dec), sigma=f_dec_err/f_dec, maxfev=800)
        decline_fit_pars_err = np.sqrt(np.diag(decline_fit_cov))
        
                
        #calc reduced chi squared chi2r for decline phase Dchi2r:
        chi2 = 0             
        for i in range(0, t_dec.shape[0]):
            D_i = log_lin(t_dec[i], decline_fit_pars[0], decline_fit_pars[1])
            d_i = np.log(f_dec[i])
            var_i = (f_dec_err[i]*f_dec_err[i])/(f_dec[i]*f_dec[i])
            chi2_i = ((d_i - D_i)*(d_i - D_i))/var_i
            chi2 += chi2_i
        N = t_dec.shape[0]-2.0-1.0    
        chi2r = chi2/N 
        
        countD = countD + 1
        ax.plot(t_dec, log_lin(t_dec, *decline_fit_pars), 'r-', label='$\\tau^{{-1}}_{{\mathrm{{D}}}}={:.3f}\pm{:.3f}$ \n $\chi^{{2}}_{{\mathrm{{red}}}}={:.1f}$'.format(decline_fit_pars[0], decline_fit_pars_err[0], chi2r)) 
 
        plt.legend(numpoints=2, loc=0, prop={'size':18})

        slopeD = decline_fit_pars[0]
        slopeerrD = decline_fit_pars_err[0]
        slopeD = np.float(slopeD)
        values_decline.append(str(slopeD))
        slopeerrD = np.float(slopeerrD)
        values_decline.append(str(slopeerrD))
        chi2r = np.float(chi2r)
        values_decline.append(str(chi2r))                
        fmax = np.float(fmax)
        values_decline.append(str(fmax))
        negnb = np.float(negnb)
        values_decline.append(str(negnb))
        values_decline.append(str(countD))
        values_decline.append(str(match))

        D.write(id.split('.txt')[0]+'\t'+'\t'.join(values_decline) + '\n')  
        print "Dec params fit:", decline_fit_pars, 'Dec params fit err:', decline_fit_pars_err, 'Peak', fmax

    except Exception as e:
        print "Decline fit failed:"
        print e
        pass


    plt.xlabel('time (days)',fontsize=17)
    plt.ylabel('ln flux (Jy)',fontsize=17)
    pylab.xticks(size=16,)
    pylab.yticks(size=16,)

    #plt.ylim([np.log(f_t_min-0.01), np.log(fmax+0.01)]) 
    ax.yaxis.grid(color='gray', linestyle=':')
    ax.xaxis.grid(color='gray', linestyle=':') 
    plt.tight_layout()  
#    plt.show()      
    plt.savefig('results/{}.png'.format(id.split('.txt')[0]))
    
    R.close()
    D.close()


def log_lin(tdata, slope, b):
    return slope * tdata + b



files = glob.glob('*.txt')
if not os.path.isdir('results'):
    os.mkdir('results')
if not os.path.isdir('results/measurements'):
    os.mkdir('results/measurements')    
for id in files:
    print id
    process_datafile(id)

#if __name__ == '__main__':
    # #### The command-line argument parser
#    parser = optparse.OptionParser(description='Exp fit to the data')
    # parser.add_option('--fln', dest="filename",
    #                   help='File with the light curve data to be processed')
#    (options, args) = parser.parse_args()
#    files = glob.glob('*.txt')
#    filename = args[0]
#    process_datafile(filename)




