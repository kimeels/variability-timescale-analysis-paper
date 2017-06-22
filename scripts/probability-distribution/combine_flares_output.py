import numpy as np
import math
import glob, os
import csv
import json
import argparse


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--jsonfilespath', action='store', type=str, help='Directory in which output files (*_flares.json) from flare finder are stored.', default='results/*/')
    parser.add_argument('-t', '--targetsfilepath', action='store', type=str, help='Directory in which target-distances-and-class.txt file is stored.', default='')
    parser.add_argument('-s', '--step', action='store', type=float, help='Logarithmic timestep over which to average the probability tables.', default=0.5)
    args = parser.parse_args()



## Combine all the separate files containing parameters for every lightcurve and split into rise and decline files.
files = sorted(glob.glob(args.jsonfilespath+'/*_flares.json'))
f_r = open('all_parameters_rise.csv', 'w')
f_r.write('#Name	Slope(1/day)	Slope_err(1/day)	Chi2	Fmax(Jy)'+'\n') 
f_d = open('all_parameters_decline.csv', 'w')
f_d.write('#Name	Slope(1/day)	Slope_err(1/day)	Chi2	Fmax(Jy)'+'\n') 
for fln in files:
    a = json.loads(open(fln).read())
    for i in a:
	rise_val=[]
        dec_val=[]
	name =  fln.split('/')[-1].split('_flares.')[0]
        slope_r = i['rise_slope']
        slope_err_r = i['rise_slope_err']
        chi2_r = i['rise_reduced_chi_sq']
        Fmax_r = i['peak_flux']
        slope_d = i['decay_slope']
        slope_err_d = i['decay_slope_err']
        chi2_d = i['decay_reduced_chi_sq']
        Fmax = i['peak_flux']
	rise_val.append(name)
	rise_val.append(str(slope_r))
	rise_val.append(str(slope_err_r))
        rise_val.append(str(chi2_r))
        rise_val.append(str(Fmax))
        dec_val.append(name)
	dec_val.append(str(slope_d))
	dec_val.append(str(slope_err_d))
        dec_val.append(str(chi2_d))
        dec_val.append(str(Fmax))
        
	f_r.write('\t'.join(rise_val)+'\n')
        f_d.write('\t'.join(dec_val)+'\n')

f_r.close()
f_d.close()
# Assign distances and classes to data in the all_parameters_rise/decline.csv file, based on the target-distances-and-class.txt file.
for phase in ('rise', 'decline'):
    MAXCOLS = 5
    parameters = [[] for _ in xrange(MAXCOLS)]

    with open('all_parameters_{}.csv'.format(phase), 'rb') as input:
        linesp = input.readlines()[1:]
        for row in csv.reader(linesp, delimiter='\t'):
            for i in xrange(MAXCOLS):
                parameters[i].append(row[i] if i < len(row) else '')


    MAXCOLS = 5
    distances = [[] for _ in xrange(MAXCOLS)]

    with open(args.targetsfilepath+'target-distances-and-class.txt', 'rb') as input:
        linesd = input.readlines()[1:]
        for row in csv.reader(linesd, delimiter='\t'):
            for i in xrange(MAXCOLS):
                distances[i].append(row[i] if i < len(row) else '')

    c = open('complete_{}'.format(phase), 'w')
    c.write('#Name Slope(day) Slope_err(day) Chi2 Fmax(Jy) Distance(Mpc) Frequency(GHz) LUMINOSITY(erg/s) Type LogSlope ErrLogSlope LogLum'+'\n')
    convert = 4*math.pi*((3*(10**6)*(10**18))**2)*(10**(-23))*(10**9)


    ### Exclude fits to rise/decline with 3 datapoints or less
    for i in range(0, len(linesp)):
        for j in range(0,len(linesd)):
            if parameters[3][i] == 'inf':
                continue
            if parameters[2][i] == 'inf':
                continue
            if parameters[3][i] == 'None':
                continue
            if phase == 'rise':
                if np.float(parameters[1][i])<0:
                    continue
            else:
                if np.float(parameters[1][i])>0:
                    continue

            if  parameters[0][i] == distances[0][j]:
                values = []
                name = parameters[0][i]
                values.append(name)
            
                slope = parameters[1][i]
                slope_inv = 1.0/np.float(slope)
                values.append(str(slope_inv))
                slope_err = parameters[2][i]
                slope_err = np.float(slope_err)/(np.float(slope)*np.float(slope))
                values.append(str(slope_err))
                chi2 = parameters[3][i]
                values.append(chi2)
                fmax = parameters[4][i]
                fm = np.float(fmax)
                values.append(fmax)
                dist = distances[1][j]
                d = np.float(dist)
                values.append(dist)
                frequency = distances[3][j]
                freq = np.float(frequency)
                values.append(frequency)
                lum = convert*d*d*fm*freq
                values.append(str(lum))
                tp = distances[2][j]
                values.append(tp)
                logslope = np.log10(abs(slope_inv))
                values.append(str(logslope))
                errlogslope = abs(slope_err)/(abs(slope_inv)*np.log(10))
                values.append(str(errlogslope))
                loglum = np.log10(lum)
                values.append(str(loglum)) 
                c.write('\t'.join(values) + '\n')
            
    c.close()    


    os.remove('all_parameters_{}.csv'.format(phase))

