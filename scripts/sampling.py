import numpy as np
import pylab as pl
import seaborn as sns
from gp_functions import *
import george
import os


def get_t0(gp,x,y,err):
    """

    Returns a starting point at which the flux of the object is greater than the noise.
    Simulating a 'detection'

    Params
    -------

    gp : Gaussian Process object
    x  : time data
    y  : flux data
    err: flux err

    """
    sig = np.mean(err)

    t0 = -1
    while t0 == -1:
        xrand = np.random.uniform(np.min(x),np.max(x))
        ytest = gp.sample_conditional(y,xrand)[0]
        if ytest > sig:
            t0 = xrand
    return t0

def get_features(gp,x,y,err):
    """

    Returns a starting point at which the flux of the object is greater than the noise.
    Simulating a 'detection'

    Params
    -------

    gp : Gaussian Process object
    x  : time data
    y  : flux data
    err: flux err

    """

    sec = 1/(24*60*60)
    mins = 1/(24*60)
    hr = 1/24
    day = 1
    week = 7*day
    month = 30*day
    yr = 365*day

    delta_t = np.array([0,2*sec,1*mins,30*mins,1*hr,2*hr,4*hr,6*hr,8*hr,12*hr,1*day,2*day,4*day,6*day,2*week,3*week,1*month,
                        2*month,4*month,6*month,8*month,1*yr,1.5*yr,2*yr])

    t0 = get_t0(gp,x,y,err)

    feature_ts = [ (t0 + dt) for dt in delta_t if (t0 + dt) < np.max(x)]
    fluxes = gp.sample_conditional(y,feature_ts)

    f0 = fluxes[0]
    feature_vec = [f0 - f for f in fluxes[1:]]
    feature_vec = np.array(feature_vec)

    pad_len = len(delta_t) - len(feature_vec)
    feature_vec = np.pad(feature_vec,(0,pad_len),'constant', constant_values=(-999,-999))

    return feature_vec


root = "../data/"
feature_root = '../features.test/'
allfiles = []
for path, subdirs, files in os.walk(root):
    for filename in files:
        f = os.path.join(path, filename)
        allfiles.append(f)


number_of_samples = 100

for i in range(len(allfiles)):
    file = allfiles[i]

    x,y,err = loadfile(file)
    gp = get_gp(file)

    for a in range(number_of_samples):
        feature_vec = get_features(gp,x,y,err)
        with open(feature_root+file[8:]+'.txt','a+') as f:
            for feature in feature_vec:
                f.write("%5.10f," %feature)
            f.write('\n')

    print(i,file)
