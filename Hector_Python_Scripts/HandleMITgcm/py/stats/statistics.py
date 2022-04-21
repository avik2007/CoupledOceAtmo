import numpy as np
from scipy.stats import binned_statistic
from scipy import stats

def binned(X,Y,bins):
    centers,_,_ = binned_statistic(X,X,statistic='mean',bins=bins)
    avg,_,_ = binned_statistic(X,Y,statistic='mean',bins=bins)
    stdevs,_,_ = binned_statistic(X,Y,statistic='std',bins=bins)
    return centers,avg,stdevs

def linearRG(X,Y):
    idx=np.isfinite(X) & np.isfinite(Y)
    slope,intercept,r_value,p_value,std_err=stats.linregress(X[idx],Y[idx])
    return slope,intercept,r_value,p_value,std_err
