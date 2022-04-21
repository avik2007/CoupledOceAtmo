def rms(x):
    """
    Root-mean-square
    """
    import numpy as np
    return np.sqrt(np.mean(x**2,axis=None))


def histogram2D(NgridA,NgridB,vminA,vmaxA,vminB,vmaxB,A,B):
    """
    Joint-Probability-density-function of A and B
    """
    import numpy as np
    grid_A = np.linspace(vminA,vmaxA,NgridA+1)
    grid_B = np.linspace(vminB,vmaxB,NgridB+1)
	
    H,xbins,ybins = np.histogram2d(A.flatten(),B.flatten(),
	            bins=(grid_A,grid_B))

    ### Normalizing H by max(H)
    H /= np.max(H)
    
    return grid_A,grid_B,H


def bining(n_bins,A,B):
    """
    Binning A as a function of B
    """
    from scipy.stats import binned_statistic
    from scipy.stats.stats import pearsonr
    bin_centers,a,b=binned_statistic(A.flatten(),B.flatten(),
	           statistic='mean',bins=n_bins)
    bin_std,a,b=binned_statistic(A.flatten(),B.flatten(),statistic='std',
	        bins=n_bins)	
    return bin_centers,bin_std,a,b 
