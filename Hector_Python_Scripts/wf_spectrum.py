def spec_est3(A,d1,d2,d3):
    import numpy as np
    l1,l2,l3 = A.shape
    df1 = 1./(l1*d1)
    df2 = 1./(l2*d2)
    df3 = 1./(l3*d3)
    f1Ny = 1./(2*d1)
    f2Ny = 1./(2*d2)
    f3Ny = 1./(2*d3)
    f1 = np.arange(-f1Ny,f1Ny,df1)
    f2 = np.arange(-f2Ny,f2Ny,df2)
    f3 = np.arange(0,l3/2+1)*df3
    # spectral window
    # first, the spatial window
    wx = np.matrix(np.hanning(l1))
    wy = np.matrix(np.hanning(l2))
    window_s = np.repeat(np.array(wx.T*wy),l3).reshape(l1,l2,l3)
    # now, the time window
    wt = np.hanning(l3)
    window_t = np.repeat(wt,l1*l2).reshape(l3,l2,l1).T
    Ahat = np.fft.rfftn(window_s*window_t*A)
    Aabs = 2 * (Ahat*Ahat.conjugate()) / (df1*df2*df3) / ((l1*l2*l3)**2)
    return np.fft.fftshift(Aabs.real,axes=(0,1)),f1,f2,f3
