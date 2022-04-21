import numpy as np
from numpy import pi
from scipy import signal
try:
    import mkl
    np.use_fastnumpy = True
except ImportError:
    pass

class coherence(object):
    """ A class that represents a single realization of the 
        one-dimensional coherence of two given fields """
    
    def __init__(self,A,B,dt):
        
        self.A = A
        self.B = B
        self.dt = dt
        self.n = A.size

        win = np.hanning(self.n)
        win = (self.n/(win**2).sum())*win

        self.A *= win
        self.B *= win

        # test if n is even
        if (self.n%2):
            self.neven = False
        else:
            self.neven = True

        # calculate frequencies
        self.calc_freq()

        # calculate coherence
        self.calc_coh()

    def calc_freq(self):
        """ calculate array of spectral variable (frequency or
                wavenumber) in cycles per unit of L """

        self.df = 1./((self.n-1)*self.dt)

        if self.neven:
            self.f = self.df*np.arange(self.n/2+1)
        else:
            self.f = self.df*np.arange( (self.n-1)/2.  + 1 )

    def calc_coh(self):
        """ compute the 1d coherence """
        
        self.Ah = np.fft.rfft(self.A)
        self.Bh = np.fft.rfft(self.B)
        
        self.Asp = 2.*(self.Ah*self.Ah.conj()).real
        self.Bsp = 2.*(self.Bh*self.Bh.conj()).real
        self.AB  = ((np.conj(self.Ah)*self.Bh).real)

        ### Coherence
        self.cohsq = np.abs(self.AB)**2 / ((self.Asp)*(self.Bsp))
        self.phase = np.angle(self.AB)

        ## zero-padding
        self.cohsq = np.fft.fftshift(self.cohsq,axes=0)
        self.Asp = np.fft.fftshift(self.Asp/self.df/self.n**2,axes=0)
        self.Bsp = np.fft.fftshift(self.Bsp/self.df/self.n**2,axes=0)
