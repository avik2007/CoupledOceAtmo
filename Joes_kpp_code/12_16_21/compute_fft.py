import math
import numpy as np
import numpy.fft
import taper_functions as tf
import stdio
from repmat import *
from scipy import signal

def compute_fft_m(f):

    return np.fft.fft(f,axis=2); 

def compute_fft_omega(f):

    tf.taper_function_5D(f,np.array([0,0,0,1,0]));
    return np.fft.fft(f,axis=3); 

def compute_highpass_omega(f,oti):

    (nx,ny,nz,nt,nf) = f.shape;

    norm = np.sum(f*f);

    dftnt = math.floor(nt/2)+1;
    Wn = oti/dftnt;
    b, a = signal.butter(4, Wn, 'high', analog=False)
    f = signal.filtfilt(b, a, f,axis=3)

    # f = np.fft.fft(f,axis=3); 
    # spec_dist_omega = np.reshape(np.linspace(1,nt,nt),(1,1,1,nt,1));
    # spec_dist_omega = np.abs(np.mod(spec_dist_omega-2+dftnt,nt)-dftnt+1)+1;
    # spec_dist_omega = repmat(spec_dist_omega,(nx,ny,nz,1,nf));
    # filt = (spec_dist_omega>oti);
    # f = f*filt;
    # f = np.fft.ifft(f,axis=3);
    # if (np.sum(np.abs(np.imag(f)))>(0.001*np.sum(np.abs(f)))):
    #     raise RuntimeError('imaginary part of high-pass field is not small!');

    print('high-pass changes field norm by ' + str((np.sum(f*f)/norm)**(0.5)))
    return np.real(f);

def compute_write_m_c8(f,fname):
    
    f = compute_fft_m(f);
    stdio.write_field_c8(f,fname);

def trim_mixed_layer(f,dz,trim_mls):

    # trim the mixed layer
    nf = f.shape[3];
    for fi in range(0,nf):
        k_mld = math.ceil(trim_mls[fi]/dz);
        f[:,:,0:k_mld,:,fi] = 0*f[:,:,0:k_mld,:,fi];

    return f













