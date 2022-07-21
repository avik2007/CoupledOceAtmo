import sys
import struct
import warnings
import numpy as np 

def rdmds(fname,iters,dims,fldname=""):
    nt = len(iters);
    field = np.empty(dims + [nt])
    if (nt>0):
        for t in range(0,nt):
            filename = fname + '.' + str(iters[t]).rjust(10,'0') + '.data';
            fid = open(filename,'rb')
            f = np.fromfile(fid, dtype=np.dtype('>f4'),count=np.prod(dims))
            if (len(dims)==3):
                field[:,:,:,t] = switch_row_column_major_Hector(f,dims);#switch_row_column_major(f,dims);
            elif (len(dims)==2):
                field[:,:,t] = switch_row_column_major_Hector(f,dims);#switch_row_column_major(f,dims);
    else:
        filename = fname + '.data';
        fid = open(filename,'rb')
        f = np.fromfile(fid, dtype=np.dtype('>f4'),count=np.prod(dims))
        if (len(dims)==3):
            field[:,:,:] = switch_row_column_major_Hector(f,dims);#switch_row_column_major(f,dims);#
        elif (len(dims)==2):
            field[:,:] = switch_row_column_major_Hector(f,dims);#switch_row_column_major(f,dims);    
        else:
            field = f;
    
    if not (len(fldname)==0):
        print('max ' + fldname + ' = ' + str(np.max(field)));
        print('min ' + fldname + ' = ' + str(np.min(field)));

    return field

def read_bin(fname,dims):
    try:
        fid=open(fname + '.bin','rb');
    except:
        fid=open(fname,'rb');
    field = np.fromfile(fid, dtype=np.dtype('>f4'),count=np.prod(dims))
    field = switch_row_column_major_Hector(field,dims);#switch_row_column_major(field,dims);#

    return field;

# f4's are currently saved as 4D fields - updated at some later time to 5D
def read_field(fname,switch_rc_major=False):
    ## Read meta file
    fid = open(fname + '.meta');
    dims = np.fromfile(fid, dtype=np.dtype('uint32')); # there should be 4 unsigned integers
    nx = dims[0]; ny = dims[1]; nz = dims[2]; nt = dims[3]; nf=1;
    if (dims.size==5):
        nf = dims[4];

    ## Read data file
    fid = open(fname + '.data');
    field = np.fromfile(fid, dtype=np.dtype('f4'),count=nx*ny*nz*nt*nf);
    if (switch_rc_major):
        if (dims.size==4):
            field = switch_row_column_major(field,[nx,ny,nz,nt]);
        elif (dims.size==5):
            field = switch_row_column_major(field,[nx,ny,nz,nt,nf]);
    else:
        field = np.reshape(field,dims);

    ## Convert NaNs
    field[field>(np.finfo(np.dtype('f4')).max**0.5)] = np.NaN;

    return field;

def write_field(f,fname):

    if (f.ndim==1):
        f = np.reshape(f.copy(),(f.shape[0],1,1,1));
    elif (f.ndim==2):
        f = np.reshape(f.copy(),(f.shape[0],f.shape[1],1,1));
    elif (f.ndim==3):
        f = np.reshape(f.copy(),(f.shape[0],f.shape[1],f.shape[2],1));
    elif (f.ndim==4):
        f = np.reshape(f.copy(),(f.shape[0],f.shape[1],f.shape[2],f.shape[3]));
    elif (f.ndim==5):
        f = np.reshape(f.copy(),(f.shape[0],f.shape[1],f.shape[2],f.shape[3],f.shape[4]));

    ## Initialize
    fmeta = np.asarray(f.shape,dtype=np.dtype('uint32'));
    if (f.ndim==5):
        fmeta = np.concatenate((fmeta,np.ones([5-fmeta.size]).astype(np.dtype('uint32'))),axis=0)
    else:
        fmeta = np.concatenate((fmeta,np.ones([4-fmeta.size]).astype(np.dtype('uint32'))),axis=0)

    ## Check for NaNs
    if (np.isnan(f).any()):
        warnings.warn('converting ' + str(np.sum(np.isnan(f))) + ' NaNs to realmax.');
        f[isnan(f)] = np.finfo(np.dtype('f4')).max;

    ## Write data file
    fida = open(fname + '.data','bw');
    f.astype(np.dtype('f4')).tofile(fida,'','f4');

    ## Write meta file
    fida = open(fname + '.meta','bw');
    fmeta.tofile(fida,'','uint32')

# c8's are saved as 5D fields
def read_field_c8(fname,switch_rc_major=False):
    ## Read meta file
    fid = open(fname + '.c8.meta');
    dims = np.fromfile(fid, dtype=np.dtype('uint32')); # there should be 5 unsigned integers
    nx = dims[0]; ny = dims[1]; nz = dims[2]; nt = dims[3]; nf = dims[4];

    ## Read data file
    fid = open(fname + '.c8.data');
    field = np.fromfile(fid, dtype=np.dtype('c8'),count=nx*ny*nz*nt*nf);
    if (switch_rc_major):
        field = switch_row_column_major(field,[nx,ny,nz,nt,nf]);
    else:
        field = np.reshape(field,dims);

    ## Convert NaNs
    field[np.abs(field)>(np.finfo(np.dtype('f4')).max**0.5)] = np.NaN;

    return field;

def write_field_c8(f,fname):

    if (f.ndim==1):
        f = np.reshape(f.copy(),(f.shape[0],1,1,1,1));
    elif (f.ndim==2):
        f = np.reshape(f.copy(),(f.shape[0],f.shape[1],1,1,1));
    elif (f.ndim==3):
        f = np.reshape(f.copy(),(f.shape[0],f.shape[1],f.shape[2],1,1));
    elif (f.ndim==4):
        f = np.reshape(f.copy(),(f.shape[0],f.shape[1],f.shape[2],f.shape[3],1));
    elif (f.ndim==5):
        f = np.reshape(f.copy(),(f.shape[0],f.shape[1],f.shape[2],f.shape[3],f.shape[4]));

    ## Initialize
    fmeta = np.asarray(f.shape,dtype=np.dtype('uint32'));
    fmeta = np.concatenate((fmeta,np.ones([5-fmeta.size]).astype(np.dtype('uint32'))),axis=0)

    ## Check for NaNs
    if (np.isnan(f).any()):
        warnings.warn('converting ' + str(np.sum(np.isnan(f))) + ' NaNs to realmax.');
        f[isnan(f)] = np.finfo(np.dtype('f4')).max;

    ## Write data file
    fida = open(fname + '.c8.data','bw');
    f.astype(np.dtype('c8')).tofile(fida,'','c8');

    ## Write meta file
    fida = open(fname + '.c8.meta','bw');
    fmeta.tofile(fida,'','uint32')

def switch_row_column_major(field,dims):

    Pfield = np.zeros(dims);
    ndims = len(dims);

    if (ndims==5):
        for f in range(0,dims[4]):
            for t in range(0,dims[3]):
                for k in range(0,dims[2]):
                    for j in range(0,dims[1]):
                        i1 = f*dims[3]*dims[2]*dims[1]*dims[0] + t*dims[2]*dims[1]*dims[0] + k*dims[1]*dims[0] + j*dims[0]
                        i2  = i1+dims[0];
                        Pfield[:,j,k,t,f] = field[i1:i2];

    elif (ndims==4):
        for t in range(0,dims[3]):
            for k in range(0,dims[2]):
                for j in range(0,dims[1]):
                    i1 = t*dims[2]*dims[1]*dims[0] + k*dims[1]*dims[0] + j*dims[0]
                    i2 = i1+dims[0];
                    Pfield[:,j,k,t] = field[i1:i2];

    elif (ndims==3):
        for k in range(0,dims[2]):
            for j in range(0,dims[1]):
                i1 = k*dims[1]*dims[0] + j*dims[0]
                i2 = i1+dims[0];
                Pfield[:,j,k] = field[i1:i2];

    elif (ndims==2):
        for j in range(0,dims[1]):
            i1 = j*dims[0]
            i2 = i1+dims[0];
            Pfield[:,j] = field[i1:i2];

    elif (ndims==1):
        Pfield = field;

    elif (ndims==0):
        Pfield = field;

    else:
        raise RuntimeError('ndims > 5 not handled!')

    return Pfield;

def switch_row_column_major_Hector(field,dims):

    Pfield = np.zeros(dims);
    ndims = len(dims);

    if (ndims==5):
        for f in range(0,dims[4]):
            for t in range(0,dims[3]):
                for k in range(0,dims[2]):
                    for j in range(0,dims[1]):
                        i1 = f*dims[3]*dims[2]*dims[1]*dims[0] + t*dims[2]*dims[1]*dims[0] + k*dims[1]*dims[0] + j*dims[0]
                        i2  = i1+dims[0];
                        Pfield[:,j,k,t,f] = field[i1:i2];

    elif (ndims==4):
        for t in range(0,dims[3]):
            for k in range(0,dims[2]):
                for j in range(0,dims[1]):
                    i1 = t*dims[2]*dims[1]*dims[0] + k*dims[1]*dims[0] + j*dims[0]
                    i2 = i1+dims[0];
                    Pfield[:,j,k,t] = field[i1:i2];

    elif (ndims==3):
        #I'm just changing this portion because I believe it's the main thing I need to deal with. this and ndims == 2
        for k in range(0,dims[0]):
            for j in range(0,dims[1]):
                i1 = k*dims[1]*dims[2] + j*dims[2]
                i2 = i1+dims[2];
                Pfield[k,j,:] = field[i1:i2];
        #Pfield = np.swapaxes(Pfield, 0, 2);        
    elif (ndims==2):
        # as if 7.12.22, I'm not sure if this needs to be changed. but i switched 0's and 1's anyways just to check
        for j in range(0,dims[0]):
            i1 = j*dims[1]
            i2 = i1+dims[1];
            Pfield[j,:] = field[i1:i2];
        #Pfield = np.swapaxes(Pfield, 0,1);
    elif (ndims==1):
        Pfield = field;

    elif (ndims==0):
        Pfield = field;

    else:
        raise RuntimeError('ndims > 5 not handled!')

    return Pfield;
    



    


