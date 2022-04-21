import numpy as np

def repmat(A_in,reps):

    A = A_in.copy();
    
    nd = len(A.shape);
    nd_out = len(reps);

    while nd<nd_out:
        A = np.expand_dims(A,-1);
        nd += 1;
        
    A = np.tile(A,reps);
    
    return A;
    
