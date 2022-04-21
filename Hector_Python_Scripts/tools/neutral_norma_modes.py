import sys
import numpy as np
from scipy import fftpack as fft
from scipy.sparse import lil_matrix, csc_matrix, csr_matrix
from scipy.sparse.linalg import eigs, inv
import warnings

def neutral_modes_from_N2_profile_raw(z, N2, f0, depth=None, **kwargs):

    nz = len(z)

    ### vertical discretization ###

    # ~~~~~ zf[0]==0, phi[0] ~~~~
    #
    # ----- zc[0], N2[0] --------
    #
    # ----- zf[1], phi[1] -------
    # ...
    # ----- zc[nz-1], N2[nz-1] ---
    #
    # ~~~~~ zf[nz], phi[nz] ~~~~~

    # just for notation's sake
    # (user shouldn't worry about discretization)
    zc = z
    dzc = np.hstack(np.diff(zc))
    # make sure z is increasing
    if not np.all(dzc > 0):
        raise ValueError('z should be monotonically increasing')
    if depth is None:
        depth = z[-1] + dzc[-1]/2
    else:
        if depth <= z[-1]:
            raise ValueError('depth should not be less than maximum z')

    dztop = zc[0]
    dzbot = depth - zc[-1]

    # put the phi points right between the N2 points
    zf = np.hstack([0, 0.5*(zc[1:]+zc[:-1]), depth ])
    dzf = np.diff(zf)

    # We want a matrix representation of the operator such that
    #    g = f0**2 * np.dot(L, f)
    # This can be put in "tridiagonal" form
    # 1) first derivative of f (defined at zf points, maps to zc points)
    #    dfdz[i] = (f[i] - f[i+1]) /  dzf[i]
    # 2) now we are at zc points, so multipy directly by f0^2/N^2
    #    q[i] = dfdz[i] / N2[i]
    # 3) take another derivative to get back to f points
    #    g[i] = (q[i-1] - q[i]) / dzc[i-1]
    # boundary condition is enforced by assuming q = 0 at zf = 0
    #    g[0] = (0 - q[0]) / dztop
    #    g[nz] = (q[nz-1] - 0) / dzbot
    # putting it all together gives
    #    g[i] = ( ( (f[i-1] - f[i]) / (dzf[i-1] * N2[i-1]) )
    #            -( (f[i] - f[i+1]) / (dzf[i] * N2[i])) ) / dzc[i-1]
    # which we can rewrite as
    #    g[i] = ( a*f[i-1] + b*f[i] +c*f[i+1] )
    # where
    #    a = (dzf[i-1] * N2[i-1] * dzc[i-1])**-1
    #    b = -( ((dzf[i-1] * N2[i-1]) + (dzf[i] * N2[i])) * dzc[i-1] )**-1
    #    c = (dzf[i] * N2[i] * dzc[i-1])**-1
    # for the boundary conditions we have
    #    g[0] =  (-f[0] + f[1]) /  (dzf[0] * N2[0] * dztop)
    #    g[nz] = (f[nz-1] - f[nz]) /  (dzf[nz-1] * N2[nz-1] *dzbot)
    # which we can rewrite as
    #    g[0] = (-a*f[0] + a*f[1])
    #           a = (dzf[0] * N2[0])**-1
    #    g[nz] = (b*f[nz-1] - b*f[nz])
    #           b = (dzf[nz-1] * N2[nz-1])**-1

    # now turn all of that into a sparse matrix

    L = lil_matrix((nz+1, nz+1), dtype=np.float64)
    for i in range(1,nz):
        a = (dzf[i-1] * N2[i-1] * dzc[i-1])**-1
        b = -(dzf[i-1] * N2[i-1]* dzc[i-1])**-1 - (dzf[i] * N2[i] * dzc[i-1])**-1
        c = (dzf[i] * N2[i] * dzc[i-1])**-1
        L[i,i-1:i+2] = [a,b,c]
    a = (dzf[0] * N2[0] * dztop)**-1
    L[0,:2] = [-a, a]
    b = (dzf[nz-1] * N2[nz-1] * dzbot)**-1
    L[nz,-2:] = [b, -b]

    # this gets the eigenvalues and eigenvectors
    if nz <= 2:
        w, v = eigs(L, k=nz-1, which='SM')
    elif nz > 2 and len(kwargs) > 0:
        w, v = eigs( L, k=kwargs['num_eigen'], which='SM', v0=kwargs['init_vector'], 
                        ncv=kwargs['num_Lanczos'], maxiter=kwargs['iteration'],
                        tol=kwargs['tolerance'])
    else:
        w, v = eigs(L, which='SM')

    # eigs returns complex values. Make sure they are actually real
    tol = 1e-20
    np.testing.assert_allclose(np.imag(v), 0, atol=tol)
    np.testing.assert_allclose(np.imag(w), 0, atol=tol)
    w = np.real(w)
    v = np.real(v)

    # they are often sorted and normalized, but not always
    # so we have to do that here
    j = np.argsort(w)[::-1]
    w = w[j]
    v = v[:,j]
    
    #########
    # w = - 1/(Rd^2 * f0^2)
    #########
    Rd = (-w)**-0.5 / np.absolute(f0)

    return zf, Rd, v
