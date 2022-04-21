import numpy as np
import spectrum as spec

dt=2.
phi = np.random.randn(n)
spec_phi = spec.Spectrum(phi,dt)

# var(p) from Fourier coefficients
P_var_spec = spec_phi.var

# var(p) in physical space
P_var_phys = phi.var()

# relative error
error = np.abs(P_var_phys - P_var_spec)/P_var_phys

assert error<rtol, " *** 1D spectrum does not satisfy Parseval's theorem"
