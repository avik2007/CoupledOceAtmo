import numpy as np

### GLOBAL MODEL PARAMETERS

# General
g = 9.81;
f0 = 0.0000685; # at 28N.  Actual value varies with latitude.
rho0 = 1027.5; # for energy output (optional)    

# Leith
viscA4D = 0.0; 
viscA4Z = 0.0; 
no_slip_sides = 1;
dt = 25; 
viscA4Max = 1e21;
viscA4Grid = 0.0;
viscA4GridMin = 0.0; # MITgcm defaults to this if not set in data file.
viscA4GridMax = 0.8;
viscC4leith = 2.15;
viscC4leithD = 2.15;
leithDivOff = False;

# KPP
viscArNr = 5.6614e-4;
Riinfty = 0.6998;
BVSQcon = -2e-5;
difm0 = 5e-3;
difs0 = 5e-3;
dift0 = 5e-3;
difmcon = 1e-1;
difscon = 1e-1;
diftcon = 1e-1;
Trho0 = 1.9;
dsfmax = 1e-2;
cstar = 1e1;



### GLOBAL NUMERICAL POSTPROCESSING PARAMETERS

ng = 2; # number of ghost cells
taper_function = 1; # 1 - Hann Window, 2 - Tukey Window (not implemented)
tukey_alpha = 0.5;
omega_threshold = 2.5*0.0000685; # filter out eddies and waves below 2.5*f0;
omega_bp_threshold = 0.8*0.0000685; # filter out eddies and waves below 0.8*f0;
