import numpy as np

# Find Beta
pi = np.pi
L = 514.5e-9 # green wavelength
m = 1.3324

def phase(psi, phi, theta):
    term1 = 1 + np.cos(psi) * np.cos(phi) * np.cos(theta/2.)
    term2 = np.sin(psi) * np.sin(theta/2.)
    return (2*pi/L) * (np.sqrt(1+m**2-m*np.sqrt(2)*np.sqrt(term1 + term2)) - 
                       np.sqrt(1+m**2-m*np.sqrt(2)*np.sqrt(term1 - term2)))

phi = pi/3 # 60 deg off-angle
theta = 2*np.arctan(25./762) # degree between beams from emitter

# ASSUME THE FOLLOWING: psi_A = 25, psi_B = 15, psi_C = -15


psi_A = np.arctan(40./750)
psi_B = np.arctan(30./750)
psi_C = np.arctan(0./750)

beta_AC = phase(psi_A, phi, theta) - phase(psi_C, phi, theta)
beta_AB = phase(psi_A, phi, theta) - phase(psi_B, phi, theta)

# For a particle of size 300 um, this gives a phase shift of
print "beta_AC = ", beta_AC
print "beta_AB = ", beta_AB
print beta_AC * 300e-6
print beta_AB * 300e-6


