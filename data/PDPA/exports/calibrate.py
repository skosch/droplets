#!/usr/bin/python

# This takes in a CSV exported from FlowSizer and finds the right values for AB
# and AC separation, so that the max value is equal to some predicted number.
import pandas as pd
import numpy as np
import matplotlib as mpl
pi = np.pi
with open("404-coinc.csv", "r") as f:
    df = pd.read_csv(f, skiprows=0, header=0)

# Let's say the relationship between AB and diameter is linear
df.columns = ['dia', 'ab', 'ac', 'mv', 't1', 't2']
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(font="Alte DIN 1451 Mittelschrift")
mpl.rc("xtick", labelsize=16)

f = plt.figure(figsize=(4.5, 3.5))

# From the remaining rows, multiply the first one by ABsep and the second one by
# ACsep. Then Take a histogram of both

df.ab = df.ab.astype("float")
df.ac = df.ac.astype("float")

# now take [-180, 0] and attach it to [0, 180] -- in other words,
# add 360 to all negative numbers
df.ab[df.ab < 0] += 360
df.ac[df.ac < 0] += 360

# add pre-calibration values
#df.ab += 5
#df.ac += 0


#####################################
# Find Beta
L = 514.5e-9 # m, green wavelength
m = 1.3324

def phase(psi, phi, theta):
    term1 = 1 + np.cos(psi) * np.cos(phi) * np.cos(theta/2.)
    term2 = np.sin(psi) * np.sin(theta/2.)
    return (2*pi/L) * (np.sqrt(1+m**2-m*np.sqrt(2)*np.sqrt(term1 + term2)) - 
                       np.sqrt(1+m**2-m*np.sqrt(2)*np.sqrt(term1 - term2)))

phi = pi/3. # 60 deg off-angle
theta = 2*np.arctan(25.e-3/762e-3) # degree between beams from emitter

# ASSUME THE FOLLOWING: psi_A = 25, psi_B = 15, psi_C = -15
dist_AB = 10.33e-3 #12.08e-3 # m
dist_AC = 37.86e-3 #43.24e-3 # m

# PART A: Find betaAB and betaAC using the full derivation
psi_A = np.arctan(-(dist_AC/2)/750e-3)
psi_B = np.arctan(-(dist_AC/2 - dist_AB)/750e-3)
psi_C = np.arctan((dist_AC/2)/750e-3)

beta_AB = (phase(psi_A, phi, theta) - phase(psi_B, phi, theta))
beta_AC = (phase(psi_A, phi, theta) - phase(psi_C, phi, theta))

beta_AB /= (2 * pi / 360.) # convert from rad to deg
beta_AC /= (2 * pi / 360.)

print("BetaAB, full calc: " + str(beta_AB))
print("BetaAC, full calc: " + str(beta_AC))

# PART B: Find betaAB and betaAC using the gamma approximation
fs = 2 * np.sin(theta/2) / L
v = np.sqrt(2 * (1 + 0.995 * np.cos(phi) * np.cos(theta/2)))
gamma = m/(v*np.sqrt(1+m**2-m*v))
print("Gamma (Auto slope) = " + str(gamma/2.0))
print("BetaAB from gamma: " + str(gamma * ((dist_AB/2)/750.e-3) * 360/fs))
print("BetaAC from gamma: " + str(gamma * ((dist_AC/2)/750.e-3) * 360/fs))
print("Gamma error basis AB = " + str(gamma * np.sin((dist_AB/2)/750.e-3) *
    np.sin(theta/2)))
print("Gamma error basis AC = " + str(gamma * np.sin((dist_AC/2)/750.e-3) *
    np.sin(theta/2)))

# PART C: Find sizes using betaAB and betaAC
# size from AB:
df['d_ab'] = df['ab'] / beta_AB

# size from AC:
# first, find the number of jumped 360's, then find the adjusted size from AC
n2pi = ( (1./360.) * (beta_AC*df['ab']/beta_AB - df['ac']) + 0.5).astype('int')
df['d_ac'] = (df['ac'] + 360*n2pi) / beta_AC

# PART D: Remove data points that have sizes differing by too much
valids = ((df.d_ab - df.d_ac).abs() < 60)
df = df[valids]


# PART E: Plot the mean measured diameter histogram

#sns.distplot(df.ab, kde=False, color="g", axlabel=False)
#sns.distplot(df.ac, kde=False, color="r", axlabel=False)
#sns.distplot(df.d_ab, kde=False, color="k", axlabel=False)
#sns.distplot(df.d_ac, kde=False, color="r", axlabel=False)
sns.distplot((df.d_ab + df.d_ac)/2, kde=True, bins=100, color="g", axlabel=False)
#kde_kws={"kernel": "gau", "bw": 0.000007}, 

plt.show()





