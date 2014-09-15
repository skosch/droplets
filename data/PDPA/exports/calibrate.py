#!/usr/bin/python

# This takes in a CSV exported from FlowSizer and finds the right values for AB
# and AC separation, so that the max value is equal to some predicted number.
import pandas as pd
import numpy as np
import matplotlib as mpl
import scipy.stats
import scipy.optimize
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(font="Alte DIN 1451 Mittelschrift")

pi = np.pi

def getDF():
    with open("seb" + str(expectedVal) + "-coinc.csv", "r") as f:
        df = pd.read_csv(f, skiprows=1, header=0)

    # Let's say the relationship between AB and diameter is linear
    df.columns = ['dia', 'ab', 'ac', 'mv', 't1', 't2']

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
    return df

def getMaxDiameter(df, dist_AB, dist_AC, plotGaussian=False):
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

    # PART A: Find betaAB and betaAC using the full derivation
    psi_A = np.arctan(-(dist_AC/2)/750e-3)
    psi_B = np.arctan(-(dist_AC/2 - dist_AB)/750e-3)
    psi_C = np.arctan((dist_AC/2)/750e-3)

    beta_AB = (phase(psi_A, phi, theta) - phase(psi_B, phi, theta))
    beta_AC = (phase(psi_A, phi, theta) - phase(psi_C, phi, theta))

    beta_AB /= (2 * pi / 360.) # convert from rad to deg
    beta_AC /= (2 * pi / 360.)

    #print("BetaAB, full calc: " + str(beta_AB))
    #print("BetaAC, full calc: " + str(beta_AC))

    ## PART B: Find betaAB and betaAC using the gamma approximation
    #fs = 2 * np.sin(theta/2) / L
    #v = np.sqrt(2 * (1 + 0.995 * np.cos(phi) * np.cos(theta/2)))
    #gamma = m/(v*np.sqrt(1+m**2-m*v))
    #print("Gamma (Auto slope) = " + str(gamma/2.0))
    #print("BetaAB from gamma: " + str(gamma * ((dist_AB/2)/750.e-3) * 360/fs))
    #print("BetaAC from gamma: " + str(gamma * ((dist_AC/2)/750.e-3) * 360/fs))
    #print("Gamma error basis AB = " + str(gamma * np.sin((dist_AB/2)/750.e-3) *
    #    np.sin(theta/2)))
    #print("Gamma error basis AC = " + str(gamma * np.sin((dist_AC/2)/750.e-3) *
    #    np.sin(theta/2)))

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

    gkde = scipy.stats.gaussian_kde((1e6*(df.d_ab + df.d_ac)/2).values)    
    def negfun(x):
        return -gkde.evaluate(x)
    opt = scipy.optimize.minimize_scalar(negfun, (0, 800))

    # PART E: Plot the mean measured diameter histogram
    if plotGaussian:
        print "Max diameter density at ", opt.x
        sns.distplot(1e6*(df.d_ab + df.d_ac)/2, kde=True, bins=100, color="g", axlabel=False)
        plt.show()
    return opt.x

def saveZeroContour():
    minAB = 8
    minAC = 34
    intervalAB = 0.2
    intervalAC = 0.2
    dist_AB_set = np.arange(minAB, 15, intervalAB)
    dist_AC_set = np.arange(minAC, 48, intervalAC)
    # Then, create an array to display the results in
    peaksizes = np.ndarray((dist_AB_set.shape[0], dist_AC_set.shape[0]))

    resx = []
    resy = []
    for AB_idx, dist_AB in enumerate(list(dist_AB_set)):
        for AC_idx, dist_AC in enumerate(list(dist_AC_set)):
            peaksizes[AB_idx, AC_idx] = getMaxDiameter(df, dist_AB * 1.e-3, dist_AC * 1.e-3)
            if(abs(peaksizes[AB_idx, AC_idx] - expectedVal) < 1):
                resx.append(dist_AC)
                resy.append(dist_AB)
                #results.append((expectedVal, dist_AB * 1e-3, dist_AC * 1e-3,
                #    peaksizes[AB_idx, AC_idx]))

    showContours = False
    if showContours:
        ax = plt.gca()
        mpl.rcParams['contour.negative_linestyle'] = 'solid'
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x,p: x*intervalAB + minAB))
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x,p: x*intervalAC + minAC))
        CS = plt.contour(peaksizes-expectedVal, origin='lower', colors='k')
        zc = CS.collections[3]
        plt.setp(zc, linewidth=4)
        plt.clabel(CS, fontsize=12, inline=1, fmt="%1.1f")
        ax.set_xlabel("Distance AC [mm]", fontsize="16")
        ax.set_ylabel("Distance AB [mm]", fontsize="16")
        plt.show()

    getContours = False
    if getContours:
        ax = plt.gca()
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x,p: x*intervalAB + minAB))
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x,p: x*intervalAC + minAC))
        CS = plt.contour(peaksizes-expectedVal, origin='lower', colors='k')
        zc = CS.collections[3]
        cyx = zc.get_paths()[0].vertices.copy()
        cyx[:,0] = cyx[:,0] * intervalAC + minAC
        cyx[:,1] = cyx[:,1] * intervalAB + minAB
        zeroContours.append(cyx)
        plt.clf()
    
    print "Done with size " + str(expectedVal)
    return (resx, resy)

results = {}
expectedVal = 0
df = None
zeroContours = []
sizes = [125, 267, 324, 354, 404]
recalcContours = True
if recalcContours:
    for i in sizes:
        expectedVal = i
        df = getDF()
        results[i] = saveZeroContour()
else:
    for i in sizes:
        zeroContours.append(np.loadtxt("zero-" + str(i)))

if False:
    expectedVal = 404
    df = getDF()
    getMaxDiameter(df, 11.866e-3, 39.7e-3, True)

print "\n".join(map(str,results))
print "Found " + str(len(results)) + " results. Plotting ..."
# Now, plot all the zero contours
for i, ls in zip(sizes, ['o', 'x', 'D', 'd', '8']):
    mpl.rc("xtick", labelsize=13)
    mpl.rc("ytick", labelsize=13)
    ax = plt.gca()
    ax.set_xlabel("Distance AC [mm]", fontsize="16")
    ax.set_ylabel("Distance AB [mm]", fontsize="16")
    line, = plt.plot(*results[i], label=str(i) + " um", marker=ls)
plt.legend()
plt.show()

