import numpy as np
import scipy.fftpack as fftpack
from skimage import filter
from skimage import io
from skimage.feature import match_template

# first, import the image
img = io.imread("image.tif", True)

# create rectangle
stripheight = 14
stripwidth = 120

pattern = 255 * np.ones((stripheight, stripwidth))

# hanning-window it
pattern *= np.hanning(stripheight)[:np.newaxis]
pattern *= np.hanning(stripwidth)[:np.newaxis].T

# correlate
cor = match_template(img, pattern)

def find_peaks(correlation_result):
    # 1. identify peak
    # 2. remove from image a rectangle until no more peaks are above a
    # threshold

    xs = []
    ys = []
    threshold = 200
    while(np.max(correlation_result) > threshold):
        x = np.argmax(correlation_result, axis=1)
        y = np.argmax(correlation_result, axis=0)
        correlation_result[y-stripheight/3:y+stripheight/3,
                x-stripwidth/3:x+stripwidth/3] = 0
        xs.append(x)
        ys.append(y)
    return (xs, ys)

# find peaks
xs, ys = find_peaks(cor)

def find_fringecount(stripimg):
    # apply hanning
    stripimg *= np.hanning(stripheight)[:np.newaxis]
    stripimg *= np.hanning(stripwidth)[:np.newaxis].T
    # pad it
    padded = np.zeros((1024,1024))
    padded[512-stripimg.shape[0]/2:512+stripimg.shape[0]/2,
           512-stripimg.shape[1]/2:512+stripimg.shape[1]/2] = stripimg
    # FFT it
    fftimg = fftpack.fftshift(fftpack.fft2(padded))

    fftheight = 40
    fftwidth = 400
    # take center
    fftimg = fftimg[512-fftheight/2:512+fftheight/2,
                    512, 512+fftwidth]
    # take mean and clip
    fftimg = np.mean(fftimg, axis=0)
    fftimg[:,0:40] = 0
    fftimg[:,150:] = 0
    # find max and get fringe count
    return stripwidth / (1024/np.argmax(fftimg))

fringecounts = []

# for every peak, take rectangle
for (x, y) in zip(xs, ys):
    # deal with edge cases later
    if (y < stripheight/2 or y > img.shape[0]-stripheight/2
        or x < stripwidth/2 or x > img.shape[1]-stripwidth/2):
        continue
    strip = img[y-stripheight/2:y+stripheight/2,
                x-stripwidth/2:x+stripwidth/2]
    fringecounts.append(find_fringecount(strip))

# plot various things
# (not shown here)

print(fringecounts)
