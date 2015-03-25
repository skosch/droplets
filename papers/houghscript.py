import numpy as np
import cv2 as cv
import cv2.cv as cvcv
from skimage.transform import hough_circle
from skimage.feature import peak_local_max
from skimage import filter
from skimage.util import img_as_ubyte

def getDiameter(fringes):
    d_a = 0.021 #m, aperture diameter
    phi = 3.14159/2.0 # 90 degrees, off - axis
    m = 1.333 # relative refractive index
    L = 532e-9 #m, wavelength
    z = 0.45 #m, camera / light sheet distance
    kappa = ((np.arcsin(d_a/(2*z))/L) * (np.cos(phi/2) +
            m * np.sin(phi/2)/np.sqrt(m**2+1-2*m*np.cos(phi/2))))
    return 1.0 e6 * fringes / kappa

def getCircles(image, radius):
    # use the circular Hough transform to find the circles
    cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cimg = cv.GaussianBlur(cimg, (5, 5), 0)
    circles = []
    # apply an edge filter first
    canimg = filter.canny(img_as_ubyte(img), sigma=1, low_threshold=120,
                    high_threshold=126)
    hough_radii = np.array([radius])
    hough_res = hough_circle(canimg, hough_radii)
    centers =[]; accums =[]; radii =[]
    numcircles = 9
    for r, h in zip(hough_radii, hough_res):
        peaks = peak_local_max(h, num_peaks = numcircles, min_distance=50, threshold_rel=0.15)
        centers.extend(peaks)
        accums.extend(h[peaks[:, 0], peaks[:, 1]])
        radii.extend ([r]* numcircles )
    for i in np.argsort(accums)[::-1][:numcircles]:
        circles.append([centers[i][1], centers[i][0], radii[i]])
    return circles

def getDiameters(image, circles):
    # Apply the Fourier transform to every Gabor - masked circle ,
    # then find the relevant peak frequency and orientation
    h, w = image.shape[:2]
    diameters = []
    for i in circles :
        mask = np.zeros((h, w), np.float)
        cv.circle(mask, (i[0], i[1]), i[2], 255, -1)
        mask = cv.GaussianBlur(mask, (21, 21), 0)
        mask /= 255.0
        maskedimg = mask * image
        f = np.fft.fftshift(np.fft.fft2(maskedimg))
        f = 20 * np.log(np.abs(f))
        8# find the maximum component
        allowable_freqs = f[(h/2)-(h/7):(h/2)+(h/7), (w/2)+10:]
        maxfreq_x_y = np.unravel_index(allowable_freqs.argmax () ,
                allowable_freqs.shape)
        # x - fringes / imagewidth : maxfreq_x_y[1]+19
        xfringes =(maxfreq_x_y[1]+10)* i[2]*2/ w
        yfringes = abs(maxfreq_x_y[0])* i[2]*2/ h
        totalfringes = np.sqrt(xfringes ** 2 + yfringes ** 2)
        print(xfringes, yfringes, totalfringes)
        # x - fringes / dropsize :(maxfreq_x_y[ 1 ] + 1 9 ) * i[2]*2/ w
        diameters.append(str(int(totalfringes)) +"("+
                        str(int(getDiameter(totalfringes ))) +"um )")
    return diameters

def drawDropletSizes(image, circles, diameters):
h, w = image.shape[:2]
circles = np.uint16(np.around(circles))
img = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
for counter, i in enumerate(circles):
    cv.circle(img ,(i[0], i[1]), i[2] + 10, (0, 255, 0), 1)
    cv.putText(img, diameters[counter] ,(i[0] - 20, i[1]) ,
    cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255))
cv.imshow("detected circles", img)
cv.waitKey(0)
cv.destroyAllWindows()
img = cv.imread("image_cropped.jpg", 0)
fullimg = cv.imread("image_cropped.tif", -1)
circles = getCircles(img, 42)
diameters = getDiameters(fullimg, circles)
drawDropletSizes(img, circles, diameters)
