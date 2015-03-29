import numpy as np
import cv2
import scipy.ndimage.morphology.grey_erosion
from skimage.draw import circle
from scipy.misc import imresize

# From the OpenCV examples:
# built-in modules
import itertools as it
from multiprocessing.pool import ThreadPool

# local modules
from common import Timer
from find_obj import init_feature, filter_matches, explore_match

# From OpenCV samples
def affine_skew(tilt, phi, img, mask=None):
    '''
    affine_skew(tilt, phi, img, mask=None) -> skew_img, skew_mask, Ai

    Ai - is an affine transform matrix from skew_img to img
    '''
    h, w = img.shape[:2]
    if mask is None:
        mask = np.zeros((h, w), np.uint8)
        mask[:] = 255
    A = np.float32([[1, 0, 0], [0, 1, 0]])
    if phi != 0.0:
        phi = np.deg2rad(phi)
        s, c = np.sin(phi), np.cos(phi)
        A = np.float32([[c,-s], [ s, c]])
        corners = [[0, 0], [w, 0], [w, h], [0, h]]
        tcorners = np.int32( np.dot(corners, A.T) )
        x, y, w, h = cv2.boundingRect(tcorners.reshape(1,-1,2))
        A = np.hstack([A, [[-x], [-y]]])
        img = cv2.warpAffine(img, A, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if tilt != 1.0:
        s = 0.8*np.sqrt(tilt*tilt-1)
        img = cv2.GaussianBlur(img, (0, 0), sigmaX=s, sigmaY=0.01)
        img = cv2.resize(img, (0, 0), fx=1.0/tilt, fy=1.0, interpolation=cv2.INTER_NEAREST)
        A[0] /= tilt
    if phi != 0.0 or tilt != 1.0:
        h, w = img.shape[:2]
        mask = cv2.warpAffine(mask, A, (w, h), flags=cv2.INTER_NEAREST)
    Ai = cv2.invertAffineTransform(A)
    return img, mask, Ai


def affine_detect(detector, img, mask=None, pool=None):
    '''
    affine_detect(detector, img, mask=None, pool=None) -> keypoints, descrs

    Apply a set of affine transormations to the image, detect keypoints and
    reproject them into initial image coordinates.
    See http://www.ipol.im/pub/algo/my_affine_sift/ for the details.

    ThreadPool object may be passed to speedup the computation.
    '''
    params = [(1.0, 0.0)]
    for t in 2**(0.5*np.arange(1,6)):
        for phi in np.arange(0, 180, 72.0 / t):
            params.append((t, phi))

    def f(p):
        t, phi = p
        timg, tmask, Ai = affine_skew(t, phi, img)
        keypoints, descrs = detector.detectAndCompute(timg, tmask)
        for kp in keypoints:
            x, y = kp.pt
            kp.pt = tuple( np.dot(Ai, (x, y, 1)) )
        if descrs is None:
            descrs = []
        return keypoints, descrs

    keypoints, descrs = [], []
    if pool is None:
        ires = it.imap(f, params)
    else:
        ires = pool.imap(f, params)

    for i, (k, d) in enumerate(ires):
        print 'affine sampling: %d / %d\r' % (i+1, len(params)),
        keypoints.extend(k)
        descrs.extend(d)

    print
    return keypoints, np.array(descrs)



img1 = cv2.imread("masked_points.jpg", 0)
img2 = cv2.imread("masked_disks.jpg", 0)

# pixel-wise mean taken over an image ensemble
mean1 = cv2.imread("masked_mean1.jpg", 0)
mean2 = cv2.imread("masked_mean2.jpg", 0)

# turn the points into disks
img1 -= mean1
img2 -= mean2

img1 = grey_erosion(img1, size=(3, 3))

# flip and scale the image
img1 = np.hflip(img1)
img1 = imresize(0.5)

# locate intensity peaks. wherever they are, black them out and replace them
with a large circle in a blank image

virtualdisks = np.zeros(img2.shape)
max_threshold = 200
virtualradius = 60
eraserradius = 13
while np.max(img1) > 200:
    cy, cx = np.argmax(img1, axis=0), np.argmax(img1, axis=1)
    rr, cc = circle(cy, cx, virtualradius)
    virtualdisks[rr, cc] = 255
    re, ce = circle(cy, cx, eraserradius)
    img1[re, ce] = 0

# now match using asift (from OpenCV samples)

detector, matcher = init_feature('orb')
pool=ThreadPool(processes = cv2.getNumberOfCPUs())
kp1, desc1 = affine_detect(detector, virtualdisks, pool=pool)
kp2, desc2 = affine_detect(detector, img2, pool=pool)
def match_and_draw(win):
    with Timer('matching'):
        raw_matches = matcher.knnMatch(desc1, trainDescriptors = desc2, k = 2) #2
    p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches)
    if len(p1) >= 4:
        H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
        print '%d / %d  inliers/matched' % (np.sum(status), len(status))
        # do not draw outliers (there will be a lot of them)
        kp_pairs = [kpp for kpp, flag in zip(kp_pairs, status) if flag]
    else:
        H, status = None, None
        print '%d matches found, not enough for homography estimation' % len(p1)

    vis = explore_match(win, img1, img2, kp_pairs, None, H)

# show the match
match_and_draw('affine find_obj')
cv2.waitKey()
cv2.destroyAllWindows()

# compute the homography
doubleSize = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]])
flipV = np.array([[1, 0, 0], [0, -1, img1.shape[0]], [0, 0, 1]])
Hh = np.dot(np.dot(H, doubleSize), flipV)

Pfoc = np.array() # from DantecStudio, or some constant
Pdef = np.dot(Hh, Pfoc)
print(Pfoc)
print(Pdef)
