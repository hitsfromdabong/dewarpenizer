import sys
import numpy as np
import cv2
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


def debug_show(name, img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 1200, 1200)
    cv2.imshow(name, img)

def show_histogram(img):
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        hist = cv2.calcHist([img],[i], None, [256],[0,256])
        plt.plot(hist, color = col)
        plt.xlim([0,256])
    plt.show()

def show_colors(img):
    h, w = img.shape[:2]
    channels = [np.empty((h,w)), np.empty((h,w)), np.empty((h,w))]
    channels = cv2.split(img)
    debug_show('blue', channels[0])
    debug_show('green', channels[1])
    debug_show('red', channels[2])

def smooth(y, pts):
    box = np.ones(pts)/pts
    return np.convolve(y, box, mode='same')

# def show_proj(img):
#     h,w = img.shape[:2]
#     proj_h = np.empty(h)
#     proj_w = np.empty(w)
#     for i in range(h):
#         proj_h[i] = 0
#         for j in range(w):
#             proj_h[i] += img[i][j]
#     for i in range(w):
#         proj_w[i] = 0
#         for j in range(h):
#             proj_h[i] += img[j][i]
#     plt.plot(proj_h)
#     plt.show()


def show_proj(img, h_bounds = [0, 0], w_bounds = [0, 0]):
     h, w = img.shape[:2]
     print h,w
     if h_bounds[1] == 0:
         h_bounds[1] = h
     if w_bounds[1] == 0:
         w_bounds[1] = w
     print h_bounds, w_bounds
     proj_h = np.empty(h)
     proj_w = np.empty(w)
     for i in range(h):
         proj_h[i] = 0
         for j in range(w_bounds[0], w_bounds[1]):
             proj_h[i] += img[i][j]
     for i in range(w):
         proj_w[i] = 0
         for j in range(h_bounds[0], h_bounds[1]):
             proj_w[i] += img[j][i]
     proj_h = np.subtract(np.ones(h), np.divide(proj_h, max(proj_h)))
     proj_w = np.subtract(np.ones(w), np.divide(proj_w, max(proj_w)))
     print(proj_h, proj_w)
     plt.figure(1)
     plt.subplot(211)
     plt.plot(proj_h)
     plt.subplot(212)
     plt.plot(proj_w)
     plt.show()






    

def sauvola(source, block):
    k = 0.01
    # m = mean of window
    # s = standard deviation of window
    # R = dynamic range of standard deviation (R = 128 with 8 bit)
    R = 128
    _ = source.copy()
    result = source.copy()
    h, w = source.shape

    means = cv2.boxFilter(source, -1, block, _, (-1, -1), True, cv2.BORDER_REPLICATE)
    m = means
    sqrMeans = cv2.sqrBoxFilter(source, -1, block, _, (-1, -1), True, cv2.BORDER_REPLICATE)

    means2 = np.multiply(means, means)
    var = sqrMeans - means2
    s = np.sqrt(var)

    for i in range(h):
        for j in range(w):
            threshold = m[i][j] * (1 + k*(s[i][j]/R - 1))
            if source[i][j] < threshold:
                result[i][j] = 0
            else:
                result[i][j] = 255

    return result

    

block = (15, 15)
img = cv2.imread(sys.argv[1])
img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

### TEST CODE GOES HERE ###


kernel = np.ones((40,40), np.uint8)
erosion = cv2.erode(grey, kernel, iterations=1)
#debug_show('erosion', erosion)

_, bw = cv2.threshold(erosion, 127, 255, cv2.THRESH_BINARY)

show_proj(bw)




### END ###

cv2.waitKey(0)
