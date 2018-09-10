import numpy as np
import sys
from cv2 import *

def debug_show(name, img):
    namedWindow(name, cv2.WINDOW_NORMAL)
    resizeWindow(name, 1200, 1200)
    imshow(name, img)


img = imread(sys.argv[1])
debug_show('img', img)

img = resize(img, (0,0), fx=0.25, fy=0.25)

gray = cvtColor(img, cv2.COLOR_BGR2GRAY)
debug_show('gray', gray)

edged = Canny(gray, 30, 200)
debug_show('edged', edged)

# _, contours, hierachy = findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# drawContours(img, contours, -1, (0,255,0), 3)
# debug_show('img_c', img)
# print hierachy

lines = HoughLines(edged, 1, np.pi/180, 200)
# for rho, theta in lines:
for v in lines:
    rho = v[0][0]
    theta = v[0][1]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*a)
    x2 = int(x0 -1000*(-b))
    y2 = int(y0 - 1000*a)

    flag = 0
    if 0.1 < theta/(np.pi) < 0.4 or 0.6 < theta/np.pi < 0.9 :
        flag = 1

    if flag == 0:
        line(img, (x1, y1), (x2, y2), (0,0,255),2)

debug_show('lines',img)


waitKey(0)
