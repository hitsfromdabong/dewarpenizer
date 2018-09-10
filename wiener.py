import sys
import numpy as np
import cv2

def debug_show(name, img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 1200, 1200)
    cv2.imshow(name, img)

def wiener(source, block):
    _ = source.copy()
    result = source.copy()
    h, w = source.shape

    means = cv2.boxFilter(source, -1, block, _, (-1, -1), True, cv2.BORDER_REPLICATE)
    sqrMeans = cv2.sqrBoxFilter(source, -1, block, _, (-1, -1), True, cv2.BORDER_REPLICATE)

    means2 = np.multiply(means, means)
    var = sqrMeans - means2

    avgVar = cv2.reduce(var, 1, cv2.REDUCE_SUM, -1)
    avgVar = cv2.reduce(avgVar, 0, cv2.REDUCE_SUM, -1)

    noise = avgVar[0][0] / (h * w)

    print means[0][0], var[0][0], noise, source[0][0]

    for i in range(h):
        for j in range(w):
            integer = means[i][j] + max(0.0, var[i][j] - noise) / max(var[i][j], noise) * (source[i][j] - means[i][j])
            if integer > 255:
                integer = 255
            result[i][j] = int(integer)

    return result


    

    

block = (5, 5)
img = cv2.imread(sys.argv[1])
grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

debug_show('gray', grey)

result = wiener(grey, block)
debug_show('result', result)
cv2.waitKey(0)
