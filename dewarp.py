import sys
import numpy as np
import cv2

# Return focal length of image in [px]
#def get_focal_length(i):
    # get focal length in [mm] from metadata

    # get CCD width in [mm]

    # focal length [px] = image width [px] * focal length [mm] / CCD width [mm]

focal_length = 0
# default intrinsic parameter matrix
K = np.array([
    [focal_length, 0, 0],
    [0, focal_length, 0],
    [0, 0, 1]], dtype = np.float32)

def main():
    if len(sys.argv) < 2:
        print('usage:', sys.argv[0], 'IMAGE1 ...')

    cv2.namedWindow('window', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('window', 1200, 1200)

    img = cv2.imread(sys.argv[1])

    # automatically calculate the thresholds
    # 1. -> calculate median -> [0.66*median, 1.33*median]
    # 2. -> adjust histogram -> calculate mean -> [0.66*mean, 1.33*mean]
    # 3. using CannyLines
    #
    # for now just use 2.

    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # grey = cv2.rgb2grey(img)

    grey_hc = cv2.equalizeHist(grey)
    summe = 0
    cols, rows, ch = img.shape

    for x in range(cols):
        for y in range(rows):
            summe += grey_hc[x, y]
    summe /= (len(grey_hc)*len(grey_hc[0]))
    print "Found mean: " + str(summe)
    print "Using thresholds (Canny): " + str(0.66*summe) +  "," + str(1.33*summe)

    # TODO bilateral filter
    #smoothed = cv2.bilateralFilter(grey_hc, 0, float(sys.argv[2]), float(sys.argv[2]))

    _ = grey_hc.copy()
    edges = cv2.Canny(grey_hc, 0.66*summe, 1.33*summe, _, int(sys.argv[2]))

#     height, width = cols, rows
#     skel = np.zeros([height,width],dtype=np.uint8)      #[height,width,3]
#     kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
#     mask = grey
#     temp_nonzero = np.count_nonzero(mask)
#     for i in range(10):
#         eroded = cv2.erode(mask,kernel)
#         cv2.imshow("window",eroded)
#         temp = cv2.dilate(eroded,kernel)
#         cv2.imshow("window",temp)
#         temp = cv2.subtract(mask,temp)
#         skel = cv2.bitwise_or(skel,temp)
#         mask = eroded.copy()


    cv2.imshow('window', edges)
    cv2.waitKey(0)
    


if __name__ == '__main__':
    main()

