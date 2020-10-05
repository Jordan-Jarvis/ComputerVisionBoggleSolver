from __future__ import print_function
from cv2 import cv2 as cv
import argparse




def CannyThresh(fileName, val):
    max_lowThreshold = 100
    window_name = 'Edge Map'
    title_trackbar = 'Min Threshold:'
    ratio = 3
    kernel_size = 3
    parser = argparse.ArgumentParser(description='Code for Canny Edge Detector tutorial.')
    parser.add_argument('--input', help='Path to input image.', default=fileName)
    args = parser.parse_args()
    src = fileName
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    low_threshold = val
    img_blur = cv.blur(src_gray, (3,3))
    detected_edges = cv.Canny(img_blur, low_threshold, low_threshold*ratio, kernel_size)
    mask = detected_edges != 0
    dst = src * (mask[:,:,None].astype(src.dtype))

    return dst


