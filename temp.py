import os
import cv2
import numpy as np
import math
scale = 4
img = cv2.imread("boggleTrain/i5.png", 0)
files = os.listdir("boggleTrain/")
for image in files:
    img = cv2.imread("boggleTrain/" + image, 0)
    img = img.reshape(math.ceil(250/scale),math.ceil(250/scale),)
    cv2.imwrite("boggleTrain/" + image, img)
img = img.reshape(math.ceil(250/scale),math.ceil(250/scale),)
print(img)
cv2.imshow("Maybe", img)
cv2.waitKey(0)
