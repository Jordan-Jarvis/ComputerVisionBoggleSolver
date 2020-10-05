import numpy as np
import cv2
import os

def rotateAndSave(img, entry):
    img = np.rot90(img)
    if(entry[1] == '0'):
        entry = entry[0] + '1' + entry[2:]
    elif(entry[1] == '1'):
        entry = entry[0] + '2' + entry[2:]
    elif(entry[1] == '2'):
        entry = entry[0] + '3' + entry[2:]
    else:
        entry = entry[0] + '0' + entry[2:]
    cv2.imwrite('boggleTrain1/' + entry, img)
    return img, entry

entries = os.listdir('boggleTrain/')
for entry in entries:
    img = cv2.imread('boggleTrain/' + entry, cv2.IMREAD_GRAYSCALE)
    img, entry = rotateAndSave(img, entry)
    img, entry = rotateAndSave(img, entry)
    img, entry = rotateAndSave(img, entry)
    img, entry = rotateAndSave(img, entry)