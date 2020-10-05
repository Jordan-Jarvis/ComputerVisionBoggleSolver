import complexWorker as cw
import cv2
import numpy as np
import math

import random
import string

def get_random_string(length):
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str
cap = cv2.VideoCapture(0)
while(True):
    ret, frame = cap.read()
    if not np.sum(frame) == 0:
        break
centerX = frame.shape[1]/2
centerY = frame.shape[0]/2
x1 = int(centerX - 170)
y1 = int(centerY - 170)
x2 = int(centerX + 170)
y2 = int(centerY + 170)
startPoint = (x1,y1)
endPoint = (x2,y2)
scale = 4
boards = []
while(True):
    while(True):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = gray[y1:y2, x1:x2]
        img, cropped, digits = cw.extract_sudoku(gray)
        cv2.imshow( "Display window", img )
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cv2.imshow( "uuu", cropped )
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        inp = input("Is this good?")
        if inp == "1" or inp == "y":
            boards.append(digits)
            break
    inp = input("Keep going?")
    if inp == "0" or inp == "n":
        break

j = 0
for digits in boards:
    for img in digits:
        img = img.reshape(math.ceil(250/scale),math.ceil(250/scale),)
        cv2.imshow("single letter", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        letter = input("What letter is this? : ")
        orientation = input("What orientation is this? : ")
        cv2.imwrite("boggleTrain/" + letter + orientation + get_random_string(3) + str(j) + '.png', img) # random stuff to allow multiple values with the same letter and orientation
        j = j + 1