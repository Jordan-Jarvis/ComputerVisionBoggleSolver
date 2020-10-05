import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import cv2 
import os
import complexWorker as cw
from BoggleSolver import boggle1 as bg
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
digits = []
trainedModel = keras.models.load_model('TrainedBoggle')
while(True):
    digits = []
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = gray[y1:y2, x1:x2]
    img, cropped, digits = cw.extract_boggle(gray)
    cv2.imshow( "Display window", img )
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.imshow( "uuu", cropped )
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    inp = input("Is this good?")
    if inp == "1" or inp == "y":
        break

yRepresentations = ('a0','a1','a2','a3','b0','b1','b2','b3','c0','c1','c2','c3','d0','d1','d2','d3','e0','e1','e2','e3','f0','f1','f2','f3','g0','g1','g2','g3','h0','h1','h2','h3','i0','i1','i2','i3','j0','j1','j2','j3','k0','k1','k2','k3','l0','l1','l2','l3','m0','m1','m2','m3','n0','n1','n2','n3','o0','o1','o2','o3','p0','p1','p2','p3','q0','q1','q2','q3','r0','r1','r2','r3','s0','s1','s2','s3','t0','t1','t2','t3','u0','u1','u2','u3','v0','v1','v2','v3','w0','w1','w2','w3','x0','x1','x2','x3','y0','y1','y2','y3','z0','z1','z2','z3','00','11')

while(True):
    cv2.imshow( "uuu", digits[1] )
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
digits = np.array(digits)
print(digits.shape)
y_test_pred = trainedModel.predict_classes(digits, verbose=0)
print('First 8 predictions: ', y_test_pred[:8])
PredictedVals = []
for index in y_test_pred:
    PredictedVals.append(yRepresentations[index][0])
print("Converted to letters: ", PredictedVals)
while(True):
    cv2.imshow( "FinalImage", cropped )
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

tempVal = 0
tempVal2 = 0

PredictedVals = [x.upper() for x in PredictedVals] 
for i in range(len(PredictedVals)):
    if PredictedVals[i] == 'Q':
        PredictedVals[i] = "Qu"
print(PredictedVals)
bg.SolveBoard(PredictedVals)
exit()