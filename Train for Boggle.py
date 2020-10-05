import numpy as np
import cv2 
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import joblib
import TF_test
import tensorflow as tf
import tensorflow.keras as keras

def train(entries):
    runFirst = 0
    nonFlattendImages = []
    images = []
    trainY = []

    yRepresentations = ('a0','a1','a2','a3','b0','b1','b2','b3','c0','c1','c2','c3','d0','d1','d2','d3','e0','e1','e2','e3','f0','f1','f2','f3','g0','g1','g2','g3','h0','h1','h2','h3','i0','i1','i2','i3','j0','j1','j2','j3','k0','k1','k2','k3','l0','l1','l2','l3','m0','m1','m2','m3','n0','n1','n2','n3','o0','o1','o2','o3','p0','p1','p2','p3','q0','q1','q2','q3','r0','r1','r2','r3','s0','s1','s2','s3','t0','t1','t2','t3','u0','u1','u2','u3','v0','v1','v2','v3','w0','w1','w2','w3','x0','x1','x2','x3','y0','y1','y2','y3','z0','z1','z2','z3','00','11')
    shapes = cv2.imread('boggleTrain1/' + entries[0]).shape
    for entry in entries:

        img = cv2.imread('boggleTrain1/' + entry, cv2.IMREAD_GRAYSCALE).flatten()
        try:
            if len(img) != len(oldImg):

                print("The entry " + entry + " is to blame. Check the file.")
        except:
            oldImg = img
        nonFlatImg = cv2.imread('boggleTrain1/' + entry, cv2.IMREAD_GRAYSCALE)
        if runFirst == 0:
            runFirst = len(img)
        else:
            if runFirst != len(img):
                print("Error: " + entry)
        splitData = entry.split()
        for i in range(20):
            images.append(img)
            nonFlattendImages.append(nonFlatImg)
            oldImg = img
            tempCheck = 0
            for j in range(len(yRepresentations)):
                    #print("Error, the file named " + entry + " does not exist in the entries for the index.")
                if yRepresentations[j] == splitData[0][0:2]:
                    trainY.append(j)
                    tempCheck = 1
            if tempCheck == 0:
                print("Error with file " + entry)
                exit()
    images = np.array(images)
    trainY = np.array(trainY)
    print(trainY)
    X_train, X_test, y_train, y_test = train_test_split(images,trainY, test_size=.5)

    i = input("type y to re-train the model before running tests, otherwise press enter: ")


    if (i == "y"):
        trainedModel = TF_test.train(X_train, X_test, y_train, y_test)
        tf.saved_model.save(trainedModel, 'TrainedBoggle')
    else:
        trainedModel = keras.models.load_model('TrainedBoggle')
        
    

    y_test_pred = trainedModel.predict_classes(X_test, verbose=0)
    print('First 3 predictions: ', yRepresentations[y_test_pred[1]])
    print('First 3 actuals:     ', y_test[:8])
    while(True):
        cv2.imshow( "uuu", X_test[1].reshape(63,63) )
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    exit()

entries = os.listdir('boggleTrain1/')
train(entries)