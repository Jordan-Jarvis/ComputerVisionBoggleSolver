import numpy as np
import cv2 
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import joblib

def train(entries):
    images = []
    trainX = []
    shapes = cv2.imread('trainingData/' + entries[0]).shape

    for entry in entries:
        img = cv2.imread('trainingData/' + entry, cv2.IMREAD_GRAYSCALE).flatten()
        splitData = entry.split()
        for i in range(20):
            images.append(img)
            trainX.append(int(splitData[0]))
    image = images[90]
    images = np.array(images)
    trainX = np.array(trainX)
    print(images)
    clf = MLPClassifier(solver='adam', alpha=.001, learning_rate='adaptive', learning_rate_init=0.01, hidden_layer_sizes=(28*28, 392,196), random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(images,trainX, test_size=0.5)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    print(pred)
    print(y_test)
    from sklearn.metrics import accuracy_score
    print("Using Gaussian the algorithm is {} accurate".format(100*accuracy_score(y_test, pred)))
    filename = 'finalized_sudoku_model.sav'
    joblib.dump(clf, filename)


entries = os.listdir('trainingData/')
train(entries)