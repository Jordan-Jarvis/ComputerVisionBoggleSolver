import numpy as np
import cv2
import glob
import os
from multiprocessing import Lock, Process, Queue, current_process
import time
import queue # imported for using queue.Empty exception
import multiprocessing as mp
from multiprocessing import shared_memory
import ctypes
from matplotlib import pyplot as plt
import cannyEdgeDetect as ce

""" 
Impliment the TODO tasks.
"""

#Load list of images in the directory chosen
def load_image_files(subpath):
    path = os.getcwd() + '/' + subpath
    files = [f for f in glob.glob(path + "**/*.png", recursive=False)]
    files.sort()
    return files

#Writes the images to a video file
def WriteToFile(fileName, vid):
    h, w = vid[1].shape[:2]
    out = cv2.VideoWriter(fileName,cv2.VideoWriter_fourcc('M','J','P','G'), 25, (w,h))
    i = 0
    print(len(vid))
    for i in range(len(vid)-1):
        out.write(vid[i]) # loops until the images are all written to the file
        i = i + 1
#loops the video when given the list of images, which frames to loop and number of 
#frames to fade

def readFileToList(fileName):
    test = []
    cap = cv2.VideoCapture(fileName)
    # Check if file is opened
    if (cap.isOpened() == False): 
        print("Unable to read file")
    success,image = cap.read()

    count = 0
    success = True
    while success:
        test.append(image) # keep reading if there is more to read
        success,image = cap.read()
        count += 1
    cap.release()
    
    # Closes all the frames
    cv2.destroyAllWindows()
    return test


def do_job(queue0, shape, datype, queue):

    existing_shm = shared_memory.SharedMemory(name='imgs')
    images = np.ndarray(shape, dtype=datype, buffer=existing_shm.buf)
    comparison_values2 = []


    while 1:
        while not queue0.empty():
            j = queue0.get()
            i = j + 1
            print('trying frames: ' + str(j) + " and ")
            for i in range(i, len(images)):
                    #append the compares along with the frames compared
                comparison_values2.append((image_compare(images[j], images[i]), j, i))
            queue.put(np.array(comparison_values2))
            print('put it on the queue')
        
        if queue0.empty():
            return
        time.sleep(1.5)


def cannyThreadProcess(img):
    """ Main function """
    
    
    #img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    img = cv2.GaussianBlur(img,(5,5),0)
#    cv2.imshow('rry image',img)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    mask = np.zeros((gray.shape),np.uint8)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))

    close = cv2.morphologyEx(gray,cv2.MORPH_CLOSE,kernel1)
    div = np.float32(gray)/(close)
    res = np.uint8(cv2.normalize(div,div,0,255,cv2.NORM_MINMAX))
    res2 = cv2.cvtColor(res,cv2.COLOR_GRAY2BGR)

    
  
    cv2.imshow('Blu',res2)
    
    
    thresh = cv2.adaptiveThreshold(res,255,0,1,19,2)
    contour,hier = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    best_cnt = None
    for cnt in contour:
        area = cv2.contourArea(cnt)
        if area > 1000:
            if area > max_area:
                max_area = area
                best_cnt = cnt

    cv2.drawContours(mask,[best_cnt],0,255,-1)
    cv2.drawContours(mask,[best_cnt],0,0,2)

    res = cv2.bitwise_and(res,mask)
    
    
    cv2.imshow('Blurry image',res)
    
    kernelx = cv2.getStructuringElement(cv2.MORPH_RECT,(2,10))

    dx = cv2.Sobel(res,cv2.CV_16S,1,0)
    dx = cv2.convertScaleAbs(dx)
    cv2.normalize(dx,dx,0,255,cv2.NORM_MINMAX)
    ret,close = cv2.threshold(dx,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    close = cv2.morphologyEx(close,cv2.MORPH_DILATE,kernelx,iterations = 1)

    contour, hier = cv2.findContours(close,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        x,y,w,h = cv2.boundingRect(cnt)
        if h/w > 5:
            cv2.drawContours(close,[cnt],0,255,-1)
        else:
            cv2.drawContours(close,[cnt],0,0,-1)
    close = cv2.morphologyEx(close,cv2.MORPH_CLOSE,None,iterations = 2)
    closex = close.copy()
    
    cv2.imshow('Blurry image',closex)
    
    kernely = cv2.getStructuringElement(cv2.MORPH_RECT,(10,2))
    dy = cv2.Sobel(res,cv2.CV_16S,0,2)
    dy = cv2.convertScaleAbs(dy)
    cv2.normalize(dy,dy,0,255,cv2.NORM_MINMAX)
    ret,close = cv2.threshold(dy,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    close = cv2.morphologyEx(close,cv2.MORPH_DILATE,kernely)

    contour, hier = cv2.findContours(close,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        x,y,w,h = cv2.boundingRect(cnt)
        if w/h > 5:
            cv2.drawContours(close,[cnt],0,255,-1)
        else:
            cv2.drawContours(close,[cnt],0,0,-1)

    close = cv2.morphologyEx(close,cv2.MORPH_DILATE,None,iterations = 2)
    closey = close.copy()
    
    cv2.imshow('Blurry image',closey)

    
    
    
    res = cv2.bitwise_and(closex,closey)

    
    
    contour, hier = cv2.findContours(res,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    centroids = []
    for cnt in contour:
        mom = cv2.moments(cnt)
        (x,y) = int(mom['m10']/mom['m00']), int(mom['m01']/mom['m00'])
        cv2.circle(img,(x,y),4,(0,255,0),-1)
        centroids.append((x,y))



    centroids = np.array(centroids,dtype = np.float32)
    c = centroids.reshape((100,2))
    c2 = c[np.argsort(c[:,1])]

    b = np.vstack([c2[i*10:(i+1)*10][np.argsort(c2[i*10:(i+1)*10,0])] for i in range(10)])
    bm = b.reshape((10,10,2))


    #output = np.zeros((450,450,3),np.uint8)
    output = np.zeros((450,450,3),np.uint8)
    for i,j in enumerate(b):
        ri = int(i/10)
        ci = int(i%10)
        if ci != 9 and ri!=9:
            src = bm[ri:ri+2, ci:ci+2 , :].reshape((4,2))
            dst = np.array( [ [ci*50,ri*50],[(ci+1)*50-1,ri*50],[ci*50,(ri+1)*50-1],[(ci+1)*50-1,(ri+1)*50-1] ], np.float32)
            retval = cv2.getPerspectiveTransform(src,dst)
            warp = cv2.warpPerspective(res2,retval,(450,450))
            output[ri*50:(ri+1)*50-1 , ci*50:(ci+1)*50-1] = warp[ri*50:(ri+1)*50-1 , ci*50:(ci+1)*50-1].copy()
 
    x = 0
    y = 0
    Squares = []
    
    ret,thresh2 = cv2.threshold(output,180,255,cv2.THRESH_BINARY_INV)
    for i in range(0,400,50):
        for j in range(0,400,50):
            Squares.append(cv2.rectangle(cv2.resize(thresh2[i:i+50,j:j+50],(28,28)), (0,0), (28,28), (0,0,0), 5))
            

    digits = np.zeros((81,), dtype=int)

    for i in range(len(Squares)):
        #cv2.imwrite("trainingData//square " + str(i) + " b.png", Squares[i])
        digits[i] = 2
    return Squares, output






























    
    #cv2.imshow('Original image',image)

    

    