import complexWorker as cw
import multiprocessing as mp
import os
import cv2 
import joblib
import numpy as np
from time import sleep
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import tensorflow.keras as keras
from BoggleSolver import boggle1 as bg

def ShowFinalImage(queue1, queue2):
    while True:

        if queue2.empty() == False:
            cropped = queue2.get()
            cv2.imshow('Final Image',cropped)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        if queue1.empty():
            sleep(0.05)
            continue
        img = queue1.get()
        cv2.imshow("Cropped (warp affine)" + str(os.getppid()), img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

#queue is cropped and black and white 
def AttemptSudokuSolve(InputQueue, ProcessedDigits, FinalImage):

    loaded_model = joblib.load('finalized_sudoku_model.sav')
    from dlxsudoku import Sudoku
    font = cv2.FONT_HERSHEY_SIMPLEX
    result_old = np.zeros(81)
    result_really_old = result_old
    result = np.zeros(81)
    while True:

        
        gray = InputQueue.get()
        img, cropped, digits = cw.extract_sudoku(gray)
        boardFound = 1
        result_really_really_old = result_really_old
        result_really_old = result_old
        result_old = result
        sleep(0.05)
		
        ProcessedDigits.put(img)
        


        result = loaded_model.predict(digits)
        resultList = (result_old, result_really_old, result_really_really_old)
        for resul in resultList:
            for i in result:
                if result[i] == 10:
                    if resul[i] != 10:
                        #print('correction2')
                        print(str(result[i]) + " changed to " + str(int(resul[i])))
                        result[i] = int(resul[i])
        temp = 0
        for i in result:
            if i == 10:
                temp = temp + 1
                result[i] = 0
        if temp >= 6:
            continue


        res = ""
        grid = np.zeros((9,9), dtype='int')
        grid = result.reshape(9,9)
        #Getting the set of numbers within a number's group------------------------------------


        for re in result:
            if re == 10:
                boardFound = 0
                break
            else:
                res += str(re)
        if boardFound == 1:
            #print(res)
            
            try:
                #start = time.time()
                s1 = Sudoku(res)
                s1.solve(verbose=False, allow_brute_force=True)
                #print("YAY!")
                stringlist = s1.to_oneliner()
                charlist = list(stringlist)
                results = list(map(int, charlist))
                #print(results)

                y = cropped.shape[0]
                cell_Spacing = y//9
                posx = cell_Spacing//2 - 4
                posy = cell_Spacing//2 + 6

                for i in range(81):
                    if results[i] != result[i]:
                        cropped = cv2.putText(cropped, charlist[i], (posx, posy), font,  
                        0.6, (0,0,255), 2, cv2.LINE_AA) 
                    if posx + cell_Spacing >= y:
                        posy = posy + cell_Spacing
                        posx = cell_Spacing//2 - 4
                    else:
                        posx = posx+cell_Spacing
                FinalImage.put(cropped)
                

            except: 
                pass
                #print("too many solutions")
        else:
            pass

def AttemptBoggleSolve(InputQueue, ProcessedDigits, FinalImage):
    trainedModel = keras.models.load_model('TrainedBoggle')
    yRepresentations = ('a0','a1','a2','a3','b0','b1','b2','b3','c0','c1','c2','c3','d0','d1','d2','d3','e0','e1','e2','e3','f0','f1','f2','f3','g0','g1','g2','g3','h0','h1','h2','h3','i0','i1','i2','i3','j0','j1','j2','j3','k0','k1','k2','k3','l0','l1','l2','l3','m0','m1','m2','m3','n0','n1','n2','n3','o0','o1','o2','o3','p0','p1','p2','p3','q0','q1','q2','q3','r0','r1','r2','r3','s0','s1','s2','s3','t0','t1','t2','t3','u0','u1','u2','u3','v0','v1','v2','v3','w0','w1','w2','w3','x0','x1','x2','x3','y0','y1','y2','y3','z0','z1','z2','z3','00','11')

    while True:
        gray = InputQueue.get()
        img, cropped, digits = cw.extract_boggle(gray)
        ProcessedDigits.put(img)
        digits = np.array(digits)
        y_test_pred = trainedModel.predict_classes(digits, verbose=0)
        PredictedVals = []
        for index in y_test_pred:
            PredictedVals.append(yRepresentations[index][0])
        if '0' in PredictedVals:
            continue
        if '1' in PredictedVals:
            continue
        tempVal = 0
        tempVal2 = 0

        PredictedVals = [x.upper() for x in PredictedVals] 
        for i in range(len(PredictedVals)):
            if PredictedVals[i] == 'Q':
                PredictedVals[i] = "Qu"
        bg.SolveBoard(PredictedVals)
        while(True):
            cv2.imshow( "FinalImage", cropped )
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
