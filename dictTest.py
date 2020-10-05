import os
import pickle
import time
def convertToInt(word):
    tempstr = ''
    for j in word:
        if(ord(j) == 10):
            break
        tempstr = tempstr + str(ord(j))
        
    return int(tempstr)

filehandler = open("tree.obj", "rb")

bstb = pickle.load(filehandler)
start = time.time()        
        bstb.find(convertToInt("terrorism"))
end = time.time() 
print((end - start))
