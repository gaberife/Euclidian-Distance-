
import pandas as pd
import math 
import numpy as np
import scipy.spatial
import timeit

# =============================================================================
# Answer to #7:
#     Euclidian methods in order of time performance from slowest to fastest 
#       1. euclideanDist3
#       2. euclideanDist4
#       3. euclideanDist1
#       4. euclideanDist5
#       5. euclideanDist2 
# 
# Answer to #8:
#         Euclidian methods in order of time performance from slowest to fastest 
#       1. euclideanDist3
#       2. euclideanDist4
#       3. euclideanDist1
#       4. euclideanDist5
#       5. euclideanDist2
# Answer to #9:
#       euclidianDist3() 
# =============================================================================

'''
Use the default value of numRows (None) to read *all* rows
'''
def readData(numRows = None):
    inputCols = ["Alcohol", "Malic Acid", "Ash", "Alcalinity of Ash", "Magnesium", "Total Phenols", "Flavanoids", "Nonflavanoid Phenols", "Proanthocyanins", "Color Intensity", "Hue", "Diluted", "Proline"]
    outputCol = 'Class'
    colNames = [outputCol] + inputCols  # concatenate two lists into one
    wineDF = pd.read_csv("data/wine.data", header=None, names=colNames, nrows = numRows)
    return wineDF, inputCols, outputCol

def main():
    df, inputCols, outputCol = readData(3)

    a = df.iloc[0, :]
    b = df.iloc[1, :]
    c = df.iloc[2, :]
    
    addRandomCols(df, 100)
    
    startTime = timeit.default_timer()
    for i in range(1000):
        euclideanDist1(a, c)
        euclideanDist1(a, b)
        euclideanDist1(b, c)
        i += 1
    elapsedTime = (timeit.default_timer() - startTime)
    print("Euclidian distance of method #1: ", elapsedTime , " seconds")
    startTime = timeit.default_timer()
    for i in range(1000):
        euclideanDist2(a, c)
        euclideanDist2(a, b)
        euclideanDist2(b, c)
        i += 1
    elapsedTime = (timeit.default_timer() - startTime)
    print("Euclidian distance of method #2: ", elapsedTime , " seconds")
    startTime = timeit.default_timer()
    for i in range(1000):
        euclideanDist3(a, c)
        euclideanDist3(a, b)
        euclideanDist3(b, c)
        i += 1
    elapsedTime = (timeit.default_timer() - startTime)
    print("Euclidian distance of method #3: ", elapsedTime , " seconds")
    startTime = timeit.default_timer()
    for i in range(1000):
        euclideanDist4(a, c)
        euclideanDist4(a, b)
        euclideanDist4(b, c)
        i += 1
    elapsedTime = (timeit.default_timer() - startTime)
    print("Euclidian distance of method #4: ", elapsedTime , " seconds")
    startTime = timeit.default_timer()
    for i in range(1000):
        euclideanDist5(a, c)
        euclideanDist5(a, b)
        euclideanDist5(b, c)
        i += 1
    elapsedTime = (timeit.default_timer() - startTime)
    print("Euclidian distance of method #5: ", elapsedTime , " seconds")
          
def addRandomCols(df, numNewCols):
    newCols = pd.Series(['rndC'+str(i) for i in range(0, numNewCols)]) 
    newCols.map(lambda colName: addRandomCol(colName, df))

def addRandomCol(colName, df):
    df.loc[:, colName] = np.random.randint(-100, 100, df.shape[0])
    
def euclideanDist1(series, series2):
    i, total = 0, 0
    for ip in series :
        sub = ip - series2.iloc[i]
        total = total + (sub * sub)
        i+=1 
    dist = math.sqrt(total)
    return dist    

def euclideanDist2(p1, p2):
    amt = map(lambda x,y: (x - y) * (x - y), p1, p2)
    total = sum(amt)
    dist = math.sqrt(total)
    return dist    

def euclideanDist3(p1,p2):
    sqDiff = (p1[:] - p2[:]) * (p1[:] - p2[:])
    total = sqDiff.sum()
    dist = math.sqrt(total)
    return dist    

def euclideanDist4(p1, p2):
    return np.linalg.norm(p1 - p2)

def euclideanDist5(p1,p2):
    return scipy.spatial.distance.euclidean(p1,p2)
    
def test04():
    df, inputCols, outputCol = readData(3)
    a = df.iloc[0, :]
    c = df.iloc[2, :]
    print(euclideanDist1(a, c))
    print(euclideanDist2(a, c))
    print(euclideanDist3(a, c))
    print(euclideanDist4(a, c))
    print(euclideanDist5(a, c))
main()
