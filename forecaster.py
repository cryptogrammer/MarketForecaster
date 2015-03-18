import csv
import math
import copy
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import QSTK.qstkutil.DataAccess as da
import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkstudy.EventProfiler as ep
import QSTK.qstkutil.qsdateutil as du
from sklearn.neighbors import KNeighborsRegressor

def simpleMovingAverage(base):
    return sum(base)/len(base)

def exponentialMovingAverage(base):
    alpha = 0.9
    sum = 0
    weight = 1.0
    for b in base:
        sum = sum + b*weight
        weight = weight*alpha
    return sum/(len(base))

def getAmplitude(base):
    return max(base)-min(base)

def rmsAmplitude(base):
    return (max(base)-simpleMovingAverage(base))/2**0.5

def getFrequency(base):
    mean = sum(base) / len(base)
    f = 0
    for i in range(0, 20):
        if base[i]<=mean and base[i+1]>mean:
            f = f + 1
        elif base[i]<mean and base[i+1]>=mean:
            f = f + 1
        elif base[i]>=mean and base[i+1]<mean:
            f = f + 1
        elif base[i]>mean and base[i+1]<=mean:
            f = f + 1
    return f

def getTrainData():
    filenames = []
    for i in range (0, 200):
        if i < 10:
            filename = 'ML4T-00'+str(i)
        elif i<=99:
            filename = 'ML4T-0'+str(i)
        else:
            filename = 'ML4T-'+str(i)           
        filenames.append(filename)
    start = dt.datetime(2001,01,01)
    end = dt.datetime(2005, 12, 31)
    timeofday = dt.timedelta(hours=16)
    timestamps = du.getNYSEdays(start, end, timeofday)
    dataobj = da.DataAccess('Yahoo')
    keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']
    data = dataobj.get_data(timestamps, filenames, keys)
    dataDic = dict(zip(keys, data))
    for key in keys:
        dataDic[key] = dataDic[key].fillna(method='ffill')
        dataDic[key] = dataDic[key].fillna(method='bfill')
        dataDic[key] = dataDic[key].fillna(1.0)
    priceTrain = dataDic['actual_close'].values
    Xtrain, Ytrain, Y = train(priceTrain)
    return Xtrain, Ytrain

def train(data):
    count = 0
    maxVal = data.shape[1]
    # initialize Xtrain, Ytrain and Y
    Xtrain = np.zeros([maxVal*(len(data)-25),3])
    Ytrain = np.zeros([maxVal*(len(data)-25),1])
    Y = np.zeros([maxVal*(len(data)-25),1])
    for i in range (0,maxVal):
        for j in range (0, len(data)-25):
            base = data[j:j+21, i]
            x1 = simpleMovingAverage(base)
            x2 = rmsAmplitude(base)
            x3 = getFrequency(base)
            Xtrain[count, 0] = x1
            Xtrain[count, 1] = x2
            Xtrain[count, 2] = x3
            Ytrain[count, 0] = (data[j+25,i]-data[j+20,i])/data[j+20,i]
            Y[count, 0] = data[j+20,i]
            count = count + 1
    return Xtrain, Ytrain, Y

def getTestData():
    start = dt.datetime(2006,01,06)
    end = dt.datetime(2007,12,07)
    timeofday = dt.timedelta(hours=16)
    timestamps = du.getNYSEdays(start, end, timeofday)
    dataobj = da.DataAccess('Yahoo')
    keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']
    filenames = []    
    filenames.append('ML4T-292')

    data = dataobj.get_data(timestamps, filenames, keys)
    dataDic = dict(zip(keys, data))  
    for key in keys:
        dataDic[key] = dataDic[key].fillna(method='ffill')
        dataDic[key] = dataDic[key].fillna(method='bfill')
        dataDic[key] = dataDic[key].fillna(1.0)
    priceTest = dataDic['actual_close'].values
    Xtest, Y, Ytest = train(priceTest)
    return Xtest, Ytest    

def rootMeanSquareError(Y, Ytest):
    total = 0
    for i in range(0, len(Y)):
        total = total + (Y[i] - Ytest[i]) * (Y[i] - Ytest[i])
    return math.sqrt(total/len(Y))

def correlationCoefficient(Y, Ytest):
    corr = np.corrcoef(Y, Ytest)
    return corr[0,1]

def createScatterPlot(xLabel, yLabel, xData, yData, filename):
    plt.clf()
    fig = plt.figure()
    plt.plot(xData, yData, 'o')
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.savefig(filename, format='pdf')

def createComparisonPlot(xLabel, yLabel, xData, y1Data, y2Data, filename, linename):
    plt.clf()
    fig = plt.figure()
    fig.add_subplot(111)
    plt.plot(xData, y1Data, color = 'blue')
    plt.plot(xData, y2Data, color = 'red')
    plt.legend(linename)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.savefig(filename, format='pdf')

def createComparisonPlotFeatures(xLabel, yLabel, xData, y1Data, y2Data, y3Data, filename, linename):
    plt.clf()
    fig = plt.figure()
    fig.add_subplot(111)
    plt.plot(xData, y1Data)
    plt.plot(xData, y2Data)
    plt.plot(xData, y3Data)
    plt.legend(linename)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.savefig(filename, format='pdf')
    
    
def test():
    Xtrain, Ytrain = getTrainData()
    Xtest, Ytest, = getTestData()
    x1_200_plot = Xtest[0:200, 0]
    x2_200_plot = Xtest[0:200, 1]
    x3_200_plot = Xtest[0:200, 2]
    Y = Ytest[:,0]
    yVals = Y[0:200]
    lastY = Y[len(Y)-200: len(Y)]
    firstDate = np.zeros([200])
    lastDate = np.zeros([200])
    for i in range (0, 200):
        firstDate[i] = i-1
        lastDate[i] = i-1 
    k = 3
    learner = KNeighborsRegressor(k)     
    predicted = learner.fit(Xtrain, Ytrain).predict(Xtest)
    predictedY = predicted[:,-1]

    for i in range (0, predictedY.shape[0]):
        predictedY[i] = (predictedY[i] + 1) * Y[i]
    unpredictedYValues = []
    unpredictedYValues.append(yVals[0])
    unpredictedYValues.append(yVals[0])
    unpredictedYValues.append(yVals[0])
    unpredictedYValues.append(yVals[0])
    unpredictedYValues.append(yVals[0])
    for i in range(5,200):
        unpredictedYValues.append(predictedY[i])
    endYVals = predictedY[len(predictedY)-200: len(predictedY)]

    rmsError = rootMeanSquareError(predictedY, Y)

    corrCoeff = correlationCoefficient(predictedY, Y)

    writer = csv.writer(open('rmsAndCorrCoeff.csv', 'wb'), delimiter=',')
    writer.writerow(['rms', rmsError])
    writer.writerow(['corr', corrCoeff])

    kIndices = np.zeros([200])
    for k in range(0, 200):
        kIndices[k] = k+1
    createComparisonPlot('Days', 'Price', kIndices, yVals, unpredictedYValues, 'first_200_comparison_plot.pdf', ['Y actual', 'Y predict'])
    createComparisonPlot('Days', 'Price', kIndices, lastY, endYVals, 'last200Comparison.pdf', ['Y actual', 'Y predict'])
    createScatterPlot('Predicted Y', 'Actual Y', predictedY, Y, 'scatter plot.pdf')
    createComparisonPlotFeatures('Dates', 'Values', firstDate, x2_200_plot, x1_200_plot, x3_200_plot, 'first200Features.pdf', ['RMSAmplitude','EMA', 'Frequency'])

if __name__ == '__main__':
    test()    




       
        


