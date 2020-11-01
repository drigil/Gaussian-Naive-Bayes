import h5py
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import pandas
import random
import math
import pickle
import itertools
import os
from itertools import product 
from scipy.special import logsumexp
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split, GridSearchCV
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from collections import OrderedDict

dir_path = os.path.dirname(os.path.realpath(__file__))

#Method for getting data from dataset 1
def getDatasetOne():
	#Get dataset
	fileOne = h5py.File(dir_path + '/part_A_train.h5', 'r')
	listDataset = list(fileOne.keys())

	#Separate features and output
	X = fileOne['X']
	Y = fileOne['Y']
	X = np.array(X[:])
	Y = np.array(Y[:])

	#Convert one hot encoded output to label encoded output
	newY = np.where(Y==1)[1]
	Y = newY

	return [X,Y]  


#Method for getting data from dataset 2
def getDatasetTwo():
	#Get dataset
	fileOne = h5py.File(dir_path + '/part_B_train.h5', 'r')
	listDataset = list(fileOne.keys())

	#Separate features and output
	X = fileOne['X']
	Y = fileOne['Y']
	X = np.array(X[:])
	Y = np.array(Y[:])

	#Convert one hot encoded output to label encoded output
	newY = np.where(Y==1)[1]
	Y = newY
	return[X, Y]


#Method for getting data from dataset 3
def getDatasetThree():
	#Get dataset
	dataframe = pandas.read_csv(dir_path + '/weight-height.csv')

	#Separate features and output
	Height = dataframe['Height'].to_numpy()
	Gender = dataframe['Gender'].to_numpy()
	Weight = dataframe['Weight'].to_numpy()
	
	#Label Encoding Gender
	labelEncoder = preprocessing.LabelEncoder()
	Gender = labelEncoder.fit_transform(Gender)

	return [Height, Weight, Gender]


#Bootstrapping on Linear Regression
#numElements is the size of bootstrap sample

# In this method we first create n bootstrap samples. Then we fit model on each sample and predict values corresponding to each x in testx.
# To find bootstrap estimate we apply the formulas for every point on x, i.e. for every point x in test we average over values obtained by each sample
# and call this this the bootstrap estimate for that x. Finally we average out the values.

# In case of MSE and variance, we similarly apply formulas for each point x in test and then average them out

def bootstrapLinear(X, Y, numElements = None, test_ratio = 0.2):

	#Use holdout method to split X and Y, we create samples on train and test on test
	trainX, testX, trainY, testY = train_test_split(X, Y, test_size = test_ratio, random_state=0)
	
	#if numElements not given take max possible value
	if(numElements is None):
		numElements = len(trainY)
	if(numElements > len(trainX)):
		print("numElements can take max value of", len(trainY))
		return None

	linearRegression = LinearRegression()
	predArr = []
	lenY = len(testY)
	
	for i in range(numElements):
		sampleX = []
		sampleY = []
		
		for j in range(numElements):
			#Construct current sample using numElements random elements with replacement  
			currIndex = random.randint(0, numElements-1)
			sampleX.append(trainX[currIndex])
			sampleY.append(trainY[currIndex])

		sampleX = np.array(sampleX)
		sampleY = np.array(sampleY)
		sampleX = sampleX.reshape(-1, 1)
		testX = testX.reshape(-1, 1)
		
		#Use linear regression to fit currently found sample
		linearRegression.fit(sampleX, sampleY)
		predArr.append(linearRegression.predict(testX))

	predArr = np.array(predArr)

	#Find the bootstrap estimate by finding mean of all the coefficients
	bootstrapEst = []
	for i in range(0, lenY):
		avgPrediction = 0

		for j in range(0, numElements):
			avgPrediction = avgPrediction + predArr[j][i]

		avgPrediction = avgPrediction / numElements
		bootstrapEst.append(avgPrediction)
	
	bootstrapEst = np.array(bootstrapEst)
	bootstrapEstAvg = np.sum(bootstrapEst)
	bootstrapEstAvg = bootstrapEstAvg / lenY

	#finding bootstrap estimate  
	print("Bootstrap estimate for each x is")
	print(bootstrapEst)

	print("Average bootstrap estimate is", bootstrapEstAvg)
	
	#Finding bias 
	bias = np.subtract(bootstrapEst, np.array(testY))
	print("Bias for each x is ")
	print(bias)

	avgBias = np.sum(bias)
	avgBias = avgBias / lenY
	print("Average Bias is", avgBias)
	
	#Find Variance
	varianceArr = []
	for i in range(0, lenY):
		temp = 0
		for j in range(numElements):
			temp = temp + np.square(np.subtract(predArr[j][i], bootstrapEst[i]))
		temp = temp / (numElements-1)
		temp = np.sqrt(temp)
		varianceArr.append(temp)

	print("Variance for each x is")
	print(np.array(varianceArr))

	varianceAvg = np.sum(varianceArr)
	varianceAvg = varianceAvg / lenY

	print("Average Variance is", varianceAvg)

	#Find MSE
	MSEArr = []
	for i in range(lenY):
		temp = 0
		for j in range(numElements):
			temp = temp + np.square(np.subtract(predArr[j][i], testY[i]))
		temp = temp / numElements
		MSEArr.append(temp)

	print("MSE for each x is")
	print(np.array(MSEArr))

	avgMSE = np.sum(MSEArr)
	avgMSE = avgMSE / lenY

	print("Average MSE is", avgMSE)

	print("MSE - Bias^2 - variance is", np.mean(MSEArr - np.square(bias) - varianceArr))

#Bootstrapping on Linear Regression using different method
#numElements is the size of bootstrap sample

# In this method we first create n bootstrap samples. Then we fit model on each sample and predict values corresponding to each x in testx.
# Now however, we compute mean value for each bootstrap and store this as an array. Estimate is the mean of the previously found means. Bias is the estimate - mean(testY)
# In case of MSE and variance, we similarly apply formulas considering theta_b to be average values for sample b


def bootstrapLinearAlt(X, Y, numElements = None, test_ratio = 0.2):

	#Use holdout method to split X and Y, we create samples on train and test on test
	trainX, testX, trainY, testY = train_test_split(X, Y, test_size = test_ratio, random_state=0)
	
	#if numElements not given take max possible value
	if(numElements is None):
		numElements = len(trainY)
	if(numElements > len(trainX)):
		print("numElements can take max value of", len(trainY))
		return None

	linearRegression = LinearRegression()
	predArr = []
	lenY = len(testY)
	
	for i in range(numElements):
		sampleX = []
		sampleY = []
		
		for j in range(numElements):
			#Construct current sample using numElements random elements with replacement  
			currIndex = random.randint(0, numElements-1)
			sampleX.append(trainX[currIndex])
			sampleY.append(trainY[currIndex])

		sampleX = np.array(sampleX)
		sampleY = np.array(sampleY)
		sampleX = sampleX.reshape(-1, 1)
		testX = testX.reshape(-1, 1)
		
		#Use linear regression to fit currently found sample
		linearRegression.fit(sampleX, sampleY)
		predArr.append(linearRegression.predict(testX))

	predArr = np.array(predArr)

	#Find the bootstrap estimate by finding mean of all the coefficients
	bootstrapEst = []
	for i in predArr:
		temp = np.sum(i)
		temp = temp / lenY
		bootstrapEst.append(temp)
	
	bootstrapEst = np.array(bootstrapEst)
	bootstrapEstAvg = np.sum(bootstrapEst)
	bootstrapEstAvg = bootstrapEstAvg / numElements

	#Finding estimate
	print("Average bootstrap estimate is", bootstrapEstAvg)
	
	#Finding bias 
	testAvg = np.sum(testY)
	testAvg = testAvg / lenY
	bias = bootstrapEstAvg - testAvg
	print("Average Bias is", bias)
	
	#Find Variance
	varianceArr = []
	temp = 0
	for i in range(numElements):
		temp = temp + np.square(np.subtract(bootstrapEst[i], bootstrapEstAvg))
	temp = temp / numElements
	temp = np.sqrt(temp)
	variance = temp

	print("Average Variance is", variance)

	#Find MSE
	MSE = 0
	for i in range(numElements):
		MSE = MSE + np.square(np.subtract(bootstrapEst[i], testAvg))
	MSE = MSE / numElements

	print("Average MSE is", MSE)

	print("MSE - Bias^2 - variance is", MSE - np.square(bias) - variance)

X, Y, Z = getDatasetThree()

print("Bootstrap values using method 1 are")
bootstrapLinear(X, Y, 1000)

# print()

# print("Bootstrap values using method 2 are")
# bootstrapLinearAlt(X, Y, 1000)
