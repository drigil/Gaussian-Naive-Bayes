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


#Function for grid search
def gridSearch(trainX, trainY, valX, valY, model, paramsDict):

	#Create all possible value combinations
	valueList = list(itertools.product(*paramsDict.values()))
	keyList = list(paramsDict.keys())
	numParams = len(paramsDict.keys())
	finalDict = {}

	for i in valueList:
		newParamsDict = {}
		for j in range(numParams):
			newParamsDict[keyList[j]] = i[j]
		model.set_params(**newParamsDict)
		model.fit(trainX, trainY)
		score = model.score(valX, valY)
		finalDict[i] = score


	return finalDict

#Function for K fold
def kfold(X, Y, model, trainRatio, valRatio, paramsDict = None, isGaussian = False):
	
	if(isGaussian == False and paramsDict is None):
			print("paramsDict is required for Decision Tree to perform Grid Search")
			return None
	if(trainRatio + valRatio >= 1):
			print("Enter appropriate ratios")
			return None
		
	#Separating out test data
	numRows = X.shape[0]
	testRatio = 1 - (trainRatio + valRatio)

	testX = X[int(numRows * (trainRatio + valRatio)):, :]
	testY = Y[int(numRows * (trainRatio + valRatio))]

	X = X[:int(numRows * (trainRatio + valRatio)), :]
	Y = Y[:int(numRows * (trainRatio + valRatio))]
	
	#Finding number of rows in validation data
	numRows = X.shape[0]
	valRatio = valRatio / (1 - testRatio)
	numValRows = int(valRatio * numRows)
	numFolds = int(1 / valRatio)

	maxAccuracy = 0
	bestFold = -1

	#for first fold
	valX = X[0:numValRows, :]
	valY = Y[0:numValRows]

	trainX = X[numValRows: , :]
	trainY = Y[numValRows:]

	if(isGaussian==False):
		currScoreDict = gridSearch(trainX, trainY, valX, valY, model, paramsDict)
		print("Accuracy for parameters for fold 1 are", currScoreDict)
		avgAccuracyDict = currScoreDict

	else:
		model.fit(trainX, trainY)
		currScore = model.score(valX, valY)
		avgAccuracy = currScore
		print("Accuracy for fold 1 is", currScore)
		if(currScore > maxAccuracy):
			maxAccuracy = currScore
			bestFold = 1

	# #For fold 2 to second last fold
	for i in range(1, numFolds - 1, 1):

		valX = X[numValRows*i:numValRows*(i+1), :]
		valY = Y[numValRows*i:numValRows*(i+1)]

		trainX_1 = X[0:numValRows*i , :]
		trainY_1 = Y[0:numValRows*i]

		trainX_2 = X[numValRows*(i+1): , :]
		trainY_2 = Y[numValRows*(i+1):]

		trainX = np.concatenate((trainX_1, trainX_2))
		trainY = np.concatenate((trainY_1, trainY_2))

		if(isGaussian == False):
			currScoreDict = gridSearch(trainX, trainY, valX, valY, model, paramsDict)
			print("Accuracy for parameters for fold " +  str(i + 1) + " are " + str(currScoreDict))
			for j in avgAccuracyDict.keys():
				avgAccuracyDict[j] = avgAccuracyDict[j] + currScoreDict[j]

		else:
			model.fit(trainX, trainY)
			currScore = model.score(valX, valY)
			avgAccuracy = currScore + avgAccuracy
			print("Accuracy for fold " +  str(i + 1) + " is " + str(currScore))
			if(currScore > maxAccuracy):
				maxAccuracy = currScore
				bestFold = i + 1

	#for last fold
	testX = X[numValRows*(numFolds-1):, :]
	testY = Y[numValRows*(numFolds-1):]

	trainX = X[0:numValRows*(numFolds-1): , :]
	trainY = Y[0:numValRows*(numFolds-1):]
	
	if(isGaussian==False):
		currScoreDict = gridSearch(trainX, trainY, valX, valY, model, paramsDict)
		print("Accuracy for parameters for fold " +  str(numFolds) + " are " + str(currScoreDict))
		for i in avgAccuracyDict.keys():
			avgAccuracyDict[i] = avgAccuracyDict[i] + currScoreDict[i]
		for i in avgAccuracyDict.keys():
			avgAccuracyDict[i] = avgAccuracyDict[i]/numFolds
		
		print("Average Accuracy for all folds is " + str(avgAccuracyDict))

		maxAccuracy = 0
	
		for i in avgAccuracyDict:
			if(avgAccuracyDict[i] > maxAccuracy):
				maxAccuracy = avgAccuracyDict[i]
				bestParams = i

		print("Best accuracy was achieved for parameters", bestParams)

		return [bestParams, testX, testY]


	else:
		model.fit(trainX, trainY)
		currScore = model.score(valX, valY)
		avgAccuracy = currScore + avgAccuracy
		print("Accuracy for fold " +  str(numFolds) + " is " + str(currScore))
		avgAccuracy = avgAccuracy / numFolds
		print("Average accuracy is", avgAccuracy)
		if(currScore > maxAccuracy):
			maxAccuracy = currScore
			bestFold = numFolds

		print("Best Accuracy of " + str(maxAccuracy) + " was achieved at fold number " + str(bestFold))
		return [bestFold, testX, testY]

 
#Plots for decision tree vs depth 
def plotAccuracy(X, Y, trainRatio, valRatio, depthArr, plotMaxDepth = True):
	#Separating out test data
	numRows = X.shape[0]
	testRatio = 1 - (trainRatio + valRatio)
	testX = X[int(numRows * (trainRatio + valRatio)):, :]
	testY = Y[int(numRows * (trainRatio + valRatio)):]

	X = X[:int(numRows * (trainRatio + valRatio)), :]
	Y = Y[:int(numRows * (trainRatio + valRatio))]

	numRows = X.shape[0]
	trainRatio = trainRatio / (1 - testRatio)
	numTrainRows = int(numRows * trainRatio)



	trainX = X[:numTrainRows, :]
	trainY = Y[:numTrainRows]
	valX = X[numTrainRows:, :]
	valY = Y[numTrainRows:]

	decisionTree = DecisionTreeClassifier(random_state = 0)
	trainAccArr = []
	valAccArr = []
	newDepthArr = []
	for i in depthArr:
		decisionTree.set_params(**{'max_depth':i})
		decisionTree.fit(trainX, trainY)
		newDepthArr.append(decisionTree.get_depth())
		trainAccArr.append(decisionTree.score(trainX, trainY))
		valAccArr.append(decisionTree.score(valX, valY))

	#Train test
	if(plotMaxDepth==True):
		plt.plot(depthArr, trainAccArr, label = "Train")
	else:
		plt.plot(newDepthArr, trainAccArr, label = "Train")
	
	#Validation test
	if(plotMaxDepth==True):
		plt.plot(depthArr, valAccArr, label = "Validation")
	else:
		plt.plot(newDepthArr, valAccArr, label = "Validation")
	plt.xlabel("Depth")
	plt.ylabel("Accuracy")
	plt.legend()
	plt.show()
	
#Saving and loading models
def saveModel(model, filename):
	# save the model to disk
	pickle.dump(model, open(filename, 'wb'))


def loadModel(filename):
	# load the model from disk
	loaded_model = pickle.load(open(filename, 'rb'))
	return loaded_model

#Function for getting accuracy
def getAccuracy(predY, testY):
	#Calculating accuracy
	accuracy = 0
	lenY = len(testY)
	for i in range(lenY):
		if(predY[i] == testY[i]):
			accuracy = accuracy + 1
	accuracy = accuracy / lenY
	return accuracy


#Precision, Recall, F1 Score using Macro Averaging 
def getPrecisionRecallF1Score(numClasses, predY, testY):
	
	lenY = len(testY)
	# using list comprehension to initializing matrix
	macroMatrix = [ [ 0 for i in range(numClasses) ] for j in range(numClasses) ]

	#Each row is predicted and column is actual value
	for i in range(lenY):
		macroMatrix[predY[i]][testY[i]] = macroMatrix[predY[i]][testY[i]] + 1

	avgPrecision = 0
	precisionDict = {}

	avgRecall = 0
	recallDict = {}

	avgF1Score = 0
	f1ScoreDict = {}

	for i in range(numClasses):
		reqVal = 0
		totVal = 0 #Rows
		totVal2 = 0 #Column

		for j in range(numClasses):
			currVal = macroMatrix[i][j]
			currVal2 = macroMatrix[j][i]
			
			if(i==j):
				reqVal = currVal

			totVal = totVal + currVal
			totVal2 = totVal2 + currVal2

		precisionDict[i] = reqVal / totVal
		recallDict[i] = reqVal / totVal2
		f1ScoreDict[i] = (2*precisionDict[i]*recallDict[i]) / (precisionDict[i] + recallDict[i]) 
		
		
		avgPrecision = avgPrecision + (reqVal / totVal)
		avgRecall = avgRecall + (reqVal / totVal2)
		avgF1Score = avgF1Score + (2*precisionDict[i]*recallDict[i]) / (precisionDict[i] + recallDict[i])
		
	avgPrecision = avgPrecision / numClasses
	avgRecall = avgRecall / numClasses
	avgF1Score = avgF1Score / numClasses

	return [avgPrecision, precisionDict, avgRecall, recallDict, avgF1Score, f1ScoreDict, macroMatrix]



#Precision, Recall, F1 Score using Micro Averaging 
def getPrecisionRecallF1ScoreMicro(numClasses, predY, testY):
	
	lenY = len(testY)

	# using list comprehension to initializing matrix
	microMatrix = [[0,0], [0,0]]
	microMatrix = np.array(microMatrix)

	#Each row is predicted and column is actual value
	for i in range(numClasses):
		tempMatrix = [[0,0], [0,0]]
		tempMatrix = np.array(tempMatrix)
		for j in range(lenY):
			if(predY[j] == i):
				if(testY[j]==i):
					tempMatrix[0][0] = tempMatrix[0][0] + 1
				else:
					tempMatrix[0][1] = tempMatrix[0][1] + 1
			else:
				if(testY[j]==i):
					tempMatrix[1][0] = tempMatrix[1][0] + 1
				else:
					tempMatrix[1][1] = tempMatrix[1][1] + 1
					
		microMatrix = np.add(microMatrix, tempMatrix)
		
	avgPrecision = microMatrix[0][0] / (microMatrix[0][0] + microMatrix[0][1])
	avgRecall = microMatrix[0][0] / (microMatrix[0][0] + microMatrix[1][0])
	avgF1Score = (2*avgPrecision*avgRecall) / (avgPrecision + avgRecall)
	
	return [avgPrecision, avgRecall, avgF1Score]


#ROC
def getROC(numClasses, probY, testY):
	
	lenY = len(testY)
	
	#Each row is predicted and column is actual value
	for i in range(numClasses):
		truePosArr = [0]
		falsePosArr = [0]
		for thresh in range(0, 1501, 1) :
			k = np.exp(-thresh/2)
			tempMatrix = [[0,0], [0,0]]
			tempMatrix = np.array(tempMatrix)
			for j in range(lenY):
				if(probY[j][i] >= k):
					if(testY[j]==i):
						tempMatrix[0][0] = tempMatrix[0][0] + 1
					else:
						tempMatrix[0][1] = tempMatrix[0][1] + 1
				else:
					if(testY[j]==i):
						tempMatrix[1][0] = tempMatrix[1][0] + 1
					else:
						tempMatrix[1][1] = tempMatrix[1][1] + 1

			truePositiveRate = tempMatrix[0][0] / (tempMatrix[0][0] + tempMatrix[1][0])
			falsePositiveRate = tempMatrix[0][1] / (tempMatrix[0][1] + tempMatrix[1][1])
			truePosArr.append(truePositiveRate)
			falsePosArr.append(falsePositiveRate)

		plt.plot(falsePosArr, truePosArr, label = "Class is "+ str(i))
		
	
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.legend()
	plt.show()


#Final function that compiles other functions
def evaluateModel(model, testX, testY):

	predY = model.predict(testX)
	probY = model.predict_proba(testX)

	accuracy = getAccuracy(predY, testY)
	avgPrecision, precisionDict, avgRecall, recallDict, avgF1Score, f1ScoreDict, confusionMatrix = getPrecisionRecallF1Score(len(model.classes_), predY, testY)
	avgPrecisionMicro, avgRecallMicro, avgF1ScoreMicro = getPrecisionRecallF1ScoreMicro(len(model.classes_), predY, testY)

	print("Confusion Matrix - ")
	for i in confusionMatrix:
		print(i)
	print("Accuracy", accuracy)
	print("Macro precission", avgPrecision)
	print("Precission for every class", precisionDict)
	print("Macro Recall",avgRecall)
	print("Recall for every class",recallDict)
	print("Macro F1 score",avgF1Score)
	print("F1 score for every class",f1ScoreDict)
	print("Micro Precission",avgPrecisionMicro)
	print("Micro Recall", avgRecallMicro)
	print("Micro F1Score", avgF1ScoreMicro)
	getROC(len(model.classes_), probY, testY) # Plots the graph directly



#Testing all functions
#Split data into training and testing
X, Y = getDatasetTwo()
trainX, testX, trainY, testY = train_test_split(X, Y, test_size = 0.2, random_state=0)


#Decision Tree
decisionTree = DecisionTreeClassifier(random_state = 0)
decisionTree.fit(trainX, trainY)

#Gaussian Naive Bayes
gaussianNaiveBayes = GaussianNB()
gaussianNaiveBayes.fit(trainX,trainY)

#(a)
# depth_arr = []
# for i in range(1, 25, 1):
# 	depth_arr.append(i)
# depth_param = {'max_depth': depth_arr}

# bestParams, testX, testY = kfold(X, Y, decisionTree, 0.6, 0.2, paramsDict=depth_param)  #Kfold implementation
# bestParams, testX, testY = kfold(X, Y, gaussianNaiveBayes, 0.6, 0.2, isGaussian = True)  #Kfold implementation


#(b)
# depth_arr = []
# for i in range(1, 200, 2):
# 	depth_arr.append(i)
# depth_param = {'max_depth': depth_arr}
# plotAccuracy(X, Y, 0.6, 0.2, depth_arr) #Plotting graph


#(c)
#Saving Model and retrieving it
#Decision Tree
# filename = dir_path + "/Weights/decisionTreeA.sav"
# decisionTree.set_params(**{'max_depth':16})	
# saveModel(decisionTree, filename)
# score = loadModel(filename).score(testX, testY)
# print(score)

# filename = dir_path + "/Weights/decisionTreeB.sav"
# decisionTree.set_params(**{'max_depth':20})	
# saveModel(decisionTree, filename)
# score = loadModel(filename).score(testX, testY)
# print(score)

#Gaussian Naive Bayes
# filename = dir_path + "/Weights/gaussianNaiveBayesA.sav"	
# saveModel(gaussianNaiveBayes, filename)
# score = loadModel(filename).score(testX, testY)
# print(score)

# filename = dir_path + "/Weights/gaussianNaiveBayesB.sav"	
# saveModel(gaussianNaiveBayes, filename)
# score = loadModel(filename).score(testX, testY)
# print(score)


#evaluateModel(decisionTree, testX, testY)



