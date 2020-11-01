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


class GaussianNaiveBayes:

	classDict = None
	meanDict = None
	varianceDict = None
	totalRows = None

	def getGNBProbability(self, x):

		classDict = self.classDict
		mean = self.meanDict
		variance = self.varianceDict

		maxProb = float('-inf')
		maxClass = -1

		for i in classDict.keys():
			probY = np.log(len(classDict[i]) / self.totalRows)
			totProb = np.multiply(-0.5, np.sum(np.log(np.multiply(np.multiply(2, np.pi), np.add(variance[i], 0.00001)))))
			totProb = np.subtract(totProb, np.multiply(0.5, np.sum(np.divide(np.square(np.subtract(x, mean[i])), np.add(variance[i], 0.00001)))))
			totProb = np.add(totProb, probY)
			
			if(totProb > maxProb):
				maxProb = totProb
				maxClass = i

		return maxClass

	def fit(self, X, Y):
		classDictionary = OrderedDict()
		numRows = Y.size

		for i in range(numRows):
			if Y[i] in classDictionary:
				classDictionary[Y[i]].append(X[i])
			else:
				classDictionary[Y[i]] = [X[i]]

		meanDict = OrderedDict()
		varianceDict = OrderedDict()

		#Finding mean for all features in each group
		for i in classDictionary.keys():
			for j in classDictionary[i]:
				if(i not in meanDict):
					meanDict[i] = j
				else:
					meanDict[i] = np.add(meanDict[i], j)

		for i in meanDict.keys():
			meanDict[i] = np.divide(meanDict[i], len(classDictionary[i]))
		
		#Finding variance of all features in each group
		for i in classDictionary.keys():
			for j in classDictionary[i]:
				if(i not in varianceDict):
					varianceDict[i] = np.square(j - meanDict[i])
				else:
					varianceDict[i] = np.add(varianceDict[i], np.square(j - meanDict[i]))
		for i in classDictionary.keys():
			if(len(classDictionary[i])>1):
				varianceDict[i] = np.divide(varianceDict[i], len(classDictionary[i]) - 1)
		
		self.meanDict = meanDict
		self.varianceDict = varianceDict
		self.classDict = classDictionary
		self.totalRows = numRows

		
	def predict(self, X):
		if(self.classDict is None or self.meanDict is None or self.varianceDict is None):
			print("Predict was run before fit or some error occurred while running fit")
			return None

		meanDict = self.meanDict
		varianceDict = self.varianceDict
		classDict = self.classDict
		totalRows = self.totalRows

		#Output array
		Y = []

		#Find class for each input row
		for i in X:
			maxProbClass = self.getGNBProbability(i)  
			Y.append(maxProbClass)

		return Y

gaussianNaiveBayes = GaussianNaiveBayes()
X, Y = getDatasetTwo()

trainX, testX, trainY, testY = train_test_split(X, Y, test_size = 0.2, random_state=0)

gaussianNaiveBayes.fit(trainX, trainY)
output = gaussianNaiveBayes.predict(testX)

counter = 0
for i in range(len(output)):
  if(output[i]==testY[i]):
    counter = counter + 1
print("Accuracy through our implementations is", counter/len(output))

#SKLearn implementation
gaussianNB = GaussianNB()
gaussianNB.fit(trainX, trainY)
print("Accuracy through sklearn's implementations is", gaussianNB.score(testX, testY))


