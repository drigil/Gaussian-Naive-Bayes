def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import h5py
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import pandas
import random
import math
import pickle
import os
import time
import itertools
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


#Method for stratified sampling and printing class frequencies
#if isPrint is true then class frequencies are printed
def stratifiedSampling(X, Y, test_size = 0.2, isPrint = False):
	#Splitting using stratified sampling
	trainInd_arr = []
	testInd_arr = []
	trainX = []
	trainY = []
	testX = []
	testY = []

	#Using skLearn's method for straified sampling and obtaining indices
	stratifiedShuffleSplit = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=0)
	for train_index, test_index in stratifiedShuffleSplit.split(X, Y):
	  trainInd_arr = train_index
	  testInd_arr = test_index

	#Using indices to create arrays and then to numpy arrays
	for i in trainInd_arr:
	  trainX.append(X[i])
	  trainY.append(Y[i])

	for i in testInd_arr:
	  testX.append(X[i])
	  testY.append(Y[i])

	trainX = np.array(trainX)
	trainY = np.array(trainY)
	testX = np.array(testX)
	testY = np.array(testY)

	if(isPrint==True):
		#Frequency percentage for train set
		(unique, counts) = np.unique(trainY, return_counts=True)
		trainYPercentages = {}
		for i in range(unique.size):
		  trainYPercentages[unique[i]] = (counts[i] / trainY.size) * 100
		print("Class Frequency in test data", trainYPercentages)

		#Frequency percentage for test set
		(unique, counts) = np.unique(testY, return_counts=True)
		testYPercentages = {}
		for i in range(unique.size):
		  testYPercentages[unique[i]] = (counts[i] / testY.size) * 100
		print("Class Frequency in training data", testYPercentages)

	return [trainX, trainY, testX, testY]


#Method for dimension reduction via PCA and applying logisctic regression and printing the accuracy
#numFeatures describes the number of features we want back
def getPCA(X, Y, numFeatures = 70, isTSNE = False):
	
	if(numFeatures > X.shape[1]):
		print("numFeatures must be less than or equal", numFeatures)
	#Using sklearn's implementation to get dimensionally reduced features using PCA
	pca = PCA(n_components=numFeatures)
	pca.fit(X)
	X = pca.transform(X)

	#Get stratified sampled test and train datasets
	trainX, trainY, testX, testY = stratifiedSampling(X, Y)

	#Apply logistic regression
	logisticRegression = LogisticRegression(verbose = 0, random_state=0, max_iter=300).fit(trainX, trainY)
	output = logisticRegression.predict(testX)
	accuracy = logisticRegression.score(testX, testY)
	print("Accuracy is " + str(accuracy))

	#Use TSNE to plot graph
	if(isTSNE == True):
		print("Scatter plot of features obtained from t-SNE")
		getTSNE(trainX, trainY)

	return [X, Y]


#Method for dimension reduction via SVD
def getSVD(X, Y, numFeatures = 70, isTSNE = False):
	
	#Using sklearn's implementation to get dimensionally reduced features using PCA
	svd = TruncatedSVD(n_components=numFeatures, random_state=0)
	svd.fit(X)
	X = svd.transform(X)

	#Get stratified sampled test and train datasets
	trainX, trainY, testX, testY = stratifiedSampling(X, Y)

	#Apply logistic regression
	logisticRegression = LogisticRegression(random_state=0, max_iter=300).fit(trainX, trainY)
	output = logisticRegression.predict(testX)
	accuracy = logisticRegression.score(testX, testY)
	print("Accuracy is " + str(accuracy))    

	#Use TSNE to plot graph
	if(isTSNE == True):
		print("Scatter plot of features obtained from t-SNE")
		getTSNE(trainX, trainY)

	return [X, Y]


#Method for t - SNE
def getTSNE(X, Y):

	#Compare with t-SNE
	X = TSNE().fit_transform(X)
	X1 = X[:, 0]
	X2 = X[:, 1]

	getScatterPlot(X1, X2, Y)

	return [X, Y]


#Method for plotting scatter plot
def getScatterPlot(X1, X2, Y):
	fig, ax = plt.subplots()
	scatter = ax.scatter(X1, X2, c = Y, label = Y)
	legend = ax.legend(*scatter.legend_elements(),loc="lower left", title="Classes")
	ax.add_artist(legend)
	plt.show()



#Calling all functions

#Get features and output
#X, Y = getDatasetOne()

#Run stratified sampling separately
#trainX, trainY, testX, testY = stratifiedSampling(X, Y, isPrint = True)

#Use PCA function
#newX, newY = getPCA(X,Y, isTSNE = False)

#Use SVD function
# newX, newY = getSVD(X,Y, isTSNE = False)
