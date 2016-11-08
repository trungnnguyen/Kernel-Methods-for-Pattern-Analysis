# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 22:59:51 2016

@author: vishal
"""
import numpy as np
from sklearn.svm import NuSVC
from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt
# load data from all.spydata
# X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
# y = np.array([1, 1, 2, 2])
class1_traintxt = np.loadtxt('../group6/class1_train.txt', delimiter=' ')
class2_traintxt = np.loadtxt('../group6/class2_train.txt', delimiter=' ')
class3_traintxt = np.loadtxt('../group6/class3_train.txt', delimiter=' ')

class1_valtxt = np.loadtxt('../group6/class1_val.txt', delimiter=' ')
class2_valtxt = np.loadtxt('../group6/class2_val.txt', delimiter=' ')
class3_valtxt = np.loadtxt('../group6/class3_val.txt', delimiter=' ')

class1_testtxt = np.loadtxt('../group6/class1_test.txt', delimiter=' ')
class2_testtxt = np.loadtxt('../group6/class2_test.txt', delimiter=' ')
class3_testtxt = np.loadtxt('../group6/class3_test.txt', delimiter=' ')

Xtrain = np.concatenate((class1_traintxt, class2_traintxt, class3_traintxt))
Xval = np.concatenate((class1_valtxt, class2_valtxt, class3_valtxt))
Xtest = np.concatenate((class1_testtxt, class2_testtxt, class3_testtxt))

trainshape = Xtrain.shape
testshape = Xtest.shape
valshape = Xval.shape

Ytrain = np.ones((trainshape[0]))
Ytrain_predict = np.ones((trainshape[0]))
Ytrain[0:250] = 1
Ytrain[250:500] = 2
Ytrain[500:750] = 3

Yval = np.ones((valshape[0]))
Yval_predict = np.ones((valshape[0]))
Yval[0:150] = 1
Yval[150:300] = 2
Yval[300:450] = 3

Ytest = np.ones((testshape[0]))
Ytest_predict = np.ones((testshape[0]))
Ytest[0:100] = 1
Ytest[100:200] = 2
Ytest[200:300] = 3

clf = NuSVC()
clf.fit(Xtrain, Ytrain)
NuSVC(cache_size=2000, class_weight=None, coef0=0.0,
      decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, nu=0.5, probability=False, random_state=None,
      shrinking=True, tol=0.00001, verbose=False)

for i in range(trainshape[0]):
    Ytrain_predict[i] = clf.predict([Xtrain[i, :]])

for i in range(valshape[0]):
    Yval_predict[i] = clf.predict([Xval[i, :]])

for i in range(testshape[0]):
    Ytest_predict[i] = clf.predict([Xtest[i, :]])

Y = np.concatenate((Ytrain, Yval, Ytest))
Y_predict = np.concatenate((Ytrain_predict, Yval_predict,
                            Ytest_predict))

cm_train = confusion_matrix(Ytrain, Ytrain_predict)
cm_val = confusion_matrix(Yval, Yval_predict)
cm_test = confusion_matrix(Ytest, Ytest_predict)
cm = confusion_matrix(Y, Y_predict)

# plt.imshow(cm)
print cm_train
print cm_val
print cm_test
print cm

x1 = np.arange(-20, 20, 1)

# plt.scatter(class1_traintxt[:, 0], class1_traintxt[:, 1], color='red')
# plt.scatter(class2_traintxt[:, 0], class2_traintxt[:, 1], color='green')
# plt.scatter(class3_traintxt[:, 0], class3_traintxt[:, 1], color='blue')
# plt.show()
zz=[]
for i in x1:
    for j in x1:
        temp = clf.predict([[i, j]])
        zz.append(temp)
        if(temp == [3.0]):
            plt.scatter(i, j, color='red')
        elif(temp == [2.0]):
            plt.scatter(i, j, color='green')
        elif(temp == [1.0]):
            plt.scatter(i, j, color='blue')
plt.show()
# plt.plot(x1, x1)
# plt.show()
