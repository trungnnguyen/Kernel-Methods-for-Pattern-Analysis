import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import convolve
from sklearn import linear_model, datasets, metrics
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
import scipy.io
from RBM import *
mat = scipy.io.loadmat('input.mat')
train_data=mat['train_x']
test_data=mat['test_x']
train_label=mat['train_y']
test_label=mat['test_y']
r1 = RBM(num_visible = 256, num_hidden = 250)

train_data1 = train_data
test_data1 = test_data

r1.train(train_data1, max_epochs = 5000)

output_train1,prob_train1=r1.run_visible(train_data1)
output_test1,prob_test1=r1.run_visible(test_data1)

r2 = RBM(num_visible = 250, num_hidden = 220)

train_data2 = output_train1
test_data2 = output_test1

r2.train(train_data2, max_epochs = 5000)

output_train2,prob_train2=r2.run_visible(train_data2)
output_test2,prob_test2=r2.run_visible(test_data2)

X_train = output_train2
X_test=output_test2
Y_train=train_label
Y_test=test_label

logistic = linear_model.LogisticRegression()
rbm = BernoulliRBM(random_state=0, verbose=True)

classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])
rbm.learning_rate = 0.1
rbm.n_iter = 5000
# More components tend to give better prediction performance, but larger
# fitting time
rbm.n_components = 200

classifier.fit(X_train, Y_train)

E=classifier.predict(X_test)
print("Logistic regression using RBM features:\n%s\n" % (
    metrics.classification_report(
        Y_test,
        classifier.predict(X_test))))



"""
r3 = RBM(num_visible = 220, num_hidden = 200)

train_data3 = output_train2
test_data3 = output_test2
r3.train(train_data3, max_epochs = 5000)

output_train3,prob_train3=r3.run_visible(train_data3)
output_test3,prob_test3=r3.run_visible(test_data3)


print output_test3
"""

