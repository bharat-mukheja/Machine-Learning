import numpy as np
import sklearn as ml

from sklearn.neural_network import MLPClassifier

fp = open('dataset.csv')
data = np.getfile(fp)
X = data[:,:-1]
y = data[:,-1]
#SMOTE technique
Xtest = [[100,200,300]]

#Visit SKlearn toy datasets


clf = MLPClassifier()
clf.fit(X,y)
predict=clf.predict(Xtest)
print("Answer=%r"%predict[0])