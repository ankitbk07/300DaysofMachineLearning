# Required Module and Dependencies
import numpy as np
np.set_printoptions(suppress= True)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
# import data
cancer = datasets.load_breast_cancer()
X_cancer = cancer['data']
y_cancer = cancer['target']
wine = datasets.load_wine()
X_wine = wine['data']
y_wine = wine['target']
#BinaryRegression
from sklearn.linear_model import LogisticRegression
binary_model = LogisticRegression(C = 10**5, max_iter = 100000)
binary_model.fit(X_cancer, y_cancer)
y_hats = binary_model.predict(X_cancer)#predict the output
p_hats = binary_model.predict_proba(X_cancer) #predict the probabalites
print(f'Training accuracy: {binary_model.score(X_cancer, y_cancer)}')#calculate the accuracy of the model
#MultiClass Regression
multiclass_model = LogisticRegression(multi_class = 'multinomial', C = 10**5, max_iter = 10**4)
multiclass_model.fit(X_wine, y_wine) #Generalize the data
y_hats =multiclass_model.predict(X_wine) #predict the output
print(multiclass_model.score(X_wine,y_wine)) #calculate the accuracy of the model
#Percepton Algorithm
from sklearn.linear_model import Perceptron #call the module of sci-learn kit
perceptron = Perceptron()
perceptron.fit(X_cancer, y_cancer)
#LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis #module of LDA from sci-learn
lda = LinearDiscriminantAnalysis(n_components = 1) 
lda.fit(X_cancer, y_cancer);
f0 = np.dot(X_cancer, lda.coef_[0])[y_cancer == 0]
f1 = np.dot(X_cancer, lda.coef_[0])[y_cancer == 1]
print('Separated:', (min(f0) > max(f1)) | (max(f0) < min(f1)))

