## Import packages
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
## Load penguins data
penguins = sns.load_dataset('penguins')
penguins = penguins.dropna().reset_index(drop = True)
X = penguins.drop(columns = 'species')
y = penguins['species']
## Train-test split
np.random.seed(1)
test_frac = 0.25
test_size = int(len(y)*test_frac)
test_idxs = np.random.choice(np.arange(len(y)), test_size, replace = False)
X_train = X.drop(test_idxs)
y_train = y.drop(test_idxs)
X_test = X.loc[test_idxs]
y_test = y.loc[test_idxs]
## Get dummies
X_train = pd.get_dummies(X_train, drop_first = True)
X_test = pd.get_dummies(X_test, drop_first = True)
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import GaussianNB
bagger1 = BaggingClassifier(n_estimators= 1000,random_state = 100)
bagger1.fit(X_train,y_train)
bagger2 = BaggingClassifier(estimator= GaussianNB(),random_state = 100)#Serch about random state it is a hyperparameter
bagger2.fit(X_train,y_train)
print(np.mean(bagger1.predict(X_test)== y_test))
print(np.mean(bagger2.predict(X_test)== y_test)) #Provides accuracy
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 1000,max_features = int(np.sqrt(X_test.shape[1])),random_state =200)
rf.fit(X_train,y_train)
print(np.mean(rf.predict(X_test)==y_test))
y_train = (y_train == 'Adelie')
y_test = (y_test == 'Adelie')
#Boosting Classification
from sklearn.ensemble import AdaBoostClassifier 
X_train = pd.get_dummies(X_train,drop_first = True)
X_test = pd.get_dummies(X_test,drop_first = True)
abc = AdaBoostClassifier(n_estimators= 50)
abc.fit(X_train,y_train)
y_test_hat = abc.predict(X_test)
np.mean(y_test_hat == y_test) #Yes it gave a 100%
#Boosting Regression
## Load penguins data
tips = sns.load_dataset('tips')
tips = tips.dropna().reset_index(drop = True)
X = tips.drop(columns = 'tip')
y = tips['tip']
## Train-test split
np.random.seed(1)
test_frac = 0.25
test_size = int(len(y)*test_frac)
test_idxs = np.random.choice(np.arange(len(y)), test_size, replace = False)
X_train = X.drop(test_idxs)
y_train = y.drop(test_idxs)
X_test = X.loc[test_idxs]
y_test = y.loc[test_idxs]
from sklearn.ensemble import AdaBoostRegressor
X_train = pd.get_dummies(X_train,drop_first = True)
X_test = pd.get_dummies(X_test,drop_first = True)
abr = AdaBoostRegressor(n_estimators = 50)
abr.fit(X_train,y_train)
y_test_hat = abr.predict(X_test)
plt.scatter(y_test,y_test_hat,color= ['#1f77b4', '#ff7f0e'])
plt.xlabel("Y_test")
plt.ylabel("Y_test_hat")
plt.show() #The variance of estimate and target is high




