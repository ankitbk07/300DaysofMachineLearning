#Installing Dependencies
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import Ridge, Lasso
import statsmodels.api as sm
california = fetch_california_housing()
X_train = california['data']
y_train = california['target']
alpha = 1
# Ridge
ridge_model = Ridge(alpha = alpha) #Instantiating Ridge Model
ridge_model.fit(X_train, y_train)#calculate required parameters
# Lasso
lasso_model = Lasso(alpha = alpha)#Instantiating Lasso Model
lasso_model.fit(X_train, y_train);#calculate required parameters
#Bayesian
bayes_model = BayesianRidge()#Instantiating Bayesian Model
bayes_model.fit(X_train, y_train);#calculate required parameters
#GLM
X_train_with_constant = sm.add_constant(X_train)
poisson_model = sm.GLM(y_train, X_train, family=sm.families.Poisson())#Instantiating Poisson Model
poisson_model.fit();#calculate required parameters