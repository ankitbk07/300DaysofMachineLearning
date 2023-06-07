## Implementating Gradient Descent
def OLS_GD(X, y, eta = 1e-3, n_iter = 1e4, add_intercept = True):   
  ## Add Intercept
  if add_intercept:
    ones = np.ones(X.shape[0]).reshape(-1, 1)
    X = np.concatenate((ones, X), 1)
    beta_hat = np.random.randn(X.shape[1]) ## Instantiate
  ## Iterate
  for i in range(int(n_iter)):
    ## Calculate Derivative
    yhat = X @ beta_hat 
    delta = -X.T @ (y - yhat) ## @ indicating dot product
    beta_hat -= delta*eta     ## adjust value of beta

## Implementing K-fold Cross Validition
housing = fetch_california_housing()
X = housing ['data']
y = housing['target']
N = X.shape[0]
## Choose alphas to consider
potential_alphas = [0, 1, 10]
error_by_alpha = np.zeros(len(potential_alphas))
## Choose the folds 
K = 5
indices = np.arange(N)
np.random.shuffle(indices)
folds = np.array_split(indices, K)
## Iterate through folds
for k in range(K):
  ## Split Train and Validation
    X_train = np.delete(X, folds[k], 0)
    y_train = np.delete(y, folds[k], 0)
    X_val = X[folds[k]]
    y_val = y[folds[k]]
  
  ## Iterate through Alphas
    for i in range(len(potential_alphas)):
        ## Train on Training Set
        model = Ridge(alpha = potential_alphas[i])
        model.fit(X_train, y_train)
        ## Calculate and Append Error
        error = np.sum( (y_val - model.predict(X_val))**2 )
        error_by_alpha[i] += error
    
error_by_alpha /= N  ## provides array with the lowest average error 
