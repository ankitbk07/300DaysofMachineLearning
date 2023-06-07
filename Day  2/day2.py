## Implementation of Linear regression
class LinearRegression:

    def fit(self, X, y, intercept = False):

        # record data and dimensions
        if intercept == False: # add intercept (if not already included)
            ones = np.ones(len(X)).reshape(len(X), 1) # column of ones 
            X = np.concatenate((ones, X), axis = 1)   #append  one 
        self.X = np.array(X)
        self.y = np.array(y)
        self.N, self.D = self.X.shape
        XtX = np.dot(self.X.T, self.X) #Estimate Parameters
        XtX_inverse = np.linalg.inv(XtX)
        Xty = np.dot(self.X.T, self.y)
        self.beta_hats = np.dot(XtX_inverse, Xty)
        self.y_hat = np.dot(self.X, self.beta_hats) # make in-sample predictions
        self.L = .5*np.sum((self.y - self.y_hat)**2)   # calculate loss
        
    def predict(self, X_test, intercept = True):
        # form predictions
        self.y_test_hat = np.dot(X_test, self.beta_hats)
housing = fetch_california_housing()  ##Loading Calfornia Housing Dataset
X = housing ['data']
y  = housing['target']
model = LinearRegression() #Instantiating
model.fit(X, y, intercept= False)   #Fitting the Regression Model
fig, ax = plt.subplots() #Plotting the graph
plt.scatter(model.y,model.y_hat , s=50, c='red', marker='s')
plt.xlabel('Y')
plt.ylabel('Y pre')
plt.title('Scatter Plot')
plt.grid(True)
plt.show()