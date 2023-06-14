import tensorflow as tf
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
cali = datasets.fetch_california_housing()
X_boston = cali['data']
y_boston = cali['target']
##Sequential Approachz
## 1. Instantiate
model = tf.keras.models.Sequential(name = 'Sequential_Model')
## 2. Add Layers
model.add(tf.keras.layers.Dense(units = 8,
                                activation = 'relu',
                                input_shape = (X_boston.shape[1], ),
                                name = 'hidden'))
model.add(tf.keras.layers.Dense(units = 1,
                                activation = 'linear',
                                name = 'output'))
## 3. Compile (and summarize)
model.compile(optimizer = 'adam', loss = 'mse')
print(model.summary())
## 4. Fit
model.fit(X_boston, y_boston, epochs = 100, batch_size = 1, validation_split=0.2, verbose = 0);
###Functional Approach
## 1. Define layers
inputs = tf.keras.Input(shape = (X_boston.shape[1],), name = "input")
hidden = tf.keras.layers.Dense(8, activation = "relu", name = "first_hidden")(inputs)
outputs = tf.keras.layers.Dense(1, activation = "linear", name = "output")(hidden)
## 2. Model
model = tf.keras.Model(inputs = inputs, outputs = outputs, name = "Functional_Model")
## 3. Compile (and summarize)
model.compile(optimizer = "adam", loss = "mse")
print(model.summary())a
## 4. Fit
model.fit(X_boston, y_boston, epochs = 100, batch_size = 1, validation_split=0.2, verbose = 0);

