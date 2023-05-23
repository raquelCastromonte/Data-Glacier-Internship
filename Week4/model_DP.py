# Importing the libraries
import numpy as np
import pandas as pd
import pickle

dataset = pd.read_csv('diabetes_prediction_dataset.csv')

del dataset["gender"]
del dataset["smoking_history"]

X = dataset.iloc[:, :6]

y = dataset.iloc[:, -1]

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X, y)

# Saving model to disk
pickle.dump(regressor, open('model_DP.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model_DP.pkl','rb'))
print(model.predict([[20, 0, 1, 25, 5.5, 110]]))
