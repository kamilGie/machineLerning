import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('dataset.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0).fit(X_train, Y_train)

Y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
#print(np.concatenate((Y_pred.reshape(len(Y_pred),1), Y_test.reshape(len(Y_test),1)),1))

from sklearn.metrics import r2_score
print("RandomForest\n\tR2 score is: ",r2_score(Y_test, Y_pred))