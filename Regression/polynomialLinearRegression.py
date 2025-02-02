import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('dataset.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,random_state=0)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
polY_reg = PolynomialFeatures(degree = 4)
X_polY = polY_reg.fit_transform(X_train)
regressor = LinearRegression().fit(X_polY, Y_train)

Y_pred = regressor.predict(polY_reg.transform(X_test))
np.set_printoptions(precision=2)
#print(np.concatenate((Y_pred.reshape(len(Y_pred),1), Y_test.reshape(len(Y_test),1)),1))

from sklearn.metrics import r2_score
print("PolynomialLinear \n\tR2 score is: ",r2_score(Y_test, Y_pred))