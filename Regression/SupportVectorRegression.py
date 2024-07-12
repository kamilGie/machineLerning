import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('dataset.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values
Y = Y.reshape(len(Y),1)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
Y_train = sc_Y.fit_transform(Y_train)

from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf').fit(X_train, Y_train.ravel())

Y_pred = sc_Y.inverse_transform(regressor.predict(sc_X.transform(X_test)).reshape(-1,1))
np.set_printoptions(precision=2)
#print(np.concatenate((Y_pred.reshape(len(Y_pred),1), Y_test.reshape(len(Y_test),1)),1))

from sklearn.metrics import r2_score
print("SupportVector\n\tR2 score is: ",r2_score(Y_test, Y_pred))