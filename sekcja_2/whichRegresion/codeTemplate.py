import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.metrics import r2_score


dataset = pd.read_csv("Data.csv")
X  = dataset.iloc[:,:-1].values
y  = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# multiple linear regression
from sklearn.linear_model import LinearRegression
Multipleregressor = LinearRegression().fit(X_train,Y_train)
Multiple_y_pred = Multipleregressor.predict(X_test)
print( r2_score(Y_test,Multiple_y_pred) )


# polynomial regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X_train)
PolynomialRegressor = LinearRegression().fit(X_poly,Y_train)
Polynomial_y_pred = PolynomialRegressor.predict(poly_reg.transform(X_test))
print( r2_score(Y_test,Polynomial_y_pred) )


# suppor Vector Regression
y_Train_SVR = Y_train.reshape(len(Y_train),1)
y_Test_SVR = Y_test.reshape(len(Y_test),1)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
X_train_SVR = sc_x.fit_transform(X_train)
y_Train_SVR = sc_y.fit_transform(y_Train_SVR)

from sklearn.svm import SVR
SVR_Regresor = SVR(kernel='rbf').fit(X_train_SVR,y_Train_SVR.ravel())
SVR_y_Pred = sc_y.inverse_transform(SVR_Regresor.predict(sc_x.transform(X_test)).reshape(-1, 1))
print( r2_score(Y_test,SVR_y_Pred) )

# Decysion Tree Regresion
from sklearn.tree import DecisionTreeRegressor
Tree_Y_pred = DecisionTreeRegressor(random_state=0).fit(X_train,Y_train).predict(X_test)
print( r2_score(Y_test,Tree_Y_pred) )


# Random forest regresion
from sklearn.ensemble import RandomForestRegressor
RandomForest_regresor = RandomForestRegressor(n_estimators=10, random_state=0).fit(X_train,Y_train)
y_randomForstPredict = RandomForest_regresor.predict(X_test)
print( r2_score(Y_test,y_randomForstPredict) )