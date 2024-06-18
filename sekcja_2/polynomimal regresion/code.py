import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 

dataset = pd.read_csv("Position_Salaries.csv")
X  = dataset.iloc[:,1:-1].values
y  = dataset.iloc[:,-1].values


from sklearn.linear_model import LinearRegression
linReg = LinearRegression()
linReg.fit(X, y)

from sklearn.preprocessing import PolynomialFeatures
poly_req = PolynomialFeatures(degree = 4)
X_poly = poly_req.fit_transform(X)
linReq_2 = LinearRegression()
linReq_2.fit(X_poly,y)

plt.scatter(X, y, color='red')
plt.plot(X, linReq_2.predict(X_poly), color= 'blue')
plt.title("Truth or Bluff (linear regresion)")
plt.xlabel("position level")
plt.ylabel("Salary")
# plt.show()

print(linReq_2.predict(poly_req.fit_transform([[6.5]])))
