import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 

dataset = pd.read_csv("Salary_Data.csv")
X  = dataset.iloc[:,:-1].values
y  = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)   

y_pred = regressor.predict(X_test)

plt.scatter(X_train, Y_train, color= 'red')
plt.plot(X_train,regressor.predict(X_train), color='blue')
plt.title('Salary vs Expieriacne')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()

plt.scatter(X_test, Y_test, color= 'red')
plt.plot(X_train,regressor.predict(X_train), color='blue')
plt.title('Salary vs Expieriacne(prime test)')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()