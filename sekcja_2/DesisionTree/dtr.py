import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 

dataset = pd.read_csv("Position_Salaries.csv")
X  = dataset.iloc[:,1:-1].values
y  = dataset.iloc[:,-1].values

from sklearn.tree import DecisionTreeRegressor
regresor = DecisionTreeRegressor(random_state=0).fit(X,y)

print(regresor.predict([[6.5]]))

# plt.scatter(X,y, color='red')
# plt.plot(X,regresor.predict(X),color='blue')
# plt.title("tree decysion")
# plt.xlabel("lvl of expirience")
# plt.ylabel("income")
# plt.show()

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regresor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (decysion Tree)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()