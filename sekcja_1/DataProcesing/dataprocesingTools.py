#import libraries
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd

#read data 
dataset = pd.read_csv("Data.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#print to screen
print(x)
print(y)

## take  care of missing data 

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
imputer.fit(x[ : ,1:3])
x[:,1:3] = imputer.transform(x[ : ,1:3])

print(x)