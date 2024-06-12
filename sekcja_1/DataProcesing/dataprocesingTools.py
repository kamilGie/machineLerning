#import libraries
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd

#read data 
dataset = pd.read_csv("Data.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

## take  care of missing data 
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
imputer.fit(x[ : ,1:3])
x[:,1:3] = imputer.transform(x[ : ,1:3])

#tranform the string to some int
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0]),('encoder2',OneHotEncoder(),[0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

#transofrm somie bool string values to bool 0 and 1 
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

#split to traing and test models
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 1)

#future scaling 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:,3:] =sc.fit_transform(X_train[:,3:])
X_test[:,3:] =sc.transform(X_test[:,3:])