# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection  import train_test_split

# Load the Iris dataset
database = pd.read_csv('iris.csv')

# Separate features and target
x = database.iloc[:,:-1].values
y = database.iloc[:,-1].values


# Split the dataset into an 80-20 training-test set
x_train, x_test, y_treing, y_test = train_test_split(x,y,train_size=0.2,random_state=1)

print(x_test)
print(y_test)
# Apply feature scaling on the training and test sets

# Print the scaled training and test sets