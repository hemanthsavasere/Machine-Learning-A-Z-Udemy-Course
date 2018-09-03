# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

# Importing the dataset and reading 
dataset = pd.read_csv('Data.csv')
# choosing the coloumns for test dataset
train = dataset[["Country", "Age", "Salary"]]
# encoding the variables and getting dummies
train["Age"] = train["Age"].fillna(train["Age"].mean())
train["Salary"] = train["Salary"].fillna(train["Salary"].mean())
train["Country"] = LabelEncoder().fit_transform(train["Country"])
train = pd.get_dummies(data = train, columns=['Country'])

# encoding test variables 
test = dataset[["Purchased"]]
test = LabelEncoder().fit_transform(test)


# splitting data to test and train dataset
X_train, X_test, Y_train, Y_test = train_test_split(train, test, test_size = 0.1, random_state = 0)

# Scaling all the values to one scale
X_train = StandardScaler().fit_transform(X_train) 