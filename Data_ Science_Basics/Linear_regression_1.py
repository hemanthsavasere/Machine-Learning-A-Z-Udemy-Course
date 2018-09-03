# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

# loading the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset[["YearsExperience"]]
Y = dataset[["Salary"]]

# splitting data to test and train dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Using Linear regression in Scikit-Learn
from sklearn.linear_model import LinearRegression
linear = LinearRegression(normalize = True)
linear.fit(X_train, Y_train)

# predictions
predictions = linear.predict(X_test)

# visulisation 
plt.scatter(X_test, Y_test, color = "Red")
plt.plot(X_test, linear.predict(X_test), color = "Blue")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Experience vs Salary")
plt.show()