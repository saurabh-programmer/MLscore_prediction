#step1:importing the libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pickle
#step2:Imports the dataset into the program
#      by the help of “Pandas” library.

dataset = pd.read_csv('student_scores.csv')

#So in our problem attribute=”Hours”, Labels=”Score”

X = dataset.iloc[:, :-1].values    #Attributes,Hours(indepedent) 
y = dataset.iloc[:, 1].values      # Labels , Score (Depend)

#Split thEEis data into training and testing data.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Train the algorithm .

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X,y)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))

print(model.predict([[2]]))