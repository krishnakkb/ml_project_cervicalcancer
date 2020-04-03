import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', 100)

data = pd.read_csv("kag_risk_factors_cervical_cancer.csv")
print(data.shape)
data.sample(5)

from sklearn.tree import DecisionTreeClassifier

#creating the model
model = DecisionTreeClassifier(class_weight="balanced")

# feeding the training data into the model
model.fit(X_train, y_train)

#predicting the test set results
y_pred = model.predict(X_test)

# Calculating the accuracies
print("Training accuracy :", model.score(X_train, y_train))
print("Testing accuracy :", model.score(X_test, y_test))

# classification report
print(classification_report(y_test, y_pred))

# confusion matrix 
print(confusion_matrix(y_test, y_pred))