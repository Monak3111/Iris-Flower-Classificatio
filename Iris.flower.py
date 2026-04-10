import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

iris = load_iris()

X = iris.data   # features (measurements)
y = iris.target # labels (species)

feature_names = iris.feature_names
target_names = iris.target_names

print(feature_names)
print(target_names)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


new_sample = [[5.1, 3.5, 1.4, 0.2]]
new_sample_scaled = scaler.transform(new_sample)

prediction = model.predict(new_sample_scaled)

print("Predicted class:", iris.target_names[prediction][0])

import pickle

pickle.dump(model, open("iris_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

model = pickle.load(open("iris_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

sample = [[6.2, 3.4, 5.4, 2.3]]
sample_scaled = scaler.transform(sample)

prediction = model.predict(sample_scaled)

print("Predicted:", prediction)