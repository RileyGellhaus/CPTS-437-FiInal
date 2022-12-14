import csv
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
#from google.colab import drive #Used for Google Colab
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score

#drive.mount('/content/gdrive', force_remount=True) #Used for Google Colab

# Load the data into a Pandas DataFrame
#df = pd.read_csv('/content/gdrive/MyDrive/Cpt_S 437/wine.csv')
  
# Open the input CSV file
#data = np.loadtxt('/content/gdrive/MyDrive/Cpt_S 437/wine.csv', dtype='float', delimiter=',') #Used for Google Colab
data = np.loadtxt('wine.csv', dtype='float', delimiter=',')


def linearReg(data):
  X_train, X_test, y_train, y_test = train_test_split(data[:, :-1], data[:, -1], test_size=0.3)
  # Create a Linear Regression model
  model = LinearRegression()
  # Train the model on the training data
  model.fit(X_train, y_train)

  # make predictions on the test set
  y_pred = model.predict(X_test)
  return r2_score(y_test, y_pred)

def percep(data):
  X_train, X_test, y_train, y_test = train_test_split(data[:, :-1], data[:, -1], test_size=0.3)
  
  #create Peceptron
  clf = Perceptron()
  #train model
  clf.fit(X_train, y_train)
  #test model
  y_pred = clf.predict(X_test)
  return accuracy_score(y_test, y_pred)

def tree(data):
  X_train, X_test, y_train, y_test = train_test_split(data[:, :-1], data[:, -1], test_size=0.3)

  clf = DecisionTreeClassifier(max_depth = 3)
  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)
  return accuracy_score(y_test, y_pred)

#kNN with bootstrapping
def knn(data):
  # Split the data into a training and testing set, using 80% of the data for training and 20% for testing
  X_train, X_test, y_train, y_test = train_test_split(data[:, :-1], data[:, -1], test_size=0.2)


  # load the original dataset
  X = np.concatenate([X_train, X_test])
  y = np.concatenate([y_train, y_test])

  # concatenate the predictor and target variables
  data = np.concatenate((X, y.reshape(-1, 1)), axis=1)

  # create an empty list to store the resampled datasets
  resampled_data = []

  # create a number of resampled datasets using bootstrapping
  for i in range(5):
    # resample the data with replacement
    resampled = resample(data, replace=True, n_samples=len(data))
    # split the resampled data into predictors and target variable
    X_resampled = resampled[:, :-1]
    y_resampled = resampled[:, -1]
    # store the resampled data
    resampled_data.append((X_resampled, y_resampled))

  # combine the resampled datasets into a single dataset
  X_bootstrapped = np.concatenate([data[0] for data in resampled_data])
  y_bootstrapped = np.concatenate([data[1] for data in resampled_data])

  # normalize the data using the StandardScaler class
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X_bootstrapped)

  # Split the data into a training and testing set, using 80% of the data for training and 20% for testing
  X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_bootstrapped, test_size=0.1)

  # Create a Linear Regression model
  model = KNeighborsClassifier(n_neighbors=3)

  # Train the model on the training data
  model.fit(X_train, y_train)

  # make predictions on the test set
  y_pred = model.predict(X_test)

  # print the accuracy of the model
  return r2_score(y_test, y_pred)



# print the accuracy of the model
linAcc = linearReg(data)
pAcc = linearReg(data)
treAcc = tree(data)
knnAcc = knn(data)
print('----------------------------------------')
print('Linear regression accuracy:', linAcc)
print('----------------------------------------')
print('Perceptron accuracy:', pAcc)
print('----------------------------------------')
print('Tree accuracy:', treAcc)
print('----------------------------------------')
print('KNN accuracy:', knnAcc)
print('----------------------------------------')