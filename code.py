# importing libaries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split 

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.metrics import classification_report

from sklearn import metrics

import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix



myFile2 = pd.read_csv('train.csv')
#print(myFile.head())

# datasets info
#print(myFile2.info())

# checking for null values
myFile2.isnull().any()


# removing and uneeded comlumns
#myFile = myFile2.drop('id', axis=1)


# defining dependent and independent variables
X = myFile2[['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g', 'int_memory', 'm_dep', 'mobile_wt',
      'n_cores', 'pc', 'px_height', 'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g', 'touch_screen', 'wifi']]
y = myFile2['price_range']


# splitting the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.25, random_state=42)



#function for ease of access
def logistic():

      # building the model
      logisticModel = LogisticRegression()
      logisticModel.fit(X_train, y_train)

      # making our predicitons
      y_pred = logisticModel.predict(X_test)

      #prints accuracy using metrics
      print("\n (metric) Accuracy:", metrics.accuracy_score(y_test, y_pred), "\n")

      # printing the classification report
      print(classification_report(y_test, y_pred))

      # plotting and printing confusion matrix
      plot_confusion_matrix(logisticModel, X_test, y_test)  
      plt.show()


      #reference: labs


def decisionTree():


      # building the model
      clf = DecisionTreeClassifier()
      clf = clf.fit(X_train,y_train)

      # making the prediction
      y_pred = clf.predict(X_test)

      #prints accuracy using metrics
      print("\n (metric) Accuracy:", metrics.accuracy_score(y_test, y_pred), "\n")

      # printing the classification report
      print(classification_report(y_test, y_pred))

      # plotting and printing confusion matrix
      plot_confusion_matrix(clf, X_test, y_test)  
      plt.show()

      
      #reference: https://stackabuse.com/decision-trees-in-python-with-scikit-learn/


#visualise decision tree in itself


def SVM():

      # building the model
      regressor  = SVC(kernel='rbf', random_state = 1)
      regressor.fit(X_train,y_train)

      #making our prediction
      y_pred = regressor.predict(X_test)

      #prints accuracy using metrics
      print("\n (metric) Accuracy: ", metrics.accuracy_score(y_test, y_pred), "\n")

      # printing the classification report
      print(classification_report(y_test, y_pred))

      # plotting and printing confusion matrix
      plot_confusion_matrix(regressor, X_test, y_test)  
      plt.show()

      #ref: https://analyticsindiamag.com/understanding-the-basics-of-svm-with-example-and-python-implementation/
      
def nb():

      # building the model
      nb = GaussianNB()
      nb.fit(X_train,y_train)

      # making predictions
      y_pred = nb.predict(X_test)
      
      #prints accuracy using metrics
      print("\n (metric) Accuracy: ", metrics.accuracy_score(y_test, y_pred), "\n")
      
      # printing the classification report 
      print(classification_report(y_test,y_pred))

      # plotting and printing confusion matrix
      plot_confusion_matrix(nb, X_test,y_pred)
      plt.show()



#logistic()
#decisionTree()
#SVM()
nb()
