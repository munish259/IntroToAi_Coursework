# importing libaries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text



# importing datasets
#myFile = pd.read_csv('train.csv')

myFile2 = pd.read_csv('train.csv')
#print(myFile.head())

# datasets info
#print(myFile2.info())

# more info
#print(myFile2.describe())

# checking for null values
myFile2.isnull().any()

# removing and uneeded comlumns
#myFile = myFile2.drop('id', axis=1)

#print(myFile2.head())

# defining dependent and independent variables
X = myFile2[['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g', 'int_memory', 'm_dep', 'mobile_wt',
      'n_cores', 'pc', 'px_height', 'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g', 'touch_screen', 'wifi']]
y = myFile2['price_range']

# splitting the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.25, random_state=42)



#function for ease of access
def linear():

      # building the model
      linearModel = LogisticRegression()
      linearModel.fit(X_train, y_train)

      # making our predicitons
      y_pred = linearModel.predict(X_test)

      # printing the classification report
      print(classification_report(y_test, y_pred))

      # plotting and printing confusion matrix
      plot_confusion_matrix(linearModel, X_test, y_test)  
      plt.show()

def decisionTree():

      # building the model
      clf = DecisionTreeClassifier()
      clf = clf.fit(X_train,y_train)
      y_pred = clf.predict(X_test)

      #prints accuracy using metrics
      print("\n Accuracy:",metrics.accuracy_score(y_test, y_pred))

      # printing the classification report
      print(classification_report(y_test, y_pred))

      # plotting and printing confusion matrix
      plot_confusion_matrix(clf, X_test, y_test)  
      plt.show()
      


linear()
#decisionTree()