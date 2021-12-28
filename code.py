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
from sklearn.metrics import plot_confusion_matrix
from scipy.stats import zscore
import seaborn as sns


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

# function for intro analysis - please run using spyder

def analyse_data():
      path = "."  #absolute or relative path to the folder containing the file. 
                #"." for current folder
    
      filename_read = os.path.join(path, "train.csv")
      df = pd.read_csv(filename_read) # for analysis

      '''Start of analysis of data'''
      # Strip non-numerics
      df = df.select_dtypes(include=['int', 'float'])
      
      print  ("\nCounting of data: \n")
      
      #details on the data in numbers 
      for col in df.columns:
          print(col + '\n_____________')
          print(df[col].value_counts())
          print('_____________________________\n')

      
      print("\nStatistics: \n")

      #display statistics 
      get_stats()

      #visualise data so that it is clear what the data is showing
      df.hist(figsize=(20,20))
      plt.show()
      #sns.pairplot(df,hue='price_range')

      
def analyse_data():
      '''Start of analysis of data'''
      # Strip non-numerics
      df = df.select_dtypes(include=['int', 'float'])

      headers = list(df.columns.values)
      fields = []
      
      #details on the data in numbers 
      for col in df.columns:
          print(col + '\n_____________')
          print(df[col].value_counts())
          print('_____________________________\n')

      #display statistics 
      get_stats()

      #visualise data so that it is clear what the data is showing
      df.hist(figsize=(20,20))
      plt.show()
      #sns.pairplot(df,hue='price_range')
      
      #heat map isa correlation matrix used to show the correlations between each field 
      plt.figure(figsize=(16,16))
      sns.heatmap(df.corr(), annot=True, fmt=".2f");
      '''End of analysis of data'''

      
      
def get_stats():
    
      headers = list(df.columns.values)
      fields = []
      # Perform basic statistics (mean, variance, standard deviation, z scores) on a dataframe.
      for field in headers:
          fields.append({
              'name' : field,
              'mean': df[field].mean(),
              'var': df[field].var(),
              'sdev': df[field].std(),# how dispersed the data is in relation to the mean
          
          })
          
          
      #display statistics 
      for field in fields:
          print(field)
          
          

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

      
analyse_data()

#analyse_data()
#logistic()
#decisionTree()
#SVM()
#nb()

