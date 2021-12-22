#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# importing libaries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# importing datasets
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# reviewing datasets
pd.set_option('display.max_columns', None)
train.head()
test.head()

# number of rows and columns
train.shape
train.shape

# datasets info
train.info()
test.info()

# checking for null values
train.isnull().any()
test.isnull().any()


train['price_range'].value_counts()
#sns.countplot('price_range',data=train)


# defining dependent and independent variables
x_train = train.drop('price_range', axis=1)
y_train = train['price_range']










































