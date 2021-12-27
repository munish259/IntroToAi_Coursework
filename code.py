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

# 2000 rows and 21 columns
train.shape
# 1000 rows and 21 columns
test.shape

# datasets info
train.info()
test.info()

# more info
train.describe()
test.describe()

# checking for null values
train.isnull().any()
test.isnull().any()

# price_range column is well balanced
train.price_range.value_counts()
#sns.countplot('price_range',data=train)


# defining dependent and independent variables
x_train = train.drop('price_range', axis=1)
y_train = train['price_range']


























