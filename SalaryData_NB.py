# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 17:18:28 2020

@author: SONY
"""
import pandas as pd
import numpy as np


salary_train=pd.read_csv("E:\\TEJAS\\EXCELR ASSIGMENTS\\COMPLETED\\NAVIES BAYIES\\SALARY\\SalaryData_Train.csv")
salary_test=pd.read_csv("E:\\TEJAS\\EXCELR ASSIGMENTS\\COMPLETED\\NAVIES BAYIES\\SALARY\\SalaryData_Test.csv")

##Here some columns  categorical##
string_columns=["workclass","maritalstatus","occupation","relationship","education","race","sex","native"]

###So the categorical columns are converted into binary format with label encoder#########
from sklearn.preprocessing import LabelEncoder
number=LabelEncoder()

for i in string_columns:
    salary_train[i]=number.fit_transform(salary_train[i])
    salary_test[i]=number.fit_transform(salary_test[i])

x_train=salary_train.iloc[:,0:12]
y_train=salary_train.iloc[:,13]
x_test=salary_test.iloc[:,0:12]
y_test=salary_test.iloc[:,13]

#######Importing the navies bayes function######
from sklearn.naive_bayes import MultinomialNB as MB
from sklearn.naive_bayes import GaussianNB as GB

classifiers_mb=MB()
classifiers_mb.fit(x_train,y_train)
train_pred_mb=classifiers_mb.predict(x_train)
train_accu_mb=np.mean(train_pred_mb==y_train)##77%
pd.crosstab(train_pred_mb,y_train)

test_pred_mb=classifiers_mb.predict(x_test)
test_accu_mb=np.mean(test_pred_mb==y_test)##77%
pd.crosstab(test_pred_mb,y_test)

classifiers_gb=GB()
classifiers_gb.fit(x_train,y_train)
train_pred_gb=classifiers_gb.predict(x_train)
train_accu_gb=np.mean(train_pred_gb==y_train)##80%
pd.crosstab(train_pred_gb,y_train)

test_pred_gb=classifiers_gb.predict(x_test)
test_accu_gb=np.mean(test_pred_gb==y_test)##80%


#####From above Gassian Navies bayes contian more accuracy w.r.t. multinomial NB.