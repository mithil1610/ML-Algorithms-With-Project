#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 20:07:26 2019

@author: dashrath
"""

import numpy as np
import seaborn as sns
import pandas as pd

data=pd.read_csv('microlending_data.csv')
datap=data.copy()
data_copy=data
datap.isnull().sum()
data.get_dtype_counts()
datap.isnull().sum()
datap.dropna(axis=0,inplace=True)
datap.isnull().sum()
datap['borrower_genders'].describe()
data['borrower_genders'].value_counts().sort_index()
pd.crosstab(index=datap['borrower_genders'],columns='status',normalize=True)
data['borrower_genders'].fillna(data['borrower_genders'].median()[0])
data['borrower_genders']=data['borrower_genders'].fillna(data['borrower_genders'].mode()[0])
data.isnull().sum()
print(data.info())
dm=data['distribution_model'].value_counts().sort_index()
print(dm)
ac=data['activity'].value_counts().sort_index()
print(ac)
pd.crosstab(index=datap['activity'],columns='status',normalize=True)
sns.countplot(y='activity',data=data)
sns.boxplot(x=data['status'],y=data['activity'])
sns.countplot(x='activity',data=data,hue='status')
data['loan_amount'].value_counts().sort_index()
data['loan_amount'].describe()
sns.boxplot(y=data['loan_amount'])
data.info()
data['sector'].value_counts().sort_index()
pd.crosstab(index=datap['sector'],columns='loan_amount',normalize=True)
sns.boxplot(y=data['sector'],x=data['loan_amount'])
datap_select=datap.select_dtypes(exclude=[object])
cor=datap_select.corr()
print(cor)
data2=datap.copy()
data2['status'].value_counts().sort_index()

data2["status"]=data2['status'].map({'funded':0,'not_funded':1})
col=['activity','country_code','distribution_model']
data2=data2.drop(columns=col,axis=1)
data_copy=data2.copy()
new_data=pd.get_dummies(data2,drop_first=True)
column_list=list(new_data.columns)
print(column_list)
features=list(set(column_list)-set(['status']))
print(features)
x=new_data[features].values
print(x)
y=new_data['status'].values
print(y)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score,accuracy_score,confusion_matrix

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
lg=LogisticRegression()
lg.fit(x_train,y_train)
pred=lg.predict(x_test)
print(pred)
confusion_matric=confusion_matrix(y_test,pred)
print(confusion_matric)
ass=accuracy_score(y_test,pred)
print(ass)
print("mis classified",(y_test != pred).sum())
