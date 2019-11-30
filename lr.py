#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 10:37:06 2019

@author: dashrath
"""

import numpy as np
import seaborn as sns
import pandas as pd

data=pd.read_csv('cars_sampled.csv')
new_data=data.copy()
print(new_data.info)
summary=new_data.describe()
print(summary)
cols=['name','dateCrawled','dateCreated','postalCode','lastSeen']
datap=new_data.drop(cols,axis=1)
datap.drop_duplicates(keep='first',inplace=True)
#data cleaning
print(datap.isnull().sum())

yor_count=datap['yearOfRegistration'].value_counts().sort_index()
sns.regplot(x='yearOfRegistration',y='price',data=datap,scatter=True)
sum(datap['yearOfRegistration']<=2019)
sum(datap['yearOfRegistration']>1950)
###1950 to 2019

p_count=datap['price'].value_counts().sort_index()
sns.distplot(datap["price"],bins=10)
datap['price'].describe()
sns.boxplot(datap['price'])
sum(datap['price']>150000)
sum(datap['price']<100)
####100 to 150000
pps=datap['powerPS'].value_counts().sort_index()
sum(datap['powerPS']>500)
sum(datap['powerPS']<10)
####clean data

datap=datap[(datap.yearOfRegistration<=2019)&
            (datap.yearOfRegistration>=1950)&
            (datap.price<=150000)&
            (datap.price>=100)&
            (datap.powerPS<=500)&
            (datap.powerPS>=10)]

datap['monthOfRegistration']/=12
datap['ages']=2019-datap['yearOfRegistration']+datap['monthOfRegistration']
datap['ages'].describe()

##drop mor and yor

datap=datap.drop(columns=['monthOfRegistration','yearOfRegistration'],axis=1)

##ages
sns.distplot(datap["ages"])
sns.boxplot(y=datap["ages"])

##prce
sns.distplot(datap["price"])
sns.boxplot(y=datap["price"])
###price and seller ##insignified
datap['seller'].value_counts().sort_index()
pd.crosstab(index=datap["seller"],columns="price",normalize=True)
sns.countplot(datap['seller'])
sns.boxplot(x='price',y='seller',data=datap)
###offertype ##insignified
datap['offerType'].value_counts().sort_index()
###abtest
datap['abtest'].value_counts().sort_index()
pd.crosstab(index=datap['abtest'],columns='price',normalize=True)
sns.countplot(datap['abtest'])
sns.boxplot(x='price',y='abtest',data=datap)
##vehicleType
datap['vehicleType'].value_counts().sort_index()
pd.crosstab(index=datap['vehicleType'],columns='price',normalize=True)
sns.countplot(datap['vehicleType'])
sns.boxplot(x='vehicleType',y='price',data=datap)
###gearbox
datap['gearbox'].value_counts().sort_index()
pd.crosstab(index=datap['gearbox'],columns='price',normalize=True)
sns.countplot(datap['gearbox'])
sns.boxplot(x='gearbox',y='price',data=datap)
###remove insignificent

col=['seller','abtest','offerType']
datap=datap.drop(columns=col,axis=1)
datap_copy=datap.copy()

#####correlation

datap_select=datap.select_dtypes(exclude=[object])
cor=datap_select.corr()
round(cor,3)
datap_select.corr().loc[:,'price'].abs().sort_values(ascending=False)[1:]
###remove missing data

data_omit=datap.dropna(axis=0)

##make categorical data

data_omit=pd.get_dummies(data_omit,drop_first=True)
#####import sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

##model buiding with omiting data science

x1=data_omit.drop(['price'],axis='columns',inplace=False)
y1=data_omit['price']
prices=pd.DataFrame({'1.Before':y1,'2.After':np.log(y1)})
prices.hist()
y1=np.log(y1)
x_train,x_test,y_train,y_test=train_test_split(x1,y1,test_size=0.3,random_state=3)
####base model

base_pred=np.mean(y_test)
print(base_pred)
base_pred=np.repeat(base_pred,len(y_test))
###finding rmse
base_rmse=np.sqrt(mean_squared_error(y_test,base_pred))
print(base_rmse)
##linear regression with omited data
lgr=LinearRegression(fit_intercept=True)
model_lin1=lgr.fit(x_train,y_train)
data_predict=model_lin1.predict(x_test)
##compute mse and rmse
lin_mse=mean_squared_error(y_test,data_predict)
lin_rmse=np.sqrt(lin_mse)
print(lin_rmse)
#r squared value
lin1=model_lin1.score(x_test,y_test)
lin_train=model_lin1.score(x_train,y_train)
print(lin1,lin_train)
##residual
residual=y_test-data_predict
sns.regplot(y=residual,x=data_predict,fit_reg=False)
residual.describe()
###############random forestregression

rf=RandomForestRegressor(n_estimators=100,max_features='auto',max_depth=100,min_samples_split=10,min_samples_leaf=4,random_state=1)
model_rf=rf.fit(x_train,y_train)
rf_pred=model_rf.predict(x_test)
lin1_mse=mean_squared_error(y_test,rf_pred)
lin1_rmse=np.sqrt(lin1_mse)
print(lin1_rmse)
lin2=model_rf.score(x_test,y_test)
lin2_train=model_rf.score(x_train,y_train)
print(lin2,lin2_train)


datap.get_dtype_counts()
#########model building with imputed data


