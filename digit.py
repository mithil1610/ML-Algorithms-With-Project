#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 10:12:53 2019

@author: dashrath
"""
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
image=img.imread('test1.jpg')
plt.imshow(image,cmap=plt.cm.gray)
image.shape
imm=image.reshape(1,-1)
imm.shape
plt.imshow(imm,cmap=plt.cm.gray)
print(imm.shape)
print(imm[0:,0:64])
import seaborn as sns
digits=load_digits()

digits.data.shape
print(digits.data[0:2])
print(digits.target.shape)
plt.figure(figsize=(20,2))
for index,(image,label) in enumerate(zip(digits.data[0:5],digits.target[0:5])):
    plt.subplot(1,8,index+1)
    plt.imshow(np.reshape(image,(8,8)),cmap=plt.cm.gray)
    
x_train,x_test,y_train,y_test=train_test_split(digits.data,digits.target,test_size=0.3,random_state=2)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

lg=linear_model.LogisticRegression()
lg.fit(x_train,y_train)
pred=lg.predict(x_test)
x_tr=x_test[0].reshape(1,-1)
print(x_tr.shape)
print(lg.predict(imm))
print("accuracy :",lg.score(x_test,y_test))
cm=confusion_matrix(y_test,pred)
print(cm)
print("mis classified",(y_test != pred).sum())
