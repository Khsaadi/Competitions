# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 08:32:19 2019

@author: LENOVO
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
#read data
train = pd.read_csv('training_set.csv')
test = pd.read_csv('testing_set.csv')



#shape of the data
train.shape, test.shape

# check index of dataframe
train.columns

target='HotelPrice_0'

train_features = train.drop(['Row_id','Date','HotelPrice_0'],axis=1)
test_features=test.drop(['Row_id','Date'],axis=1)
features = pd.concat([train_features, test_features]).reset_index(drop=True)

features=features.replace(-1,features.mean())

for i in features:
   k=features[i].values
   for j in range( len(k)):
       if k[j]<0:
          print(k[j])
          k[j]=k.mean()


features['revenu total']=features['Revenue_1']+features['Revenue_0']+features['Revenue_7']+features['Revenue_14']+features['Revenue_21']+features['Revenue_28']
features['total BKg']=features['BKG_1']+features['BKG_0']+features['BKG_7']+features['BKG_14']+features['BKG_21']+features['BKG_28']
features['totalprice1']=features['PriceComp1_0']+features['PriceComp2_0']+features['PriceComp3_0']+features['PriceComp4_0']
features['totalprice14']=features['PriceComp1_14']+features['PriceComp2_14']+features['PriceComp3_14']+features['PriceComp4_14']
features['totalprice1']=features['PriceComp1_21']+features['PriceComp2_21']+features['PriceComp3_21']+features['PriceComp4_21']
features['totalprice1']=features['PriceComp1_28']+features['PriceComp2_28']+features['PriceComp3_28']+features['PriceComp4_28']


sns.distplot(train[target])

A=train_features.isnull().sum()  
B=test_features.isnull().sum() 
from sklearn.metrics import r2_score


l=[] 
for i in features:
   counts =features[i].value_counts()
   print(counts)
   l.append(counts)
   

features.describe()
d={}
features=pd.DataFrame(features)
for i in(features):
   j=features[i].corr(train[target])
   if abs(j)<0.1:
    features = features.drop([i], axis=1)
    print(i,j)
    d[i]=j
    
pca = PCA(n_components=30)
pca.fit(features)
features= pca.transform(features)    

#for i in features:
     #features[features[i] < 0] = 0
         
features[features< 0]=0
features=features.replace(features[features< 0],features.mean())    
features.describe()

# 
#= log10(Y + 1 - min(Y)); /* translate, then transform */

features=features.loc[:,""].mean()
for i in features:
    if(features[i]<0):
        
        

 acc=r2_score(train[target,])
#applying Maxminscaler to normalize the data
scaler = MinMaxScaler()
scaler.fit(features)
features=scaler.transform(features) 

#separtae the train features and test features
features=pd.DataFrame(features)
train_features= features.iloc[:len(train[target]), :]
test_features=features.iloc[len(train[target]):, :]

# split data test and train
#from sklearn.model_selection import cross_validate
#from sklearn.model_selection import train_test_split
#feature_train, feature_test, target_train, target_test = train_test_split(train_features,train[target], test_size=0.35, random_state=42
# apllying the linear classifier
import math
from sklearn import linear_model
clf = linear_model.Lasso(alpha=100)
clf.fit(train_features,train[target])
predictions=clf.predict(test_features) 

regr = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100)
regr.fit(train_features, train[target])
predictions=regr.predict(test_features) 

 
#predictions=pd.DataFrame(predictions)
#target_test=pd.DataFrame(target_test)
#target_test=target_test.astype(float64)
#from sklearn.metrics import accuracy_score
score=clf.score(train_features,train[target])


submission = pd.DataFrame({'Row_id':test['Row_id'],'HotelPrice_0':predictions})
#Visualize the first 5 rows
submission.head()
filename = 'predictions1.csv'
submission.to_csv(filename,index=False)
print('Saved file: ' + filename)





 
