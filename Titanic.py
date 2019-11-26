# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 07:33:34 2019

@author: LENOVO
"""
import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#Drop features we are not going to use
train = train.drop(['Name', 'Ticket','SibSp','Parch','Fare',  'Cabin'],axis=1)
test = test.drop(['Name','Ticket','Ticket','SibSp' , 'Fare','Cabin'],axis=1)

#Look at the first 3 rows of our training data
train.head(3)
for df in [train,test]:
    df['Sex_binary']=df['Sex'].map({'male':1,'female':0})
    df['Embarked']=df['Embarked'].map({'S':1,'Q':2,'C':3})
    
    
#Fill in missing age values with 0 (presuming they are a baby if they do not have a listed age)
train['Age'] = train['Age'].fillna(0)
test['Age'] = test['Age'].fillna(0)
#test['Fare'] = test['Fare'].fillna(0)

#Select feature column names and target variable we are going to use for training
features = ['Pclass','Age','Sex_binary']
target = 'Survived'

#Look at the first 3 rows (we have over 800 total rows) of our training data.; 
#This is input which our classifier will use as an input.
train[features].head(3)
for df in [train,test]:
    df['Sex_binary']=df['Sex'].map({'male':1,'female':0})
    df['Embarked']=df['Embarked'].map({'S':1,'Q':2,'C':0})
#Fill in missing age values with 0 (presuming they are a baby if they do not have a listed age)
train['Age'] = train['Age'].fillna(0)
test['Age'] = test['Age'].fillna(0)
train['Embarked'] = train['Embarked'].fillna(0)
test['Embarked'] = test['Embarked'].fillna(0)

#Select feature column names and target variable we are going to use for training
features = ['Pclass','Age','Sex_binary','Embarked']
target = 'Survived'

#Look at the first 3 rows (we have over 800 total rows) of our training data.; 
#This is input which our classifier will use as an input.
#from sklearn.tree import DecisionTreeClassifier

#Create classifier object with default hyperparameters
from sklearn.metrics import accuracy_score
from sklearn.learning_curve import validation_curve

k=[]
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train[features],train[target], test_size=0.33, random_state=42)
C_param_range = [0.0001,0.001,0.01,0.1111,0.1,1,10,100]
sepal_acc_table = pd.DataFrame(columns = ['C_parameter','Accuracy'])
sepal_acc_table['C_parameter'] = C_param_range
j = 0
for i in C_param_range:
    
    # Apply logistic regression model to training data
    lr = LogisticRegression(penalty = 'l2', C = i,random_state = 0,class_weight='balanced')
    lr.fit(X_train,y_train)
    
    # Predict using model
    y_pred_sepal = lr.predict(X_test)
    
    # Saving accuracy score in table
    sepal_acc_table.iloc[j,1] = accuracy_score(y_test,y_pred_sepal)
    j += 1
    
  #rd =LogisticRegression(penalty='l2', dual=True, tol=0.0001, 
                          # C=1, fit_intercept=True, intercept_scaling=1.0, 
                          # class_weight=None, random_state=None)

  

#from sklearn.neural_network import MLPClassifier
#mlp = MLPClassifier(hidden_layer_sizes=(15,15,15),max_iter=500)
#mlp.fit(train[features],train[target])
#clf.fit(train[features],train[target])
#pred=clf.predict(X_test)
#from sklearn.metrics import accuracy_score
#acc = accuracy_score(pred, y_test)
#print(round (acc,3))
#clf = DecisionTreeClassifier()  

#Fit our classifier using the training features and the training target values
#clf.fit(train[features],train[target]) 
#train[features].head(3)
predictions = lr.predict(test[features])

#Display our predictions - they are either 0 or 1 for each training instance 
#depending on whether our algorithm believes the person survived or not.
predictions
submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':predictions})

#Visualize the first 5 rows
submission.head()
filename = 'Titanic Predictions 1.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)
