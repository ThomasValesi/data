#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 16:12:49 2023

@author: thomasvalesi
"""
##Import all I need 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from joblib import dump


def flowers():
    ##Import the database
    data = pd.read_csv("/Users/thomasvalesi/Downloads/IRIS_ Flower_Dataset.csv")

    caract=data.drop("species",axis=1) ##we keep all the mesure 
    target=data["species"] ##we keep all what species it is to check

    ##I create the tab to do the train and the test 20% of the database are for the test
    caract_train, caract_test, target_train, target_test = train_test_split(caract, target, test_size=0.20, random_state=42)

    ##this function create a random forest with 200 trees
    rf = RandomForestClassifier(n_estimators=200, random_state=42)

    rf.fit(caract_train, target_train)##it train the algorithm
    
    ##dump(rf, 'random_forest_model_flowers.joblib')

    target_pred = rf.predict(caract_test)##it test the random forest

    accuracy = accuracy_score(target_test, target_pred)
    print(f"The precision of the model is : {accuracy}") 
    
    
    ##Graf 1
    plt.bar(data['species'],data['petal_width'])
    plt.savefig('graf1_flowers.png')
    plt.show()
    
    ##Graf 2
    conf_matrix = confusion_matrix(target_test, target_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Predict 0', 'Predict 1'], yticklabels=['True 0', 'True 1'])
    plt.xlabel('Prediction')
    plt.ylabel('true label')
    plt.title('Confusion matrix')
    plt.savefig('graf2_flowers.png')
    plt.show()
    
    ##Graf 3
    importances = rf.feature_importances_
    feature_names = caract.columns
    indices = np.argsort(importances)
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importance of caracteristics')
    plt.title('Importance of caracteristics in random forest')
    plt.savefig('graf3_flowers.png')
    plt.show()
    
    ##Graph 4
    plt.figure(figsize=(10,5))
    sns.heatmap(caract_train.corr())
    plt.savefig('graf4_flowers.png')

    ##I add this part to allow enter values of a iris and it uses the random forest to predict what species is
    sl=float(input("Enter the value of the sepal length : "))
    sw=float(input("Enter the value of the sepal width : "))
    pl=float(input("Enter the value of the petal length : "))
    pw=float(input("Enter the value of the petal width : "))

    values_test = pd.DataFrame({
        'sepal_length': [sl],
        'sepal_width': [sw],
        'petal_length': [pl],
        'petal_width': [pw]
    })

    tool_pred=rf.predict(values_test)

    print(values_test)
    print(tool_pred)


def titanic():
  
    from sklearn.model_selection import GridSearchCV
  
    data_train = pd.read_csv("/Users/thomasvalesi/Downloads/titanic/train.csv", index_col='PassengerId')
    data_test = pd.read_csv("/Users/thomasvalesi/Downloads/titanic/test.csv", index_col='PassengerId')
    data_test_survived = pd.read_csv("/Users/thomasvalesi/Downloads/titanic/gender_submission.csv", index_col='PassengerId')


    ##we replace any values for it's simpler than before 
    data_train['Sex'] = data_train['Sex'].map({'male': 0, 'female': 1})
    data_test['Sex'] = data_test['Sex'].map({'male': 0, 'female': 1})

    data_train['Embarked'] = data_train['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    data_test['Embarked'] = data_test['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    
    ##data_train['Cabin'] = data_train['Cabin'].apply(lambda x: 1 if pd.notna(x) and x != '' else 0)
    ##data_test['Cabin'] = data_test['Cabin'].apply(lambda x: 1 if pd.notna(x) and x != '' else 0)


    columns_to_remove_caract_train=["Survived","Name","Ticket","Cabin"]
    caract_train=data_train.drop(columns_to_remove_caract_train,axis=1)
    caract_train=caract_train.fillna(0)


    target_train=data_train["Survived"]

    columns_to_remove_caract_test=["Name","Ticket","Cabin"]
    caract_test=data_test.drop(columns_to_remove_caract_test,axis=1)
    caract_test=caract_test.fillna(0)


    # define the hyperparameters to test
    param_grid = {
        'n_estimators': [100, 500, 1000],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    

    # initialize the random forest
    rf = RandomForestClassifier(random_state=42)

    # find thee best hyperparameters
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)
    grid_search.fit(caract_train, target_train)

    best_params = grid_search.best_params_
    print(f"Best hyperparameters: {best_params}")

    # use the best hyperparameter to train the model
    best_rf = RandomForestClassifier(n_estimators=best_params['n_estimators'],
                                     max_depth=best_params['max_depth'],
                                     min_samples_split=best_params['min_samples_split'],
                                     random_state=42)
    best_rf.fit(caract_train, target_train)
    
    ##dump(best_rf, 'random_forest_model_titanic.joblib')

    # predict with the best hyperparameters 
    target_pred = best_rf.predict(caract_test)
    accuracy = accuracy_score(data_test_survived, target_pred)
    print(f"The precision of the model is : {accuracy}")
    
    ## Graf 1
    importances = best_rf.feature_importances_
    feature_names = caract_train.columns
    indices = np.argsort(importances)
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importance of caracteristics')
    plt.title('Importance of caracteristics in random forest')
    plt.savefig('graf1_titanic.png')
    plt.show()
    
    ##Graph 2
    plt.figure(figsize=(10,5))
    sns.heatmap(caract_train.corr())
    plt.savefig('graf2_titanic.png')


    Pclass=float(input("Enter the class (1 or 2 or 3) : "))
    Sex=float(input("Enter the sex (0:male 1:female): "))
    Age=float(input("Enter the age (if you don't know enter 0') : "))
    SibSp=float(input("Enter the number of siblings and spouses : "))
    Parch=float(input("Enter the number of parent children : "))
    Fare=float(input("Enter the Fare : "))
    Embarked=float(input("Enter the embarked S:0, C:1, Q:2 : "))

    values_test = pd.DataFrame({
        'Pclass': [Pclass],
        'Sex': [Sex],
        'Age': [Age],
        'SibSp': [SibSp],
        'Parch': [Parch],
        'Fare': [Fare],
        'Embarked': [Embarked]
    })

    tool_pred=best_rf.predict(values_test)

    print(values_test)
    if(tool_pred==0):
        print("Survived")
    else:
        print("Dead")
        
        
        
    
    

    
    
    

    
    
    
    
    
    
    
    
    
    
    
