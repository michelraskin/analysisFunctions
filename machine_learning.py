import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from analysisFunctions.analysis_functions import *
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, roc_auc_score, make_scorer, auc, roc_curve
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from xgboost import XGBClassifier, XGBRegressor, plot_tree, plot_importance, to_graphviz
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from graphviz import Source
from sklearn import tree
from sklearn.svm import SVC
import re

DefaultGrid = [
  {
    'clf': [XGBClassifier(eval_metric='logloss'), RandomForestClassifier()],
    'clf__n_estimators': [5, 10, 50], 
    'clf__max_depth': [2, 5, 10]
  },
  {
    'clf': [DecisionTreeClassifier()],
    'clf__max_depth': [2, 5, 20],
    'clf__class_weight': [None, 'balanced']
  },
  {
    'clf': [LogisticRegression(penalty='l1'), SVC(probability=True, kernel='linear')],
    'clf__C': [0.1, 1, 10]
  }
]

def gridSearchKFoldClassification(X_train, X_test, y_train, y_test, aScore = 'roc_auc', aGrid = DefaultGrid, aPreprocessor = None, aNumericalColumns = None):
  kf = StratifiedKFold(n_splits=10, shuffle=True)
  
  myPreprocessor = aPreprocessor
  if aPreprocessor is None:
    myPreprocessor = ColumnTransformer(
      transformers=[
        ('num', StandardScaler(), aNumericalColumns)
      ],
      remainder='passthrough'
    )
  
  myPipeline = Pipeline([('preprocessor', myPreprocessor), ('clf', XGBClassifier())])
  myGridSearchCv = GridSearchCV(myPipeline, aGrid, cv=kf, scoring=aScore, n_jobs=-1, verbose=0)
  myGridSearchCv.fit(X_train, y_train)
  myBestModel = myGridSearchCv.best_estimator_
  y_pred_proba = myBestModel.predict_proba(X_test)[:, 1]
  y_pred = myBestModel.predict(X_test)
  print(f'Best parameters: {myGridSearchCv.best_params_}')
  print(f'Best cross val roc auc score: {myGridSearchCv.best_score_:.4f}')
  print(f'Area under the receiver operating curve is {roc_auc_score(y_test, y_pred_proba):.4f}')
  print(f'Accuracy score is {accuracy_score(y_test, y_pred):.4f}')
  return myGridSearchCv

def getTopFeatures(aGridSearch, aColumnNames):
  myBestModel = aGridSearch.best_estimator_
  if hasattr(myBestModel.named_steps['clf'], 'coef_'):
    myImportances = myBestModel.named_steps['clf'].coef_[0]
  else:
    myImportances = myBestModel.named_steps['clf'].feature_importances_
  myFeatureImportancesDf = pd.DataFrame({
    'Feature': aColumnNames,
    'Importance': myImportances
  })
  myFeatureImportancesDf.sort_values(by='Importance', ascending=False, inplace=True)
  myTopFeatures = myFeatureImportancesDf.head(50)
  plt.figure(figsize=(8, 8))
  sns.barplot(x = 'Importance', y= 'Feature', data=myTopFeatures)
  plt.title(f'Importances for {aGridSearch.best_params_}')

def plotRocAucCuve(aGridSearchCv, y_test):
  myBestModel = aGridSearchCv.best_estimator_
  y_pred_proba = myBestModel.predict_proba(X_test)[:, 1]
  myFpr, myTpr, myThresholds = roc_curve(y_test, y_pred_proba)
  myRocAuc = auc(myFpr, myTpr)
  sns.set(style='whitegrid')
  plt.figure(figsize=(8,6))
  sns.scatterplot(x= myFpr, y= myTpr, color='darkorange', lw=2, label = f'ROC curve {myRocAuc:.2f}')
  sns.lineplot(x= [0, 1], y= [0,1], color='navy', lw=2, linestyle = '--')
  plt.xlim([0, 1.0])
  plt.ylim([0, 1.05])
  plt.xlabel('False positive rate')
  plt.ylabel('True Positive Rate')
  plt.legend(loc='lower right')
  plt.title(f'ROC Curve for {myGridSearchCv.best_params_}')
  plt.show()

