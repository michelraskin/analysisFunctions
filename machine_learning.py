import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from analysis_functions import *
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, roc_auc_score, make_scorer, auc, roc_curve
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from xgboost import XGBClassifier, XGBRegressor, plot_tree, plot_importance, to_graphviz
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from graphviz import Source
from sklearn import tree
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.svm import SVC, SVR
import re
from sklearn.inspection import permutation_importance
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2
from sklearn.decomposition import PCA

DefaultGrid = [
    {
        'clf': [XGBClassifier(eval_metric='logloss'), RandomForestClassifier()],
        'clf__n_estimators': [5, 10, 50, 200], 
        'clf__max_depth': [2, 5, 10, 25, None]
    },
    {
        'clf': [DecisionTreeClassifier()],
        'clf__max_depth': [2, 5, 20, 50, None],
        'clf__class_weight': [None, 'balanced']
    },
    {
        'clf': [LogisticRegression(), SVC(probability=True, kernel='linear')],
        'clf__C': [0.1, 1, 10]
    }
]

def getDefaultPreprocessor(aNumericalColumns, aBinaryColumns):
    return    ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), aNumericalColumns), 
                ('bin', 'passthrough', aBinaryColumns), 
            ],
            remainder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        )

def getDefaultPipelineSteps(X_train):
    myNumericalColumns = X_train.columns[(X_train.nunique() > 10) & (X_train.dtypes != object)]
    myBinaryColumns = X_train.columns[X_train.nunique() == 2]
    myPreprocessor = getDefaultPreprocessor(aNumericalColumns=myNumericalColumns, aBinaryColumns=myBinaryColumns)
    return [('preprocessor', myPreprocessor), ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean'))]#, ('pca', PCA(n_components=0.95))]

def gridSearchKFoldClassification(X_train, X_test, y_train, y_test, aScore = 'roc_auc', aGrid = DefaultGrid):
    kf = StratifiedKFold(n_splits=10, shuffle=True)
    myPipelineSteps = getDefaultPipelineSteps(X_train = X_train)
    myPipelineSteps.append(('clf', XGBClassifier()))
    myPipeline = Pipeline(myPipelineSteps)
    myGridSearchCv = GridSearchCV(myPipeline, aGrid, cv=kf, scoring=aScore, n_jobs=-1, verbose=3)
    myGridSearchCv.fit(X_train, y_train)
    myBestModel = myGridSearchCv.best_estimator_
    y_pred_proba = myBestModel.predict_proba(X_test)[:, 1]
    y_pred = myBestModel.predict(X_test)
    print(f'Best parameters: {myGridSearchCv.best_params_}')
    print(f'Best cross val {aScore} score: {myGridSearchCv.best_score_:.4f}')
    print(f'Area under the receiver operating curve on test set is {roc_auc_score(y_test, y_pred_proba):.4f}')
    print(f'Accuracy score on test set is {accuracy_score(y_test, y_pred):.4f}')
    return myGridSearchCv

def getTopFeatures(aGridSearch, aColumnNames, X_train= None, y_train = None):
    myBestModel = aGridSearch.best_estimator_
    if hasattr(myBestModel.named_steps['clf'], 'coef_'):
        myImportances = myBestModel.named_steps['clf'].coef_[0]
    elif hasattr(myBestModel.named_steps['clf'], 'feature_importances_'):
        myImportances = myBestModel.named_steps['clf'].feature_importances_
    else: 
        myImportances = permutation_importance(myBestModel, X_train, y_train).importances_mean
        aColumnNames = X_train.columns
    
    myFeatureImportancesDf = pd.DataFrame({
        'Feature': aColumnNames,
        'Importance': myImportances
    })
    myFeatureImportancesDf.sort_values(by='Importance', ascending=False, inplace=True)
    myTopFeatures = myFeatureImportancesDf.head(50)
    plt.figure(figsize=(8, 8))
    sns.barplot(x = 'Importance', y= 'Feature', data=myTopFeatures)
    plt.title(f'Importances for {aGridSearch.best_params_}')
    return myFeatureImportancesDf

DefaulGridRegression = [
    {
        'clf': [XGBRegressor(), RandomForestRegressor()],
        'clf__n_estimators': [5, 10, 200], 
        'clf__max_depth': [2, 5, 10, None]
    },
    {
        'clf': [KNeighborsRegressor()],
        'clf__n_neighbors': [2, 5, 10],
    },
    {
        'clf': [LogisticRegression(), SVR()],
        'clf__C': [0.1, 1, 10]
    }
]

def gridSearchKFoldRegression(X_train, X_test, y_train, y_test, aScore = 'r2', aGrid = DefaulGridRegression):
    kf = StratifiedKFold(n_splits=5, shuffle=True)
    
    myPipelineSteps = getDefaultPipelineSteps(X_train=X_train)
    myPipelineSteps.append(('clf', XGBRegressor()))
    myPipeline = Pipeline(myPipelineSteps)
    myGridSearchCv = GridSearchCV(myPipeline, aGrid, cv=kf, scoring=aScore, n_jobs=-1, verbose=0)
    myGridSearchCv.fit(X_train, y_train)
    myBestModel = myGridSearchCv.best_estimator_
    y_pred = myBestModel.predict(X_test)
    print(f'Best parameters: {myGridSearchCv.best_params_}')
    print(f'Best cross val {aScore} score: {myGridSearchCv.best_score_:.4f}')
    print(f'R2 score    on test set: {r2_score(y_test, y_pred)}')
    return myGridSearchCv

def plotRocAucCuve(aGridSearchCv, X_test, y_test):
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
    plt.title(f'ROC Curve for {aGridSearchCv.best_params_}')
    plt.show()

def getPredictedThirds(aDf):
    lower_third = aDf['predicted_effect'].quantile(1/3)
    upper_third = aDf['predicted_effect'].quantile(2/3)
    if upper_third == lower_third:
        print(f'No effect difference')
        return 1, 1, aDf
    aDf['predicted_effect_group'] = pd.cut(
        aDf['predicted_effect'],
        bins=[-float('inf'), lower_third, upper_third, float('inf')],
        labels=['Lower', 'Middle', 'Upper']
    )
    return lower_third, upper_third, aDf

def plotPredictedTreatmentEffect(myNewDf, aCategory = 'CPC12'):
    lower_third, upper_third, myNewDf = getPredictedThirds(myNewDf)
    if lower_third == upper_third:
        plt.scatter(x = range(len(y_pred_proba1)), y = myNewDf['predicted_effect'].sort_values())
    else:
        myNewDf.sort_values(['predicted_effect'], inplace=True)
        myNewDf.reset_index(inplace=True)
        for group in myNewDf['predicted_effect_group']:
            myFilter = myNewDf['predicted_effect_group'] == group
            plt.scatter(x = myNewDf[myFilter]['predicted_effect'].index, y = myNewDf[myFilter]['predicted_effect'])
        plt.legend(myNewDf['predicted_effect_group'].unique())
    plt.title(f'Predicted treatment effect diff between hypothermia and normothermia for {aCategory}')
    return lower_third, upper_third, myNewDf

def getPredictedTreatmentEffectSupervisedClassif(X_train, aModel, aCategory, aGroup):
    myXValueModified1 = X_train.copy()
    myXValueModified1[aGroup] = 1.0
    myXValueModified2 = X_train.copy()
    myXValueModified2[aGroup] = 0.0
    if hasattr(aModel, 'predict_proba'):
        y_pred_proba1 = aModel.predict_proba(myXValueModified1)[:, 1]
        y_pred_proba2 = aModel.predict_proba(myXValueModified2)[:, 1]
    else:
        y_pred_proba1 = aModel.predict(myXValueModified1)
        y_pred_proba2 = aModel.predict(myXValueModified2)
    myNewDf = pd.DataFrame()
    myNewDf['predicted_effect'] = (y_pred_proba1 - y_pred_proba2)
    return plotPredictedTreatmentEffect(myNewDf=myNewDf, aCategory=aCategory)
    

def getTreatmentEffectDiff(X_train, y_train, aModel, aCategory = 'CPC12', aGroup = 'groupe'):
    lower_third, upper_third, myNewDf = getPredictedTreatmentEffectSupervisedClassif(X_train, aModel, aCategory, aGroup)
    if upper_third == lower_third:
        print(f'No effect difference')
        return 1
    myData = pd.concat([X_train[aGroup].reset_index(), myNewDf['predicted_effect_group'].reset_index(), y_train.reset_index()], axis=1)
    model1 = smf.logit(
        f'{aCategory} ~ predicted_effect_group + {aGroup}',
        data=myData
    ).fit()

    model2 = smf.logit(
        f'{aCategory} ~ predicted_effect_group * {aGroup}',
        data=myData
    ).fit()

    llr = -2*(model1.llf - model2.llf)
    df_diff = model2.df_model - model1.df_model
    p_value = chi2.sf(llr, df_diff)

    print(f'Likelihood ratio of test results:')
    print(f'Chi square statistic: {llr}')
    print(f'p-value: {p_value}')
    print(f'Degress of freedom: {df_diff}')
    return p_value

def getTreatmentEffectDiffUnsupervised(aX, aY, aGroups, aCategory = 'CPC12', aGroup = 'groupe'):
        myNewDf = pd.DataFrame()
        myNewDf['predicted_effect_group'] = aGroups
        myData = pd.concat([aX[aGroup].reset_index(), myNewDf['predicted_effect_group'].reset_index(), aY.reset_index()], axis=1)
        model1 = smf.logit(
        f'{aCategory} ~ predicted_effect_group + {aGroup}',
        data=myData
        ).fit()

        model2 = smf.logit(
        f'{aCategory} ~ predicted_effect_group * {aGroup}',
        data=myData
        ).fit()

        llr = -2*(model1.llf - model2.llf)
        df_diff = model2.df_model - model1.df_model
        p_value = chi2.sf(llr, df_diff)

        print(f'Likelihood ratio of test results:')
        print(f'Chi square statistic: {llr}')
        print(f'p-value: {p_value}')
        print(f'Degress of freedom: {df_diff}')
        return p_value, model2, myData

def plotPredictedEffectDiff(aData, aBestModel, aCategory = 'CPC12', aGroup = 'groupe'):
    predicted_effect_groups = aData['predicted_effect_group'].unique()
    predicted_effect_groups.sort()
    groupe_values = aData[aGroup].unique()
    groupe_values.sort()
    groupe_values = list(filter(lambda x: not np.isnan(x), groupe_values))
    
    predicted_effect_groups = list(filter(lambda x: not np.isnan(x), predicted_effect_groups))

    # Prepare the DataFrame for prediction
    predictions = []
    for groupe in groupe_values:
            for effect in predicted_effect_groups:
                    temp_df = pd.DataFrame({
                            'predicted_effect_group': [effect],
                            'groupe': [groupe]
                    })
                    # Predict the probability
                    temp_df['predicted_prob'] = aBestModel.predict(temp_df)
                    predictions.append(temp_df)

    # Concatenate all predictions
    predictions_df = pd.concat(predictions)

    # Create a bar plot
    plt.figure(figsize=(8, 6))
    for i, groupe in enumerate(groupe_values):
            subset = predictions_df[predictions_df['groupe'] == groupe]
            plt.bar(
                    subset['predicted_effect_group'] + (i * 0.2) - 0.1,    # Shift bars slightly for better visualization
                    subset['predicted_prob'],
                    width=0.2,
                    label=f'Groupe {groupe}'
            )
            
    x_positions = np.arange(len(predicted_effect_groups)) 

    plt.xticks(x_positions)
    plt.xlabel('Predicted Effect Group')
    plt.ylabel(f'Predicted Probability of {aCategory}')
    plt.title(f'Predicted Probability of {aCategory} by Predicted Effect Group and Groupe')
    plt.legend(title='Groupe')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
