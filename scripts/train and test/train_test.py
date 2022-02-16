# pip install --upgrade imblearn xgboost=1.0.0 lightgbm shap

import numpy as np
import pandas as pd 
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix,precision_score,recall_score
import plotly.graph_objs as go
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import shap
import sys
import csv
import os
import glob
from sklearn.decomposition import PCA
from matplotlib import pyplot


names = [
    'timestamp', 'mean_nni', 'sdnn', 'sdsd', 'rmssd', 'median_nni',
    'nni_50', 'pnni_50', 'nni_20', 'pnni_20', 'range_nni', 'cvsd', 'cvnni',
    'mean_hr', 'max_hr', 'min_hr', 'std_hr', 'total_power', 'vlf', 'lf',
    'hf', 'lf_hf_ratio', 'lfnu', 'hfnu', 'csi', 'cvi', 'Modified_csi',
    'triangular_index', 'tinn', 'sd1', 'sd2', 'ratio_sd2_sd1', 'sampen',
    'gt_self_report', 'gt_timestamp', 'gt_pss_control', 'gt_pss_difficult',
    'gt_pss_confident', 'gt_pss_yourway', 'gt_likert_stresslevel',
    'gt_score', 'gt_label'
]
selected_features = [
    'mean_nni', 'sdnn', 'rmssd', 'nni_50', 'lf', 'hf', 'lf_hf_ratio', 'sampen', 'ratio_sd2_sd1', 'sd2'
] #sd1, 'sd2'

params_all_users = {}

with open("score.csv", "w+") as w:
    w.write('Participant,Balanced Accuracy,F1 score,ROC_AUC,TPR,TNR\n')

cwd = os.getcwd()
os.chdir('11. combined-filter-v2')
for fname in glob.glob('*.csv'):
    print('participant', fname)
    # train dataset
    DATA_SET = pd.read_csv(fname, skiprows=1, names=names).replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    FEATURE = DATA_SET.iloc[:,1:-9].copy()
    FEATURE = FEATURE[selected_features]
    LABEL = DATA_SET.iloc[:,-1].copy()
    LABEL = LABEL.astype(int)
    CATEGORY = []

    # print('# Feature data')
    # print('- # rows = {} / # columns = {}'.format(FEATURE.shape[0], FEATURE.shape[1]))
    # print(FEATURE.head(), '\r\n')

    # print('# Class label')
    # print(LABEL.head(), '\r\n')

    # print('# Label distribution')
    # print('- # label = 0: {}'.format(np.count_nonzero(LABEL == 0)))
    # print('- # label = 1: {}'.format(np.count_nonzero(LABEL == 1)), '\r\n')

    K_FOLDS = []

    # n_splits: the number of folds.
    # shuffle: whether data are shuffled before splitting.
    splitter = StratifiedKFold(n_splits=5, shuffle=True) 

    # Here, 'train_indices' and 'test_indices' is numpy's array indicating indices of data.
    for idx, (train_indices, test_indices) in enumerate(splitter.split(FEATURE, LABEL)): 
        X_train = FEATURE.iloc[train_indices] 
        y_train = LABEL.iloc[train_indices]

        X_test = FEATURE.iloc[test_indices]
        y_test = LABEL.iloc[test_indices]

        # Here, we store train and test set of each fold into dictionary.
        K_FOLDS.append((X_train, y_train, X_test, y_test))

    K_FOLDS_RESAMPLE = []

    for idx, (X_train, y_train, X_test, y_test) in enumerate(K_FOLDS):
        # categorical_features: masked arrays indicating where categorical feature is placed.
        sampler = SMOTE()

        # 'fit_resample' conducts over-sampling data in the minority class.
        # Again, resampling should be only conducted in train set.
        X_sample, y_sample = sampler.fit_resample(X_train, y_train)

        # Because SMOTENC.fit_resample() returns a tuple of numpy's array (not DataFrame or Series!),
        # We need to again build DataFrame and Series from resampled data.
        X_sample = pd.DataFrame(X_sample, columns=X_train.columns)
        y_sample = pd.Series(y_sample)

        K_FOLDS_RESAMPLE.append((X_sample, y_sample, X_test, y_test))

    K_FOLDS_SCALED = []

    for X_train, y_train, X_test, y_test in K_FOLDS_RESAMPLE:
        scaler = MinMaxScaler()

        # StandardScaler.fit() finds characteristics of data distribution (i.e., min, max) in train set.
        scaler.fit(X_train)

        # Transform numeric data within train and test set.
        X_train_scale = scaler.transform(X_train)
        X_test_scale = scaler.transform(X_test)

        # Because MinMaxScaler.transform() returns a tuple of numpy's array (not DataFrame or Series!),
        # We need to again build DataFrame from scaled numeric data.
        X_train = pd.DataFrame(
            X_train_scale, index=X_train.index, columns=X_train.columns
        )
        X_test = pd.DataFrame(
            X_test_scale, index=X_test.index, columns=X_test.columns
        )

        K_FOLDS_SCALED.append((X_train, y_train, X_test, y_test))

    # This is used to store models for each fold.
    XGB_MODELS = []

    # Balanced accuracy, F1 score, and ROC-AUC score.
    scores = {
        'acc': [],
        'f1': [],
        'roc_auc': [],
        'TPR' : [],
        'TNR' : []
    }

    score = []
    score.append(fname)
    # 2 X 2 confusion matrix
    conf_mtx = np.zeros((2, 2))
    
    params_all_users[fname]={}

    dtrain = xgb.DMatrix(data=FEATURE, label=LABEL.to_numpy())

    params_all_users[fname] = {
      # Parameters that we are going to tune.
      'max_depth':6,
      'min_child_weight': 1,
      'eta':.3, 
      'subsample': 1,
      'colsample_bytree': 1,
      # Other parameters
      'objective':'binary:logistic',
      'booster':'gbtree',
      'verbosity':0
    }

    params_all_users[fname]['eval_metric'] = "auc"
    
    cv_results = xgb.cv(
      params_all_users[fname],
      dtrain,
      num_boost_round=1000,
      nfold=5,
      metrics={'auc'},
      early_stopping_rounds=25
    )

    gridsearch_params = [
      (max_depth, min_child_weight)
      for max_depth in range(0,12)
      for min_child_weight in range(0,8)
    ]

    min_mae = -float("Inf")
    best_params = None
    for max_depth, min_child_weight in gridsearch_params:
        # Update our parameters
        params_all_users[fname]['max_depth'] = max_depth
        params_all_users[fname]['min_child_weight'] = min_child_weight
        # Run CV
        cv_results = xgb.cv(
          params_all_users[fname],
          dtrain,
          nfold=5,
          metrics={'auc'},
          early_stopping_rounds=25
        )
        # Update best MAE
        mean_mae = cv_results['test-auc-mean'].max()
        boost_rounds = cv_results['test-auc-mean'].argmax()
        if mean_mae > min_mae:
            min_mae = mean_mae
            best_params = (max_depth,min_child_weight)

    params_all_users[fname]['max_depth'] = best_params[0]
    params_all_users[fname]['min_child_weight'] = best_params[1]



    gridsearch_params = [
      (subsample, colsample)
      for subsample in [i/10. for i in range(7,11)]
      for colsample in [i/10. for i in range(7,11)]
    ]

    min_mae = -float("Inf")
    best_params = None
    # We start by the largest values and go down to the smallest
    for subsample, colsample in reversed(gridsearch_params):
        # We update our parameters
        params_all_users[fname]['subsample'] = subsample
        params_all_users[fname]['colsample_bytree'] = colsample
        # Run CV
        cv_results = xgb.cv(
          params_all_users[fname],
          dtrain,
          num_boost_round=1000,
          nfold=5,
          metrics={'auc'},
          early_stopping_rounds=25
        )
        mean_mae = cv_results['test-auc-mean'].max()
        boost_rounds = cv_results['test-auc-mean'].argmax()
        if mean_mae > min_mae:
            min_mae = mean_mae
            best_params = (subsample,colsample)
    params_all_users[fname]['subsample'] = best_params[0]
    params_all_users[fname]['colsample_bytree'] = best_params[1]

    min_mae = -float("Inf")
    best_params = None
    for eta in [.3, .2, .1, .05, .01, .005]:
        # We update our parameters
        params_all_users[fname]['eta'] = eta
        # Run and time CV
        cv_results = xgb.cv(
          params_all_users[fname],
          dtrain,
          num_boost_round=1000,
          nfold=5,
          metrics=['auc'],
          early_stopping_rounds=25
        )
        # Update best score
        mean_mae = cv_results['test-auc-mean'].max()
        boost_rounds = cv_results['test-auc-mean'].argmax()
        if mean_mae > min_mae:
            min_mae = mean_mae
            best_params = eta
    params_all_users[fname]['eta'] = best_params

    min_mae = -float("Inf")
    best_params = None
    gamma_range = [i/10.0 for i in range(0,25)]
    for gamma in gamma_range : 
    # We update our parameters
        params_all_users[fname]['gamma'] = gamma
        # Run and time CV
        cv_results = xgb.cv(
            params_all_users[fname],
            dtrain,
            num_boost_round=1000,
            nfold=5,
            metrics=['auc'],
            early_stopping_rounds=25
          )
        # Update best score
        mean_mae = cv_results['test-auc-mean'].max()
        boost_rounds = cv_results['test-auc-mean'].argmax()
        if mean_mae > min_mae:
            min_mae = mean_mae
            best_params = gamma
    params_all_users[fname]['gamma'] = best_params


    for X_train, y_train, X_test, y_test in K_FOLDS_SCALED:
        results = {}
        MAX_ITER = 1000
        ETA_BASE = 0.03
        ETA_MIN = 0.001
        ETA_DECAY = np.linspace(ETA_BASE, ETA_MIN, MAX_ITER).tolist()
        # XGBoost requires a special data structure, xgboost.DMatrix()
        # Here we build DMatrix for train and test set.
        dtrain = xgb.DMatrix(data=X_train, label=y_train.to_numpy())
        dtest = xgb.DMatrix(data=X_test, label=y_test.to_numpy())

        # xgboost.train() conducts actual model training and returns a trained model.
        # For detailed parameter setting, please check: https://xgboost.readthedocs.io/en/latest/parameter.html
        booster = xgb.train(
            params_all_users[fname],
            # dtrain: DMatrix of training data
            dtrain=dtrain,
            # num_boost_round: the number of boosted trees
            num_boost_round=1000, 
            # early_stopping_rounds: early stop generating trees when eval_metric is not improved
            early_stopping_rounds=25,
            # evals: evaluation set to check early stooping
            evals=[(dtest, 'test')],#,(dtrain,'train')],
            verbose_eval=False,
            evals_result = results
            #callbacks = [xgb.callback.reset_learning_rate(ETA_DECAY)]
            #learning_rates=ETA_DECAY

        )

        # epochs = len(results['test']['auc'])
        # x_axis = range(0, epochs)
        # fig, ax = pyplot.subplots()
        # #ax.plot(x_axis, results['train']['auc'], label='Train')
        # ax.plot(x_axis, results['test']['auc'], label='Test')
        # ax.legend()
        # pyplot.ylabel('AUC')
        # pyplot.title('XGBoost training plot')
        # pyplot.show()


        # predict() returns probability of a positive label (label == 1)
        y_pred = booster.predict(
            # dtest: DMatrix of test data set
            data=dtest, 
            # ntree_limit: the number of boosted trees used for prediction.
            # booster.best_ntree_limit returns the number of trees that show best performance.
            ntree_limit=booster.best_ntree_limit)

        # Because predict() returns probability, we should change them into class labels.
        # Here, we set cur-off as 0.5: positive label when a probability is higher than 0.5.
        y_pred_class = np.where(y_pred > 0.5, 1, 0)

        acc = balanced_accuracy_score(y_test, y_pred_class)
        f1 = f1_score(y_test, y_pred_class, average='macro')
        roc_auc = roc_auc_score(y_test, y_pred)
        tpr = recall_score(y_test, y_pred_class)
        tnr = recall_score(y_test, y_pred_class,pos_label = 0)

        scores['acc'].append(acc)
        scores['f1'].append(f1)
        scores['roc_auc'].append(roc_auc)
        scores['TPR'].append(tpr)
        scores['TNR'].append(tnr)

        conf_mtx += confusion_matrix(y_test, y_pred_class)

        XGB_MODELS.append(booster)


    # print('# Classification results')
    for k, v in scores.items():
        # print('- {}: {}'.format(k.upper(), np.mean(v)))
        score.append(np.mean(v))

    with open("%s/score.csv" % cwd, "a+") as w:
        w.write(','.join([str(val) for val in score]) + '\n')

    # fig = go.Figure(
    #     go.Heatmap(
    #         x=['Pred. Label = 0', 'Pred Label = 1'],
    #         y=['True. Label = 1', 'True. Label = 0'],
    #         z=np.flip(conf_mtx, axis=0)
    #     )
    # )
    # fig.update_layout(
    #     title_text='Confusion matrix for XGBoost evaluation',
    #     xaxis_title_text='Prediction',
    #     yaxis_title_text='True'
    # )
    # fig.show()

    # feature_importances = {}

    # for model in XGB_MODELS:
    #     # get_fscore() returns a dictionary, where a key is a feature name and a value is feature importance.
    #     imp = model.get_fscore()
    #     for k, v in imp.items():
    #         # If a given feature name already exists, add feature importance into existing importance.
    #         if k in feature_importances:
    #             feature_importances[k] += v
    #         else:
    #             feature_importances[k] = v

    # feature_importances = [
    #     # Averaging feature importance
    #     (k, float(v) / 5)
    #     for k, v in feature_importances.items()
    # ]


    # Build dataframe from feature importances, and sort them by importances in descending order.
    # And then, select top 50 features.
    # df = pd.DataFrame(
    #     feature_importances, 
    #     columns=['name', 'importance']
    # ).sort_values('importance', ascending=False).head(50)

    # feature_importance_user = {}
    # feature_importance_user[fname] = df

    # fig = go.Figure(
    #     go.Bar(
    #         x=df.loc[:, 'name'],
    #         y=df.loc[:, 'importance']
    #     )
    # )
    # fig.update_layout(
    #     title_text='Top features for XGBoost',
    #     yaxis_title_text='Feature importances'
    # )
    # fig.show()


    # print('shap')
    # SHAP_VALUES = []
    # TEST_FEATURES = []

    # for (X_train, y_train, X_test, y_test), model in zip(K_FOLDS_SCALED, XGB_MODELS):
    #     # shap.TreeExplainer() is used to explain tree-based ensemble models.
    #     explainer = shap.TreeExplainer(model)

    #     # TreeExplainer.shap_values indicate how features of each sample contribute to predicted output.
    #     shap_value = explainer.shap_values(X_test, tree_limit=model.best_ntree_limit)

    #     SHAP_VALUES.append(shap_value)
    #     TEST_FEATURES.append(X_test)

    # # Merging all shap values and train data.
    # # SHAP_VALUES = np.vstack(SHAP_VALUES)
    # # TEST_FEATURES = pd.concat(TEST_FEATURES, axis=0)

    # # shap.summary_plot(SHAP_VALUES, TEST_FEATURES)

    # shap_features = {}
    # shap_features[fname] = [SHAP_VALUES, TEST_FEATURES]
