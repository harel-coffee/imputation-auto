#!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3
import numpy as np
import pandas as pd
import csv
import os
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import matplotlib.pyplot as plt 
import matplotlib.path as mpath
import matplotlib.patches as patches  
from matplotlib.lines import Line2D  
from matplotlib.gridspec import GridSpec
import matplotlib.collections as collections

import operator
import itertools
import time
import math
import random
import codecs
import pydotplus 
import copy
import pickle

# sklearn
import tensorflow as tf
from sklearn import linear_model as lm
from sklearn import svm
from sklearn import feature_selection as fs
from sklearn import model_selection as ms
from sklearn import ensemble as es
from sklearn import tree
from sklearn import pipeline
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics.ranking import roc_auc_score
 
from scipy import stats 
from functools import partial
from datetime import datetime
from numpy import inf
from cgi import log
from decimal import *
from collections import Counter
sns.set(rc={'axes.facecolor':'#C0C0C0'}) 

import alm_fun

class alm_es:

    def __init__(self, es_init_params):
        for key in es_init_params:
            setattr(self, key, es_init_params[key])
 
    def run_backup(self, features, compare_features, ml_type, train_x, train_y, test_x, test_y, extra_train_x=None, extra_train_y=None, use_extra_train_data=0):
         
        if any(isinstance(i, list) for i in features):  # if features are nested list
            features = list(itertools.chain(*features))
        features = list(set(features))       
                
        if self.estimator != None:  # if estimator is None, there is no need to train the model 
            # fit the estimator        
            if use_extra_train_data == 1:  # use extra training data, predict the label of the extra training data first
                self.estimator.fit(train_x[features], train_y)
                extra_train_data_y = self.estimator.predict_proba(extra_train_x[features])[range(len(extra_train_y)), 1].round()           
                self.estimator.fit(pd.concat([train_x[features], extra_train_x[features]]), np.hstack((train_y, extra_train_y)))
            if use_extra_train_data == 2:  # use extra training data directly, no prediction
                self.estimator.fit(pd.concat([train_x[features], extra_train_x[features]]), np.hstack((train_y, extra_train_y)))
            if use_extra_train_data == 0:
                self.estimator.fit(train_x[features], train_y)   
            # record feature importance
            if self.feature_importance_name == 'coef_':
                feature_importance = np.squeeze(self.estimator.coef_)
            if self.feature_importance_name == 'feature_importances_' :  
                feature_importance = np.squeeze(self.estimator.feature_importances_)
            if self.feature_importance_name == 'booster' :
                if len(features) == 1:
                    feature_importance = np.zeros(len(features))
                else:
                    feature_importance = np.array(pd.DataFrame.from_dict(self.estimator.booster().get_fscore(), 'index').loc[features, 0])
            if self.feature_importance_name == 'none' :
                feature_importance = np.zeros(len(features))
 
            if feature_importance.ndim == 0:
                feature_importance = pd.DataFrame([feature_importance], index=features).transpose()
            else:
                if feature_importance.shape[0] != len(features):
                    feature_importance = pd.DataFrame(np.transpose(feature_importance), index=features).transpose()
                else:
                    feature_importance = pd.DataFrame(feature_importance, index=features).transpose()      
        else:
            feature_importance = pd.DataFrame(np.zeros(len(features)), index=features).transpose()   
                               
        # get score      
        if ml_type == "regression":   
        # make predictions
            null_rows_idx = test_x[features].isnull().any(axis=1)
            if (null_rows_idx.sum() == 0):
                if self.estimator == None:
                    test_y_predicted = np.squeeze(test_x[features])
                else:
                    test_y_predicted = self.estimator.predict(test_x[features])
            else:
                test_y_predicted = np.full(test_x.shape[0], np.nan)
                test_x_notnull = test_x[features].loc[~null_rows_idx, :]
                 
                if self.estimator == None:
                    test_y_notnull_predicted = test_x_notnull[features]
                else:
                    test_y_notnull_predicted = self.estimator.predict(test_x_notnull)
                 
                test_y_predicted[~null_rows_idx] = test_y_notnull_predicted  
             
            if self.estimator == None:
                train_y_predicted = np.squeeze(train_x[features]) 
            else:
                train_y_predicted = self.estimator.predict(train_x[features]) 
         
            # test score 
            test_score_df = pd.DataFrame(np.zeros(len(compare_features) * 2 + 2), index=[s + '_pcc' for s in compare_features] + [s + '_rmse' for s in compare_features] + ['pcc', 'rmse']).transpose()               
            for s_feature in compare_features:
                # calculate the auprc and auroc                
                pcc = alm_fun.pcc_cal(test_y, test_x[s_feature])
                test_score_df[s_feature + '_pcc'] = pcc
                 
                rmse = alm_fun.rmse_cal(test_y, test_x[s_feature])
                test_score_df[s_feature + '_rmse'] = rmse
                
            rmse = alm_fun.rmse_cal(test_y, test_y_predicted)  
            pcc = alm_fun.pcc_cal(test_y, test_y_predicted)         
            test_score_df['rmse'] = rmse
            test_score_df['pcc'] = pcc
             
            # train score 
            train_score_df = pd.DataFrame(np.zeros(len(compare_features) * 2 + 2), index=[s + '_pcc' for s in compare_features] + [s + '_rmse' for s in compare_features] + ['pcc', 'rmse']).transpose()               
            for s_feature in compare_features:
                # calculate the auprc and auroc                
                pcc = alm_fun.pcc_cal(train_y, train_x[s_feature])
                train_score_df[s_feature + '_pcc'] = pcc
                 
                rmse = alm_fun.rmse_cal(train_y, train_x[s_feature])
                train_score_df[s_feature + '_rmse'] = rmse
                
            rmse = alm_fun.rmse_cal(train_y, train_y_predicted)  
            pcc = alm_fun.pcc_cal(train_y, train_y_predicted)         
            train_score_df['rmse'] = rmse
            train_score_df['pcc'] = pcc
                
        if ml_type == "classification_binary":            
            # validation or test set
            null_rows_idx = test_x[features].isnull().any(axis=1)
            if (null_rows_idx.sum() == 0):
                if self.estimator == None:
                    test_y_predicted = np.squeeze(test_x[features])
                else:
                    try:
                        test_y_predicted = self.estimator.predict_proba(test_x[features])[:, 1]
                    # in case use regression result as classification    
                    except:
                        test_y_predicted = self.estimator.predict(test_x[features])
            else:
                test_y_predicted = np.full(test_x.shape[0], np.nan)
                test_x_notnull = test_x[features].loc[~null_rows_idx, :]
                 
                if self.estimator == None:
                    test_y_notnull_predicted = test_x_notnull[features]
                else:
                    try:
                        test_y_notnull_predicted = self.estimator.predict_proba(test_x_notnull)[:, 1]
                    except:
                        test_y_notnull_predicted = self.estimator.predict(test_x_notnull)
                test_y_predicted[~null_rows_idx] = np.squeeze(test_y_notnull_predicted)
                
            # make predictions on validation set               
            test_score_df = pd.DataFrame(np.zeros(6), index=['size', 'prior', 'auroc', 'auprc', 'pfr', 'rfp']).transpose()      
            [best_y_predicted, metric, multiclass_metrics] = alm_fun.classification_metrics(test_y, test_y_predicted)
            test_score_df['size'] = len(test_y)
            test_score_df['auroc'] = metric['auroc']
            test_score_df['auprc'] = metric['auprc']
            test_score_df['prior'] = metric['prior']
            test_score_df['pfr'] = metric['precision_fixed_recall']
            test_score_df['rfp'] = metric['recall_fixed_precision']
  
            # training set 
            if self.estimator == None:
                train_y_predicted = np.squeeze(train_x[features]) 
            else:
                try:
                    train_y_predicted = self.estimator.predict_proba(train_x[features])[:, 1]
                except:
                    train_y_predicted = self.estimator.predict(train_x[features]) 
            # in case train_y is not categorical 
            new_train_y = train_y.copy()
            new_train_y[new_train_y <= 0.5] = 0
            new_train_y[new_train_y > 0.5] = 1
             
            # make predictions on training set              
            train_score_df = pd.DataFrame(np.zeros(6), index=['size', 'prior', 'auroc', 'auprc', 'pfr', 'rfp']).transpose()                                   
            [best_y_predicted, metric, multiclass_metrics] = alm_fun.classification_metrics(new_train_y, train_y_predicted)
            train_score_df['size'] = len(new_train_y)
            train_score_df['auroc'] = metric['auroc']
            train_score_df['auprc'] = metric['auprc']
            train_score_df['prior'] = metric['prior']
            train_score_df['pfr'] = metric['precision_fixed_recall']
            train_score_df['rfp'] = metric['recall_fixed_precision']

        if ml_type == "classification_multiclass":
            # make predictions
            test_y_predicted_probs = self.estimator.predict_proba(test_x[features])  
            test_y_predicted = self.estimator.predict(test_x[features])
            train_y_predicted_probs = self.estimator.predict_proba(train_x[features])  
            train_y_predicted = self.estimator.predict(train_x[features])
                         
            train_score_df = pd.DataFrame(np.zeros(1), index=['neg_log_loss']).transpose()                             
            train_score_df['neg_log_loss'] = alm_fun.get_classification_metrics('neg_log_loss', 4, train_y, train_y_predicted_probs)
               
            test_score_df = pd.DataFrame(np.zeros(1), index=['neg_log_loss']).transpose()                             
            test_score_df['neg_log_loss'] = alm_fun.get_classification_metrics('neg_log_loss', 4, test_y, test_y_predicted_probs) 
            
        train_score_df = round(train_score_df, self.round_digits)
        test_score_df = round(test_score_df, self.round_digits)
                     
#             #export decision trees 
#             if (self.name in ['xgb_r','xgb_c','rf_r','rf_c','gbt_r','gbt_c']):
#                 for i in range(self.estimator.n_estimators):
#                     tree_dot = tree.export_graphviz(self.estimator.estimators_[i][0], out_file=None, 
#                                              feature_names=features,  
#                                              class_names= train_y.unique(), 
#                                              filled=True, rounded=True,  
#                                              special_characters=True)  
#                     graph = pydotplus.graph_from_dot_data(tree_dot) 
#                     graph.write_pdf('tree' + str(i) +'.pdf') 
#                     #Image(graph.create_png())
        return [train_y_predicted, train_score_df, test_y_predicted, test_score_df, feature_importance]

    def run(self, features, dependent_variable, ml_type, train, test, extra_train=None, use_extra_train_data=0):
        if self.if_feature_engineer:
            [train,test] = self.feature_engineer(train,test)
            
        train_y = train[dependent_variable]   
        train_x = train.drop(dependent_variable, axis=1)  
        test_y = test[dependent_variable]   
        test_x = test.drop(dependent_variable, axis=1) 
 
        if any(isinstance(i, list) for i in features):  # if features are nested list
            features = list(itertools.chain(*features))
#         features = list(set(features))       
                
        if (self.estimator == None) | ((self.single_feature_as_prediction == 1) & (len(features) == 1)): # if estimator is None, there is no need to train the model 
            feature_importance = pd.DataFrame(np.zeros(len(features)), index=features).transpose() 
        else:
            # fit the estimator        
            if use_extra_train_data == 1:  # use extra training data, predict the label of the extra training data first
                self.estimator.fit(train_x[features], train_y)
                extra_train_data_y = self.estimator.predict_proba(extra_train_x[features])[range(len(extra_train_y)), 1].round()           
                self.estimator.fit(pd.concat([train_x[features], extra_train_x[features]]), np.hstack((train_y, extra_train_y)))
            if use_extra_train_data == 2:  # use extra training data directly, no prediction
                self.estimator.fit(pd.concat([train_x[features], extra_train_x[features]]), np.hstack((train_y, extra_train_y)))
            if use_extra_train_data == 0:
                self.estimator.fit(train_x[features], train_y)   
            # record feature importance
            if self.feature_importance_name == 'coef_':
                feature_importance = np.squeeze(self.estimator.coef_)
            if self.feature_importance_name == 'feature_importances_' :  
                feature_importance = np.squeeze(self.estimator.feature_importances_)
            if self.feature_importance_name == 'booster' :
                if len(features) == 1:
                    feature_importance = np.zeros(len(features))
                else:
                    feature_importance_df = pd.DataFrame.from_dict(self.estimator.get_booster().get_score(importance_type='gain'), 'index')
                    if len(feature_importance_df.columns) == 1:
                        feature_importance = np.array(feature_importance_df.loc[features, 0])
                    else:
                        feature_importance = np.zeros(len(features))
            if self.feature_importance_name == 'none' :
                feature_importance = np.zeros(len(features))
 
            if feature_importance.ndim == 0:
                feature_importance = pd.DataFrame([feature_importance], index=features).transpose()
            else:
                if feature_importance.shape[0] != len(features):
                    feature_importance = pd.DataFrame(np.transpose(feature_importance), index=features).transpose()
                else:
                    feature_importance = pd.DataFrame(feature_importance, index=features).transpose()      
  
                                  
        # get score      
        if ml_type == "regression":   
#         #make predictions
#             null_rows_idx = test_x[features].isnull().any(axis=1)
#             if (null_rows_idx.sum() == 0):
#                 if (self.estimator == None) & ((self.single_feature_as_prediction == 1) & (len(features) == 1)):
#                     test_y_predicted = np.squeeze(test_x[features])
#                 else:
#                     test_y_predicted = self.estimator.predict(test_x[features])
#             else:
#                 test_y_predicted = np.full(test_x.shape[0],np.nan)
#                 test_x_notnull = test_x[features].loc[~null_rows_idx,:]
#                  
#                 if (self.estimator == None) & ((self.single_feature_as_prediction == 1) & (len(features) == 1)):
#                     test_y_notnull_predicted = test_x_notnull[features]
#                 else:
#                     test_y_notnull_predicted = self.estimator.predict(test_x_notnull)
#                  
#                 test_y_predicted[~null_rows_idx] = test_y_notnull_predicted 

#             test_y_predicted[~null_rows_idx] = test_y_notnull_predicted 
#             test_score_df = pd.DataFrame(np.zeros(2),index =  ['pcc','rmse']).transpose()                               
#             rmse = alm_fun.rmse_cal(test_y,test_y_predicted)  
#             pcc = alm_fun.pcc_cal(test_y,test_y_predicted)         
#             test_score_df['rmse'] = rmse
#             test_score_df['pcc'] = pcc
                
            if (self.estimator == None) | ((self.single_feature_as_prediction == 1) & (len(features) == 1)):
                test_y_predicted = test_x[features]
            else:
                test_y_predicted = self.estimator.predict(test_x[features])
            test_score_df = pd.DataFrame(np.zeros(2), index=['pcc', 'rmse']).transpose()                               
            rmse = alm_fun.rmse_cal(test_y, test_y_predicted)  
            pcc = alm_fun.pcc_cal(test_y, test_y_predicted)         
            test_score_df['rmse'] = rmse
            test_score_df['pcc'] = pcc
             
            # train score 
            if (self.estimator == None) & ((self.single_feature_as_prediction == 1) & (len(features) == 1)):
#                 train_y_predicted = np.squeeze(train_x[features]) 
                train_y_predicted = [1]*train_x.shape[0]
            else:
                train_y_predicted = self.estimator.predict(train_x[features]) 
            
            train_score_df = pd.DataFrame(np.zeros(2), index=['pcc', 'rmse']).transpose()                               
            rmse = alm_fun.rmse_cal(train_y, train_y_predicted)  
            pcc = alm_fun.pcc_cal(train_y, train_y_predicted)         
            train_score_df['rmse'] = rmse
            train_score_df['pcc'] = pcc
                
        if ml_type == "classification_binary":            
#             #validation or test set
#             null_rows_idx = test_x[features].isnull().any(axis=1)
#             if (null_rows_idx.sum() == 0):
#                 if (self.estimator == None) | ((self.single_feature_as_prediction == 1) & (len(features) == 1)):
#                     test_y_predicted = np.squeeze(test_x[features])
#                 else:
#                     try:
#                         test_y_predicted = self.estimator.predict_proba(test_x[features])[:,1]
#                     #in case use regression result as classification    
#                     except:
#                         test_y_predicted = self.estimator.predict(test_x[features])
#             else:
#                 test_y_predicted = np.full(test_x.shape[0],np.nan)
#                 test_x_notnull = test_x[features].loc[~null_rows_idx,:]
#                  
#                 if (self.estimator == None) | ((self.single_feature_as_prediction == 1) & (len(features) == 1)):
#                     test_y_notnull_predicted = test_x_notnull[features]
#                 else:
#                     try:
#                         test_y_notnull_predicted = self.estimator.predict_proba(test_x_notnull)[:,1]
#                     except:
#                         test_y_notnull_predicted = self.estimator.predict(test_x_notnull)
#                 test_y_predicted[~null_rows_idx] = np.squeeze(test_y_notnull_predicted)
             
            if (self.estimator == None) | ((self.single_feature_as_prediction == 1) & (len(features) == 1)):
                test_y_predicted = np.array(list(np.squeeze(test_x[features])))
            else:
                try:
                    test_y_predicted = self.estimator.predict_proba(test_x[features])[:, 1]
                # in case use regression result as classification    
                except:
                    test_y_predicted = self.estimator.predict(test_x[features])
#                 
            # make predictions on validation set               
            test_score_df = pd.DataFrame(np.zeros(6), index=['size', 'prior', 'auroc', 'auprc', 'pfr', 'rfp']).transpose()      
            [best_y_predicted, metric, multiclass_metrics] = alm_fun.classification_metrics(test_y, test_y_predicted)
            test_score_df['size'] = len(test_y)
            test_score_df['auroc'] = metric['auroc']
            test_score_df['auprc'] = metric['auprc']
            test_score_df['prior'] = metric['prior']
            test_score_df['pfr'] = metric['precision_fixed_recall']
            test_score_df['rfp'] = metric['recall_fixed_precision']
  
            # training set 
            if (self.estimator == None) | ((self.single_feature_as_prediction == 1) & (len(features) == 1)):
#                 train_y_predicted = np.array(list(np.squeeze(train_x[features])))
                train_y_predicted = [1]*train_x.shape[0]
            else:
                try:
                    train_y_predicted = self.estimator.predict_proba(train_x[features])[:, 1]
                except:
                    train_y_predicted = self.estimator.predict(train_x[features]) 
            # in case train_y is not categorical 
            new_train_y = train_y.copy()
            new_train_y[new_train_y <= 0.5] = 0
            new_train_y[new_train_y > 0.5] = 1
             
            # make predictions on training set              
            train_score_df = pd.DataFrame(np.zeros(6), index=['size', 'prior', 'auroc', 'auprc', 'pfr', 'rfp']).transpose()                                   
            [best_y_predicted, metric, multiclass_metrics] = alm_fun.classification_metrics(new_train_y, train_y_predicted)
            train_score_df['size'] = len(new_train_y)
            train_score_df['auroc'] = metric['auroc']
            train_score_df['auprc'] = metric['auprc']
            train_score_df['prior'] = metric['prior']
            train_score_df['pfr'] = metric['precision_fixed_recall']
            train_score_df['rfp'] = metric['recall_fixed_precision']

        if ml_type == "classification_multiclass":
            # make predictions
            test_y_predicted_probs = self.estimator.predict_proba(test_x[features])  
            test_y_predicted = self.estimator.predict(test_x[features])
            train_y_predicted_probs = self.estimator.predict_proba(train_x[features])  
            train_y_predicted = self.estimator.predict(train_x[features])
                         
            train_score_df = pd.DataFrame(np.zeros(1), index=['neg_log_loss']).transpose()                             
            train_score_df['neg_log_loss'] = alm_fun.get_classification_metrics('neg_log_loss', 4, train_y, train_y_predicted_probs)
               
            test_score_df = pd.DataFrame(np.zeros(1), index=['neg_log_loss']).transpose()                             
            test_score_df['neg_log_loss'] = alm_fun.get_classification_metrics('neg_log_loss', 4, test_y, test_y_predicted_probs) 
            
        train_score_df = round(train_score_df, self.round_digits)
        test_score_df = round(test_score_df, self.round_digits)
                     
#             #export decision trees 
#             if (self.name in ['xgb_r','xgb_c','rf_r','rf_c','gbt_r','gbt_c']):
#                 for i in range(self.estimator.n_estimators):
#                     tree_dot = tree.export_graphviz(self.estimator.estimators_[i][0], out_file=None, 
#                                              feature_names=features,  
#                                              class_names= train_y.unique(), 
#                                              filled=True, rounded=True,  
#                                              special_characters=True)  
#                     graph = pydotplus.graph_from_dot_data(tree_dot) 
#                     graph.write_pdf('tree' + str(i) +'.pdf') 
#                     #Image(graph.create_png())

        test_y_predicted = pd.Series(test_y_predicted,index = test_x.index)
        
        predicted_test = test.copy()
        
        predicted_test[dependent_variable] = test_y_predicted
        
        return_dict = {}
        return_dict ['train_y_predicted'] = train_y_predicted
        return_dict ['train_score_df'] = train_score_df
        return_dict ['test_y_predicted'] = test_y_predicted
        return_dict ['test_score_df'] = test_score_df
        return_dict ['feature_importance'] = feature_importance
        return_dict ['train_y_predicted'] = train_y_predicted
        return_dict ['predicted_df'] = predicted_test

        return (return_dict)

