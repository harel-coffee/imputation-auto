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
    
class alm_data:

    def __init__(self, data_init_params):
        
        for key in data_init_params:
            setattr(self, key, data_init_params[key])
        alm_fun.show_msg (self.log,self.verbose,'Class: [alm_data] [__init__] ' + self.name + ' ......done @' + str(datetime.now()))
 
    def refresh_data(self): 

#         self.verbose = verbose
        
        if self.load_from_disk == 0 :                    
            # load data (set initial features, handel onehot features,remove samples without valid dependent variable)
            self.load_data()
            msg = "[load_data]\n" + self.data_msg()
            alm_fun.show_msg(self.log, self.verbose, msg)                           
            # slice data
            self.slice_data()            
            msg = "[slice_data]\n" + self.data_msg()
            alm_fun.show_msg(self.log, self.verbose, msg)  
            # filter data
            self.filter_data()
            msg = "[filter_data]\n" + self.data_msg()
            alm_fun.show_msg(self.log, self.verbose, msg)  
            
            #split data
            self.split_data()
            msg = "[split_data]\n" + self.data_msg()
            alm_fun.show_msg(self.log, self.verbose, msg)           
 
            # gradient reshape
            if self.if_gradient == 1:   
                self.gradient_data()
                msg = "[gradient_data]\n" + self.data_msg()
                alm_fun.show_msg(self.log, self.verbose, msg)  
                        
            # engineer data
            if self.if_engineer == 1:
                self.engineer_data()
                msg = "[egineer_data]\n" + self.data_msg()
                alm_fun.show_msg(self.log, self.verbose, msg)    
            
            if self.save_to_disk == 1:
                    
                self.dict_savedata = {}
                self.dict_savedata['extra_train_data_df'] = self.extra_train_data_df
                self.dict_savedata['train_data_df'] = self.train_data_df
                self.dict_savedata['test_data_df'] = self.test_data_df
                self.dict_savedata['target_data_df'] = self.target_data_df
                
                self.dict_savedata['train_data_index_df'] = self.train_data_index_df
                self.dict_savedata['validation_data_index_df'] = self.validation_data_index_df                
                self.dict_savedata['test_data_index_df'] = self.test_data_index_df
                self.dict_savedata['target_data_index_df'] = self.target_data_index_df

                self.dict_savedata['train_data_for_target_df'] = self.train_data_for_target_df
                self.dict_savedata['target_data_for_target_df'] = self.target_data_for_target_df
                self.dict_savedata['validation_data_for_target_df'] = self.validation_data_for_target_df                
                
                self.dict_savedata['train_splits_df'] = self.train_splits_df
                self.dict_savedata['test_splits_df'] = self.test_splits_df
                
                self.dict_savedata['train_cv_splits_df'] = self.train_cv_splits_df
                self.dict_savedata['validation_cv_splits_df'] = self.validation_cv_splits_df
                
                if self.if_gradient:
                    self.dict_savedata['gradients'] = self.gradients
                    
                pickle_out = open(self.path + 'output/' + self.name + '_savedata.npy', 'wb')
                pickle.dump(self.dict_savedata, pickle_out) 
                pickle_out.close()        
                    
                
                if self.if_engineer:
                    self.dict_savedata_engineered = {}
                    self.dict_savedata_engineered['train_data_for_target_engineered_df'] = self.train_data_for_target_engineered_df
                    self.dict_savedata_engineered['target_data_for_target_engineered_df'] = self.target_data_for_target_engineered_df
                    self.dict_savedata_engineered['validation_data_for_target_engineered_df'] = self.validation_data_for_target_engineered_df    
                    
                    self.dict_savedata_engineered['train_splits_engineered_df'] = self.train_splits_engineered_df
                    self.dict_savedata_engineered['test_splits_engineered_df'] = self.test_splits_engineered_df
                    
                    self.dict_savedata_engineered['train_cv_splits_engineered_df'] = self.train_cv_splits_engineered_df
                    self.dict_savedata_engineered['validation_cv_splits_engineered_df'] = self.validation_cv_splits_engineered_df

                    if self.if_gradient:
                        self.dict_savedata_engineered['gradients'] = self.gradients                

                    pickle_out = open(self.path + 'output/' + self.name + '_savedata_engineered.npy','wb')
                    pickle.dump(self.dict_savedata_engineered, pickle_out) 
                    pickle_out.close()        
                     
        else:

            self.dict_savedata = np.load(self.path + 'output/' + self.name + '_savedata.npy')

            self.extra_train_data_df = self.dict_savedata['extra_train_data_df'] 
            self.train_data_df = self.dict_savedata['train_data_df'] 
            self.test_data_df = self.dict_savedata['test_data_df'] 
            self.target_data_df = self.dict_savedata['target_data_df'] 
            
            self.train_data_index_df = self.dict_savedata['train_data_index_df'] 
            self.validation_data_index_df = self.dict_savedata['validation_data_index_df']
            self.test_data_index_df = self.dict_savedata['test_data_index_df'] 
            self.target_data_index_df = self.dict_savedata['target_data_index_df'] 
            
            self.train_data_for_target_df = self.dict_savedata['train_data_for_target_df'] 
            self.target_data_for_target_df = self.dict_savedata['target_data_for_target_df'] 
            self.validation_data_for_target_df = self.dict_savedata['validation_data_for_target_df']                 
            
            self.train_splits_df = self.dict_savedata['train_splits_df'] 
            self.test_splits_df = self.dict_savedata['test_splits_df'] 
            
            self.train_cv_splits_df = self.dict_savedata['train_cv_splits_df'] 
            self.validation_cv_splits_df = self.dict_savedata['validation_cv_splits_df']  
            
            if self.if_gradient:
                self.gradients = self.dict_savedata['gradients']
            
            if self.if_engineer == 1:
                self.dict_savedata_engineered = np.load(self.path + 'output/' + self.name + '_savedata_engineered.npy')
                self.train_data_for_target_engineered_df = self.dict_savedata_engineered['train_data_for_target_engineered_df'] 
                self.target_data_for_target_engineered_df = self.dict_savedata_engineered['target_data_for_target_engineered_df'] 
                self.validation_data_for_target_engineered_df  = self.dict_savedata_engineered['validation_data_for_target_engineered_df']   
                
                self.train_splits_engineered_df = self.dict_savedata_engineered['train_splits_engineered_df'] 
                self.test_splits_engineered_df = self.dict_savedata_engineered['test_splits_engineered_df'] 
                
                self.train_cv_splits_engineered_df = self.dict_savedata_engineered['train_cv_splits_engineered_df'] 
                self.validation_cv_splits_engineered_df = self.dict_savedata_engineered['validation_cv_splits_engineered_df'] 
                
            msg = "[refresh_data] -- load from disk --" + self.data_msg()
            alm_fun.show_msg(self.log, self.verbose, msg)  

    def reload_data(self, ctuoff=np.nan):
        # Read training (extra training) and test data from files
        self.read_data()        
        # refresh data
        self.refresh_data()
         
    def read_data(self): 
        self.train_data_original_df = pd.read_csv(self.path + self.train_file)
        self.test_data_original_df = pd.read_csv(self.path + self.test_file) 
        self.target_data_original_df = pd.read_csv(self.path + self.target_file)
        if self.use_extra_train_data != 0:
            self.extra_train_data_original_df = pd.read_csv(self.path + self.extra_train_file) 
        else:
            self.extra_train_data_original_df = None
                             
    def load_data(self):                  
        # loading original training data , add random feature , remove the ones without label   
        self.train_data_working_df = self.train_data_original_df.copy()
        self.train_data_working_df['random_feature'] = np.random.uniform(0, 1, self.train_data_working_df.shape[0])
        self.train_data_working_df = self.train_data_working_df.loc[self.train_data_working_df[self.dependent_variable].notnull(), :]
        
        # loading original test data, if there is no dependent variable than add one, otherwise remove the records without label
        self.test_data_working_df = self.test_data_original_df.copy()
        if self.test_data_original_df.shape[0] != 0 :
            self.test_data_working_df['random_feature'] = np.random.uniform(0, 1, self.test_data_working_df.shape[0])
            self.test_data_working_df = self.test_data_working_df.loc[self.test_data_working_df[self.dependent_variable].notnull(), :]

        # loading original target data
        self.target_data_working_df = self.target_data_original_df.copy()
        if self.target_data_original_df.shape[0] != 0 :
            self.target_data_working_df['random_feature'] = np.random.uniform(0, 1, self.target_data_working_df.shape[0])
#             self.target_data_working_df = self.target_data_working_df.loc[self.target_data_working_df[self.dependent_variable].notnull(), :]
        
        self.extra_train_data_working_df = self.extra_train_data_original_df.copy()
        if self.extra_train_data_original_df.shape[0] != 0:
            self.extra_train_data_working_df['random_feature'] = np.random.uniform(0, 1, self.extra_train_data_working_df.shape[0])
            self.extra_train_data_working_df = self.extra_train_data_working_df.loc[self.extra_train_data_working_df[self.dependent_variable].notnull(), :]

         
        #*****************************************************************************************
#       # Take care onehot_features
        #*****************************************************************************************
#         if len(self.onehot_features) != 0:
#             # need to concatenate training, test, extra_training dataset to handle onehot features
#             self.train_data_working_df['temp_dtype'] = 'TR'
#             self.test_data_working_df['temp_dtype'] = 'TE'
#             if self.use_extra_train_data != 0:
#                 self.extra_train_data_working_df['temp_dtype'] = 'ETR'
#              
#             if self.use_extra_train_data != 0:
#                 self.data_working_df = pd.concat([self.train_data_working_df, self.test_data_working_df, self.extra_train_data_working_df], axis=0)
#             else:
#                 self.data_working_df = pd.concat([self.train_data_working_df, self.test_data_working_df], axis=0)
#                  
#             self.data_working_df[self.onehot_features] = self.data_working_df[self.onehot_features].astype(str) 
#             self.data_working_df = pd.get_dummies(self.data_working_df, columns=self.onehot_features, prefix_sep='~',)   
#              
#             self.train_data_working_df = self.data_working_df.loc[self.data_working_df['temp_dtype'] == 'TR']
#             self.test_data_working_df = self.data_working_df.loc[self.data_working_df['temp_dtype'] == 'TE']
#             if self.use_extra_train_data != 0:
#                 self.extra_train_data_working_df = self.data_working_df.loc[self.data_working_df['temp_dtype'] == 'ETR']             
#      
#             # use train original data set fill the onehot feature dictionary
#             for onehot_feature in self.onehot_features:
#                 r = re.compile(onehot_feature + '~')
#                 self.dict_onehot_features[onehot_feature] = list(filter(r.match, self.train_data_working_df.columns))
#              
#             # reconstruct the initial_features, train_features,compare_fetures,interaction features
#             for onehot_feature in self.onehot_features:
#                 if onehot_feature in self.initial_features:
#                     self.initial_features.remove(onehot_feature)
#                     self.initial_features += self.dict_onehot_features[onehot_feature]
#                 if onehot_feature in self.train_features: 
#                     self.train_features.remove(onehot_feature)
#                     self.train_features += self.dict_onehot_features[onehot_feature]
#                 if onehot_feature in self.compare_features:    
#                     self.compare_features.remove(onehot_feature)
#                     self.compare_features += self.dict_onehot_features[onehot_feature]
#                 if onehot_feature in self.interaction_features:
#                     self.interaction_features.remove(onehot_feature)
#                     self.interaction_features += self.dict_onehot_features[onehot_feature]
                   
#         if len(self.initial_features) != 0:
#             self.train_data_df = self.train_data_working_df[self.initial_features]
#             self.test_data_df = self.test_data_working_df[self.initial_features] 
#             self.target_data_df = self.target_data_working_df[self.initial_features]
#         else:

        self.train_data_df = self.train_data_working_df.copy()
        self.test_data_df = self.test_data_working_df.copy()  
        self.target_data_df = self.target_data_working_df.copy()
        self.extra_train_data_df = self.extra_train_data_working_df
                       
        self.n_features = self.train_data_df.shape[1] - 1
        self.feature_names = self.train_data_df.columns.get_values()
        
        # it is possible that self.test_data_df , target_data_df has less columns than self.train_data_df because of the onehot encoding 
#         self.test_feature_diff = list(set(self.feature_names) - set(self.test_data_df.columns.get_values()))
#         for i in range(len(self.test_feature_diff)):
#             self.test_data_df[self.test_feature_diff[i]] = 0
#         self.target_feature_diff = list(set(self.feature_names) - set(self.target_data_df.columns.get_values()))
#         for i in range(len(self.target_feature_diff)):
#             self.target_data_df[self.target_feature_diff[i]] = 0    
                     
        self.train_counts = self.train_data_df.shape[0]
        self.test_counts = self.test_data_df.shape[0]
        self.target_counts = self.target_data_df.shape[0]
         
        if self.use_extra_train_data != 0:
            self.extra_train_data_df = self.extra_train_data_working_df.copy()
            self.extra_train_counts = self.extra_train_data_df.shape[0]
#             if len(self.initial_features) != 0:
#                 self.extra_train_data_df = self.extra_train_data_working_df[self.initial_features]
#             else:
#                 self.extra_train_data_df = self.extra_train_data_working_df.copy()                
#             if len(self.onehot_features) != 0:
#                 self.extra_train_data_df[self.onehot_features] = self.extra_train_data_df[self.onehot_features].astype(basestring)   
#                 self.extra_train_data_df = pd.get_dummies(self.extra_train_data_df)     
        
        self.train_data_for_target_df = None
        self.train_cv_splits_df = None
        self.validation_cv_splits_df = None
          
    def slice_data(self):
        [self.target_data_df, self.train_data_df, self.test_data_df, self.extra_train_data_df] = self.data_slice(self.name, self.target_data_df, self.train_data_df, self.test_data_df, self.extra_train_data_df)        
           

    def engineer_data(self):              

        self.train_data_for_target_engineered_df = copy.deepcopy(self.train_data_for_target_df)
        self.target_data_for_target_engineered_df = copy.deepcopy(self.target_data_for_target_df)
        if (self.independent_testset == 1) & (self.validation_from_testset == 1):
            self.validation_data_for_target_engineered_df = copy.deepcopy(self.validation_data_for_target_df)
        else:
            self.validation_data_for_target_engineered_df = None
        
        self.train_splits_engineered_df = copy.deepcopy(self.train_splits_df)
        self.test_splits_engineered_df = copy.deepcopy(self.test_splits_df)
        
        self.train_cv_splits_engineered_df = copy.deepcopy(self.train_cv_splits_df)
        self.validation_cv_splits_engineered_df = copy.deepcopy(self.validation_cv_splits_df)
                   
        # take care of the train_data_for_target_df, target_data_for_target_df, validation_data_for_target_df
        for key in self.train_data_for_target_df.keys():
            [self.train_data_for_target_engineered_df[key],self.target_data_for_target_engineered_df[key]] = \
            self.feature_engineer(self.train_data_index_df.loc[self.train_data_for_target_df[key],:],self.target_data_index_df.loc[self.target_data_for_target_df[key],:])
        
        if (self.independent_testset == 1) & (self.validation_from_testset == 1):
            for key in self.train_data_for_target_df.keys():
                [self.train_data_for_target_engineered_df[key],self.validation_data_for_target_engineered_df[key]] = \
                self.feature_engineer(self.train_data_index_df.loc[self.train_data_for_target_df[key],:],self.validation_data_index_df.loc[self.validation_data_for_target_df[key],:])
            
        # take care of train_splits_df and test_splits_df
        for i in range(self.test_split_folds):
            for key in self.train_splits_df[i].keys():                    
                print('i:' + str(i) + '-' + key)
                [self.train_splits_engineered_df[i][key],self.test_splits_engineered_df[i][key]] = \
                self.feature_engineer(self.train_data_index_df.loc[self.train_splits_df[i][key],:],self.test_data_index_df.loc[self.test_splits_df[i][key],:])
                
        #take care of train_cv_splits_df and validation_cv_splites_df
        for i in range(self.test_split_folds):
            for j in range(self.cv_split_folds):
                for key in self.train_cv_splits_engineered_df[i][j].keys():
                    [self.train_cv_splits_engineered_df[i][j][key],self.validation_cv_splits_engineered_df[i][j][key]] = \
                    self.feature_engineer(self.train_data_index_df.loc[self.train_cv_splits_df[i][j][key],:],self.validation_data_index_df.loc[self.validation_cv_splits_df[i][j][key],:])

    def filter_data(self):
        if self.filter_train == 1:   
            train_null_idx = self.train_data_df[self.train_features].isnull().any(axis=1)
            train_null_count = train_null_idx[train_null_idx].shape[0]
            self.train_data_df = self.train_data_df[-train_null_idx]
            
#             if self.train_data_for_target_df is not None:
#                 train_null_idx = self.train_data_for_target_df.isnull().any(axis=1)
#                 train_null_count = train_null_idx[train_null_idx].shape[0]
#                 self.train_data_for_target_df = self.train_data_for_target_df[-train_null_idx]
#             
            if self.use_extra_train_data != 0:
                extra_train_null_idx = self.extra_train_data_df[self.train_features].isnull().any(axis=1)
                extra_train_null_count = extra_train_null_idx[extra_train_null_idx].shape[0]
                self.extra_train_data_df = self.extra_train_data_df[-extra_train_null_idx]
                  
        if self.filter_test == 1:     
            test_null_idx = self.test_data_df[self.train_features].isnull().any(axis=1)
            test_null_count = test_null_idx[test_null_idx].shape[0]
            self.test_data_df = self.test_data_df[-test_null_idx]    
            
        if self.filter_target == 1:     
            target_null_idx = self.target_data_df[self.train_features].isnull().any(axis=1)
            target_null_count = target_null_idx[target_null_idx].shape[0]
            self.target_data_df = self.target_data_df[-target_null_idx]    
    
#         for fold_id in range(self.cv_split_folds):
#             if self.filter_train == 1:  
#                 train_null_idx = self.train_cv_splits_df[fold_id].isnull().any(axis=1)
#                 train_null_count = train_null_idx[train_null_idx].shape[0]
#                 self.train_cv_splits_df[fold_id] = self.train_cv_splits_df[fold_id][-train_null_idx]
#             if self.filter_validation == 1:
#                 validation_null_idx = self.validation_cv_splits_df[fold_id].isnull().any(axis=1)
#                 validation_null_count = validation_null_idx[validation_null_idx].shape[0]
#                 self.validation_cv_splits_df[fold_id] = self.validation_cv_splits_df[fold_id][-validation_null_idx]

    def gradient_data(self):             
        #first to get the graidents 
        self.gradients = self.setup_gradients(self.train_data_df)
                   
        #take care of the train_data_for_target_df, target_data_for_target_df, validation_data_for_target_df
        train_dict = self.gradient_reshape(self.train_data_index_df.loc[self.train_data_for_target_df['no_gradient'],:],self.gradients)
        self.train_data_for_target_df.update(train_dict)
        target_dict = {x:self.target_data_for_target_df['no_gradient'] for x in self.gradients}
        self.target_data_for_target_df.update(target_dict)
         
        if (self.independent_testset == 1) & (self.validation_from_testset == 1):
            validation_dict = {x:self.validation_data_for_target_df['no_gradient'] for x in self.gradients}
            self.validation_data_for_target_df.update(validation_dict)
        
        #take care of train_splits_df and test_splits_df
        for i in range(self.test_split_folds):
            train_splits_dict = self.gradient_reshape(self.train_data_index_df.loc[self.train_splits_df[i]['no_gradient'],:],self.gradients)
            self.train_splits_df[i].update(train_splits_dict)
            test_dict = {x:self.test_splits_df[i]['no_gradient'] for x in self.gradients}
            self.test_splits_df[i].update(test_dict)
            
        #take care of train_cv_splits_df and validation_cv_splites_df
        for i in range(self.test_split_folds):
            for j in range(self.cv_split_folds):
                train_cv_splits_dict = self.gradient_reshape(self.train_data_index_df.loc[self.train_cv_splits_df[i][j]['no_gradient'],:],self.gradients)
                self.train_cv_splits_df[i][j].update(train_cv_splits_dict)
                validation_dict = {x:self.validation_cv_splits_df[i][j]['no_gradient'] for x in self.gradients}
                self.validation_cv_splits_df[i][j].update(validation_dict)
                                
    def split_data(self):
        self.train_data_for_target_df = {}        
        self.train_data_for_target_df['no_gradient'] = self.train_data_df.index        
        self.target_data_for_target_df = {}        
        self.target_data_for_target_df['no_gradient'] = self.target_data_df.index
        
        self.target_data_index_df = self.target_data_df
        self.train_data_index_df = self.train_data_df
        
        if self.independent_testset == 1:
            self.test_data_index_df = self.test_data_df
        else:
            self.test_data_index_df = self.train_data_df
            
        if self.validation_from_testset == 1:
            self.validation_data_index_df = self.test_data_df
        else:
            self.validation_data_index_df = self.train_data_df                
        
        if (self.independent_testset == 1) & (self.validation_from_testset == 1):            
            self.validation_data_for_target_df = {}        
            self.validation_data_for_target_df['no_gradient'] = self.test_data_df.index
        else:
            self.validation_data_for_target_df = None
            
        #split test, validation, training dataset
        #case1: all test, validation, training from one original training set 
        if self.independent_testset == 0:
            #Split the original training set into test_split_folds folds.  (training - test)
            #output two list train_splits_df and test_splits_df
            self.train_splits_df = [{} for i in range(self.test_split_folds)]      
            self.test_splits_df = [{} for i in range(self.test_split_folds)]
            self.train_cv_splits_df = [[{} for i in range(self.cv_split_folds)] for j in range(self.test_split_folds)]
            self.validation_cv_splits_df = [[{} for i in range(self.cv_split_folds)] for j in range(self.test_split_folds)]
             
            if self.test_split_method == 0 :   
                if self.test_split_folds == 1:
                    kf_list = []
                    if self.test_split_ratio == 0: 
                        kf_list.append((range(len(self.train_data_df.index)),None))
                    else:
                        kf_folds = int(1/self.test_split_ratio)
                        kf = ms.KFold(n_splits=kf_folds, shuffle=True) 
                        kf_list.append(list(kf.split(self.train_data_df))[0])
                else:
                    kf = ms.KFold(n_splits=self.test_split_folds, shuffle=True) 
                    kf_list = list(kf.split(self.train_data_df))  
            # stratified split (keep prior)
            if self.test_split_method == 1 :   
                if self.test_split_folds == 1:
                    kf_list = []                    
                    if self.test_split_ratio == 0: 
                        kf_list.append((self.train_data_df.index,None))
                    else:
                        kf_folds = int(1/self.test_split_ratio)
                        kf = ms.StratifiedKFold(n_splits=kf_folds, shuffle=True) 
                        kf_list.append(list(kf.split(self.train_data_df))[0])
                else:                
                    kf = ms.StratifiedKFold(n_splits=self.test_split_folds, shuffle=True)   
                    kf_list = list(kf.split(self.train_data_df,self.train_data_df[self.dependent_variable]))
            # customized split   
            if self.test_split_method == 2 :
                kf_list = self.test_split(self.train_data_df) 
                 
            test_split_fold_id = 0 
            for train_index_split, test_index_split in kf_list:
                train_index = self.train_data_df.index[train_index_split]
                self.train_splits_df[test_split_fold_id]['no_gradient'] = train_index
                if test_index_split is None:
                    test_index = None
                else:
                    test_index = self.train_data_df.index[test_index_split] 
                self.test_splits_df[test_split_fold_id]['no_gradient'] = test_index                                    
                test_split_fold_id += 1 
                
            #Split each training set into cv_split_folds folds  (training - validation)
            for i in range(self.test_split_folds):
                cur_train_data_df = self.train_data_df.loc[self.train_splits_df[i]['no_gradient'],:]
                if self.cv_split_method == 0 : 
                    if self.cv_split_folds == 1:
                        kf_folds = int(1/self.cv_split_ratio)
                        kf = ms.KFold(n_splits=kf_folds, shuffle=True) 
                        kf_list = []
                        kf_list.append(list(kf.split(cur_train_data_df))[0])
                    else:  
                        kf = ms.KFold(n_splits=self.cv_split_folds, shuffle=True) 
                        kf_list = list(kf.split(cur_train_data_df))
                # stratified split (keep prior)
                if self.cv_split_method == 1 : 
                    if self.cv_split_folds == 1:
                        kf_folds = int(1/self.cv_split_ratio)
                        kf = ms.StratifiedKFold(n_splits=kf_folds, shuffle=True) 
                        kf_list = []
                        kf_list.append(list(kf.split(cur_train_data_df,cur_train_data_df[self.dependent_variable]))[0])
                    else:    
                        kf = ms.StratifiedKFold(n_splits=self.cv_split_folds, shuffle=True)   
                        kf_list = list(kf.split(cur_train_data_df,cur_train_data_df[self.dependent_variable]))
                # customized split   
                if self.cv_split_method == 2 :
                    kf_list = self.cv_split(cur_train_data_df)  
                    
                cv_split_fold_id = 0 
                for train_index_split, validation_index_split in kf_list:
                    train_index = cur_train_data_df.index[train_index_split]
                    validation_index = cur_train_data_df.index[validation_index_split]
                    self.train_cv_splits_df[i][cv_split_fold_id]['no_gradient'] = train_index
                    self.validation_cv_splits_df[i][cv_split_fold_id]['no_gradient'] = validation_index                   
#                     self.train_cv_splits_df[i][cv_split_fold_id]['no_gradient'] = self.train_splits_df[i].loc[train_index, :]
#                     self.validation_cv_splits_df[i][cv_split_fold_id]['no_gradient'] = self.train_splits_df[i].loc[validation_index, :] 
                    cv_split_fold_id += 1 
                    
        
        #case2: training from one set, validation and test from another set (independent_testset and validation_from_testset parameters) 
        if self.independent_testset == 1:            
#             if self.validation_from_testset:
                #validation from testset will force the cv_split_folds = 1
                #self.cv_split_folds = 1
#             else:
                #validation not from testset will force the test_split_folds = 1
#                 self.test_split_folds  = 1
                           
            self.train_splits_df = [{} for i in range(self.test_split_folds)]      
            self.test_splits_df = [{} for i in range(self.test_split_folds)]
            self.train_cv_splits_df = [[{} for i in range(self.cv_split_folds)] for j in range(self.test_split_folds)]
            self.validation_cv_splits_df = [[{} for i in range(self.cv_split_folds)] for j in range(self.test_split_folds)]
            
            #split testset to test_split_folds folds (validation - test)
            if self.validation_from_testset:                 
                if self.validation_equal_testset:
                    self.test_split_folds = 1 #special case that validation set is the same as test set
                    kf_list = [(range(self.test_data_df.shape[0]),range(self.test_data_df.shape[0]))]
                else:                
                    if self.test_split_method == 0 :  
                        if self.test_split_folds == 1:
                            kf_folds = int(1/self.test_split_ratio)
                            kf = ms.KFold(n_splits=kf_folds, shuffle=True) 
                            kf_list = []
                            kf_list.append(list(kf.split(self.test_data_df))[0])
                        else:                                         
                            kf = ms.KFold(n_splits=self.test_split_folds, shuffle=True) 
                            kf_list = list(kf.split(self.test_data_df))  
                    # stratified split (keep prior)
                    if self.test_split_method == 1 : 
                        if self.test_split_folds == 1:
                            kf_folds = int(1/self.test_split_ratio)
                            kf = ms.StratifiedKFold(n_splits=kf_folds, shuffle=True) 
                            kf_list = []
                            kf_list.append(list(kf.split(self.test_data_df,self.test_data_df[self.dependent_variable]))[0])
                        else:   
                            kf = ms.StratifiedKFold(n_splits=self.test_split_folds, shuffle=True)   
                            kf_list = list(kf.split(self.test_data_df,self.test_data_df[self.dependent_variable]))
                    # customized split   
                    if self.test_split_method == 2 :
                        kf_list = self.test_split(self.test_data_df.copy())

                test_split_fold_id = 0 
                for validation_index_split , test_index_split in kf_list:
                    validation_index = self.test_data_df.index[validation_index_split]
                    test_index = self.test_data_df.index[test_index_split]  
                    self.train_splits_df[test_split_fold_id]['no_gradient'] = self.train_data_df.index
                    self.test_splits_df[test_split_fold_id]['no_gradient'] = test_index
                    
                    cv_validation_index = np.array_split(validation_index,self.cv_split_folds)
                    
                    for j in range(self.cv_split_folds):

                        self.train_cv_splits_df[test_split_fold_id][j]['no_gradient'] = self.train_data_df.index
                        self.validation_cv_splits_df[test_split_fold_id][j]['no_gradient'] = cv_validation_index[j]                                                          
    #                     self.train_splits_df[test_split_fold_id]['no_gradient'] = self.train_data_df
    #                     self.test_splits_df[test_split_fold_id]['no_gradient'] = self.test_data_df.loc[test_index, :]
    #                     self.train_cv_splits_df[test_split_fold_id][0]['no_gradient'] = self.train_data_df
    #                     self.validation_cv_splits_df[test_split_fold_id][0]['no_gradient'] = self.test_data_df.loc[validation_index, :] 
                    test_split_fold_id += 1 
                print('done')
            else:
                self.train_splits_df[0]['no_gradient'] = self.train_data_df.index
                self.test_splits_df[0]['no_gradient'] = self.test_data_df.index
                
                #Split each training set into cv_split_folds folds  (training - validation)
                for i in range(self.test_split_folds):
                    cur_train_data_df = self.train_data_df.loc[self.train_splits_df[i]['no_gradient'],:]
                    if self.cv_split_method == 0 :   
                        kf = ms.KFold(n_splits=self.cv_split_folds, shuffle=True) 
                        kf_list = list(kf.split(cur_train_data_df))  
                    # stratified split (keep prior)
                    if self.cv_split_method == 1 :   
                        kf = ms.StratifiedKFold(n_splits=self.cv_split_folds, shuffle=True)   
                        kf_list = list(kf.split(cur_train_data_df))
                    # customized split   
                    if self.cv_split_method == 2 :
                        kf_list = self.cv_split(self.name, self.cv_split_folds, cur_train_data_df)  
                        
                    cv_split_fold_id = 0 
                    for train_index_split, validation_index_split in kf_list:
                        train_index = cur_train_data_df.index[train_index_split]
                        validation_index = cur_train_data_df.index[validation_index_split]
                        self.train_cv_splits_df[i][cv_split_fold_id]['no_gradient'] = train_index
                        self.validation_cv_splits_df[i][cv_split_fold_id]['no_gradient'] = validation_index                     
#                         self.train_cv_splits_df[i][cv_split_fold_id]['no_gradient'] = self.train_splits_df[i].loc[train_index, :]
#                         self.validation_cv_splits_df[i][cv_split_fold_id]['no_gradient'] = self.train_splits_df[i].loc[validation_index, :] 
                        cv_split_fold_id += 1 
  
    def data_msg (self):
        msg = 'train_data' + ' [' + str(self.train_data_df.shape[0]) + ',' + str(self.train_data_df.shape[1] - 1) + ']\n' + \
              'test_data' + ' [' + str(self.test_data_df.shape[0]) + ',' + str(self.test_data_df.shape[1] - 1) + ']\n' + \
              'target_data' + ' [' + str(self.target_data_df.shape[0]) + ',' + str(self.target_data_df.shape[1] - 1) + ']\n' 
        if self.use_extra_train_data == 1:
            msg += 'extra_train_data' + ' [' + str(self.extra_train_data_df.shape[0]) + ',' + str(self.extra_train_data_df.shape[1] - 1) + ']\n'
        
#         if self.train_data_for_target_df is not None:
#             msg += 'train_data_for_target' + ' [' + str(self.train_data_for_target_df['no_gradient'].shape[0]) + ',' + str(self.train_data_index_df.shape[1]) + ']\n'
#         
#         if self.train_cv_splits_df is not None:
#             msg += 'cv_data\n'
#             for fold_id in range(self.cv_split_folds):
#                 msg += 'fold_id:' + str(fold_id) + ' [' + str(self.train_cv_splits_df[fold_id].shape[0]) + ',' + str(self.train_cv_splits_df[fold_id].shape[1] - 1) + '] ' + ' [' + str(self.validation_cv_splits_df[fold_id].shape[0]) + ',' + str(self.validation_cv_splits_df[fold_id].shape[1] - 1) + ']\n'
#                 
        return (msg)     

