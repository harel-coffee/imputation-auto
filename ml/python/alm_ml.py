#!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3
import numpy as np
import pandas as pd
import csv
import os
import re
import operator
import itertools
import time
import math
import random
import codecs
import copy
import pickle
from datetime import datetime

import alm_fun
               
class alm_ml:

    def __init__(self, ml_init_params):
        for key in ml_init_params:
            setattr(self, key, ml_init_params[key])
        
        msg = "Class: [alm_ml] [__init__]......done @" + str(datetime.now())
        alm_fun.show_msg(self.log, self.verbose, msg)    
         
    def grid_search(self, alm_estimator, alm_dataset):     
           
        if len(alm_estimator.gs_range.keys()) == 0: 
            return (['N/A', 'N/A', 'N/A'])
        else:
            # generate all params permutation for current estimator
            key_names = ['dummy']
            key_types = [type(0)]
            params = np.array(['0']).reshape([1, 1])
            for key in alm_estimator.gs_range:
                key_names.append(key)
                cur_params = params 
                first_param = 1               
                for param in alm_estimator.gs_range[key]:
                    new_params = np.vstack([cur_params, np.repeat(param, cur_params.shape[1])])
                    if first_param == 1:
                        key_types.append(type(param))
                        params = new_params                       
                        first_param = 0
                    else:
                        params = np.hstack([params, new_params])
                     
            # loop through all permutation to check cv performance 
            for j in range(params.shape[1]):
                cur_params_dict = {}
                cur_params_instance = params[:, j]
                # form a dictionary 
                for k in range(len(key_names)):
                    if k == 0: 
                        continue
                    else:                      
                        cur_params_dict[key_names[k]] = key_types[k](cur_params_instance[k])
                alm_estimator.estimator.set_params(**cur_params_dict)
 
                r = self.run_cv_prediction(alm_estimator, alm_dataset)
                validation_cv_score = r['validation_cv_score']
                validation_cv_result = r['validation_cv_result']
                train_cv_score = r['train_cv_score']
 
                if j == 0:
                    all_validation_cv_score = validation_cv_score
                    all_train_score = train_cv_score
                    gs_opt_cv_score = validation_cv_score
                    gs_opt_result = validation_cv_result
                    gs_opt_params = cur_params_dict 
                else:        
                    all_validation_cv_score = np.vstack([all_validation_cv_score, validation_cv_score])
                    all_train_score = np.vstack([all_train_score, train_cv_score])
                    if (alm_estimator.score_direction == 0 and float(validation_cv_score) < float(gs_opt_cv_score)) or (alm_estimator.score_direction == 1 and float(validation_cv_score) > float(gs_opt_cv_score)):
                        gs_opt_cv_score = validation_cv_score
                        gs_opt_cv_result = validation_cv_result
                        gs_opt_params = cur_params_dict                 
                msg = "[grid search] current parameters: " + str(cur_params_dict) + " validation_cv_score: " + str(validation_cv_score) + " training_score : " + str(train_cv_score)
                alm_fun.show_msg(self.log, self.verbose, msg) 
            alm_estimator.estimator.set_params(**gs_opt_params)   
            gs_results = pd.DataFrame(np.transpose(np.vstack([params, np.transpose(all_validation_cv_score), np.transpose(all_train_score)])), columns=key_names + ['validation_' + alm_estimator.score_name, 'train_' + alm_estimator.score_name ]).drop('dummy', axis=1)
            
            result_dict = {}
            result_dict['gs_results'] = gs_results
            result_dict['gs_opt_params'] = gs_opt_params
            result_dict['gs_opt_cv_result'] = gs_opt_cv_result
            result_dict['gs_opt_cv_score'] = gs_opt_cv_score
     
            return (result_dict)

    def run_test_prediction(self, alm_estimator,alm_dataset,features=None,nofit = 0):        
        if features == None:
            features = alm_dataset.train_features 
            
        if alm_dataset.if_engineer:
            cur_train_df = alm_dataset.train_splits_engineered_df[alm_dataset.cur_test_split_fold][alm_dataset.cur_gradient_key]
            cur_test_df = alm_dataset.test_splits_engineered_df[alm_dataset.cur_test_split_fold][alm_dataset.cur_gradient_key]
        else:
            cur_train_df = alm_dataset.train_data_index_df.loc[alm_dataset.train_splits_df[alm_dataset.cur_test_split_fold][alm_dataset.cur_gradient_key],:]
            cur_test_df = alm_dataset.test_data_index_df.loc[alm_dataset.test_splits_df[alm_dataset.cur_test_split_fold][alm_dataset.cur_gradient_key],:]
        
            
        if alm_dataset.prediction_bootstrapping == 1:       
            bs_n = alm_dataset.bootstrapping_num
            test_size = cur_test_df.shape[0]
            # bootstrapping test set   
            if alm_estimator.ml_type == 'classification_binary':
                positive_idx = cur_test_df.loc[cur_test_df[alm_dataset.dependent_variable] == 1, :].index
                test_positive_size = len(positive_idx)
                negative_idx = cur_test_df.loc[cur_test_df[alm_dataset.dependent_variable] == 0, :].index
                test_negative_size = len(negative_idx)            
                bs_positive_idx = np.random.choice(positive_idx, (bs_n, test_positive_size), replace=True)
                bs_negative_idx = np.random.choice(negative_idx, (bs_n, test_negative_size), replace=True)
                bs_idx = np.hstack([bs_positive_idx, bs_negative_idx])
            if alm_estimator.ml_type == 'regression':
                bs_idx = np.random.choice(cur_test_df.index, (bs_n, test_size), replace=True)
            for i in range(bs_n):
                cur_test_df_bs = cur_test_df.loc[bs_idx[i, :], :]       
                cur_test_df_bs.to_csv('/Users/joewu/test_bs.csv')
                 
                r = alm_estimator.run(features, alm_dataset.dependent_variable, alm_estimator.ml_type, cur_train_df, cur_test_df_bs, alm_dataset.extra_train_data_df,alm_dataset.use_extra_train_data,nofit)
        
                if 'test_bs_results' not in locals():
                    test_bs_results = r['test_score_df']   
                else:
                    test_bs_results = pd.concat([test_bs_results, r['test_score_df']])
     
            test_bs_results = test_bs_results.reset_index(drop=True)  
            test_bs_result_mean = round(test_bs_results.mean(axis=0), alm_estimator.round_digits)
            test_bs_result_ste = round(test_bs_results.std(axis=0) / np.sqrt(alm_dataset.cv_split_folds), alm_estimator.round_digits)
            test_bs_result_ste.index = [x + '_ste' for x in test_bs_result_ste.index]
            test_bs_result = pd.DataFrame(pd.concat([test_bs_result_mean, test_bs_result_ste], axis=0)).transpose()
     
            test_bs_score = test_bs_result[[alm_estimator.score_name, alm_estimator.score_name + '_ste']]
            test_bs_score.columns = ['mean', 'ste']
            r = alm_estimator.run(features, alm_dataset.dependent_variable, alm_estimator.ml_type, cur_train_df, cur_test_df, alm_dataset.extra_train_data_df)
            test_y_predicted = r['test_y_predicted']
            feature_importance = r['feature_importance']            
        else:             
            r = alm_estimator.run(features, alm_dataset.dependent_variable, alm_estimator.ml_type, cur_train_df, cur_test_df ,alm_dataset.extra_train_data_df, alm_dataset.use_extra_train_data,nofit)            
            test_y_predicted = r['test_y_predicted']
            feature_importance = r['feature_importance']   
            
            test_bs_results = r['test_score_df']
            test_bs_result_mean = round(test_bs_results.mean(axis=0), alm_estimator.round_digits)
            test_bs_result_ste = round(test_bs_results.std(axis=0) / np.sqrt(alm_dataset.cv_split_folds), alm_estimator.round_digits)
            test_bs_result_ste.index = [x + '_ste' for x in test_bs_result_ste.index]
            test_bs_result = pd.DataFrame(pd.concat([test_bs_result_mean, test_bs_result_ste], axis=0)).transpose()
    
            test_bs_score = test_bs_result[[alm_estimator.score_name, alm_estimator.score_name + '_ste']]
            test_bs_score.columns = ['mean', 'ste']

        result_dict = {}
        result_dict['test_y_predicted'] = test_y_predicted
        result_dict['feature_importance'] = feature_importance
        result_dict['test_bs_results'] = test_bs_results
        result_dict['test_bs_result'] = test_bs_result
        result_dict['test_bs_score'] = test_bs_score
        return (result_dict)
    
    def run_target_prediction(self, alm_estimator, alm_dataset, features=None,nofit = 0):        
        if features == None:
            features = alm_dataset.train_features 
            
        if alm_dataset.if_engineer:
            cur_train_df = alm_dataset.train_data_for_target_engineered_df[alm_dataset.cur_gradient_key]
            cur_target_df = alm_dataset.target_data_for_target_engineered_df[alm_dataset.cur_gradient_key]
        else:
            cur_train_df = alm_dataset.train_data_index_df.loc[alm_dataset.train_data_for_target_df[alm_dataset.cur_gradient_key],:]
            cur_target_df = alm_dataset.target_data_index_df.loc[alm_dataset.target_data_for_target_df[alm_dataset.cur_gradient_key],:]
             
        r = alm_estimator.run(features, alm_dataset.dependent_variable, alm_estimator.ml_type, cur_train_df, cur_target_df, alm_dataset.extra_train_data_df, alm_dataset.use_extra_train_data,nofit)  
        alm_dataset.predicted_target_df = r['predicted_df']    
        target_results = r['test_score_df']
        target_result_mean = round(target_results.mean(axis=0), alm_estimator.round_digits)
        target_result_ste = round(target_results.std(axis=0) / np.sqrt(alm_dataset.cv_split_folds), alm_estimator.round_digits)
        target_result_ste.index = [x + '_ste' for x in target_result_ste.index]
        target_result = pd.DataFrame(pd.concat([target_result_mean, target_result_ste], axis=0)).transpose()

        target_score = target_result[[alm_estimator.score_name, alm_estimator.score_name + '_ste']]
        target_score.columns = ['mean', 'ste']
        
        result_dict = {}
        result_dict['target_y_predicted'] = r['test_y_predicted']
        result_dict['feature_importance'] = r['feature_importance']
        result_dict['target_results'] = target_results
        result_dict['target_result'] = target_result
        result_dict['target_score'] = target_score

        return (result_dict)
    
    def run_cv_prediction(self,alm_estimator,alm_dataset,features=None,nofit = 0):
        if features == None:
            features = alm_dataset.train_features  
                            
        for i in range(alm_dataset.cv_split_folds):            
            if alm_dataset.if_engineer:
                cur_train_cv_df = alm_dataset.train_cv_splits_engineered_df[alm_dataset.cur_test_split_fold][i][alm_dataset.cur_gradient_key]
                cur_validation_cv_df = alm_dataset.validation_cv_splits_engineered_df[alm_dataset.cur_test_split_fold][i][alm_dataset.cur_gradient_key]               
            else:
                cur_train_cv_df = alm_dataset.train_data_index_df.loc[alm_dataset.train_cv_splits_df[alm_dataset.cur_test_split_fold][i][alm_dataset.cur_gradient_key],:]
                cur_validation_cv_df = alm_dataset.validation_data_index_df.loc[alm_dataset.validation_cv_splits_df[alm_dataset.cur_test_split_fold][i][alm_dataset.cur_gradient_key],:]
                        
            if ((alm_dataset.innerloop_cv_fit_once == 1) & (i != 0 ) | (nofit == 1)):
                r = alm_estimator.run(features, alm_dataset.dependent_variable , alm_estimator.ml_type, cur_train_cv_df, cur_validation_cv_df,alm_dataset.extra_train_data_df,alm_dataset.use_extra_train_data,nofit =1)            
            else:    
                r = alm_estimator.run(features, alm_dataset.dependent_variable , alm_estimator.ml_type, cur_train_cv_df, cur_validation_cv_df,alm_dataset.extra_train_data_df,alm_dataset.use_extra_train_data)

            train_y_splits_predicted = r['train_y_predicted']
            train_score_df = r['train_score_df']
            validation_y_splits_predicted = r['test_y_predicted']
            validation_score_df = r['test_score_df']
            feature_importance = r['feature_importance']
              
            if 'validation_y_predicted' not in locals():
                validation_y_predicted = validation_y_splits_predicted
            else:
                validation_y_predicted = np.hstack([validation_y_predicted, validation_y_splits_predicted])
             
            if 'validation_y_truth' not in locals():
                validation_y_truth = cur_validation_cv_df[alm_dataset.dependent_variable]
            else:
                validation_y_truth = np.hstack([validation_y_truth, cur_validation_cv_df[alm_dataset.dependent_variable]])
 
            if 'train_cv_results' not in locals():
                train_cv_results = train_score_df   
            else:
                train_cv_results = pd.concat([train_cv_results, train_score_df])
             
            if 'validation_cv_results' not in locals():
                validation_cv_results = validation_score_df   
            else:
                validation_cv_results = pd.concat([validation_cv_results, validation_score_df])
     
            if 'cv_feature_importances' not in locals():
                cv_feature_importances = feature_importance
            else:
                cv_feature_importances = pd.concat([cv_feature_importances, feature_importance])
 
        train_cv_results = train_cv_results.reset_index(drop=True)  
        train_cv_result_mean = round(train_cv_results.mean(axis=0), alm_estimator.round_digits)
        train_cv_result_ste = round(train_cv_results.std(axis=0) / np.sqrt(alm_dataset.cv_split_folds), alm_estimator.round_digits)
        train_cv_result_ste.index = [x + '_ste' for x in train_cv_result_ste.index]
        train_cv_result = pd.DataFrame(pd.concat([train_cv_result_mean, train_cv_result_ste], axis=0)).transpose()
        
        validation_cv_results = validation_cv_results.reset_index(drop=True)  
        validation_cv_result_mean = round(validation_cv_results.mean(axis=0), alm_estimator.round_digits)
        validation_cv_result_ste = round(validation_cv_results.std(axis=0) / np.sqrt(alm_dataset.cv_split_folds), alm_estimator.round_digits)
        validation_cv_result_ste.index = [x + '_ste' for x in validation_cv_result_ste.index]
        validation_cv_result = pd.DataFrame(pd.concat([validation_cv_result_mean, validation_cv_result_ste], axis=0)).transpose()

        cv_feature_importances = cv_feature_importances.reset_index(drop=True)
        cv_feature_importance = round(cv_feature_importances.mean(axis=0), alm_estimator.round_digits)
        cv_feature_importance = cv_feature_importance.sort_values(ascending=False)
         
#         if cv_feature_importance.shape[0] > 50:
#             cv_feature_importance = cv_feature_importance[0:50]         
        train_cv_score = train_cv_result[[alm_estimator.score_name, alm_estimator.score_name + '_ste']]
        train_cv_score.columns = ['mean', 'ste']
        validation_cv_score = validation_cv_result[[alm_estimator.score_name, alm_estimator.score_name + '_ste']]
        validation_cv_score.columns = ['mean', 'ste']
        
        return_dict = {}
        return_dict['validation_y_predicted'] = validation_y_predicted
        return_dict['validation_y_truth'] = validation_y_truth
        return_dict['cv_feature_importance'] = cv_feature_importance
        return_dict['train_cv_results'] = train_cv_results
        return_dict['validation_cv_results'] = validation_cv_results        
        return_dict['train_cv_result'] = train_cv_result
        return_dict['validation_cv_result'] = validation_cv_result
        return_dict['train_cv_score'] = train_cv_score['mean'][0]
        return_dict['validation_cv_score'] = validation_cv_score['mean'][0]
        return_dict['train_cv_score_ste'] = train_cv_score['ste'][0]
        return_dict['validation_cv_score_ste'] = validation_cv_score['ste'][0]
        
  
        return (return_dict)
    
    def feature_enrich_analysis(self, feature, type, p_value_cutoff, feature_cutoff=np.nan):
#         [best_y_predicted,best_mcc,best_cutoff,metrics_dict] = classification_metrics(self.train_data_y,self.train_data_x_df[feature],feature_cutoff)
         
        [best_y_predicted, metrics_dict] = classification_metrics(self.train_data_y, self.train_data_x_df[feature], feature_cutoff)        
        best_mcc = metrics_dict['best_mcc']
        best_cutoff = metrics_dict['best_cutoff']
         
        pred = self.train_data_x_df.copy()
        pred['label'] = self.train_data_y
        pred['predicted_label'] = best_y_predicted
        pred['predicted_label'] = pred['predicted_label'].astype(int)
         
        set_truth_p = pred.loc[(pred.label == 1), :]
        set_truth_p = set_truth_p.sort([feature])
        set_truth_n = pred.loc[(pred.label == 0), :]
        set_truth_n = set_truth_n.sort([feature], ascending=False)
            
        set_truth_p_wrong = set_truth_p.loc[(set_truth_p.predicted_label == 0), :]
        set_truth_n_wrong = set_truth_n.loc[(set_truth_n.predicted_label == 1), :]
        set_truth_wrong = pd.concat([set_truth_n_wrong, set_truth_p_wrong]) 
            
        set_truth_p_right = set_truth_p.loc[(set_truth_p.predicted_label == 1), :]
        set_truth_n_right = set_truth_n.loc[(set_truth_n.predicted_label == 0), :]
        set_truth_right = pd.concat([set_truth_n_right, set_truth_p_right]) 
                 
        confusion_matrix = [set_truth_p_right.shape[0], set_truth_p_wrong.shape[0], set_truth_n_right.shape[0], set_truth_n_wrong.shape[0]]        
 
        set_pred_p = pred.loc[(pred.predicted_label == 1), :]
        set_pred_p = set_pred_p.sort([feature])
        set_pred_n = pred.loc[(pred.predicted_label == 0), :]
        set_pred_n = set_pred_n.sort([feature], ascending=False)
            
        set_pred_p_wrong = set_pred_p.loc[(set_pred_p.label == 0), :]
        set_pred_n_wrong = set_pred_n.loc[(set_pred_n.label == 1), :]
        set_pred_wrong = pd.concat([set_pred_n_wrong, set_pred_p_wrong]) 
            
        set_pred_p_right = set_pred_p.loc[(set_pred_p.label == 1), :]
        set_pred_n_right = set_pred_n.loc[(set_pred_n.label == 0), :]
        set_pred_right = pd.concat([set_pred_n_right, set_pred_p_right]) 
         
        if type == 'precision':
            bk_pred = set_pred_p
            wrong_pred = set_pred_p_wrong    
             
        if type == 'npv':
            bk_pred = set_pred_n
            wrong_pred = set_pred_n_wrong
         
        if type == 'recall':
            bk_pred = set_truth_p
            wrong_pred = set_truth_p_wrong  
             
        if type == 'specificity':
            bk_pred = set_truth_n
            wrong_pred = set_truth_n_wrong                         
 
        if type == 'accuracy':
            bk_pred = pred
            wrong_pred = set_pred_wrong
 
        col_names = ['feature', 'feature_value', 'p_value', 'p_value_direction', 'effect_size', 'bk_total_counts', 'bk_value_counts', 'wrong_total_counts', 'wrong_value_counts', 'right_value_counts', 'wrong_no_value_count', 'right_no_value_counts', 'entropy', 'conditional_entropy', 'mutual information']
        hg_sig_features = pd.DataFrame(columns=col_names)
 
        hypothesis_count = 0 
        for col in wrong_pred.columns.get_values():
            print(col)
            bk_counts = bk_pred[col].value_counts()
            bk_total_count = bk_counts.sum()
            wrong_counts = wrong_pred[col].value_counts()
            wrong_total_count = wrong_counts.sum()
             
            if (wrong_counts.shape[0] == 2) | (col == 'aa_ref') | (col == 'aa_alt') | (col == 'aa_ref_alt') | (col == 'essential_ref_alt') :
                # correct for multiple testing
                # hypothesis_count  += wrong_counts.shape[0]    
                 
                for v in bk_counts.index:
                    v_count = bk_counts[v]
                     
                    if v not in wrong_counts.index:
                        wrong_v_count = 0
                    else:                       
                        wrong_v_count = wrong_counts[v] - 1
    
                    # adjust wrong_v_count
                     
                    if wrong_v_count == 0:
                        wrong_v_count = 1  # to remove "fake" depletion ,only select the depletion that won't affect by the adjust
                     
                    if wrong_v_count == v_count :
                        wrong_v_count = wrong_v_count - 1  # to remove "fake" enrichment ,only select the enrichment that won't affect by the adjust
 
                    # effect size
                    no_v_count = bk_total_count - v_count 
                    wrong_no_v_count = wrong_total_count - wrong_v_count
                    right_no_v_count = no_v_count - wrong_no_v_count                    
                    right_v_count = v_count - wrong_v_count
                     
                    hg_effectsize = (wrong_v_count * right_no_v_count) / (wrong_no_v_count * right_v_count)
                     
                    # hypergeometric test pvalue
                    hg_cdf = stats.hypergeom.cdf(wrong_v_count, bk_total_count, v_count, wrong_total_count)
                    hg_pvalue = -1
                    if hg_cdf > 0.5:
                        hg_pvalue = 1 - hg_cdf
                        hg_pvalue_direction = 'enrich'
                    else:
                        hg_pvalue = hg_cdf
                        hg_pvalue_direction = 'depletion'
                        hg_effectsize = (-1) / hg_effectsize
                         
                    hg_pvalue = hg_pvalue * bk_counts.shape[0]     
 
                    # calculate entropy and mutual information
                    p_w_no_v = wrong_no_v_count / no_v_count 
                    p_r_no_v = 1 - p_w_no_v
                    entropy_no_v = 0 - (p_w_no_v * np.log2(p_w_no_v) + p_r_no_v * np.log2(p_r_no_v))
                                 
                    p_w_v = wrong_v_count / v_count
                    p_r_v = 1 - p_w_v
                    entropy_v = 0 - (p_w_v * np.log2(p_w_v) + p_r_v * np.log2(p_r_v))
                     
                    p_w = wrong_total_count / bk_total_count 
                    p_r = 1 - p_w
                    entropy = 0 - (p_w * np.log2(p_w) + p_r * np.log2(p_r))
                     
                    v_fraction = v_count / bk_total_count
                    no_v_fraction = 1 - v_fraction
                    conditional_entropy = v_fraction * entropy_v + no_v_fraction * entropy_no_v
                     
                    mutual_information = entropy - conditional_entropy
                    cur_sig_feature = pd.DataFrame(data=[col, v, hg_pvalue, hg_pvalue_direction, hg_effectsize, bk_total_count, v_count, wrong_total_count, wrong_v_count, right_v_count, wrong_no_v_count, right_no_v_count, entropy, conditional_entropy, mutual_information]).transpose()
                    cur_sig_feature.columns = col_names
                    hg_sig_features = pd.concat([hg_sig_features, cur_sig_feature])
     
        # correct for multiple testing
        # hg_sig_features.p_value = hg_sig_features.p_value*hypothesis_count        
        # select the significant feature and feature value 
        if p_value_cutoff != -1:
            hg_sig_features = hg_sig_features.loc[(hg_sig_features.p_value < p_value_cutoff), :]
        hg_sig_features = hg_sig_features.sort(['p_value_direction', 'effect_size'])
        return [hg_sig_features, pred, confusion_matrix]

    def feature_evaluation(self, feature_plot=0, extra_name=''):        
        self.feature_names = self.train_data_x_df.columns.get_values()        
        if self.ml_type == 'regression':
            self.mutual_info = fs.mutual_info_regression(self.train_data_x_df, self.train_data_y, discrete_features='auto', n_neighbors=3, copy=True, random_state=None)
            self.F_corr = fs.f_regression(np.array(self.train_data_x_df), self.train_data_y, center=True) 
            self.feature_eval = pd.DataFrame(np.transpose(np.vstack((self.feature_names, self.mutual_info, self.F_corr[0], self.F_corr[1], self.train_data_x_df.mean(axis=0), self.train_data_x_df.std(axis=0)))), columns=['feature', 'mi', 'F_score', 'F_pvalue', 'mean', 'std'])
        if self.ml_type == 'classification':
            self.mutual_info = np.round(fs.mutual_info_classif(self.train_data_x_df, self.train_data_y, discrete_features='auto', n_neighbors=3, copy=True, random_state=None), 4)
            # [self.auprc_single_feature,self.auroc_single_feature,self.best_cutoff_single_feature] = self.train_data_x_df.apply(lambda x:[classification_metrics(self.train_data_y,x)[3]['mean'][x1] for x1 in ['auprc','auroc','best_cutoff']],axis = 0)
            self.eval_metrics = ['size', 'prior', 'precision', 'recall', 'recall_fixed_precision', 'precision_fixed_recall', 'auprc', 'auroc', 'best_cutoff', 'reverse']
            self.single_feature_metrics = self.train_data_x_df.apply(lambda x:[classification_metrics(self.train_data_y, x)[1][x1] for x1 in self.eval_metrics], axis=0)
            self.single_feature_metrics = pd.DataFrame.from_records(list(self.single_feature_metrics)).transpose()
            # self.auroc_single_feature = self.train_data_x_df.apply(lambda x: classification_metrics(self.train_data_y,x)['roc'],axis = 0)
            self.feature_eval = pd.DataFrame(np.transpose(np.vstack((self.feature_names, self.train_data_x_df.mean(axis=0), self.train_data_x_df.std(axis=0), self.mutual_info, self.single_feature_metrics))), columns=['feature', 'mean', 'std', 'mi'] + self.eval_metrics)
            self.feature_eval = self.feature_eval.sort(columns='auroc')
        if self.ml_type == 'classification_multiclass':
            self.mutual_info = np.round(fs.mutual_info_classif(self.train_data_x_df, self.train_data_y, discrete_features='auto', n_neighbors=3, copy=True, random_state=None), 4)
            self.feature_eval = pd.DataFrame(np.transpose(np.vstack((self.feature_names, self.train_data_x_df.mean(axis=0), self.train_data_x_df.std(axis=0), self.mutual_info))), columns=['feature', 'mean', 'std', 'mi'])
            self.feature_eval = self.feature_eval.sort(columns='mi')    
             
        self.feature_eval.to_csv(self.path + self.name + extra_name + "_feature_evaluation.csv")  
        msg = "[feature_evaluation] Saving mutual information and linear relationship between each feature and test to " + self.path + self.name + "_feature_evaluation.csv"
        alm_fun.show_msg(self.log, self.verbose, msg)
        if feature_plot == 1:
            # plot data distibution for each feature
            fig = plt.figure(figsize=(14, 10))
            for feature in self.feature_names:
                plt.clf()
                # plt.suptitle('Feature: [' + feature +'] evaluation' ,size = 20 )
                feature_reverse = 0
                 
                # feature distribution for each class
                ax = plt.subplot(2, 3, (4, 6))
                red_patch = patches.Patch(color='red', label='Positive')
                green_patch = patches.Patch(color='green', label='Negative')
                plt.legend(handles=[red_patch, green_patch])
                n_idx = (self.train_data_y == 0)
                p_idx = (self.train_data_y == 1)
                n_plot = sns.distplot(self.train_data_x_df.loc[n_idx, feature], color='green')
                p_plot = sns.distplot(self.train_data_x_df.loc[p_idx, feature], color='red')
                # ax.set_xlim (-2500,10000)
                ax.set_title('Distribution for each class' + ' [' + 'Mutual information:' + str(np.array(self.feature_eval.loc[self.feature_eval.feature == feature, 'mi'])[0]) + ']', size=15)
     
                # Dataset information
                ax = plt.subplot(2, 3, 1)
                ax.pie([self.prior, 1 - self.prior], autopct='%1.1f%%', labels=['positive', 'negative'], colors=['red', 'green'])
                ax.set_title('Dataset:' + '[' + self.name + ']' + ' size: [' + str(self.n_train) + ']' , size=15)
                 
                # AUROC plot
                metric = classification_metrics(self.train_data_y, self.train_data_x_df[feature])
     
                ax = plt.subplot(2, 3, 2)            
                ax.plot(metric['fprs'], metric['tprs'])      
                if metric['reverse'] == 1:      
                    ax.set_title(' AUROC: ' + str(metric['roc']) + ' [reverse]', size=15)
                else:
                    ax.set_title(' AUROC: ' + str(metric['roc']), size=15)
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                 
                # AUROC plot
                ax = plt.subplot(2, 3, 3)            
                ax.plot(metric['recalls'], metric['precisions'])      
                if metric['reverse'] == 1:      
                    ax.set_title(' AUPRC: ' + str(metric['prc']) + ' [reverse]', size=15)
                else:
                    ax.set_title(' AUPRC: ' + str(metric['prc']), size=15)
                ax.set_xlabel('Recall')
                ax.set_ylabel('Precision') 
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                fig.tight_layout()
                plt.savefig(self.path + self.name + '/' + 'feature_eval_' + self.name + extra_name + '_[' + feature + ']')
 
        # create a feature correclation matrix
        feature_names = list(self.feature_eval.feature)
        self.feature_cor_matrix = np.repeat(np.nan, np.power(len(feature_names), 2)).reshape(len(feature_names), len(feature_names))
        self.feature_cor_matrix = pd.DataFrame(self.feature_cor_matrix, columns=feature_names)
        self.feature_cor_matrix.index = feature_names
        for feature1 in feature_names:
            for feature2 in feature_names:
                self.feature_cor_matrix.loc[feature1, feature2] = pcc_cal(self.train_data_x_df[feature1], self.train_data_x_df[feature2])
        return [self.feature_eval, self.feature_cor_matrix]
         
    def feature_interaction(self, alm_estimator, alm_dataset, interaction_features, interaction_features_name):                        
        # get all single and double features 
        single_features_idx = range(len(interaction_features_name))
        single_features = interaction_features
        double_features_idx = itertools.combinations(single_features_idx, 2)
        n_features = len(single_features_idx)
         
        # get the score using all features
        wt_score_backward = self.run_cv_prediction(alm_estimator, alm_dataset, 'feature_iteraction', single_features)[3]['mean']
 
        # Create interaction score matrices
        score_interaction_forward = pd.DataFrame(np.zeros((n_features, n_features)), columns=single_features_idx, index=single_features_idx)
        score_interaction_backward = score_interaction_forward.copy()
        epsilon_interaction_forward = score_interaction_forward.copy()
        epsilon_interaction_backward = score_interaction_forward.copy()        
 
        # Get all the scores 
      
        for i in single_features_idx:
            cur_feature_backward = single_features.copy() 
            cur_feature_forward = []              
            feature = single_features[i]
            cur_feature_forward.append(feature)
            cur_feature_backward.remove(feature)
            cur_score_forward = self.run_cv_prediction(alm_estimator, alm_dataset, 'feature_iteraction', cur_feature_forward)[3]['mean']
            cur_score_backward = self.run_cv_prediction(alm_estimator, alm_dataset, 'feature_iteraction', cur_feature_backward)[3]['mean']    
            score_interaction_forward.loc[i, i] = cur_score_forward
            score_interaction_backward.loc[i, i] = cur_score_backward
            msg = "[feature interaction] Single feature: (" + interaction_features_name[i] + ")," + " forward score: " + str(cur_score_forward) + " backward score: " + str(cur_score_backward)
            alm_fun.show_msg(self.log, self.verbose, msg)      
          
        for i in double_features_idx: 
            i_list = list(i)   
            cur_feature_backward = single_features.copy() 
            cur_feature_forward = []          
            for f in i_list:
                cur_feature_forward.append(single_features[f])
                # print (f)
                cur_feature_backward.remove(single_features[f])  
                                           
            cur_score_forward = self.run_cv_prediction(alm_estimator, alm_dataset, 'feature_iteraction', cur_feature_forward)[3]['mean']
            score_multiplicative_forward = score_interaction_forward.loc[i_list[0], i_list[0]] * score_interaction_forward.loc[i_list[1], i_list[1]] / (wt_score_backward * wt_score_backward)
 
            cur_score_backward = self.run_cv_prediction(alm_estimator, alm_dataset, 'feature_iteraction', cur_feature_backward)[3]['mean']
            score_multiplicative_backward = score_interaction_backward.loc[i_list[0], i_list[0]] * score_interaction_backward.loc[i_list[1], i_list[1]] / (wt_score_backward * wt_score_backward)
             
            score_interaction_forward.loc[i] = cur_score_forward
            score_interaction_backward.loc[i] = cur_score_backward
            epsilon_interaction_forward.loc[i] = cur_score_forward / wt_score_backward - score_multiplicative_forward
            epsilon_interaction_backward.loc[i] = cur_score_backward / wt_score_backward - score_multiplicative_backward
             
            score_interaction_forward.loc[i[::-1]] = score_interaction_forward.loc[i]
            score_interaction_backward.loc[i[::-1]] = score_interaction_backward.loc[i]
            epsilon_interaction_forward.loc[i[::-1]] = epsilon_interaction_forward.loc[i]
            epsilon_interaction_backward.loc[i[::-1]] = epsilon_interaction_backward.loc[i]
            msg = "[feature interaction] Double features: (" + str(interaction_features_name[i_list[0]]) + " - " + str(interaction_features_name[i_list[1]]) + ")," + " forward score: " + str(cur_score_forward) + " backward score: " + str(cur_score_backward)
            alm_fun.show_msg(self.log, self.verbose, msg)                    
         
        score_interaction_forward.columns = interaction_features_name
        score_interaction_forward.index = interaction_features_name
        epsilon_interaction_forward.columns = interaction_features_name
        epsilon_interaction_forward.index = interaction_features_name
        score_interaction_backward.columns = interaction_features_name
        score_interaction_backward.index = interaction_features_name
        epsilon_interaction_backward.columns = interaction_features_name
        epsilon_interaction_backward.index = interaction_features_name
                 
        score_interaction_forward.to_csv(alm_estimator.name + '_score_interaction_forward.csv')
        epsilon_interaction_forward.to_csv(alm_estimator.name + '_epsilon_interaction_forward.csv')
        score_interaction_backward.to_csv(alm_estimator.name + '_score_interaction_backward.csv')
        epsilon_interaction_backward.to_csv(alm_estimator.name + '_epsilon_interaction_backward.csv')
                        
        return [score_interaction_forward, epsilon_interaction_forward, score_interaction_backward, epsilon_interaction_backward]
 
    def feature_selection(self, alm_estimator, alm_dataset, type, start_features=[], args=0):
#         results = None
        if self.fs_type == 'forward':
            all_features = alm_dataset.train_features
            left_features = list(set(all_features) - set(start_features)) 
            start_score = inf
            fs_df = pd.DataFrame(columns=['feature', 'sign', 'score', 'accept'])
            results = self.fs(alm_estimator, alm_dataset, start_features, left_features, start_score, fs_df)
        if self.fs_type == 'backward':
            start_features = ['dummy'] + alm_dataset.train_features
            start_score = inf
            bs_df = pd.DataFrame(columns=['feature', 'sign', 'score', 'accept'])
            results = self.bs(alm_estimator, alm_dataset, start_features, start_score, bs_df)
        if self.fs_type == 'local search':
            start_features = args['start_features']
            all_features = alm_dataset.train_features
            left_features = list(set(all_features) - set(start_features)) 
            start_score = inf
            all_feature_states = [1] * len(start_features) + [0] * len(left_features)  # track the state of each feature  
            all_features = start_features + left_features
            self.fs_results = pd.DataFrame(columns=['T', 'k', 'sign', 'feature', 'features', 'score', 'delta_score','validation_score','prob', 'accept', 'successive_rejection','successive_noincrease'])
            T = args['T']
            alpha = args['alpha']
            K = args['K']
            epsilon = args['epsilon']
            k = K
            self.ls(alm_estimator, alm_dataset, start_features, start_score, all_features, all_feature_states, T, alpha, K, k, epsilon, 0,0)            
        return (self.fs_results)
     
    def bs(self, alm_estimator, alm_dataset, start_features, start_score, bs_df):     
        prev_score = start_score        
        start_features_cp = list(start_features)
        for feature in start_features_cp:
            start_features.remove(feature)
            cur_score = self.run_cv_prediction(alm_estimator, alm_dataset, 'backward_selection', start_features, plot=False)[-1]
            if (alm_estimator.score_direction == 0 and cur_score < start_score) or (alm_estimator.score_direction == 1 and cur_score > start_score) or feature == 'dummy':
                best_feature = feature
                start_score = cur_score   
                bs_df.loc[bs_df.shape[0]] = [feature, "+", cur_score, "?"]
            else:
                bs_df.loc[bs_df.shape[0]] = [feature, "+", cur_score, ""]      
                start_features.append(feature)    
            msg = "[feature selection] Backward selection: Check feature: (" + feature + ")," + " score: " + str(cur_score)
            alm_fun.show_msg(self.log, self.verbose, msg)                             
        if prev_score == start_score:
            return bs_df
        msg = "[feature selection] Backward selection: Remove feature: " + best_feature + ")," + " score: " + str(start_score)
        alm_fun.show_msg(self.log, self.verbose, msg)  
        bs_df.loc[bs_df.shape[0]] = [best_feature, "+", start_score, ""]        
        start_features.remove(best_feature)
        self.bs(alm_estimator, alm_dataset, start_features, start_score, bs_df)
     
    # forward feature selection     
     
    def fs(self, alm_estimator, alm_dataset, start_features, left_features, start_score, fs_df):        
        prev_score = start_score
        for feature in left_features:
            start_features.append(feature)
            cur_score = self.run_cv_prediction(alm_estimator, alm_dataset, 'forward_selection', start_features, plot=False)[-1]
            if (alm_estimator.score_direction == 0 and cur_score < start_score) or (alm_estimator.score_direction == 1 and cur_score > start_score):
                best_feature = feature
                start_score = cur_score   
                fs_df.loc[fs_df.shape[0]] = [feature, "+", cur_score, "?"]
            else:
                fs_df.loc[fs_df.shape[0]] = [feature, "+", cur_score, ""]
             
            start_features.remove(feature)
            # msg =  "[feature selection] Forward selection: Check feature: (" + feature + ")," + " score: " + str(cur_score)
            # alm_fun.show_msg(self.log,self.verbose,msg)                         
        if prev_score == start_score:
            return fs_df
        msg = "[feature selection] Forward selection: Add feature: (" + best_feature + ")," + " score: " + str(start_score)
        alm_fun.show_msg(self.log, self.verbose, msg)  
        fs_df.loc[fs_df.shape[0]] = [best_feature, "+", start_score, ""]        
        start_features.append(best_feature)
        left_features.remove(best_feature)  
        self.fs(alm_estimator, alm_dataset, start_features, left_features, start_score, fs_df)
         
    # local search feature selection    
     
    def ls(self, alm_estimator, alm_dataset, start_features, start_score, all_features, all_feature_states, T, alpha, K, k, epsilon, successive_rejection_count,successive_noincrease_count):        
        k -= 1                      
        if start_score == inf:  # this is first time run 
            if len(start_features) != 0:
                alm_dataset.train_features = start_features
                cv_results = self.run_cv_prediction(alm_estimator, alm_dataset)
                cur_score = cv_results['train_cv_score']  
                msg = "[feature selection] Local search: start from feature (" + str(start_features) + "), score: " + str(cur_score)
                start_score = cur_score 
                alm_fun.show_msg(self.log, self.verbose, msg)
            else:
                if (alm_estimator.score_direction == 0):            
                    start_score = inf
                if (alm_estimator.score_direction == 1):            
                    start_score = -inf
         
        # random pick one feature from all features 
        selected_idx = random.randint(0, len(all_features) - 1)
        selected_feature = all_features[selected_idx]
        if all_feature_states[selected_idx] == 0:
            sign = "add"
            start_features.append(selected_feature)
        else:
            sign = "remove"
            start_features.remove(selected_feature)
        if len(start_features) != 0:
            alm_dataset.train_features = start_features
            cv_results = self.run_cv_prediction(alm_estimator, alm_dataset)
            cur_score = cv_results['train_cv_score']    
            cur_validation_score = cv_results['validation_cv_score']  
            if (alm_estimator.score_direction == 0):            
                delta_score = start_score - cur_score
            if (alm_estimator.score_direction == 1):            
                delta_score = cur_score - start_score 
            
            if delta_score > 0 :   
                successive_noincrease_count = 0
            else:
                successive_noincrease_count += 1
                
            prob = np.exp(delta_score / T)
            if prob > 1: 
                prob = 1
            if random.uniform(0, 1) <= prob:
                # accept the selected feature
                successive_rejection_count = 0        
                all_feature_states[selected_idx] = 1 - all_feature_states[selected_idx]
                self.fs_results.loc[self.fs_results.shape[0]] = [T, k, sign, selected_feature, start_features.copy(), cur_score, delta_score, cur_validation_score,prob, "Yes", successive_rejection_count,successive_noincrease_count]
#                 self.fs_results = self.fs_results.append(pd.Series([T,k,sign,selected_feature,start_features,cur_score,delta_score,prob,"Yes",successive_rejection_count],index = self.fs_results.columns),ignore_index = True)
                msg = "[feature selection] Local search: Accept " + sign + " feature (" + selected_feature + ") with probability " + str(prob) + ", score: " + str(cur_score) + ", delta score: " + str(delta_score) + ", validation_score: " + str(cur_validation_score) + ", T: " + str(T) + ", k: " + str(k)
                alm_fun.show_msg(self.log, self.verbose, msg)       
                start_score = cur_score        
            else:
                successive_rejection_count += 1 
                self.fs_results.loc[self.fs_results.shape[0]] = [T, k, sign, selected_feature, start_features.copy(), cur_score, delta_score, cur_validation_score, prob, "No", successive_rejection_count,successive_noincrease_count]   
#                 self.fs_results = self.fs_results.append(pd.Series([T,k,sign,selected_feature,start_features,cur_score,delta_score,prob,"No",successive_rejection_count],index = self.fs_results.columns),ignore_index = True)
#                 msg = "[feature selection] Local search: Reject " + sign + " feature (" + selected_feature + ") with probability " + str(prob) + ", score: " + str(cur_score) + ", delta score: " + str(delta_score) + ", validation_score: " + str(cur_validation_score) + ", T: " + str(T) + ", k: " + str(k)
#                 alm_fun.show_msg(self.log, self.verbose, msg)  
                if all_feature_states[selected_idx] == 0:
                   start_features.remove(selected_feature)
                else:
                    start_features.append(selected_feature) 
  
        if (T < epsilon) | (successive_noincrease_count > K):
            return 0
        if k == 0:
            k = K
            T = alpha * T
        self.ls(alm_estimator, alm_dataset, start_features, start_score, all_features, all_feature_states, T, alpha, K, k, epsilon, successive_rejection_count,successive_noincrease_count)

