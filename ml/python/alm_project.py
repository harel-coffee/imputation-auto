import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import time
import itertools
import pickle
import copy
from sklearn import neighbors as knn
from sklearn import linear_model as lm
from sklearn import feature_selection as fs
from sklearn import model_selection as ms
from sklearn.decomposition import nmf
from sklearn import svm
from sklearn import ensemble
from sklearn import tree
from sklearn import pipeline
from sklearn import preprocessing
from sklearn import metrics
import alm_data
import alm_ml
import alm_es
import alm_fun
from datetime import datetime
from numpy import gradient


class alm_project:

    def __init__(self, project_init_params, data_init_params,ml_init_params,es_init_params):        
        ####*************************************************************************************************************************************************************
        # Define Project specific parameters and initialize the project 
        ####*************************************************************************************************************************************************************
        # *project_name: project name
        # *project_path: project root path
        # *ml_type: classification or regression
        # *cv_split_method: [0] k_folds [1]Stratified k_folds
        # *cv_split_folds: n folds cross-validataion
        # *data_name: name of the datasets
        # *test_file: the file path for the test dataset
        # *train_file: the file path for the training dataset
        # *extra_train_file: the file path for the extra training dataset
        # *use_extra_train_data: whether you are going to use the extra training dataset    
        # *dependent_variable: the column name for the dependent variable
        # *onehot_features: features are categorical (may be can be detected automatically later)
        # *initial_features: features used in the analysis
        # *train_features: features used for training
        # *compare_features: features used to compare performance individually
        # *interaction_features: features used to study feature interaction
        # *interaction_features_name: name of the features or feature groups used to study feature interaction
        # *percent_min_feature: remove the feature if the missing data of this feature is lower than percent_min_feature%  
        ####*************************************************************************************************************************************************************
        for key in project_init_params:
            setattr(self, key, project_init_params[key])
            
        #***************************************************************************************************************************************************************
        # initialize alphame ml object
        #***************************************************************************************************************************************************************
        self.ml = alm_ml.alm_ml(ml_init_params)
        
        #***************************************************************************************************************************************************************
        # initialize alphame estimators
        #***************************************************************************************************************************************************************
        self.estimators = self.construct_estimators(es_init_params)
        
        #***************************************************************************************************************************************************************
        # initialize alphame dataset parameters (share with different datasets)
        #***************************************************************************************************************************************************************
        self.data = {}
        for i in range(len(self.data_names)): 
            data_init_params['name'] = self.data_names[i]   
            if self.input_data_type[i] == 'file':
                if self.target_data[i] is None:
                    data_init_params['target_data_original_df'] = pd.DataFrame()
                else:    
                    data_init_params['target_data_original_df'] = pd.read_csv(self.target_data[i])
                
                if self.train_data[i] is None:                   
                    data_init_params['train_data_original_df'] = pd.DataFrame()
                else:
                    data_init_params['train_data_original_df'] = pd.read_csv(self.train_data[i])
                
                if self.test_data[i] is None:    
                    data_init_params['test_data_original_df'] = pd.DataFrame()
                else:
                    data_init_params['test_data_original_df'] = pd.read_csv(self.test_data[i])
                    
                if self.extra_train_data[i] is None:
                    data_init_params['extra_train_data_original_df'] = pd.DataFrame()
                else:                    
                    data_init_params['extra_train_data_original_df'] = pd.read_csv(self.extra_train_data[i])
                     
                data_init_params['use_extra_train_data'] =  self.use_extra_train_data[i]  
                      
            if self.input_data_type[i] == 'dataframe':
                data_init_params['target_data_original_df'] = self.target_data[i]
                data_init_params['train_data_original_df'] = self.train_data[i]
                data_init_params['test_data_original_df'] = self.test_data[i]
                data_init_params['extra_train_data_original_df'] = self.extra_train_data[i]
                data_init_params['use_extra_train_data'] =  self.use_extra_train_data[i] 
            self.data[self.data_names[i]] = alm_data.alm_data(data_init_params)

        alm_fun.show_msg(self.log,self.verbose,'Class: [alm_project] [__init__]......done @' + str(datetime.now()))

    def construct_estimators(self, es_init_params):
        es = []
        es_names = []
        es_gs_range = []
        es_scores = []
        es_score_directions = []
        es_importance = []
        es_type = []
        estimators = {}
        
        #***************************************************************************************************************************************************************
        # Regression
        #***************************************************************************************************************************************************************
        # None Regressor
        es.append(None)
        es_scores.append('rmse')
        es_score_directions.append(0)
        es_gs_range.append({})
        es_names.append("None")
        es_importance.append('none')
        es_type.append('regression')   
        
        #Decision Tree Regressor
        es.append(tree.DecisionTreeRegressor(**{'max_depth':3}))   
        es_scores.append('rmse')
        es_score_directions.append(0)
        es_gs_range.append({'max_depth': np.arange(1, 10, 1)})
        es_names.append("dct_r")
        es_importance.append('feature_importances_')
        es_type.append('regression')    
                     
        # kNN Regressor
        es.append(knn.KNeighborsRegressor(**{'n_neighbors': 7, 'weights': 'uniform', 'n_jobs':-1}))
        es_scores.append('rmse')
        es_score_directions.append(0)
        es_gs_range.append({'n_neighbors': np.arange(1, 100, 1)})
        es_names.append("knn_r")
        es_importance.append('none')
        es_type.append('regression')
        
        # Bayesian Ridge Regression  
        es.append(lm.BayesianRidge())
        es_scores.append('rmse')
        es_score_directions.append(0)
        es_gs_range.append({})
        es_names.append("brr_r")
        es_importance.append('coef_')   
        es_type.append('regression')  
    
        # xgb Regressor
        es.append(xgb.XGBRegressor(**{'subsample': 0.8, 'colsample_bytree': 1, 'max_depth': 3, 'n_estimators': 100, 'learning_rate': 0.02, 'n_jobs': 8}))
        es_scores.append('rmse')
        es_score_directions.append(0)
        es_gs_range.append({'learning_rate':np.arange(0.01, 0.11, 0.01), 'max_depth': np.arange(3, 6, 1), 'n_estimators':range(100, 500, 100)})
        es_names.append("xgb_r")
        es_importance.append('feature_importances_')
        es_type.append('regression')
        
        # Random Forest Regressor
        es.append(ensemble.RandomForestRegressor(**{'n_jobs':-1, 'n_estimators': 200, 'max_features': 'auto'}))
        es_scores.append('rmse')
        es_score_directions.append(0)
        es_gs_range.append({'n_estimators':range(100, 500, 100), 'max_features':np.arange(0.1, 1.0, 0.1)})
        es_names.append("rf_r")
        es_importance.append('feature_importances_')   
        es_type.append('regression')
        
        # ElasticNet Regressor
        es.append(lm.ElasticNet(alpha=0.01, l1_ratio=0.5))
        es_scores.append('rmse')
        es_score_directions.append(0)
        es_gs_range.append({'alpha':np.arange(0, 1, 0.1), 'l1_ratio':np.arange(0, 1, 0.1)})
        es_names.append("en_r")
        es_importance.append('coef_')   
        es_type.append('regression')  
            
        #SVM Regressor
        es.append(svm.SVR(C=1.0, epsilon=0.1,kernel='linear'))
        es_scores.append('rmse')
        es_score_directions.append(0)
        es_gs_range.append({})
        es_names.append("svm_r")
        es_importance.append('coef_')  
        es_type.append('regression')  
        
        #AdaBoost ElasticNet Regressor
        es.append(ensemble.AdaBoostRegressor(lm.ElasticNet(alpha=0.1, l1_ratio=0.5),n_estimators=500, random_state=0))
        es_scores.append('rmse')
        es_score_directions.append(0)
        es_gs_range.append({})
        es_names.append("ada_en_r")
        es_importance.append('none')    
        es_type.append('regression')  
        
        #Keras regressor for classification        
        es.append(None)
        es_scores.append('rmse')
        es_score_directions.append(0)
        es_gs_range.append(None)
        es_names.append("keras_r")
        es_importance.append('none')
        es_type.append('regression')
        
        
        #***************************************************************************************************************************************************************
        # Binary classification 
        #***************************************************************************************************************************************************************        
        # None Classification
        es.append(None)
        es_scores.append('auroc')
        es_score_directions.append(1)
        es_gs_range.append({})
        es_names.append("None")
        es_importance.append('none')
        es_type.append('classification_binary')  
        
        
        #Decision tree regressor for classification
        es.append(tree.DecisionTreeRegressor(**{'max_depth':5}))   
        es_scores.append('auprc')
        es_score_directions.append(1)
        es_gs_range.append({'max_depth': np.arange(1, 10, 1)})
        es_names.append("dct_r_c")
        es_importance.append('feature_importances_')
        es_type.append('classification_binary') 
        
        #Decision tree classifier
        es.append(tree.DecisionTreeClassifier(**{'max_depth':5}))   
        es_scores.append('auprc')
        es_score_directions.append(1)
        es_gs_range.append({'max_depth': np.arange(1, 10, 1)})
        es_names.append("dct_c")
        es_importance.append('feature_importances_')
        es_type.append('classification_binary')  
        
        # Gradient boosted tree regressor for classification
#         es.append(xgb.XGBRegressor(**{'n_jobs': 8,'subsample': 0.8, 'colsample_bytree': 1, 'max_depth': 3, 'n_estimators': 100, 'learning_rate': 0.02}))
        es.append(xgb.XGBRegressor(**{'n_jobs': 8}))
        es_scores.append('auprc')
        es_score_directions.append(1)
        es_gs_range.append({'learning_rate':np.arange(0.01, 0.06, 0.01), 'max_depth': np.arange(3, 5, 1), 'n_estimators':range(100, 400, 100)})
        es_names.append("xgb_r_c")
        es_importance.append('booster')
        es_type.append('classification_binary')
        
        # Gradient boosted tree Classifier
        es.append(xgb.XGBClassifier())
        es_scores.append('auroc')
        es_score_directions.append(1)
        es_gs_range.append({'learning_rate':np.arange(0.01, 0.1, 0.01), 'max_depth': np.arange(3, 6, 1), 'n_estimators':range(100, 500, 100)})
        es_names.append("xgb_c")
        es_importance.append('booster')
        es_type.append('classification_binary')
        
        # Random Forest regressor for classification
#         es.append(ensemble.RandomForestRegressor(**{'n_jobs':-1, 'n_estimators': 200, 'max_features': 'auto'}))
        es.append(ensemble.RandomForestRegressor())
        es_scores.append('auroc')
        es_score_directions.append(1)
        es_gs_range.append({'max_features':range(10, 100, 10), 'n_estimators':range(100, 200, 100), 'test_bs_result':['True', 'False']})
        es_names.append("rf_r_c")
        es_importance.append('feature_importances_')
        es_type.append('classification_binary')
        
        # Random Forest Classifier
        es.append(ensemble.RandomForestClassifier(**{'n_jobs':-1, 'n_estimators': 200, 'max_features': 'auto'}))
        es_scores.append('auroc')
        es_score_directions.append(1)
        es_gs_range.append({'max_features':range(10, 100, 10), 'n_estimators':range(100, 200, 100), 'test_bs_result':['True', 'False']})
        es_names.append("rf_c")
        es_importance.append('feature_importances_')
        es_type.append('classification_binary')   
        
        
        # ElasticNet Regressor for classification
#         es.append(lm.ElasticNet(alpha=0.01, l1_ratio=0.5))
        es.append(lm.ElasticNet())
        es_scores.append('auroc')
        es_score_directions.append(0)
        es_gs_range.append({'alpha':np.arange(0, 1, 0.1), 'l1_ratio':np.arange(0, 1, 0.1)})
        es_names.append("en_r_c")
        es_importance.append('coef_')   
        es_type.append('classification_binary')  
        
             
        # Logistic Regression Classifier (binary)
        es.append(lm.LogisticRegression())
        es_scores.append('auroc')
        es_score_directions.append(1)
        es_gs_range.append({})
        es_names.append("lgr_c")
        es_importance.append('coef_')
        es_type.append('classification_binary')
        
        # KNN Classifier (binary) 
        es.append(knn.KNeighborsClassifier(**{'n_neighbors': 10, 'weights': 'distance', 'n_jobs':-1}))
        es_scores.append('auroc')
        es_score_directions.append(1)
        es_gs_range.append({})
        es_names.append("knn_c")
        es_importance.append('none')
        es_type.append('classification_binary')
        
        # SVM Regressor for classification 
        es.append(svm.SVR(C=1.0, epsilon=0.1,kernel='linear'))
        es_scores.append('auroc')
        es_score_directions.append(1)
        es_gs_range.append({})
        es_names.append("svm_r_c")
        es_importance.append('coef_')  
        es_type.append('classification_binary')  
               
        # SVM Classifier    
        es.append(svm.SVC(**{'C': 1.0}))
        es_scores.append('auroc')
        es_score_directions.append(1)
        es_gs_range.append({})
        es_names.append("svm_c")
        es_importance.append('coef_')
        es_type.append('classification_binary')
        
        #Keras regressor for classification        
        es.append(None)
        es_scores.append('auprc')
        es_score_directions.append(1)
        es_gs_range.append(None)
        es_names.append("keras_r_c")
        es_importance.append('none')
        es_type.append('classification_binary')
        
        
#         #Tensor flow classifier
#         es.append(alphame_ml.alm_tf(**{'estimator_name': 'DNNClassifier', 'loss_name': 'cross_entropy', 'hidden_units': [100],
#                                 'activation_fn': tf.nn.sigmoid,'n_classes': 2, 'batch_gd': 1,'batch_size': 0,'num_epochs': 20000, 'learning_rate': 0.0003})    )
#         es_scores.append('auroc')
#         es_score_directions.append(1)
#         es_gs_range.append({})
#         es_names.append("tf_c")    
#         es_importance.append('none')
#         es_type.append('classification_binary')
                
#         #Neural Network (Tensorflow)
#         es.append(alphame_ml.alm_tf())
#         es_scores.append('neg_log_loss')
#         es_score_directions.append(0)
#         es_gs_range.append({'leraning_rate':[0.01,0.1]})
#         es_names.append("nn_c")
#         es_importance.append('none')
#         es_type.append('classification_multiclass')
        
        #***************************************************************************************************************************************************************
        # multi-class classification 
        #***************************************************************************************************************************************************************        
        # xgb 
        es.append(xgb.XGBClassifier(**{'subsample': 0.9, 'colsample_bytree': 1, 'max_depth': 5, 'n_estimators': 200, 'learning_rate': 0.05}))
        es_scores.append('neg_log_loss')
        es_score_directions.append(0)
        es_gs_range.append({'learning_rate':[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1], 'max_depth': [3, 5]})
    #     es_gs_range.append({ 'subsample':[i/10.0 for i in range(6,10)],'colsample_bytree':[i/10.0 for i in range(6,10)]})
        es_names.append("xgb_c")
        es_importance.append('feature_importances_')
        es_type.append('classification_multiclass')

        # Random Forest Classifier
        es.append(ensemble.RandomForestClassifier(**{'n_jobs':-1, 'n_estimators': 200, 'max_features': 'auto'}))
        es_scores.append('neg_log_loss')
        es_score_directions.append(0)
        es_gs_range.append({'max_features':range(10, 100, 10), 'n_estimators':range(100, 200, 100), 'test_bs_result':['True', 'False']})
        es_names.append("rf_c")
        es_importance.append('feature_importances_')
        es_type.append('classification_multiclass')

        # Gradient Boost Tree Classifier
        es.append(ensemble.GradientBoostingClassifier(**{'n_estimators': 200, 'max_features': 'auto', 'max_depth': 3}))
        es_scores.append('neg_log_loss')
        es_score_directions.append(0)
        es_gs_range.append({})
        es_names.append("gbt_c")   
        es_importance.append('feature_importances_') 
        es_type.append('classification_multiclass')
    
        # Logistic Regression Classifier (multi-class)
        es.append(lm.LogisticRegression())
        es_scores.append('neg_log_loss')
        es_score_directions.append(1)
        es_gs_range.append({})
        es_names.append("lgr_c")
        es_importance.append('coef_')
        es_type.append('classification_multiclass')
               
        # KNN Classifier    (multi-class)
        es.append(knn.KNeighborsClassifier(**{'n_neighbors': 10, 'weights': 'distance', 'n_jobs':-1}))
        es_scores.append('neg_log_loss')
        es_score_directions.append(1)
        es_gs_range.append({})
        es_names.append("knn_c")
        es_importance.append('none')
        es_type.append('classification_multiclass')
        
        # SVM Classifier    
        es.append(svm.SVC(**{'C': 1.0, 'kernel': 'linear', 'probability': True}))
        es_scores.append('neg_log_loss')
        es_score_directions.append(1)
        es_gs_range.append({})
        es_names.append("svm_c")
        es_importance.append('coef_')
        es_type.append('classification_multiclass')
        
        for i in range(len(es)):
            if es_type[i] == es_init_params['ml_type']:
                es_init_params['single_feature_as_prediction'] = 1
                es_init_params['estimator'] = es[i]
                es_init_params['name'] = es_names[i]
                es_init_params['gs_range'] = es_gs_range[i]
                es_init_params['score_name'] = es_scores[i]
                es_init_params['score_direction'] = es_score_directions[i]
                es_init_params['feature_importance_name'] = es_importance[i]
                es_init_params['prediction_transformation'] = None
                estimators[es_names[i]] = alm_es.alm_es(es_init_params)
        return estimators
    
    def run(self,refresh_data = 0, nofit = 0):               
        return_objs = {}
        self.estimators[self.run_estimator_name].score_name = self.run_estimator_scorename   
   
        # refresh data first     
    
        for data_name in self.run_data_names:        
            stime1 = time.time()
            self.data[data_name].train_features = self.train_features     
            if refresh_data == 1:   
                self.data[data_name].refresh_data()
            etime1 = time.time()
#             alm_fun.show_msg(self.log,self.verbose,"Class: [alm_project] Fun: [run] -- Current Modes: " + str(self.modes) + " Current Data: " + data_name + ", data preparation time was %g seconds" % (etime1 - stime1)) 
    
        for mode in self.modes: 
            return_objs[mode] = {}  
            for data_name in self.run_data_names:
                stime2 = time.time()
                #**************************************************************************
                # run project in different mode 
                #**************************************************************************
                if mode == 'target_prediction': 
                    r = self.ml.run_target_prediction(self.estimators[self.run_estimator_name], self.data[data_name],nofit = nofit)
                    return_objs[mode][data_name] = r['target_y_predicted']
                    
                if mode == 'test_prediction':
                    if self.grid_search_on == 1: 
                        r = self.ml.grid_search(self.estimators[self.run_estimator_name], self.data[data_name])
                        alm_fun.show_msg(self.log,self.verbose,'grid search best socre:' + str(r['gs_opt_cv_score']) + 'and best parameters:' + str(r['gs_opt_params']))                      
                    
                    r = self.ml.run_test_prediction(self.estimators[self.run_estimator_name], self.data[data_name],nofit = nofit)
                    test_y_predicted = r['test_y_predicted']                                                 
                    test_bs_result = r['test_bs_result']
                    test_bs_result.columns = ['test_' + x for x in test_bs_result.columns]      
                    feature_importance = r['feature_importance']
#                     alm_fun.show_msg(self.log,self.verbose,str(test_bs_result))
#                     alm_fun.show_msg(self.log,self.verbose,str(feature_importance))                                        
                    test_bs_result.to_csv(self.project_path + data_name +'_test_results' + '_fold_' + str(self.data[data_name].cur_test_split_fold) + '_' + str(self.data[data_name].cur_gradient_key) +'.csv')                              
                    return_objs[mode][data_name] = [test_bs_result,feature_importance,test_y_predicted]
                                        
                if mode == 'test_prediction_all_folds':
                    tp_results = None
                    tp_predictions = None
                    feature_importance = None
                                        
                    for j in range(self.data[data_name].test_split_folds):
                        self.data[data_name].cur_test_split_fold = j  
                        self.data[data_name].train_features = self.cv_selected_features[j]

                        if self.grid_search_on == 1: 
                            r = self.ml.grid_search(self.estimators[self.run_estimator_name], self.data[data_name])
                            alm_fun.show_msg(self.log,self.verbose,'grid search best socre:' + str(r['gs_opt_cv_score']) + 'and best parameters:' + str(r['gs_opt_params']))                      
                        
                        if (self.outloop_cv_fit_once  == 1) & (j!= 0):                        
                            r = self.ml.run_test_prediction(self.estimators[self.run_estimator_name], self.data[data_name], nofit = 1)                            
                        else:
                            r = self.ml.run_test_prediction(self.estimators[self.run_estimator_name], self.data[data_name], nofit = 0)
                                
                        feature_fold_importance = r['feature_importance'].reset_index()
                        feature_fold_importance.columns = ['feature',str(j)]                                                       
                        tp_fold_predictions = r['test_y_predicted']                                                 
                        tp_fold_results = r['test_bs_result']
                        tp_fold_results.columns = ['test_' + x for x in tp_fold_results.columns]

                        if feature_importance is None:
                            feature_importance =  feature_fold_importance
                        else:
                            feature_importance = pd.merge(feature_importance,feature_fold_importance)
                                                
                        if tp_results is None:
                            tp_results = tp_fold_results
                        else:
                            tp_results = pd.concat([tp_results,tp_fold_results])   
      
                        if self.data[data_name].if_engineer:
                            predition_labels = self.data[data_name].test_splits_engineered_df[self.data[data_name].cur_test_split_fold][self.data[data_name].cur_gradient][self.data[data_name].dependent_variable]                                    
                        else:
                            predition_labels = self.data[data_name].test_data_index_df.loc[self.data[data_name].test_splits_df[self.data[data_name].cur_test_split_fold][self.data[data_name].cur_gradient_key],self.data[data_name].dependent_variable]   
                              
                        tp_fold_predictions = pd.concat([tp_fold_predictions, predition_labels], axis=1)

                        if tp_predictions is None:
                            tp_predictions = tp_fold_predictions
                        else:
                            tp_predictions = pd.concat([tp_predictions,tp_fold_predictions])
                                                
                    tp_final_results = tp_results.reset_index()   
                    tp_final_results.to_csv(self.project_path + data_name +'_test_predition_folds_results.csv')                              

#                     alm_fun.show_msg(self.log,self.verbose,str(tp_results))                         
                    tp_final_results = tp_final_results.groupby(['index'])['test_auroc','test_auprc','test_rfp','test_prior'].agg(['mean','std'])                                                
                    tp_final_results.columns = ['test_macro_auroc_mean','test_macro_auroc_std','test_macro_auprc_mean','test_macro_auprc_std','test_macro_rfp_mean','test_macro_rfp_std','test_prior_mean','test_prior_std']
                    tp_final_results.index = [self.predictor_name]
                    
                    tp_predictions.columns = [self.predictor_name,'label']
                    tp_micro_preditions = tp_predictions.apply(lambda x: np.array([alm_fun.classification_metrics(np.array(tp_predictions['label']),np.array(x))[1][y]  for y in ['auroc','auprc','recall_fixed_precision']]),axis = 0)
                    tp_micro_preditions.drop(columns = {'label'},inplace = True)
                    tp_micro_preditions = tp_micro_preditions.transpose()

                    tp_micro_preditions.columns = ['test_micro_auroc','test_micro_auprc','test_micro_rfp']                    
                    tp_final_results = pd.concat([tp_final_results,tp_micro_preditions],axis = 1)                                                      
                    alm_fun.show_msg(self.log,self.verbose,str(tp_final_results))
                                                            
                    tp_final_results.to_csv(self.project_path + data_name +'_test_predition_all_folds_results.csv')                              
                    return_objs[mode][data_name] = [tp_results,tp_predictions,feature_importance,tp_final_results]
                    
                if mode == "cross_validation_all_folds":
                    cv_results = None
                    for j in range(self.data[data_name].test_split_folds):
                        self.data[data_name].cur_test_split_fold = j  
                        self.data[data_name].train_features = self.cv_selected_features[j]
                        if self.grid_search_on == 1:                        
                            [gs_opt_params, validation_cv_fold_result, gs_fold_results] = self.ml.grid_search(self.estimators[self.run_estimator_name], self.data[data_name])
                            [test_y_predicted, feature_importance, test_bs_fold_result, test_bs_score] = self.ml.run_test_prediction(self.estimators[self.run_estimator_name], self.data[data_name])
                            alm_fun.show_msg(self.log,self.verbose,'all_features - cv:' + str(validation_cv_fold_result['mean']) + ' ' + str(validation_cv_fold_result['ste']) + ' test:' + str(test_bs_score['mean']) + ' ' + str(test_bs_score['ste']) + " parameters:" + str(gs_opt_params))
                        else:
                            if ((self.outloop_cv_fit_once == 1) & (j!= 0)): 
                                cv_fold_result = self.ml.run_cv_prediction(self.estimators[self.run_estimator_name], self.data[data_name],nofit = 1)
                            else:
                                cv_fold_result = self.ml.run_cv_prediction(self.estimators[self.run_estimator_name], self.data[data_name],nofit = 0)
                            train_cv_fold_result = cv_fold_result['train_cv_result']      
                            train_cv_fold_result.columns = ['train_' + x for x in train_cv_fold_result.columns]
                                                                             
                            validation_cv_fold_result = cv_fold_result['validation_cv_result']      
                            validation_cv_fold_result.columns = ['validation_' + x for x in validation_cv_fold_result.columns]
                                                        
#                           test_result = self.ml.run_test_prediction(self.estimators[self.run_estimator_name], self.data[data_name],nofit = 1)
#                           test_bs_result = test_result['test_bs_result']
#                           test_bs_result.columns = ['test_' + x for x in test_bs_result.columns]
                            
                        cv_fold_results = pd.concat([train_cv_fold_result,validation_cv_fold_result],axis = 1)  
                        cv_fold_results['fold'] = j                          
                        if cv_results is None:
                            cv_results = cv_fold_results
                        else:
                            cv_results = pd.concat([cv_results,cv_fold_results])   
                            
#                     alm_fun.show_msg(self.log,self.verbose,str(cv_results))
                    return_objs[mode][data_name] = [cv_results]
                        
                                                
                if mode == 'cross_validation':
                    if self.grid_search_on == 1:                        
                        [gs_opt_params, validation_cv_result, gs_results] = self.ml.grid_search(self.estimators[self.run_estimator_name], self.data[data_name])
                        [test_y_predicted, feature_importance, test_bs_result, test_bs_score] = self.ml.run_test_prediction(self.estimators[self.run_estimator_name], self.data[data_name])
                        alm_fun.show_msg(self.log,self.verbose,'all_features - cv:' + str(validation_cv_result['mean']) + ' ' + str(validation_cv_result['ste']) + ' test:' + str(test_bs_score['mean']) + ' ' + str(test_bs_score['ste']) + " parameters:" + str(gs_opt_params))
                    else:
                        cv_result = self.ml.run_cv_prediction(self.estimators[self.run_estimator_name], self.data[data_name])                                                                   
                        validation_cv_result = cv_result['validation_cv_result']      
                        validation_cv_result.columns = ['cv_' + x for x in validation_cv_result.columns]
                        train_cv_result = cv_result['train_cv_result']      
                        train_cv_result.columns = ['cv_' + x for x in train_cv_result.columns]
                                                                      
                    return_objs[mode][data_name] = [validation_cv_result, train_cv_result]
                    
                if mode == 'gradient_comparison':                    
                    gc_results = pd.DataFrame(columns = ['params','gradient','cv_score','cv_score_ste'])
                    for gradient in ['no_gradient'] + self.data[data_name].gradients:
                        self.data[data_name].cur_gradient_key = gradient
                        if self.grid_search_on == 1:                        
                            [gs_opt_params, validation_cv_result, gs_results] = self.ml.grid_search(self.estimators[self.run_estimator_name], self.data[data_name],self.data[data_name].cur_test_split_fold,gradient,self.data[data_name].if_engineer)                                                        
                            cur_params = gs_opt_params
                            cur_cv_score = validation_cv_result.get_values()[0]                                                        
                        else:
                            cv_result = self.ml.run_cv_prediction(self.estimators[self.run_estimator_name], self.data[data_name])
                            cur_cv_score = cv_result['validation_cv_score']
                            cur_cv_score_ste = cv_result['validation_cv_score_ste']
                            cur_params = self.estimators[self.run_estimator_name].estimator.get_params()
  
                        gc_results.loc[gradient,'params']  = str(cur_params)
                        gc_results.loc[gradient,'gradient'] = gradient
                        gc_results.loc[gradient,'cv_score'] = cur_cv_score
                        gc_results.loc[gradient,'cv_score_se'] = cur_cv_score_ste
        
                    if self.estimators[self.run_estimator_name].score_direction == 1:                      
                        opt_score = max(gc_results['cv_score'])
                    else:
                        opt_score = min(gc_results['cv_score'])
                        
                    opt_gradient = gc_results.loc[gc_results['cv_score'] == opt_score, 'gradient'].get_values()[0]
                    alm_fun.show_msg(self.log,self.verbose,str(gc_results))                                        
                    gc_results.to_csv(self.project_path + data_name +'_gradient_comparison_results.csv', encoding='utf-8')                              
                    return_objs[mode][data_name] = [gc_results,opt_score,opt_gradient]
                 
                if mode == 'feature_selection':    
                    args = {}
                    args['start_features'] = self.ml.fs_start_features
                    args['T'] = self.ml.fs_T
                    args['alpha'] = self.ml.fs_alpha
                    args['K'] = self.ml.fs_K
                    args['epsilon'] = self.ml.fs_epsilon
                                                       
                    fs_results = self.ml.feature_selection(self.estimators[self.run_estimator_name], self.data[data_name], type='local search', args=args) 
                    
                    max_score = max(fs_results['score'])
                    opt_features = fs_results.loc[fs_results['score'] == max_score,'features']
                                        
                    fs_results.to_csv(self.project_path + data_name +'_feature_selection_results_' + str(self.data[data_name].cur_test_split_fold) + '.csv', encoding='utf-8')
                    return_objs[mode][data_name] = [fs_results,max_score,opt_features]
                    
                if mode == 'method_comparison':
                    mc_results = None
                    methods = list(self.estimators.keys())
                    if 'None' in methods:
                        methods.remove('None')

                    for method in self.compare_methods: 
                        
                        if self.grid_search_on == 1: 
                            [gs_opt_params, validation_cv_result, gs_results] = self.ml.grid_search(self.estimators[method], self.data[data_name])
   
                        cv_result = self.ml.run_cv_prediction(self.estimators[method], self.data[data_name])
                        test_result = self.ml.run_test_prediction(self.estimators[method], self.data[data_name])                                                 
                        validation_cv_result = cv_result['validation_cv_result']      
                        validation_cv_result.index = [method]
                        validation_cv_result.columns = ['cv_' + x for x in validation_cv_result.columns]
                        
                        test_bs_result = test_result['test_bs_result']
                        test_bs_result.index = [method]
                        test_bs_result.columns = ['test_' + x for x in test_bs_result.columns]                            
                            
                        if  mc_results is None:                  
                            mc_results = pd.concat([validation_cv_result, test_bs_result], axis=1)
                        else:
                            mc_results = pd.concat([mc_results, pd.concat([validation_cv_result, test_bs_result], axis=1)])       
                        
                    mc_results.to_csv(self.project_path + data_name +'_method_comparison_results.csv', encoding='utf-8')  
                    alm_fun.show_msg(self.log,self.verbose,str(mc_results))
                    return_objs[mode][data_name] = mc_results
                    
                if mode == 'grid_search':
                    [gs_opt_params, gs_opt_score, gs_results] = self.ml.grid_search(self.estimators[self.run_estimator_name], self.data[data_name])
                    max_score = gs_opt_score['mean'].get_values()[0]
                    gs_results.to_csv(self.project_path + 'grid_search_results.csv', encoding='utf-8')
                    return_objs[mode][data_name] = [gs_results,max_score,gs_opt_params]

                if mode == 'feature_comparison_test':
                    fc_results = None
                    fc_predictions_xyz = None
                    
                    for j in range(self.data[data_name].test_split_folds):
                        self.data[data_name].cur_test_split_fold = j  
                        fc_fold_results = None
                        fc_fold_predictions = None                                                       
                        for i in range(len(self.compare_features)):
                            self.data[data_name].train_features = self.compare_features[i]
                            r = self.ml.run_test_prediction(self.estimators[self.run_estimator_name], self.data[data_name])      
                            test_bs_result = r['test_bs_result']
                            test_predictions = r['test_y_predicted']
#                             test_predictions = r['test_bs_result']
                            
                            test_bs_result.index = [self.compare_features_name[i]]
                            test_bs_result.columns = ['test_' + x for x in test_bs_result.columns]  
                                                                                                             
                            if  fc_fold_results is None:                  
                                fc_fold_results = test_bs_result
                            else:
                                fc_fold_results = pd.concat([fc_fold_results, test_bs_result])
                                                              
                            if  fc_fold_predictions is None:                  
                                fc_fold_predictions = test_predictions
                            else:
                                fc_fold_predictions = pd.concat([fc_fold_predictions, test_predictions], axis=1)                         
                            
                            alm_fun.show_msg(self.log,self.verbose,self.compare_features_name[i])
                            
                        if self.data[data_name].if_engineer:
                            predition_labels = self.data[data_name].test_splits_engineered_df[self.data[data_name].cur_test_split_fold][self.data[data_name].cur_gradient][self.data[data_name].dependent_variable]                                    
                        else:
                            predition_labels = self.data[data_name].test_data_index_df.loc[self.data[data_name].test_splits_df[self.data[data_name].cur_test_split_fold][self.data[data_name].cur_gradient_key],self.data[data_name].dependent_variable]   
                              
                        fc_fold_predictions = pd.concat([fc_fold_predictions, predition_labels], axis=1)
                    
                        if fc_results is None:
                            fc_results = fc_fold_results
                        else:
                            fc_results = pd.concat([fc_results,fc_fold_results])   
    
                        if fc_predictions_xyz is None:
                            fc_predictions_xyz = fc_fold_predictions
                        else:
                            fc_predictions_xyz = pd.concat([fc_predictions_xyz,fc_fold_predictions])
                            
                    fc_results = fc_results.reset_index()                            
                    fc_results = fc_results.groupby(['index'])['test_auroc','test_auprc','test_rfp','test_prior'].agg(['mean','std'])                                                
                    fc_results.columns = ['test_macro_auroc_mean','test_macro_auroc_std','test_macro_auprc_mean','test_macro_auprc_std','test_macro_rfp_mean','test_macro_rfp_std','test_prior_mean','test_prior_std']

                    fc_predictions_xyz.columns = self.compare_features_name + ['fitness']                    
                    fc_micro_preditions_xyz = fc_predictions_xyz.apply(lambda x: np.array([alm_fun.classification_metrics(np.array(fc_predictions_xyz['fitness']),np.array(x))[1][y]  for y in ['auroc','auprc','recall_fixed_precision']]),axis = 0)
                    fc_micro_preditions_xyz.drop(columns = {'fitness'},inplace = True)
                    fc_micro_preditions_xyz = fc_micro_preditions_xyz.transpose()
                    fc_micro_preditions_xyz.columns = ['test_micro_auroc','test_micro_auprc','test_micro_rfp']
                    
                    fc_results = pd.concat([fc_results,fc_micro_preditions_xyz],axis = 1)
                    fc_results = fc_results.sort_values('test_micro_auprc',ascending = False)
                    fc_predictions_xyz.to_csv(self.project_path + 'output/' + data_name +'_fc_predictions.csv',index = False)
                    alm_fun.show_msg(self.log,self.verbose,str(fc_results))
                                                
                if mode == 'feature_comparison':
                    fc_results = None
                    fc_predictions = None
                                                                          
                    for i in range(len(self.compare_features)):
                        self.data[data_name].train_features = self.compare_features[i]
                        if self.grid_search_on == 1:                        
                            r = self.ml.grid_search(self.estimators[self.run_estimator_name], self.data[data_name])
                            gs_opt_params = r['gs_opt_params']
                            validation_cv_result = r['gs_opt_cv_result']
                            gs_results = r['gs_results']     
                            alm_fun.show_msg(self.log,self.verbose, self.compare_features_name[i] + ' - best params: ' + str(gs_opt_params))                                                
                        else:
                            r = self.ml.run_cv_prediction(self.estimators[self.run_estimator_name], self.data[data_name])                                                                             
                            validation_cv_result = r['validation_cv_result']
                        
                        validation_cv_result.index = [self.compare_features_name[i]]
                        validation_cv_result.columns = ['cv_' + x for x in validation_cv_result.columns]
                         
                        r = self.ml.run_test_prediction(self.estimators[self.run_estimator_name], self.data[data_name])      
                        test_bs_result = r['test_bs_result']
                        test_predicitons = pd.Series(r['test_y_predicted'], name=self.compare_features_name[i])
                        test_bs_result.index = [self.compare_features_name[i]]
                        test_bs_result.columns = ['test_' + x for x in test_bs_result.columns]  
                            
                                                                             
                        if  fc_results is None:                  
                            fc_results = pd.concat([validation_cv_result, test_bs_result], axis=1)
                        else:
                            fc_results = pd.concat([fc_results, pd.concat([validation_cv_result, test_bs_result], axis=1)])                              
                        if  fc_predictions is None:                  
                            fc_predictions = test_predicitons
                        else:
                            fc_predictions = pd.concat([fc_predictions, test_predicitons], axis=1)                           
                        
                        alm_fun.show_msg(self.log,self.verbose,self.compare_features_name[i])
                                                                                    
                    alm_fun.show_msg(self.log,self.verbose,str(fc_results))
                    fc_results.to_csv(self.project_path + data_name +'_feature_comparison_results' + '_fold_' + str(self.data[data_name].cur_test_split_fold) + '_' + str(self.data[data_name].cur_gradient_key) +'.csv')
                    
                    if self.data[data_name].if_engineer:
                        predition_labels = self.data[data_name].test_splits_engineered_df[self.data[data_name].cur_test_split_fold][self.data[data_name].cur_gradient][self.data[data_name].dependent_variable]                                    
                    else:
                        predition_labels = self.data[data_name].test_data_index_df.loc[self.data[data_name].test_splits_df[self.data[data_name].cur_test_split_fold][self.data[data_name].cur_gradient_key],self.data[data_name].dependent_variable]   
                        
                    fc_predictions = pd.concat([fc_predictions, predition_labels], axis=1)  
                    fc_predictions.to_csv(self.project_path + data_name +'_fc_predictions' + '_fold_' + str(self.data[data_name].cur_test_split_fold) + '_' + str(self.data[data_name].cur_gradient_key) +'.csv',index = False)
                    if self.estimators[self.run_estimator_name].ml_type == 'classification_binary':
                        auprc_plotname = self.project_path + 'output/' + mode + '_' + data_name + '_fold_' + str(self.data[data_name].cur_test_split_fold) + '_' + str(self.data[data_name].cur_gradient_key) + '_auprc.png'
                        auroc_plotname = self.project_path + 'output/' + mode + '_' + data_name + '_fold_' + str(self.data[data_name].cur_test_split_fold) + '_' + str(self.data[data_name].cur_gradient_key) + '_auroc.png'
                        alm_fun.plot_prc(predition_labels, fc_predictions[self.compare_features_name_forplot], auprc_plotname, 20, 10, None, 0.9, 0.9, 'AUPRC Comparison')
                        alm_fun.plot_roc(predition_labels, fc_predictions[self.compare_features_name_forplot], auroc_plotname, 20, 10, None, 0.9, 0.9, 'AUROC Comparison')
                    return_objs[mode][data_name] = fc_results
                    
                etime2 = time.time()
#                 alm_fun.show_msg(self.log,self.verbose,"Class: [alm_project] Fun: [run] -- Current Mode: " + "[" + mode + "]" + " Current Data: " + data_name + ", running time was %g seconds" % (etime2 - stime2))                 

        return (return_objs)
    
    def project_plot(self, data, mode, data_name, x_label, y_label, ylim_min=0, ylim_max=0.5, fig_w=20 , fig_h=5):
        title = mode + ' (' + data_name + ')' 
        plot_name = self.project_path + 'output/' + mode + '_' + data_name + '.png'        
        title_size = 30
        label_size = 20
        tick_size = 15       
        # alphame_ml.plot_barplot(data,fig_w,fig_h,title,title_size,x_label,y_label,label_size,tick_size,ylim_min,ylim_max,plot_name)

    def plots(self):
        if plot:
            if alm_dataset.ml_type == 'classification':        
                fig = plt.figure(figsize=(20, 10))
                plt.clf()
                 
                ax = plt.subplot(2, 2, 1)
                ax.pie([alm_dataset.prior, 1 - alm_dataset.prior], autopct='%1.1f%%', labels=['positive', 'negative'], colors=['red', 'green'])
                ax.set_title('Dataset:' + '[' + alm_dataset.name + ']' + ' size: [' + str(alm_dataset.n_train) + ']' , size=15)
                 
                plt.subplot(2, 2, 2)
                ax = sns.barplot(cv_feature_importance.index, cv_feature_importance)
                ax.set_title('Feature importance', size=15)
                ax.set_ylabel('Importance')
                ax.tick_params(labelsize=10) 
                 
                plt.subplot(2, 2, 3)
                cv_auroc_result = validation_cv_result[[s + '_auroc' for s in alm_dataset.compare_features] + ['auroc']]
                cv_auroc_result = cv_auroc_result.sort_values()
                predictors = list(cv_auroc_result.index)
                 
                for i in range(len(predictors)):
                    if '_auroc' in predictors[i]:
                        predictors[i] = predictors[i][:-6]
                predictors[predictors.index('auroc')] = predictor_name 
                 
                ax = sns.barplot(predictors, cv_auroc_result)
                ax.set_title('Predictor performance AUROC' + ' [' + str(alm_dataset.cv_split_folds) + ' folds]', size=15)
                ax.set_ylabel('AUROC')
                ax.tick_params(labelsize=10)
                ax.set_ylim(0, 1)
                i = 0
                for p in ax.patches:
                    height = p.get_height()
                    ax.text(p.get_x() + p.get_width() / 2., height + 0.005, np.array(cv_auroc_result)[i], ha="center") 
                    i += 1
                     
                plt.subplot(2, 2, 4)
                cv_auprc_result = validation_cv_result[[s + '_auprc' for s in alm_dataset.compare_features] + ['auprc']]
                cv_auprc_result = cv_auprc_result.sort_values()
                predictors = list(cv_auprc_result.index)
                 
                for i in range(len(predictors)):
                    if '_auprc' in predictors[i]:
                        predictors[i] = predictors[i][:-6]
                predictors[predictors.index('auprc')] = predictor_name 
                ax = sns.barplot(predictors, cv_auprc_result)
                ax.set_title('Predictor performance AURPC' + ' [' + str(alm_dataset.cv_split_folds) + ' folds]', size=15)
                ax.set_ylabel('AUPRC')
                ax.tick_params(labelsize=10)
                ax.set_ylim(0, 1)
                i = 0
                for p in ax.patches:
                    height = p.get_height()
                    ax.text(p.get_x() + p.get_width() / 2., height + 0.005, np.array(cv_auprc_result)[i], ha="center") 
                    i += 1    
                     
                fig.tight_layout()
                plt.savefig(self.path + 'cv_' + alm_dataset.name + 'png')    
                     
            if alm_dataset.ml_type == 'regression':
                fig = plt.figure(figsize=(16, 10))
                plt.clf()
     
                plt.subplot(3, 1, 3)
                ax = sns.barplot(cv_feature_importance.index[:10], cv_feature_importance[:10])
                ax.set_title('Feature importance', size=15)
                ax.set_ylabel('Importance')
                ax.tick_params(labelsize=10)
     
                plt.subplot(3, 1, 1)
                cv_pcc_result = validation_cv_result[[s + '_pcc' for s in alm_dataset.compare_features] + ['pcc']]
                cv_pcc_result = cv_pcc_result.sort_values()
                predictors = list(cv_pcc_result.index)
                 
                for i in range(len(predictors)):
                    if '_pcc' in predictors[i]:
                        predictors[i] = predictors[i][:-4]
                predictors[predictors.index('pcc')] = predictor_name 
                ax = sns.barplot(predictors, cv_pcc_result)
                ax.set_title('Predictor performance PCC' + ' [' + str(alm_dataset.cv_split_folds) + ' folds]', size=15)
                ax.set_ylabel('pcc')
                ax.tick_params(labelsize=10)
                # ax.set_ylim(0,1)
                i = 0
                for p in ax.patches:
                    height = p.get_height()
                    ax.text(p.get_x() + p.get_width() / 2., height + 0.005, np.array(cv_pcc_result)[i], ha="center") 
                    i += 1  
                pass    
             
                plt.subplot(3, 1, 2)
                cv_rmse_result = validation_cv_result[[s + '_rmse' for s in alm_dataset.compare_features] + ['rmse']]
                cv_rmse_result = cv_rmse_result.sort_values()
                predictors = list(cv_rmse_result.index)
                 
                for i in range(len(predictors)):
                    if '_rmse' in predictors[i]:
                        predictors[i] = predictors[i][:-5]
                predictors[predictors.index('rmse')] = predictor_name 
                ax = sns.barplot(predictors, cv_rmse_result)
                ax.set_title('Predictor performance rmse' + ' [' + str(alm_dataset.cv_split_folds) + ' folds]', size=15)
                ax.set_ylabel('rmse')
                ax.tick_params(labelsize=10)
                i = 0
                for p in ax.patches:
                    height = p.get_height()
                    ax.text(p.get_x() + p.get_width() / 2., height + 0.005, np.array(cv_rmse_result)[i], ha="center") 
                    i += 1    
                pass       
                fig.tight_layout()
                plt.savefig(self.path + 'cv_' + alm_dataset.name + '.png')   
    
    def remove_features(self, features, i):
        remove_features = features[i]
        if any(isinstance(i, list) for i in features):  # if features are nested list
            features = list(itertools.chain(*features))   
        features = list(set(features)) 
        if isinstance(remove_features, list):     
            for x in remove_features:
                features.remove(x)
        else:
            features.remove(remove_features)
        features = list(set(features))    
        return (features)
        
