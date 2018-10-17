import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import time
import itertools
import pickle
import copy
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
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
                data_init_params['target_data_original_df'] = pd.read_csv(self.target_data[i])
                data_init_params['train_data_original_df'] = pd.read_csv(self.train_data[i])
                data_init_params['test_data_original_df'] = pd.read_csv(self.test_data[i])                
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
        es.append(xgb.XGBRegressor(**{'subsample': 0.8, 'colsample_bytree': 1, 'max_depth': 3, 'n_estimators': 100, 'learning_rate': 0.02}))
        es_scores.append('rmse')
        es_score_directions.append(0)
        es_gs_range.append({'learning_rate':np.arange(0.01, 0.1, 0.01), 'max_depth': np.arange(3, 6, 1), 'n_estimators':range(100, 500, 100)})
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
        es.append(xgb.XGBRegressor(**{'n_jobs': -1,'subsample': 0.8, 'colsample_bytree': 1, 'max_depth': 3, 'n_estimators': 100, 'learning_rate': 0.02}))
        es_scores.append('auprc')
        es_score_directions.append(1)
        es_gs_range.append({'learning_rate':np.arange(0.01, 0.06, 0.01), 'max_depth': np.arange(3, 5, 1), 'n_estimators':range(100, 400, 100)})
        es_names.append("xgb_r_c")
        es_importance.append('booster')
        es_type.append('classification_binary')
        
        # Gradient boosted tree Classifier
        es.append(xgb.XGBClassifier(**{'subsample': 0.9, 'colsample_bytree': 1, 'max_depth': 3, 'n_estimators': 200, 'learning_rate': 0.02}))
        es_scores.append('auroc')
        es_score_directions.append(1)
        es_gs_range.append({'learning_rate':np.arange(0.01, 0.1, 0.01), 'max_depth': np.arange(3, 6, 1), 'n_estimators':range(100, 500, 100)})
        es_names.append("xgb_c")
        es_importance.append('booster')
        es_type.append('classification_binary')
        
        # Random Forest regressor for classification
        es.append(ensemble.RandomForestRegressor(**{'n_jobs':-1, 'n_estimators': 200, 'max_features': 'auto'}))
        es_scores.append('auroc')
        es_score_directions.append(1)
        es_gs_range.append({'max_features':range(10, 100, 10), 'n_estimators':range(100, 200, 100), 'test_score':['True', 'False']})
        es_names.append("rf_r_c")
        es_importance.append('feature_importances_')
        es_type.append('classification_binary')
        
        # Random Forest Classifier
        es.append(ensemble.RandomForestClassifier(**{'n_jobs':-1, 'n_estimators': 200, 'max_features': 'auto'}))
        es_scores.append('auroc')
        es_score_directions.append(1)
        es_gs_range.append({'max_features':range(10, 100, 10), 'n_estimators':range(100, 200, 100), 'test_score':['True', 'False']})
        es_names.append("rf_c")
        es_importance.append('feature_importances_')
        es_type.append('classification_binary')   
        
        
        # ElasticNet Regressor for classification
        es.append(lm.ElasticNet(alpha=0.01, l1_ratio=0.5))
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
        es_gs_range.append({'max_features':range(10, 100, 10), 'n_estimators':range(100, 200, 100), 'test_score':['True', 'False']})
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
                estimators[es_names[i]] = alm_es.alm_es(es_init_params)
        return estimators
    
    def run(self,refresh_data = 0):               
        return_objs = {}
        self.estimators[self.run_estimator_name].score_name = self.run_estimator_scorename           
        # refresh data first     
        if refresh_data == 1:       
            for data_name in self.run_data_names:        
                stime1 = time.time()
                self.data[data_name].train_features = self.train_features
                self.data[data_name].refresh_data()
                etime1 = time.time()
                alm_fun.show_msg(self.log,self.verbose,"Class: [alphame_project] Fun: [run] -- Current Modes: " + str(self.modes) + " Current Data: " + data_name + ", data preparation time was %g seconds" % (etime1 - stime1)) 
        
        for mode in self.modes: 
            return_objs[mode] = {}  
            for data_name in self.run_data_names:
                stime2 = time.time()
                #**************************************************************************
                # run project in different mode 
                #**************************************************************************
                if mode == 'prediction': 
                    prediction_results = self.ml.run_target_prediction(self.estimators[self.run_estimator_name], self.data[data_name])
                    return_objs[mode][data_name] = prediction_results['target_y_predicted']
                    
                if mode == 'cross_validation':
                    if self.grid_search_on == 1:                        
                        [gs_opt_params, validation_cv_score, gs_results] = self.ml.grid_search(self.estimators[self.run_estimator_name], self.data[data_name])
                        [test_y_predicted, feature_importance, test_bs_result, test_bs_score] = self.ml.run_test_prediction(self.estimators[self.run_estimator_name], self.data[data_name])
                        alm_fun.show_msg(self.log,self.verbose,'all_features - cv:' + str(validation_cv_score['mean']) + ' ' + str(validation_cv_score['ste']) + ' test:' + str(test_bs_score['mean']) + ' ' + str(test_bs_score['ste']) + " parameters:" + str(gs_opt_params))
                    else:
                        cv_result = self.ml.run_cv_prediction(self.estimators[self.run_estimator_name], self.data[data_name])
                        test_result = self.ml.run_test_prediction(self.estimators[self.run_estimator_name], self.data[data_name])                                                 
                        validation_cv_score = cv_result['validation_cv_result']      
                        validation_cv_score.columns = ['cv_' + x for x in validation_cv_score.columns]

                        test_score = test_result['test_bs_result']
                        test_score.columns = ['test_' + x for x in test_score.columns]

                        feature_importance = test_result['feature_importance'].transpose()
                        feature_importance = feature_importance.sort_values([0])

                    feature_importance.to_csv(self.project_path + data_name +'_feature_importance.csv', encoding='utf-8')
                    alm_fun.show_msg(self.log,self.verbose,  data_name + "\n" + str(pd.concat([validation_cv_score, test_score], axis=1)))
                    alm_fun.show_msg(self.log,self.verbose,str(feature_importance))
                                                                      
                    return_objs[mode][data_name] = [str(validation_cv_score), str(test_score),feature_importance]
                    
                if mode == 'gradient_comparison':                    
                    gc_results = pd.DataFrame(columns = ['params','gradient','cv_score','cv_score_ste'])
                    for gradient in ['no_gradient'] + self.data[data_name].gradients:
                        self.data[data_name].cur_gradient_key = gradient
                        if self.grid_search_on == 1:                        
                            [gs_opt_params, validation_cv_score, gs_results] = self.ml.grid_search(self.estimators[self.run_estimator_name], self.data[data_name],self.data[data_name].cur_test_split_fold,gradient,self.data[data_name].if_engineer)                                                        
                            cur_params = gs_opt_params
                            cur_cv_score = validation_cv_score.get_values()[0]                                                        
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
                    args['start_features'] = self.fs_start_features
                    args['T'] = self.fs_T
                    args['alpha'] = self.fs_alpha
                    args['K'] = self.fs_K
                    args['epsilon'] = self.fs_epsilon
                                                       
                    fs_results = self.ml.feature_selection(self.estimators[self.run_estimator_name], self.data[data_name], type='local search', args=args) 
                    
                    max_score = max(fs_results['score'])
                    opt_features = fs_results.loc[fs_results['score'] == max_score,'features']
                                        
                    fs_results.to_csv(self.project_path + data_name +'_feature_selection_results.csv', encoding='utf-8')
                    return_objs[mode][data_name] = [fs_results,max_score,opt_features]
                    
                if mode == 'method_comparison':
                    mc_results = None
                    methods = list(self.estimators.keys())
                    if 'None' in methods:
                        methods.remove('None')

                    for method in self.compare_methods: 
                        
                        if self.grid_search_on == 1: 
                            [gs_opt_params, validation_cv_score, gs_results] = self.ml.grid_search(self.estimators[method], self.data[data_name])
                            test_score = self.ml.run_test_prediction(self.estimators[method], self.data[data_name])[-1]
                            alm_fun.show_msg(self.log,self.verbose,method + ' - cv:' + str(validation_cv_score) + ' test:' + str(test_score) + " parameters:" + str(gs_opt_params))
                        else:
                            cv_result = self.ml.run_cv_prediction(self.estimators[method], self.data[data_name])
                            test_result = self.ml.run_test_prediction(self.estimators[method], self.data[data_name])                                                 
                            validation_cv_score = cv_result['validation_cv_result']      
                            validation_cv_score.index = [method]
                            validation_cv_score.columns = ['cv_' + x for x in validation_cv_score.columns]
                            
                            test_score = test_result['test_bs_result']
                            test_score.index = [method]
                            test_score.columns = ['test_' + x for x in test_score.columns]                            
                            
                        if  mc_results is None:                  
                            mc_results = pd.concat([validation_cv_score, test_score], axis=1)
                        else:
                            mc_results = pd.concat([mc_results, pd.concat([validation_cv_score, test_score], axis=1)]) 
                            
                        
                    mc_results.to_csv(self.project_path + data_name +'_method_comparison_results.csv', encoding='utf-8')  
                    alm_fun.show_msg(self.log,self.verbose,str(mc_results))
                    return_objs[mode][data_name] = mc_results
                    
                if mode == 'grid_search':
                    [gs_opt_params, gs_opt_score, gs_results] = self.ml.grid_search(self.estimators[self.run_estimator_name], self.data[data_name])
                    max_score = gs_opt_score['mean'].get_values()[0]
                    gs_results.to_csv(self.project_path + 'grid_search_results.csv', encoding='utf-8')
                    return_objs[mode][data_name] = [gs_results,max_score,gs_opt_params]
            
                if mode == 'feature_comparison':
                    #******************************************
                    # train_features can be nested list
                    #******************************************
#                     column_name = [data_name + '_' + self.run_estimator_name +'_cv_' + self.estimators[self.run_estimator_name].score_name,
#                                    data_name + '_' + self.run_estimator_name +'_cv_' + self.estimators[self.run_estimator_name].score_name + '_ste',
#                                    data_name + '_' + self.run_estimator_name +'_test_' + self.estimators[self.run_estimator_name].score_name,
#                                    data_name + '_' + self.run_estimator_name +'_test_' + self.estimators[self.run_estimator_name].score_name + '_ste']
# 
#                     fc_results = pd.DataFrame(np.empty((len(self.compare_features)+2,4))*np.nan,index = [self.train_features_name,self.start_features_name] + self.compare_features_name, columns = [column_name])      
#                     fc_results = pd.DataFrame(np.empty((len(self.compare_features)+1,2))*np.nan,index = ['start_features'] + self.compare_features_name, columns = column_name)
                    fc_results = None
                    fc_predictions = None
                    if len(self.start_features) == 0:
                        alm_fun.show_msg(self.log,self.verbose,'start_features: N/A')
                        # fc_results.loc[self.start_features_name,column_name] = np.nan
                    else:   
                        self.data[data_name].train_features = self.start_features
                        if self.grid_search_on == 1:                        
                            [gs_opt_params, validation_cv_score, gs_results] = self.ml.grid_search(self.estimators[self.run_estimator_name], self.data[data_name])
                            test_score = self.ml.run_test_prediction(self.estimators[self.run_estimator_name], self.data[data_name],self.data[data_name].cur_test_split_fold,self.data[data_name].cur_gradient_key,self.data[data_name].if_engineer)[-1]
                            alm_fun.show_msg(self.log,self.verbose,'start_features - cv:' + str(validation_cv_score['mean']) + ' ' + str(validation_cv_score['ste']) + ' test:' + str(test_score['mean']) + ' ' + str(test_score['ste']) + " parameters:" + str(gs_opt_params))
                        else:       
                            cv_result = self.ml.run_cv_prediction(self.estimators[self.run_estimator_name], self.data[data_name],'',self.data[data_name].cur_test_split_fold,self.data[data_name].cur_gradient_key,self.data[data_name].if_engineer)
                            test_result = self.ml.run_test_prediction(self.estimators[self.run_estimator_name], self.data[data_name],self.data[data_name].cur_test_split_fold,self.data[data_name].cur_gradient_key,self.data[data_name].if_engineer)                                                 
                            validation_cv_scores = cv_result[-3]      
                            test_scores = test_result[-2]
                            test_predicitons = pd.Series(test_result[0], name='start_features')
                            validation_cv_scores.index = ['start_features']
                            validation_cv_scores.columns = ['cv_' + x for x in validation_cv_scores.columns]
                            test_scores.index = ['start_features']
                            test_scores.columns = ['test_' + x for x in test_scores.columns] 
                            validation_cv_score = cv_result[-1]      
                            test_score = test_result[-1]
#                             alm_fun.show_msg(self.log,self.verbose,'start_features - cv:' + str(validation_cv_score['mean']) + ' ' + str(validation_cv_score['ste'])  + ' test:' + str(test_score['mean']) + ' ' + str(test_score['ste']))
                        if  fc_results is None:                  
                            fc_results = pd.concat([validation_cv_scores, test_scores], axis=1)
                        else:
                            fc_results = pd.concat([fc_results, pd.concat([validation_cv_scores, test_scores], axis=1)])   
                                                    
                        if  fc_predictions is None:                  
                            fc_predictions = test_predicitons
                        else:
                            fc_predictions = pd.concat([fc_predictions, test_predicitons], axis=1)                         
                    if len(self.train_features) == 0:
                        alm_fun.show_msg(self.log,self.verbose,'all_features: N/A')
#                         fc_results.loc[self.train_features_name,column_name] = np.nan
                    else:   
                        self.data[data_name].train_features = self.train_features
                        if self.grid_search_on == 1:                        
                            [gs_opt_params, validation_cv_score, gs_results] = self.ml.grid_search(self.estimators[self.run_estimator_name], self.data[data_name])
                            test_score = self.ml.run_test_prediction(self.estimators[self.run_estimator_name], self.data[data_name],self.data[data_name].cur_test_split_fold,self.data[data_name].cur_gradient_key,self.data[data_name].if_engineer)[-1]
#                             alm_fun.show_msg(self.log,self.verbose,'all_features - cv:' + str(validation_cv_score['mean']) + ' ' + str(validation_cv_score['ste'])  + ' test:' + str(test_score['mean']) + ' ' + str(test_score['ste']) + " parameters:" + str(gs_opt_params))
                        else:
                            cv_result = self.ml.run_cv_prediction(self.estimators[self.run_estimator_name], self.data[data_name])
                            test_result = self.ml.run_test_prediction(self.estimators[self.run_estimator_name], self.data[data_name])                                                 
                            validation_cv_scores = cv_result[-3]      
                            test_scores = test_result[-2]
                            test_predicitons = pd.Series(test_result[0], name='all_features')
                            validation_cv_scores.index = ['all_features']
                            validation_cv_scores.columns = ['cv_' + x for x in validation_cv_scores.columns]
                            test_scores.index = ['all_features']
                            test_scores.columns = ['test_' + x for x in test_scores.columns] 
                            validation_cv_score = cv_result[-1]      
                            test_score = test_result[-1]
#                             alm_fun.show_msg(self.log,self.verbose,'all_features - cv:' + str(validation_cv_score['mean']) + ' ' + str(validation_cv_score['ste'])  + ' test:' + str(test_score['mean']) + ' ' + str(test_score['ste']))  
                        if  fc_results is None:                  
                            fc_results = pd.concat([validation_cv_scores, test_scores], axis=1)
                        else:
                            fc_results = pd.concat([fc_results, pd.concat([validation_cv_scores, test_scores], axis=1)])
                            
                        if  fc_predictions is None:                  
                            fc_predictions = test_predicitons
                        else:
                            fc_predictions = pd.concat([fc_predictions, test_predicitons], axis=1)   
#                         fc_results.loc[self.train_features_name,column_name] = [validation_cv_score['mean'],validation_cv_score['ste'],test_score['mean'],test_score['ste']]                                             
                    for i in range(len(self.compare_features)):
                        if self.feature_compare_direction == 0:
                            self.data[data_name].train_features = self.start_features + [self.compare_features[i]]
                        else:
                            compare_features_copy = self.compare_features.copy()   
                            compare_features_copy = self.remove_features(compare_features_copy, i)   
                            self.data[data_name].train_features = compare_features_copy

                        if self.grid_search_on == 1:                        
                            [gs_opt_params, validation_cv_score, gs_results] = self.ml.grid_search(self.estimators[self.run_estimator_name], self.data[data_name])
                            test_score = self.ml.run_test_prediction(self.estimators[self.run_estimator_name], self.data[data_name])
                            alm_fun.show_msg(self.log,self.verbose,self.compare_features_name[i] + ' - cv:' + str(validation_cv_score['mean']) + ' ' + str(validation_cv_score['ste']) + ' test:' + str(test_score['mean']) + ' ' + str(test_score['ste']) + " parameters:" + str(gs_opt_params))
                        else:
                            cv_result = self.ml.run_cv_prediction(self.estimators[self.run_estimator_name], self.data[data_name])
                            test_result = self.ml.run_test_prediction(self.estimators[self.run_estimator_name], self.data[data_name])                                                 
                            validation_cv_scores = cv_result[-3]      
                            test_scores = test_result[-2]
                            test_predicitons = pd.Series(test_result[0], name=self.compare_features_name[i])
                            validation_cv_scores.index = [self.compare_features_name[i]]
                            validation_cv_scores.columns = ['cv_' + x for x in validation_cv_scores.columns]
                            test_scores.index = [self.compare_features_name[i]]
                            test_scores.columns = ['test_' + x for x in test_scores.columns]                         
                            validation_cv_score = cv_result[-1]      
                            test_score = test_result[-1]                            
                        if  fc_results is None:                  
                            fc_results = pd.concat([validation_cv_scores, test_scores], axis=1)
                        else:
                            fc_results = pd.concat([fc_results, pd.concat([validation_cv_scores, test_scores], axis=1)])  
                            alm_fun.show_msg(self.log,self.verbose,self.compare_features_name[i])                              
                        if  fc_predictions is None:                  
                            fc_predictions = test_predicitons
                        else:
                            fc_predictions = pd.concat([fc_predictions, test_predicitons], axis=1)                                                      
#                             alm_fun.show_msg(self.log,self.verbose,self.compare_features_name[i] +' - cv:' + str() + ' ' + str(validation_cv_score['ste'])  + ' test:' + str(test_score['mean']) + ' ' + str(test_score['ste']))                        
#                         fc_results.loc[self.compare_features_name[i],column_name] = [validation_cv_score['mean'][0],validation_cv_score['ste'][0],test_score['mean'][0],test_score['ste'][0]]    
                    alm_fun.show_msg(self.log,self.verbose,fc_results[[x  for x in fc_results.columns if 'ste' not in x]])
                    fc_results.to_csv(self.project_path + data_name +'_feature_comparison_results' + '_fold_' + str(self.data[data_name].cur_test_split_fold) + '_' + str(self.data[data_name].cur_gradient_key) +'.csv')
                    
                    if self.data[data_name].if_engineer:
                        predition_labels = self.data[data_name].test_splits_engineered_df[self.data[data_name].cur_test_split_fold][self.data[data_name].cur_gradient][self.data[data_name].dependent_variable]                                    
                    else:
                        predition_labels = self.data[data_name].test_data_index_df.loc[self.data[data_name].test_splits_df[self.data[data_name].cur_test_split_fold][self.data[data_name].cur_gradient_key],self.data[data_name].dependent_variable]   
                        
                    fc_predictions = pd.concat([fc_predictions, predition_labels], axis=1)  
                    fc_predictions.to_csv(self.project_path + data_name +'_fc_predictions' + '_fold_' + str(self.data[data_name].cur_test_split_fold) + '_' + str(self.data[data_name].cur_gradient_key) +'.csv',index = False)
                    auprc_plotname = self.project_path + 'output/' + mode + '_' + data_name + '_fold_' + str(self.data[data_name].cur_test_split_fold) + '_' + str(self.data[data_name].cur_gradient_key) + '_auprc.png'
                    auroc_plotname = self.project_path + 'output/' + mode + '_' + data_name + '_fold_' + str(self.data[data_name].cur_test_split_fold) + '_' + str(self.data[data_name].cur_gradient_key) + '_auroc.png'
                    alm_fun.plot_prc(predition_labels, fc_predictions[self.compare_features_name_forplot], auprc_plotname, 20, 10, None, 0.9, 0.9, 'AUPRC Comparison')
                    alm_fun.plot_roc(predition_labels, fc_predictions[self.compare_features_name_forplot], auroc_plotname, 20, 10, None, 0.9, 0.9, 'AUROC Comparison')

                    return_objs[mode][data_name] = fc_results
                etime2 = time.time()
                alm_fun.show_msg(self.log,self.verbose,"Class: [alphame_project] Fun: [run] -- Current Mode: " + "[" + mode + "]" + " Current Data: " + data_name + ", running time was %g seconds" % (etime2 - stime2))                 
#                 if (mode != 'prediction') & (mode != 'cross_validation') & (mode != 'gradient_comparison') :                
# #                    plot for figure for the current result   
#                    plot_column_names = [return_objs[mode][data_name].columns[x] for x in self.plot_columns]
# #                    sort by the first column in the plot
#                    return_objs[mode][data_name] = return_objs[mode][data_name].sort_values(return_objs[mode][data_name].columns[self.plot_columns[0]])                             
#                    self.project_plot(return_objs[mode][data_name][plot_column_names], mode, data_name, x_label=mode, y_label=self.ml_score, ylim_max=self.plot_vmax, ylim_min=self.plot_vmin, fig_w=self.fig_w, fig_h=self.fig_h)
         
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
        
