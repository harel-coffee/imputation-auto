import sys
import os
import funregressor
import numpy as np
import pandas as pd
import pickle
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import seaborn as sns
import subprocess
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
import keras.backend as K 
from sklearn import tree
from sklearn import preprocessing
from setuptools.dist import Feature
from xgboost import plot_tree
from xgboost import plot_importance
warnings.filterwarnings("ignore")
python_path = '/usr/local/projects/ml/python/'
project_path = '/usr/local/projects/imputation/gwt/www/'
humandb_path = '/usr/local/database/humandb/'
sys.path.append(python_path)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
os.environ["PATH"] += os.pathsep + '/usr/local/Cellar/graphviz/2.40.1/bin/'
def scatter_plots(ax,x,y,x_target_name,y_target_name,hue,hue_name,title_extra,marker_size = 160 ):
    if hue is None:
        ax.scatter(x, y, cmap = 'Blues',s = marker_size)
    else:
        ax.scatter(x, y, c = hue, cmap = 'Blues',s = marker_size)
   
    ax.set_title(x_target_name +' VS ' + y_target_name + ' [pcc:' + str(round(alphame_ml.pcc_cal(x,y),3)) + 
                     '][spc:' + str(round(alphame_ml.spc_cal(x,y),3)) + '][color: ' + hue_name + '] ' + title_extra,size = 20)
    ax.set_ylabel(y_target_name,size = 15)
    ax.set_xlabel(x_target_name,size = 15) 
    ax.tick_params(size = 20)
    return(ax)

dict_dms_uniportid = {}
dict_dms_uniportid['P63279'] = 'UBE2I'
dict_dms_uniportid['P63165'] = 'SUMO1'
dict_dms_uniportid['Q9H3S4'] = 'TPK1'
dict_dms_uniportid['P0DP23'] = 'CALM1'
dict_dms_uniportid['P62166'] = 'NCS1'        
dict_dms_uniportid['P35520'] = 'CBS'
dict_dms_uniportid['Q9NZ01'] = 'TECR'
dict_dms_uniportid['P31150'] = 'GDI1'
dict_dms_uniportid['P42898'] = 'MTHFR'
dict_dms_uniportid['P04035'] = 'HMGCR'

#***************************************************************************************************************************************************************
# FunRegressor initialization
#***************************************************************************************************************************************************************


#project init parameters for alm_project class 
project_params = {}
project_params['project_name'] = 'funregressor'
project_params['project_path'] = '/usr/local/projects/imputation/gwt/www/'
project_params['humandb_path'] = '/usr/local/database/humandb/'
project_params['verbose'] = 1

#the reason the following parameters don't belong to data class is we may want to create multiple data instance in one project instance 
project_params['data_name'] = ['dms'] 
project_params['train_data'] = [project_params['humandb_path'] + 'funregressor/funregressor_training_final.csv']        
project_params['test_data'] = [project_params['humandb_path'] + 'funregressor/funregressor_test_final_processed.csv']
project_params['target_data'] = [project_params['humandb_path'] + 'funregressor/funregressor_test_final.csv']
project_params['extra_train_data'] = [project_params['humandb_path'] + 'funregressor/funregressor_test_final.csv']
project_params['use_extra_train_data'] = [0]
project_params['input_data_type'] = ['file']

project_params['run_data_names'] = ['dms']
project_params['run_estimator_name'] = 'xgb_r'
project_params['run_estimator_scorename'] = 'rmse'
project_params['grid_search_on'] = 0

project_params['modes'] = None
project_params['train_features'] = None
project_params['train_features'] = None
project_params['train_features_name'] = None
project_params['start_features'] = None
project_params['start_features_name'] = None  
project_params['compare_features'] = None
project_params['compare_features_name'] = None
project_params['compare_features_name_forplot'] = None
project_params['feature_compare_direction'] = 0 
project_params['compare_methods'] = None 

project_params['plot_columns'] = [0, 1]
project_params['plot_vmin'] = 0.5
project_params['plot_vmax'] = 1
project_params['fig_w'] = 20
project_params['fig_h'] = 5


#data init parameters for alm_data class
data_params = {}
data_params['name'] = None 
data_params['target_data_original_df'] = None
data_params['train_data_original_df'] = None
data_params['test_data_original_df'] = None                
data_params['extra_train_data_original_df'] = None 
data_params['use_extra_train_data'] =  None
data_params['path'] = project_params['project_path']  
data_params['log'] = open(project_params['project_path'] + 'log/alm_data[' + project_params['project_name'] + '].log', 'w') 
data_params['verbose'] = 1
data_params['independent_testset'] = 0
data_params['cv_split_method'] = 0
data_params['test_split_method'] = 0
data_params['test_split_folds'] = 5
# data_params['test_split_ratio'] = 0.2
data_params['cv_split_folds'] = 5
data_params['validation_from_testset'] = False
data_params['percent_min_feature'] = 1

data_params['dependent_variable'] = 'fitness'
data_params['filter_target'] = 1
data_params['filter_test'] = 0
data_params['filter_train'] = 0
data_params['filter_validation'] = 1
data_params['prediction_bootstrapping'] = 0
data_params['bootstrapping_num'] = 3

data_params['if_gradient'] = 0
data_params['if_engineer'] = 0
data_params['load_from_disk'] = 0
data_params['save_to_disk'] = 0
data_params['cur_test_split_fold'] = 0
data_params['cur_gradient_key'] = 'no_gradient'

data_params['onehot_features'] = []

#ml init parameters for alm_ml class
ml_params = {}
ml_params['log'] = open(project_params['project_path'] + 'log/alm_ml[' + project_params['project_name'] + '].log', 'w') 
ml_params['verbose'] = 1
ml_params['run_grid_search'] = 0
ml_params['fs_start_features'] = []
ml_params['fs_T'] = 0.001
ml_params['fs_alpha'] = 0.8
ml_params['fs_K'] = 100
ml_params['fs_epsilon'] = 0.00001 

#es init parameters for es_ml class
es_params = {}
es_params['ml_type'] = 'regression'
es_params['single_feature_as_prediction'] = 1
es_params['estimator'] = None
es_params['name'] = None
es_params['gs_range'] = None
es_params['score_name'] = None
es_params['score_direction'] = None
es_params['feature_importance_name'] = None
es_params['round_digits'] = 4

#FunRegressor level parameters for funregressor class
funregressor_params = {}
funregressor_params['project_extra'] = 0
funregressor_params['run_data_preprocess'] = 0
funregressor_params['humandb_path'] = '/usr/local/database/humandb/'
funregressor_params['quality_cutoff'] = 0
funregressor_params['pos_importance'] = 1
funregressor_params['project_params'] = project_params
funregressor_params['data_params'] = data_params
funregressor_params['ml_params'] = ml_params
funregressor_params['es_params'] = es_params

#create a funregressor instance 
fr_proj = funregressor.funregressor(funregressor_params)

#***************************************************************************************************************************************************************
# Pre-defined features
#***************************************************************************************************************************************************************
envision_features = fr_proj.aa_name_features + fr_proj.envision_features
all_features_no_scores = ['blosum100','in_domain','asa_mean','accessibility']  + \
                        fr_proj.aa_psipred_features + fr_proj.aa_name_features + \
                        fr_proj.aa_physical_ref_features + fr_proj.aa_physical_alt_features + \
                        fr_proj.kmer_name_features + fr_proj.kmer_physical_features + fr_proj.aa_physical_delta_features 
all_features_all_scores = ['polyphen_new_score','provean_new_score','sift_new_score','evm_epistatic_score','envi_delta_psic']  + all_features_no_scores
all_features_four_scores = ['polyphen_new_score','provean_new_score','sift_new_score','envi_delta_psic']  + all_features_no_scores
all_features_three_scores = ['polyphen_new_score','provean_new_score','envi_delta_psic']  + all_features_no_scores
all_features_with_polyphen = ['polyphen_new_score']  + all_features_no_scores
all_features_with_provean = ['provean_new_score']  + all_features_no_scores
all_features_with_evmutation = ['provean_new_score']  + all_features_no_scores

selected_features_no_scores_prc = ['pkb_alt', 'blosum100', 'mw_delta', 'vadw_alt', 'asa_mean', 'pbr_10_delta', 'asa_ref', 'pka_alt', 'avbr_100_alt', 'hi_alt', 'hbond_alt', 'size_ref', 'positive_ref', 'aliphatic_delta', 'aa_ref_encode', 'aliphatic_alt', 'accessibility', 'hydrophobic_alt','pka_ref', 'pbr_alt', 'polar_ref', 'asa_100_delta'] + fr_proj.aa_psipred_features
selected_features_no_scores_rfp = ['pkb_alt', 'charge_ref', 'avbr_100_alt', 'negative_delta', 'ionizable_delta', 'hydrophobic_delta', 'ionizable_alt', 'aa_psipred_encode', 'funsum_fitness_mean_aa_asa', 'avbr_100_delta', 'pkb_ref', 'hydrophobic_alt', 'accessibility', 'pbr_delta', 'asa_alt', 'ionizable_ref', 'asa_mean', 'hi_delta']
selected_features_all_scores_prc = ['aliphatic_delta', 'negative_delta', 'pka_delta', 'polyphen_new_score', 'sulfur_ref', 'vadw_100_delta', 'pbr_10_delta', 'pbr_10_alt', 'evm_epistatic_score', 'accessibility', 'pka_alt', 'pkb_delta', 'sulfur_delta', 'pkb_alt', 'avbr_ref', 'hbond_alt', 'hydrophobic_ref', 'sift_new_score', 'asa_100_ref', 'pbr_ref']
selected_features_all_scores_rfp = ['aliphatic_delta', 'negative_delta', 'pka_delta', 'polyphen_new_score', 'sulfur_ref', 'vadw_100_delta', 'pbr_10_delta', 'pbr_10_alt', 'evm_epistatic_score', 'accessibility', 'pka_alt', 'pkb_delta', 'sulfur_delta', 'pkb_alt', 'avbr_ref', 'hbond_alt', 'hydrophobic_ref', 'sift_new_score', 'asa_100_ref', 'pbr_ref']
selected_features_four_scores = ['ionizable_ref', 'asa_mean', 'envi_delta_psic', 'hbond_ref', 'hydrophobic_alt', 'aa_ref_encode', 'pbr_alt', 'avbr_alt', 'essential_ref', 'sift_new_score', 'pka_ref', 'blosum100']
selected_features_three_scores = ['aromatic_alt', 'avbr_alt', 'hbond_alt', 'positive_ref', 'accessibility', 'pbr_ref', 'ionizable_ref', 'pka_ref', 'hi_alt', 'in_domain', 'charge_alt', 'asa_alt', 'sulfur_alt', 'avbr_ref', 'hydrophobic_ref', 'pbr_alt', 'envi_delta_psic', 'mw_ref', 'avbr_100_alt', 'essential_alt', 'pka_alt']
selected_features_with_provean = ['provean_new_score']  + all_features_no_scores
selected_features_with_polyphen= ['polyphen_new_score']  + all_features_no_scores

# funregressor_debug_features = ['pkb_alt', 'blosum100', 'mw_delta', 'vadw_alt', 'pbr_10_delta', 'asa_ref', 'pka_alt', 'avbr_100_alt', 'hi_alt', 'hbond_alt', 'size_ref', 'positive_ref', 'aliphatic_delta', 'aa_ref_encode', 'aliphatic_alt', 'accessibility', 'hydrophobic_alt', 'pka_ref', 'pbr_alt', 'polar_ref', 'asa_100_delta']
funregressor_debug_features = ['pkb_alt', 'blosum100', 'mw_delta', 'vadw_alt', 'pbr_10_delta', 'asa_ref', 'pka_alt', 'avbr_100_alt', 'hi_alt', 'hbond_alt', 'size_ref', 'positive_ref', 'aliphatic_delta', 'aliphatic_alt', 'accessibility', 'hydrophobic_alt',  'pka_ref', 'pbr_alt', 'polar_ref', 'asa_100_delta'] + fr_proj.aa_name_features + fr_proj.kmer_name_features + fr_proj.aa_psipred_features
funregressor_debug_features = ['blosum100','in_domain','asa_mean','accessibility','funsum_fitness_mean', 'funsum_fitness_mean_in_domain']  + \
                        fr_proj.aa_psipred_features + fr_proj.aa_name_features + fr_proj.aa_physical_ref_features + fr_proj.aa_physical_alt_features + fr_proj.aa_physical_delta_features + \
                        fr_proj.kmer_name_features + fr_proj.kmer_physical_features
selected_features = selected_features_four_scores  


all_features_no_scores = ['blosum100','in_domain','accessibility']  + \
                        fr_proj.aa_psipred_features + fr_proj.aa_name_features + \
                        fr_proj.aa_physical_ref_features + fr_proj.aa_physical_alt_features + \
                        fr_proj.kmer_name_features + fr_proj.kmer_physical_features + fr_proj.aa_physical_delta_features 
#***************************************************************************************************************************************************************
# Project RUN
#***************************************************************************************************************************************************************
gs_opt_params = {'learning_rate': 0.04, 'max_depth': 3, 'n_estimators': 100}
gc_opt_quality_cutoff = 250
gc_opt_pos_importance = 0.72

#*************************************************************
# STEP0: methods comparison
#*************************************************************
#bulid kreas estimator
def pcc_loss_function(y_true, y_pred):
    l = K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))
#     l =  0-K.mean((y_true - K.mean(y_true))*(y_pred - K.mean(y_pred)),axis = -1)/(K.std(y_true)*K.std(y_pred))
    return l
 
def keras_model():
    dimension = len(all_features_no_scores)    
    # create model
    model = Sequential()
    model.add(Dense(dimension, input_dim=dimension, kernel_initializer='normal', activation='relu'))
    model.add(Dense(10, kernel_initializer='normal', activation='relu'))
    model.add(Dense(10, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss = pcc_loss_function, optimizer='adam')
    return model
keras_regressor_estimator = KerasRegressor(build_fn=keras_model, epochs=10, batch_size=10, verbose=1)
fr_proj.project.estimators['keras_r'].estimator = keras_regressor_estimator          
 
fr_proj.project.data['dms'].filter_train = 1
fr_proj.project.data['dms'].filter_test = 1
fr_proj.project.data['dms'].save_to_disk = 1
fr_proj.project.estimators['xgb_r'].estimator.max_depth = 5
fr_proj.project.estimators['xgb_r'].estimator.n_estimators = 200
fr_proj.project.estimators['dct_r'].estimator.max_depth = 5
# fr_proj.project.compare_methods = ['keras_r','en_r','dct_r','rf_r','xgb_r']
fr_proj.project.compare_methods = ['dct_r','rf_r']
fr_proj.project.modes = ['method_comparison']
fr_proj.project.train_features = all_features_no_scores
fr_proj.project.train_features_name = 'all_features'

run_return = fr_proj.project_run()
  
# tree.export_graphviz(fr_proj.project.estimators['dct_r_c'].estimator,project_params['project_path'] + 'dct.dot',feature_names = fr_proj.project.train_features )
# subprocess.run(['dot', '-Tpng', 'dct.dot', '-o', 'dct.png', '-Gdpi=300'], cwd = project_params['project_path'])
 
 
feature_imp = pd.DataFrame()
  
feature_imp['feature_name'] = fr_proj.project.train_features
feature_imp['dct_r'] = fr_proj.project.estimators['dct_r'].estimator.feature_importances_
# feature_imp['en_r_c'] = fr_proj.project.estimators['en_r_c'].estimator.coef_
# feature_imp['xgb_r_c'] = fr_proj.project.estimators['xgb_r_c'].estimator.feature_importances_
feature_imp['rf_r'] = fr_proj.project.estimators['rf_r'].estimator.feature_importances_
feature_imp.sort_values(['rf_r'])

print ('OK')

#******************************************************************************************************************************************************************
# STEP1: Feature selection using local search (default hyper-parameters) 
#******************************************************************************************************************************************************************
# project_run_params['load_from_disk'] = 0
# project_run_params['if_gradient'] = 0
# project_run_params['save_to_disk'] = 1
# project_run_params['grid_search_on'] = 0
# project_run_params['run_estimator_name'] = 'xgb_r_c'
# project_run_params['train_features'] = all_features_all_scores
# project_run_params['train_features_name'] = 'all features'
# project_run_params['modes'] = ['feature_selection']
# fs_return = fr_proj.project_run(feature_engineer_params, data_slice_params, data_split_params, project_run_params)
# fs_max_score = fs_return['feature_selection']['dms'][1]
# selected_features = fs_return['feature_selection']['dms'][2].get_values()[0]
# print ("\nFeature selection: best score is " + str(fs_max_score) + ", best features:" + str(selected_features) +"\n")
#                  
# #plot the feature selection 
# feature_selection_results = pd.read_csv(project_path + 'feature_selection_results.csv',index_col = False) 
# feature_selection_results_accepted = feature_selection_results.loc[feature_selection_results['accept'] == 'Yes',:]
# feature_selection_results_rejected= feature_selection_results.loc[feature_selection_results['accept'] == 'No',:]
# fig = plt.figure(figsize=(20,10))  
# ax = plt.subplot()
# ax.plot(feature_selection_results.index,feature_selection_results['score'],c = 'black',lw = 0.5)
# ax.scatter(feature_selection_results_rejected.index,feature_selection_results_rejected['score'],c = 'red',s = 25)
# ax.scatter(feature_selection_results_accepted.index,feature_selection_results_accepted['score'],c = 'green',s = 25)
# ax.set_xlabel('Local search iterations',size = 15)
# ax.set_ylabel('AUPRC',size  = 15)
# ax.set_title('FunRegressor feature selection',size = 20)   
# fig.tight_layout()
# plt.subplots_adjust(top=0.9)
# plt.savefig(project_path + 'feature_selection.png')


#******************************************************************************************************************************************************************
# STEP2: Training data gradient (with hyper-parameter tuning)
#******************************************************************************************************************************************************************
# project_run_params['load_from_disk'] = 1
# project_run_params['if_gradient'] = 1
# project_run_params['save_to_disk'] = 1
# project_run_params['grid_search_on'] = 1
# project_run_params['modes'] = ['gradient_comparison']
#    
# project_run_params['gradient_tolerance'] = 0.001 
# project_run_params['run_estimator_name'] = 'xgb_r_c'
# project_run_params['train_features'] =  all_features_no_scores
# gc_return = fr_proj.project_run(feature_engineer_params, data_slice_params, data_split_params, project_run_params)
#              
# gc_max_score = gc_return['gradient_comparison']['dms'][1]
# gc_opt_cutoff = gc_return['gradient_comparison']['dms'][2].get_values()[0]
# print ("\nGradient Comparision: best score is " + str(gc_max_score) + ", best cutoff:" + str(gc_opt_cutoff) + "\n")
# project_run_params['quality_cutoff'] = gc_opt_quality_cutoff
# project_run_params['pos_importance'] = gc_opt_pos_importance
#       
# gradient_comparison_results = pd.read_csv(project_path + 'gradient_comparison_results.csv')
# gradient_comparison_results = gradient_comparison_results.sort_values(['gradient'],ascending= False)
# gradient_comparison_results.reset_index(inplace = True) 
#                                         
# fig = plt.figure(figsize=(20,10))  
# ax = plt.subplot()
#      
# #  # plot quality cutoff
# #  ax.plot(gradient_comparison_results['gradient'],gradient_comparison_results['validation_score'],c = 'black',lw = 0.5)
# #  ax.scatter(gradient_comparison_results['gradient'],gradient_comparison_results['validation_score'],c = 'red',s = 15)
# #  ax.set_xlabel('Quality Score Cutoffs',size = 15)
# #  ax.set_ylabel('AUPRC',size  = 15)
# #  ax.set_title('Training data quality score cutoff selection',size = 20) 
# #  plt.subplots_adjust(top=0.9)
# #  plt.savefig(project_path + 'quality_socre_cutoff.png')
# #  
# #  # plot position importance 
# #  ax.plot(gradient_comparison_results['gradient'],gradient_comparison_results['validation_score'],c = 'black',lw = 0.5)
# #  ax.scatter(gradient_comparison_results['gradient'],gradient_comparison_results['validation_score'],c = 'red',s = 15)
# #  ax.set_xlabel('Position Importance Cutoffs',size = 15)
# #  ax.set_ylabel('AUPRC',size  = 15)
# #  ax.set_title('Training data position importance cutoff selection',size = 20) 
# #  plt.subplots_adjust(top=0.9)
# #  plt.savefig(project_path + 'position_importance.png')
# #       
# # plot quality cutoff and position importance
# gradient_comparison_results['pos_importance_graident'] = gradient_comparison_results['gradient'].apply(lambda x: round(float(x[1:-1].split(',')[0]),2))
# gradient_comparison_results['quality_cutoff_graident'] = gradient_comparison_results['gradient'].apply(lambda x: int(x[1:-1].split(',')[1]))
#   
# #  #Position importance
# #  for quality_cutoff in gradient_comparison_results['quality_cutoff_graident'].unique():
# #      cutoff_df = gradient_comparison_results.loc[gradient_comparison_results['quality_cutoff_graident'] == quality_cutoff,: ]
# #      cutoff_df = cutoff_df.sort_values(['pos_importance_graident'])
# #      ax.plot(cutoff_df['pos_importance_graident'],cutoff_df['validation_score'],c = 'black',lw = 0.5)
# #      ax.scatter(cutoff_df['pos_importance_graident'],cutoff_df['validation_score'],c = 'red',s = 15)
# #  pass
# #  ax.set_xlabel('Position Importance Cutoffs',size = 15)
# #  ax.set_ylabel('AUPRC on validation set',size  = 15)
# #  
# #  #quality cutoff 
# #  for pos_importance in gradient_comparison_results['pos_importance_graident'].unique():
# #      cutoff_df = gradient_comparison_results.loc[gradient_comparison_results['pos_importance_graident'] == pos_importance,: ]
# #      cutoff_df = cutoff_df.sort_values(['quality_cutoff_graident'])
# #      ax.plot(cutoff_df['quality_cutoff_graident'],cutoff_df['validation_score'],c = 'black',lw = 0.5)
# #      ax.scatter(cutoff_df['quality_cutoff_graident'],cutoff_df['validation_score'],c = 'red',s = 15)
# #  pass
# #  ax.set_xlabel('Quality Score Cutoffs',size = 15)
# #  ax.set_ylabel('AUPRC on validation set',size  = 15)
# #  
# # quality cutoff  and position importance
# ax.scatter(gradient_comparison_results['pos_importance_graident'],gradient_comparison_results['quality_cutoff_graident'],c = gradient_comparison_results['validation_score'], cmap = 'Blues',s = 100)
# ax.set_xlabel('Position Importance Cutoffs',size = 15)
# ax.set_ylabel('Quality Score Cutoffs',size  = 15)
# 
# 
# ax.set_title('Training data Filtering',size = 20) 
# plt.subplots_adjust(top=0.9)
# plt.savefig(project_path + 'quality_cutoff.png')
# 
# # plot individual protein training 
# ax.bar(gradient_comparison_results['gradient'],gradient_comparison_results['validation_score'],color = 'royalblue')
# for i in range(gradient_comparison_results.shape[0]):
#     ax.text(i*50 ,gradient_comparison_results.loc[i,'validation_score'] + 0.01, str(gradient_comparison_results.loc[i,'validation_score']) + "(" + str(gradient_comparison_results.loc[i,'train_samplesize']) + ")", color='royalblue', fontweight='bold',size = 12)
# pass
# ax.set_xlabel('Individual MAVE protein',size = 15)
# ax.set_ylabel('AUPRC',size  = 15)
# ax.set_title('FunRegressor individual MAVE data performance',size = 20)   
# plt.subplots_adjust(top=0.9)
# plt.savefig(project_path + 'individual_protein_performance.png')
# 
# print("OK")

#******************************************************************************************************************************************************************
# STEP3: HyperParameter Tuning
#******************************************************************************************************************************************************************
# project_run_params['load_from_disk'] = 1
# project_run_params['run_estimator_name'] = 'xgb_r_c'
# project_run_params['train_features'] = selected_features
# project_run_params['train_features_name'] = 'selected features'
# project_run_params['modes'] = ['grid_search']
# gs_return = fr_proj.project_run(feature_engineer_params, data_slice_params, data_split_params, project_run_params)
#       
# gs_max_score = gs_return['grid_search']['dms'][1]
# gs_opt_params = gs_return['grid_search']['dms'][2]
# print ("\nGrid Search: best score is " + str(gs_max_score) + ", best parameters:" + str(gs_opt_params) + "\n")
# fr_proj.project.estimators[project_run_params['run_estimator_name']].estimator.set_params(**gs_opt_params)

#******************************************************************************************************************************************************************
# STEP4: Feature Importance
#******************************************************************************************************************************************************************
# project_run_params['load_from_disk'] = 1
# project_run_params['run_estimator_name'] = 'xgb_r_c'
# project_run_params['train_features'] = selected_features_all_scores
# project_run_params['train_features_name'] = 'all features'
# project_run_params['modes'] = ['cross_validation']
# cv_return = fr_proj.project_run(feature_engineer_params, data_slice_params, data_split_params, project_run_params)
# feature_importance_df = cv_return['cross_validation']['dms'][2].transpose()
# feature_importance_df.columns = ['Gain']
# feature_importance_df.to_csv(project_path + 'feature_importance.csv')
#       
# # plot feature importance
# feature_importance_df = pd.read_csv(project_path + 'feature_importance.csv')
# feature_importance_df.columns = ['feature','gain']
# feature_importance_df = feature_importance_df.loc[feature_importance_df['gain'].notnull(),:]
# feature_importance_df = feature_importance_df.sort_values(['gain'],ascending = False)
# fig = plt.figure(figsize=(20,10))  
# ax = plt.subplot()
# ax.bar(feature_importance_df['feature'],feature_importance_df['gain'],color = 'royalblue',lw = 0.5)
# # ax.scatter(feature_selection_results_rejected.index,feature_selection_results_rejected['score'],c = 'red',s = 15)
# # ax.scatter(feature_selection_results_accepted.index,feature_selection_results_accepted['score'],c = 'green',s = 15)
# ax.set_xlabel('Features',size = 15)
# ax.set_ylabel('Average Gain',size  = 15)
# ax.set_title('FunRegressor feature importance',size = 20)   
# fig.tight_layout()
# plt.xticks(rotation=10)
# plt.subplots_adjust(top=0.9)
# plt.savefig(project_path + 'feature_importance.png')
# print("OK")

#******************************************************************************************************************************************************************
# SETP5: Performance comparison 
#******************************************************************************************************************************************************************
# #funregressor class run parameters
# # fr_proj.quality_cutoff = gc_opt_quality_cutoff
# # fr_proj.pos_importance = gc_opt_pos_importance
# fr_proj.project.estimators[fr_proj.project.run_estimator_name].estimator.set_params(**gs_opt_params)
# 
# # fr_proj.project.data['dms'].filter_train = 1
# # fr_proj.project.data['dms'].filter_test = 1
# 
# #project class run parameters
# fr_proj.project.cur_test_split_fold = 0
# fr_proj.project.run_estimator_name = 'xgb_r_c'
# fr_proj.project.feature_compare_direction = 0
# fr_proj.project.modes = ['feature_comparison']
# fr_proj.project.train_features = ['polyphen_new_score']
# fr_proj.project.train_features_name = 'all features'
# fr_proj.project.start_features = []
# fr_proj.project.train_features_name = 'all features'
# fr_proj.project.start_features_name = 'Start feature'                        
# fr_proj.project.compare_features = [selected_features_all_scores_prc,selected_features_all_scores_rfp,all_features_all_scores,selected_features_four_scores,all_features_four_scores,selected_features_three_scores,all_features_three_scores,selected_features_with_provean,all_features_with_provean,selected_features_with_polyphen,all_features_with_polyphen,selected_features_no_scores_prc,selected_features_no_scores_rfp,all_features_no_scores,['evm_epistatic_score'], envision_features,['envi_Envision_predictions'], ['polyphen_new_score'], ['provean_new_score'],['sift_new_score'],['envi_delta_psic']]
# fr_proj.project.compare_features_name = ['FunRegressor_All_FS_prc','FunRegressor_All_FS_rfp','FunRegressor_All','FunRegressor_Four_FS','FunRegressor_Four','FunRegressor_Three_FS','FunRegressor_Three','FunRegressor_Provean_FS','FunRegressor_Provean','FunRegressor_Polyphen_FS','FunRegressor_Polyphen','FunRegressor_FS_prc','FunRegressor_FS_rfp','FunRegressor','EVmutation','Envision_features','Envision','Polyphen','Provean','SIFT','PSIC']
# fr_proj.project.compare_features_name_forplot = ['FunRegressor_All_FS_prc','FunRegressor_All_FS_rfp','FunRegressor_All','FunRegressor_Four_FS','FunRegressor_Four','FunRegressor_Three_FS','FunRegressor_Three','FunRegressor_Provean_FS','FunRegressor_Provean','FunRegressor_Polyphen_FS','FunRegressor_Polyphen','FunRegressor_FS_prc','FunRegressor_FS_rfp','FunRegressor','EVmutation','Envision_features','Envision','Polyphen','Provean','SIFT','PSIC']
#   
# # selected figure scores 
# fr_proj.project.compare_features = [selected_features_all_scores_prc,selected_features_no_scores_prc,funregressor_debug_features,['evm_epistatic_score'], ['polyphen_new_score'], ['provean_new_score'],['sift_new_score'],['envi_Envision_predictions'],['primateai_score'],['VEST3_selected_score'],['FATHMM_selected_score'],['MutationTaster_selected_score'],['CADD_raw'],['M-CAP_score'],['REVEL_score'],['MutPred_score'],['DANN_score'],['fathmm-MKL_coding_score'],['Eigen-raw'],['GenoCanyon_score'],['integrated_fitCons_score'],'GERP++_RS']
# fr_proj.project.compare_features_name = ['FunRegressor+','FunRegressor','FunRegressor-debug','EVmutation','Polyphen','Provean','SIFT','Envision','PrimateAI','VEST3','FATHMM','MutationTaster','CADD','M-CAP','REVEL','MutPred','DANN','fathmm-MKL','Eigen','GenoCanyon','fitCons','GERP++']
# fr_proj.project.compare_features_name_forplot = ['FunRegressor+','FunRegressor','FunRegressor-debug','EVmutation','Polyphen','Provean','SIFT','Envision','PrimateAI','VEST3','FATHMM','MutationTaster','CADD','M-CAP','REVEL','MutPred','DANN','fathmm-MKL','Eigen','GenoCanyon','fitCons','GERP++']
#  
# # selected figure scores 
# fr_proj.project.compare_features = [selected_features_all_scores_prc,selected_features_no_scores_prc,['polyphen_new_score'], ['provean_new_score'],['sift_new_score'],['evm_epistatic_score'],['envi_Envision_predictions'],['primateai_score'],['MutationTaster_selected_score'],['CADD_raw']]
# fr_proj.project.compare_features_name = ['FunRegressor+','FunRegressor','Polyphen','Provean','SIFT','EVmutation','Envision','PrimateAI','MutationTaster2','CADD']
# fr_proj.project.compare_features_name_forplot = ['FunRegressor+','FunRegressor','Polyphen','Provean','SIFT','EVmutation','Envision','PrimateAI','MutationTaster2','CADD']
# 
# # selected figure scores 
# # fr_proj.project.compare_features = [all_features_no_scores,all_features_with_evmutation,selected_features_all_scores_prc,['polyphen_new_score'], ['provean_new_score'],['sift_new_score'],['evm_epistatic_score'],['envi_Envision_predictions'],['primateai_score'],['MutationTaster_selected_score'],['CADD_raw']]
# # fr_proj.project.compare_features_name = ['FunRegressor','FunRegresspr_EVmutation','FunRegressor_FS','Polyphen','Provean','SIFT','EVmutation','Envision','PrimateAI','MutationTaster2','CADD']
# # fr_proj.project.compare_features_name_forplot = ['FunRegressor','FunRegresspr_EVmutation','FunRegressor_FS','Polyphen','Provean','SIFT','EVmutation','Envision','PrimateAI','MutationTaster2','CADD']
# # #   
# #    
# 
# # selected figure scores 
# # project_run_params['compare_features'] = [selected_features_all_scores_prc,selected_features_no_scores_prc,['polyphen_new_score'], ['provean_new_score'],['sift_new_score'],['evm_epistatic_score'],['envi_Envision_predictions'],['primateai_score'],['MutationTaster_selected_score'],['CADD_raw'],['M-CAP_score']]
# # project_run_params['compare_features_name'] = ['FunRegressor+','FunRegressor','Polyphen','Provean','SIFT','EVmutation','Envision','PrimateAI','MutationTaster2','CADD','M-CAP']
# # project_run_params['compare_features_name_forplot'] = ['FunRegressor+','FunRegressor','Polyphen','Provean','SIFT','EVmutation','Envision','PrimateAI','MutationTaster2','CADD','M-CAP']
# 
#            
# #data class run parameters
# fr_proj.project.data['dms'].load_from_disk = 0
# fr_proj.project.data['dms'].save_to_disk = 0
# fr_proj.project.data['dms'].if_gradient = 1
# fr_proj.project.data['dms'].cur_test_split_fold = 0
# fr_proj.project.data['dms'].cur_gradient_key = 'no_gradient'
# 
# for i in range(fr_proj.project.data['dms'].test_split_folds):
#     fr_proj.project.cur_test_split_fold = i
#     run_return = fr_proj.project_run()
# print("OK")


#******************************************************************************************************************************************************************
# STEP7: Plot Gradient Boost Tree
#******************************************************************************************************************************************************************
# project_run_params['load_from_disk'] = 1
# project_run_params['run_estimator_name'] = 'xgb_r_c'
# project_run_params['train_features'] = selected_features_all_scores
# project_run_params['train_features_name'] = 'selected features'
# project_run_params['modes'] = ['prediction']
# prediction_return = fr_proj.project_run(feature_engineer_params, data_slice_params, data_split_params, project_run_params)
# num_trees = 1
# plot_tree(fr_proj.project.estimators[project_run_params['run_estimator_name']].estimator,num_trees = num_trees)
# plt.show()
# plt.savefig(project_path + 'plot_tree_' + str(num_trees) + '.png',dpi = 600)



#******************************************************************************************************************************************************************
# STEP8: Training/Test data analysis (CBS performs very well)
#******************************************************************************************************************************************************************
# train_df = pd.read_csv(humandb_path + 'dms/funregressor_training_final.csv',index_col = False)
# train_dms_genes = ['P63279','P63165','P62166','Q9H3S4','P0DP23','Q9NZ01','P31150','P42898','P35520']
# # train_df = train_df.loc[(train_df['aa_ref'] != train_df['aa_alt']) & train_df['fitness'].notnull(), :]
# gene_info = pd.DataFrame(columns = ['gene','gene_name','sample_size','asa_count','in_domain_count','out_domain_count','provean_count','polyphen_count','sift_count','envision_count','evmutation_count','provean_pcc','polyphen_pcc','sift_pcc','provean_spc','polyphen_spc','sift_spc'])
# for dms_gene in train_dms_genes:
#     cur_gene_info_dict = {}
#     cur_gene_info_dict['gene'] = dms_gene
#     cur_gene_info_dict['gene_name'] =  dict_dms_uniportid.get(dms_gene,np.nan)
#     cur_gene_info_dict['sample_size'] = train_df.loc[train_df['p_vid'] == dms_gene,:].shape[0]
#     cur_gene_info_dict['asa_count'] = train_df.loc[train_df['asa_mean'].notnull() & (train_df['p_vid'] == dms_gene),:].shape[0]
#     cur_gene_info_dict['in_domain_count'] = train_df.loc[(train_df['in_domain'] == 1) & (train_df['p_vid'] == dms_gene),:].shape[0]
#     cur_gene_info_dict['out_domain_count'] = train_df.loc[(train_df['in_domain'] == 0) & (train_df['p_vid'] == dms_gene),:].shape[0]
#      
#     cur_gene_info_dict['provean_count'] = train_df.loc[train_df['provean_new_score'].notnull() & (train_df['p_vid'] == dms_gene),:].shape[0]
#     cur_gene_info_dict['polyphen_count'] = train_df.loc[train_df['polyphen_new_score'].notnull() & (train_df['p_vid'] == dms_gene),:].shape[0]
#     cur_gene_info_dict['sift_count'] = train_df.loc[train_df['sift_new_score'].notnull() & (train_df['p_vid'] == dms_gene),:].shape[0]
#     cur_gene_info_dict['envision_count'] = train_df.loc[train_df['envi_Envision_predictions'].notnull() & (train_df['p_vid'] == dms_gene),:].shape[0]
#     cur_gene_info_dict['evmutation_count'] = train_df.loc[train_df['evm_epistatic_score'].notnull() & (train_df['p_vid'] == dms_gene),:].shape[0]
#      
#     cur_gene_info_dict['provean_pcc'] = alphame_ml.pcc_cal(train_df.loc[(train_df['p_vid'] == dms_gene),'fitness'],train_df.loc[(train_df['p_vid'] == dms_gene),'provean_new_score'])    
#     cur_gene_info_dict['polyphen_pcc'] = alphame_ml.pcc_cal(train_df.loc[(train_df['p_vid'] == dms_gene),'fitness'],train_df.loc[(train_df['p_vid'] == dms_gene),'polyphen_new_score'])
#     cur_gene_info_dict['sift_pcc'] = alphame_ml.pcc_cal(train_df.loc[(train_df['p_vid'] == dms_gene),'fitness'],train_df.loc[(train_df['p_vid'] == dms_gene),'sift_new_score'])
#     cur_gene_info_dict['provean_spc'] = alphame_ml.spc_cal(train_df.loc[(train_df['p_vid'] == dms_gene),'fitness'],train_df.loc[(train_df['p_vid'] == dms_gene),'provean_new_score'])
#     cur_gene_info_dict['polyphen_spc'] = alphame_ml.spc_cal(train_df.loc[(train_df['p_vid'] == dms_gene),'fitness'],train_df.loc[(train_df['p_vid'] == dms_gene),'polyphen_new_score'])
#     cur_gene_info_dict['sift_spc'] = alphame_ml.spc_cal(train_df.loc[(train_df['p_vid'] == dms_gene),'fitness'],train_df.loc[(train_df['p_vid'] == dms_gene),'sift_new_score'])
#     gene_info = gene_info.append(cur_gene_info_dict,ignore_index = True)   
# pass   
# print (gene_info)
# 
# test_df = pd.read_csv(humandb_path + 'clinvar/csv/clinvar_plus_gnomad_final_funregressor.csv')
# # test_df = test_df.loc[test_df['clinvar_gene'] == 1,:]
# test_dms_genes = test_df['p_vid'].unique()
# gene_info = pd.DataFrame(columns = ['gene','gene_name','sample_size','clinvar_gene_pathogenic_count','clinvar_gene_benign_count','asa_count','in_domain_count','out_domain_count','provean_count','polyphen_count','sift_count','envision_count','evmutation_count','provean_pcc','polyphen_pcc','sift_pcc','provean_spc','polyphen_spc','sift_spc'])
# cur_gene_info_dict = {}
# cur_gene_info_dict['gene'] = "ALL"
# cur_gene_info_dict['gene_name'] =  "ALL"
# cur_gene_info_dict['sample_size'] = test_df.shape[0]
# cur_gene_info_dict['clinvar_gene_pathogenic_count'] = test_df.loc[(test_df['clinvar_gene'] == 1) & (test_df['fitness'] == 1),:].shape[0]
# cur_gene_info_dict['clinvar_gene_benign_count'] = test_df.loc[(test_df['clinvar_gene'] == 1) & (test_df['fitness'] == 0),:].shape[0]
# cur_gene_info_dict['asa_count'] = test_df.loc[test_df['asa_mean'].notnull(),:].shape[0]
# cur_gene_info_dict['in_domain_count'] = test_df.loc[(test_df['in_domain'] == 1),:].shape[0]
# cur_gene_info_dict['out_domain_count'] = test_df.loc[(test_df['in_domain'] == 0),:].shape[0]
#  
# cur_gene_info_dict['provean_count'] = test_df.loc[test_df['provean_new_score'].notnull(),:].shape[0]
# cur_gene_info_dict['polyphen_count'] = test_df.loc[test_df['polyphen_new_score'].notnull(),:].shape[0]
# cur_gene_info_dict['sift_count'] = test_df.loc[test_df['sift_new_score'].notnull(),:].shape[0]
# cur_gene_info_dict['envision_count'] = test_df.loc[test_df['envi_Envision_predictions'].notnull(),:].shape[0]
# cur_gene_info_dict['evmutation_count'] = test_df.loc[test_df['evm_epistatic_score'].notnull(),:].shape[0]
#  
# cur_gene_info_dict['provean_pcc'] = alphame_ml.pcc_cal(test_df['fitness'],test_df['provean_new_score'])    
# cur_gene_info_dict['polyphen_pcc'] = alphame_ml.pcc_cal(test_df['fitness'],test_df['polyphen_new_score'])
# cur_gene_info_dict['sift_pcc'] = alphame_ml.pcc_cal(test_df['fitness'],test_df['sift_new_score'])
# cur_gene_info_dict['provean_spc'] = alphame_ml.spc_cal(test_df['fitness'],test_df['provean_new_score'])
# cur_gene_info_dict['polyphen_spc'] = alphame_ml.spc_cal(test_df['fitness'],test_df['polyphen_new_score'])
# cur_gene_info_dict['sift_spc'] = alphame_ml.spc_cal(test_df['fitness'],test_df['sift_new_score'])
# gene_info = gene_info.append(cur_gene_info_dict,ignore_index = True)   
# 
# print (gene_info)

#*********************************************************************************
# STEP9: Feature Interaction
#*********************************************************************************
# project_run_params['run_estimator_name'] = 'xgb_r_c'
# project_run_params['train_features'] = otherscores
# project_run_params['train_features_name'] = 'all features'
# project_run_params['start_features'] = []
# project_run_params['start_features_name'] = 'Start feature'  
# 
# 
# polyphen_provean = ['polyphen_new_score','provean_new_score']
# polyphen_sift = ['polyphen_new_score','sift_new_score']
# polyphen_evmutation = ['polyphen_new_score','evm_epistatic_score']
# provean_sift = ['provean_new_score','sift_new_score']
# provean_evmutation = ['provean_new_score','evm_epistatic_score']
# sift_evmutation = ['sift_new_score','evm_epistatic_score']
# 
# provean_sift_evmutation = ['provean_new_score', 'sift_new_score','evm_epistatic_score']
# polyphen_provean_evmutation = ['polyphen_new_score','provean_new_score','evm_epistatic_score']
# polyphen_sift_evmutation = ['polyphen_new_score', 'sift_new_score','evm_epistatic_score']
# polyphen_sift_provean = ['provean_new_score', 'sift_new_score','polyphen_new_score']
#     
# project_run_params['compare_features'] = [['evm_epistatic_score'],['polyphen_new_score'], ['provean_new_score'], ['sift_new_score'],polyphen_provean,polyphen_sift,polyphen_evmutation,provean_sift,provean_evmutation,sift_evmutation,provean_sift_evmutation,polyphen_provean_evmutation,polyphen_sift_evmutation,polyphen_sift_provean]
# project_run_params['compare_features_name'] = ['evmutation','polyphen','provean','sift','polyphen_provean','polyphen_sift','polyphen_evmutation','provean_sift','provean_evmutation','sift_evmutation','provean_sift_evmutation','polyphen_provean_evmutation','polyphen_sift_evmutation','polyphen_sift_provean']
# project_run_params['compare_features_name_forplot'] = ['evmutation','polyphen','provean','sift','polyphen_provean','polyphen_sift','polyphen_evmutation','provean_sift','provean_evmutation','sift_evmutation','provean_sift_evmutation','polyphen_provean_evmutation','polyphen_sift_evmutation','polyphen_sift_provean']
# 
# project_run_params['feature_compare_direction'] = 0
# project_run_params['modes'] = ['feature_comparison']
# run_return = fr_proj.project_run(feature_engineer_params, data_slice_params, data_split_params, project_run_params)
# print("OK")



#*********************************************************************************
# STEP11: MAVE data Performance for all DMS maps 
#*********************************************************************************
# funregressor_training_final = pd.read_csv(humandb_path + 'dms/funregressor_training_final.csv')
#merger with clinvar and ganomad
# clinvar = pd.read_csv(humandb_path + 'clinvar/csv/clinvar_srv.csv')
# funregressor_training_final = pd.merge(funregressor_training_final,clinvar[['p_vid','aa_ref','aa_alt','aa_pos','clinsig_level','review_star','label']],how = 'left')
#  
# gnomad_aa = pd.read_csv(humandb_path  + 'gnomad/gnomad_output_snp_aa_uniprot.txt', sep='\t', dtype={"chr": str})
# gnomad_aa['p_vid'] = gnomad_aa['uniprot_id']
# funregressor_training_final = pd.merge(funregressor_training_final,gnomad_aa[['p_vid', 'aa_pos', 'aa_ref', 'aa_alt','gnomad_af','gnomad_gc_homo_alt']],how = 'left')
#  
# funregressor_training_final.to_csv(humandb_path + 'dms/funregressor_training_final.csv',index = False)

# all_dms_df = pd.read_csv(humandb_path + 'dms/funregressor_training_final.csv')
# all_dms_df = all_dms_df.loc[all_dms_df['aa_ref'] != all_dms_df['aa_alt']]
# all_dms_df = all_dms_df.loc[all_dms_df['aa_alt'] != '*']
# all_dms_df = all_dms_df.loc[all_dms_df['fitness'].notnull(),:]
# 
#  
# all_dms_df.loc[(all_dms_df['gnomad_gc_homo_alt'] > 1) & (all_dms_df['gnomad_af'] <0.0001) , 'label'] = 0
# dms_columns = ['p_vid','aa_ref','aa_alt','aa_pos','fitness','polyphen_new_score','provean_new_score','sift_new_score','gnomad_af','gnomad_gc_homo_alt','review_star','clinsig_level','label']
# all_dms_benign = all_dms_df.loc[(all_dms_df['label'] == 0) & all_dms_df['review_star'].isnull() , dms_columns] 
# all_dms_delterious = all_dms_df.loc[(all_dms_df['label'] == 1) & (all_dms_df['review_star'] >= 2), dms_columns] 
# 
# all_dms_clinvar_gnomad = pd.concat([all_dms_benign,all_dms_delterious])
# 
# features_for_plot = ['fitness','polyphen_new_score','provean_new_score','sift_new_score']
# 
# #
# auprc_plotname = project_path + 'output/all_dms_auprc.png'
# auroc_plotname = project_path + 'output/all_dms_auroc.png'
# alphame_ml.plot_prc(all_dms_clinvar_gnomad['label'], all_dms_clinvar_gnomad[features_for_plot], auprc_plotname, 20, 10, None, 0.7, 0.9, 'AUPRC Comparison')
# alphame_ml.plot_roc(all_dms_clinvar_gnomad['label'], all_dms_clinvar_gnomad[features_for_plot], auroc_plotname, 20, 10, None, 0.7, 0.9, 'AUROC Comparison')


