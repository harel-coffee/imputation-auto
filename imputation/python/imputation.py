#!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3
import sys
import numpy as np
import pandas as pd
import random
import smtplib
from email.message import EmailMessage
import matplotlib
from cmath import inf
matplotlib.use('Agg') 
import matplotlib.pyplot as plt 
import matplotlib.path as mpath
import seaborn as sns
import matplotlib.patches as patches  
from matplotlib.lines import Line2D  
from matplotlib.gridspec import GridSpec
import matplotlib.collections as collections
import codecs
import re
import math
import pickle
import gensim
import time
import traceback
import warnings
import glob
import os
import subprocess
from scipy import stats
import xgboost as xgb
from sklearn import feature_selection as fs
from sklearn import linear_model as lm
from sklearn import neighbors as knn
from sklearn import ensemble
from datetime import datetime
from statistics import mode

python_path = '/usr/local/projects/ml/python/'
project_path = '/usr/local/projects/imputation/project/'
sys.path.append(python_path)
import alm_project
import alm_fun
sns.set(rc={'axes.facecolor':'#C0C0C0'}) 
warnings.filterwarnings("ignore")

class imputation:
    def __init__(self, imputation_params):
        for key in imputation_params:
            setattr(self, key, imputation_params[key])
        alm_fun.show_msg (self.log,self.verbose,'Class: [imputation] [__init__]......starts @' + str(datetime.now()))  
         
        self.lst_nt = ['A', 'T', 'C', 'G']
        self.lst_aa = ["S", "A", "V", "R", "D", "F", "T", "I", "L", "K", "G", "Y", "N", "C", "P", "E", "M", "W", "H", "Q", "U", "*", '_']
        self.lst_aa_21 = ["S", "A", "V", "R", "D", "F", "T", "I", "L", "K", "G", "Y", "N", "C", "P", "E", "M", "W", "H", "Q", "*"]
        self.lst_aa_20 = ["S", "A", "V", "R", "D", "F", "T", "I", "L", "K", "G", "Y", "N", "C", "P", "E", "M", "W", "H", "Q"]
        self.lst_aa3 = ["Ser", "Ala", "Val", "Arg", "Asp", "Phe", "Thr", "Ile", "Leu", "Lys", "Gly", "Tyr", "Asn", "Cys", "Pro", "Glu", "Met", "Trp", "His", "Gln", "Sec", "Ter", 'Unk']
        self.lst_aaname = ["Serine", "Alanine", "Valine", "Arginine", "Asparitic Acid", "Phenylalanine", "Threonine", "Isoleucine", "Leucine", "Lysine", "Glycine", "Tyrosine", "Asparagine", "Cysteine", "Proline", "Glutamic Acid", "Methionine", "Tryptophan", "Histidine", "Glutamine", "Selenocysteine", "Stop", "Unknown"]
        self.lst_chr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13''14', '15', '16', '17', '18', '19', '20', '21', '22', 'X', 'Y', 'MT']

        self.dict_aaencode = {}
        for i in range(len(self.lst_aa)):
            self.dict_aaencode[self.lst_aa[i]] = i
        
        #data preprocessing and update data_params              
        self.data_init()
        self.es_params['feature_engineer'] = self.feature_engineer
        self.project = alm_project.alm_project(self.project_params,self.data_params,self.ml_params,self.es_params)
        self.feature_init()  
        self.create_time = time.time()        
        alm_fun.show_msg (self.log,self.verbose,'Class: [imputation] Fun: [__init__] .... done @' + str(datetime.now()) + "\n")                      
                 
    def data_init(self):
        alm_fun.show_msg (self.log,self.verbose,'Class: [imputation] Fun: [data_init] .... starts @' + str(datetime.now()))
        self.data_params['feature_engineer'] = self.feature_engineer
        self.data_params['data_slice'] = self.data_slice
        self.data_params['cv_split'] = self.cv_split
        self.data_params['test_split'] = self.test_split
        self.data_params['gradient_reshape'] = self.gradient_reshape
        self.data_params['setup_gradients'] = self.setup_gradients
        
        all_dms_df = None
        if self.run_data_preprocess == 1:            
            for i in range(len(self.dms_landscape_files)):
                [cur_train_dms_df, cur_target_dms_df, cur_extra_train_dms_df] = self.data_preprocess(self.dms_landscape_files[i], self.dms_fasta_files[i], self.dms_protein_ids[i], self.data_names[i], self.raw_processed[i], self.normalized_flags[i], self.proper_num_replicates[i], self.reverse_flags[i], self.floor_flags[i],self.quality_cutoffs[i])                
                self.project_params['data_names'].append(self.data_names[i])
                self.project_params['target_data'].append(cur_target_dms_df)
                self.project_params['test_data'].append(pd.DataFrame(columns=cur_train_dms_df.columns))
                self.project_params['train_data'].append(cur_train_dms_df)                
                self.project_params['extra_train_data'].append(cur_extra_train_dms_df)  
                self.project_params['extra_train_data'].append(cur_extra_train_dms_df)
                self.project_params['use_extra_train_data'].append(0)
                self.project_params['input_data_type'].append('dataframe')             
                # create combined dms 
                if self.combine_dms == 1:
                    if self.combine_flags[i] == 1:
#                         cur_combin_dms_df = cur_train_dms_df.copy()
#                         cur_combin_dms_df['gene_name'] = self.data_params['data_name'][i]
                        if all_dms_df is None:
                            all_dms_df = cur_train_dms_df
                        else:
                            all_dms_df = pd.concat([all_dms_df, cur_train_dms_df])
            if all_dms_df is not None:
                all_dms_df = all_dms_df.reset_index()
                all_dms_df['clinvar_gene'] = 0
                all_dms_df.to_csv(self.humandb_path + 'dms/all_dms_maps')   
        else:
            self.project_params['target_data'] = [self.project_path + 'output/' + x + '_target' for x in self.dms_landscape_files] + [self.humandb_path + 'clinvar/csv/clinvar_final.csv']    
            self.project_params['test_data'] = [self.project_path + 'output/' + x + '_test' for x in self.dms_landscape_files] + [self.humandb_path + 'dms/all_dms_maps'] 
            self.project_params['train_data'] = [self.project_path + 'output/' + x + '_train' for x in self.dms_landscape_files] + [self.humandb_path + 'dms/all_dms_maps']            
            self.project_params['extra_train_data'] = [self.project_path + 'output/' + x + '_train' for x in self.dms_landscape_files] + [self.humandb_path + 'dms/all_dms_maps']
            self.project_params['use_extra_train_data'] = [0] * (len(self.dms_landscape_files) + 1)
            project_params['data_name'] = self.data_names + ['all_dms_maps']
            self.data_params['input_data_type'] = ['file'] * (len(self.dms_landscape_files) + 1)
        alm_fun.show_msg (self.log,self.verbose,'Class: [imputation] Fun: [data_init] .... done @' + str(datetime.now()) + "\n")  

    def imputation_rawdata_process_old(self, dms_gene_raw_fitness, data_name): 
        
        ####*************************************************************************************************************************************************************
        # raw data analysis
        ####*************************************************************************************************************************************************************            
        dms_gene_raw_fitness = dms_gene_raw_fitness.groupby(['wt_aa', 'mut_aa', 'pos', 'annotation'])['nonselect1', 'nonselect2', 'select1', 'select2', 'controlNS1', 'controlNS2', 'controlS1', 'controlS2'].sum().reset_index()
        
        cur_data_idx = self.data_names.index(data_name)
        proper_replicate_counts = self.proper_num_replicates[cur_data_idx]                      
        for value_name in ['controlS', 'controlNS']:
            value1 = value_name + '1'
            value2 = value_name + '2'
            value_mean = value_name + '_mean'
            value_sd = value_name + '_sd'
            value_count = value_name + '_count'
            value_pseudo_count = value_name + '_pseudo_count'
            value_mean_3sd_plus = value_name + '_mean+3sd'
            value_mean_3sd_minus = value_name + '_mean-3sd'
#             dms_gene_raw_fitness[value_mean] = dms_gene_raw_fitness[[value1,value2]].apply(lambda x: np.nanmean(x),axis = 1)
#             dms_gene_raw_fitness[value_sd] = dms_gene_raw_fitness[[value1,value2]].apply(lambda x: np.nanstd(x),axis = 1)
#             dms_gene_raw_fitness[value_count] = dms_gene_raw_fitness[[value1,value2]].apply(lambda x: 2-np.isnan(x).sum(),axis = 1)
            dms_gene_raw_fitness[value_mean] = dms_gene_raw_fitness[[value1, value2]].mean(axis=1)
            dms_gene_raw_fitness[value_sd] = dms_gene_raw_fitness[[value1, value2]].std(axis=1)
            dms_gene_raw_fitness[value_count] = dms_gene_raw_fitness[[value1, value2]].count(axis=1)
            dms_gene_raw_fitness[value_pseudo_count] = proper_replicate_counts - dms_gene_raw_fitness[value_count]
            dms_gene_raw_fitness[value_mean_3sd_plus] = dms_gene_raw_fitness[value_mean] + 3 * dms_gene_raw_fitness[value_sd]
            dms_gene_raw_fitness[value_mean_3sd_minus] = dms_gene_raw_fitness[value_mean] - 3 * dms_gene_raw_fitness[value_sd]
            
        # filter out the replicates when there are relative high sequencing error, nonselect      
        dms_gene_raw_fitness.loc[dms_gene_raw_fitness['nonselect1'] <= dms_gene_raw_fitness['controlNS_mean+3sd'], 'nonselect1'] = np.nan
        dms_gene_raw_fitness.loc[dms_gene_raw_fitness['nonselect2'] <= dms_gene_raw_fitness['controlNS_mean+3sd'], 'nonselect2'] = np.nan
        # filter out the replicates when there are relative high sequencing error, select
        dms_gene_raw_fitness.loc[dms_gene_raw_fitness['select1'] <= dms_gene_raw_fitness['controlS_mean-3sd'],'select1'] = np.nan
        dms_gene_raw_fitness.loc[dms_gene_raw_fitness['select2'] <= dms_gene_raw_fitness['controlS_mean-3sd'],'select2'] = np.nan
# 
#         #
#         dms_gene_raw_fitness.loc[dms_gene_raw_fitness['select1'].isnull(),'nonselect1'] = np.nan
#         dms_gene_raw_fitness.loc[dms_gene_raw_fitness['nonselect1'].isnull(),'select1'] = np.nan
#         dms_gene_raw_fitness.loc[dms_gene_raw_fitness['select2'].isnull(),'nonselect2'] = np.nan
#         dms_gene_raw_fitness.loc[dms_gene_raw_fitness['nonselect2'].isnull(),'select2'] = np.nan
                                    
        for value_name in ['select', 'nonselect']:
            value1 = value_name + '1'
            value2 = value_name + '2'
            value_mean = value_name + '_mean'
            value_sd = value_name + '_sd'
            value_count = value_name + '_count'
            value_pseudo_count = value_name + '_pseudo_count'
            dms_gene_raw_fitness[value_mean] = dms_gene_raw_fitness[[value1, value2]].mean(axis=1)
            dms_gene_raw_fitness[value_sd] = dms_gene_raw_fitness[[value1, value2]].std(axis=1)            
            dms_gene_raw_fitness[value_count] = dms_gene_raw_fitness[[value1, value2]].count(axis=1)
            # in the case value_count = 1 , we set std = 0
            dms_gene_raw_fitness.loc[dms_gene_raw_fitness[value_count] == 1, value_sd] = 0                         
            dms_gene_raw_fitness[value_pseudo_count] = proper_replicate_counts - dms_gene_raw_fitness[value_count]

        # B&L regularization on wt, nonselect and select condition               
        regression_df = dms_gene_raw_fitness.loc[dms_gene_raw_fitness['controlS_mean'].notnull() & dms_gene_raw_fitness['controlNS_mean'].notnull() & dms_gene_raw_fitness['select_mean'].notnull() & dms_gene_raw_fitness['nonselect_mean'].notnull()]
        regression_df.to_csv(self.project_path + 'output/' + data_name + '_raw_processed_regression_df.csv') 
        
        
        n = regression_df.shape[0] 
        regressor = xgb.XGBRegressor(**{'subsample': 0.8, 'colsample_bytree': 1, 'max_depth': 3, 'n_estimators': 100, 'learning_rate': 0.08})        
        for value_name in ['controlS', 'controlNS', 'select', 'nonselect']:
            value_mean = value_name + '_mean'
            value_sd = value_name + '_sd'
            value_sd_prior = value_name + '_sd_prior'
            value_sd_reg = value_name + '_sd_reg'
            value_se_reg = value_name + '_se_reg'
            value_count = value_name + '_count'
            value_pseudo_count = value_name + '_pseudo_count'
            notnull_idx = dms_gene_raw_fitness.index[dms_gene_raw_fitness[value_mean].notnull()]  
            regressor.fit(np.array(regression_df[value_mean]).reshape(n, 1), np.array(regression_df[value_sd]))
            dms_gene_raw_fitness.loc[notnull_idx, value_sd_prior] = regressor.predict(np.array(dms_gene_raw_fitness.loc[notnull_idx, [value_mean]]))
            dms_gene_raw_fitness.loc[notnull_idx, value_sd_reg] = np.sqrt((dms_gene_raw_fitness[value_pseudo_count] * (dms_gene_raw_fitness[value_sd_prior] ** 2) + (dms_gene_raw_fitness[value_count] - 1) * (dms_gene_raw_fitness[value_sd] ** 2)) / (dms_gene_raw_fitness[value_pseudo_count] + dms_gene_raw_fitness[value_count] - 1))
            dms_gene_raw_fitness.loc[notnull_idx, value_se_reg] = dms_gene_raw_fitness.loc[notnull_idx, value_sd_reg] / np.sqrt(proper_replicate_counts)
            
        # Error propagation  
        dms_gene_raw_fitness['select-control_mean'] = np.nan
        dms_gene_raw_fitness['select-control_se'] = np.nan
        dms_gene_raw_fitness['nonselect-control_mean'] = np.nan
        dms_gene_raw_fitness['nonselect-control_se'] = np.nan
        dms_gene_raw_fitness['foldchange'] = np.nan
        dms_gene_raw_fitness['foldchange_se'] = np.nan
        dms_gene_raw_fitness['logfoldchange'] = np.nan
        dms_gene_raw_fitness['logfoldchange_se'] = np.nan                          
        dms_gene_raw_fitness[['select-control_mean', 'select-control_se']] = dms_gene_raw_fitness[['select_mean', 'select_se_reg', 'controlS_mean', 'controlS_se_reg']].apply(lambda x: alm_fun.error_propagation_operation(x['select_mean'], x['select_se_reg'], x['controlS_mean'], x['controlS_se_reg'], 'subtraction'), axis=1)
        # add pseudocount to the select-control_mean if it is 0
        dms_gene_raw_fitness.loc[dms_gene_raw_fitness['select-control_mean'] <= 0 , 'select-control_mean'] = 1                
        dms_gene_raw_fitness[['nonselect-control_mean', 'nonselect-control_se']] = dms_gene_raw_fitness[['nonselect_mean', 'nonselect_se_reg', 'controlNS_mean', 'controlNS_se_reg']].apply(lambda x: alm_fun.error_propagation_operation(x['nonselect_mean'], x['nonselect_se_reg'], x['controlNS_mean'], x['controlNS_se_reg'], 'subtraction'), axis=1)                
        dms_gene_raw_fitness[['foldchange', 'foldchange_se']] = dms_gene_raw_fitness[['select-control_mean', 'select-control_se', 'nonselect-control_mean', 'nonselect-control_se']].apply(lambda x: alm_fun.error_propagation_operation(x['select-control_mean'], x['select-control_se'], x['nonselect-control_mean'], x['nonselect-control_se'], 'division'), axis=1)
        dms_gene_raw_fitness[['logfoldchange', 'logfoldchange_se']] = dms_gene_raw_fitness[['foldchange', 'foldchange_se']].apply(lambda x: alm_fun.error_propagation_fun(x['foldchange'], x['foldchange_se'], 'log'), axis=1)
        dms_gene_raw_fitness['logfoldchange_sd'] = dms_gene_raw_fitness['logfoldchange_se'] * np.sqrt(proper_replicate_counts)   
                   
        dms_gene_raw_fitness['fitness_input'] = dms_gene_raw_fitness['logfoldchange'] 
        dms_gene_raw_fitness['fitness_input_se'] = dms_gene_raw_fitness['logfoldchange_se']
        dms_gene_raw_fitness['fitness_input_sd'] = dms_gene_raw_fitness['logfoldchange_sd']
  
        dms_gene_raw_fitness.rename(columns={'wt_aa':'aa_ref', 'mut_aa':'aa_alt', 'pos':'aa_pos'}, inplace=True)        
        dms_gene_raw_fitness['num_replicates'] = proper_replicate_counts
        dms_gene_raw_fitness['quality_score'] = dms_gene_raw_fitness['nonselect-control_mean']
        dms_gene_raw_fitness.to_csv(self.project_path + 'output/' + data_name + '_raw_processed.csv')   
        dms_gene_fitness = dms_gene_raw_fitness.copy()
        return(dms_gene_fitness)
        
    def imputation_rawdata_process(self, dms_gene_raw_fitness, data_name): 
        def count_notnans(x):
            return sum(~np.isnan(x))
        
        def std_notnans(x):
            return np.nanstd(x,ddof = 1)
        
        ####*************************************************************************************************************************************************************
        # raw data analysis
        ####*************************************************************************************************************************************************************            
        dms_gene_raw_fitness = dms_gene_raw_fitness.groupby(['wt_aa', 'mut_aa', 'pos', 'annotation','replicate_id'])['select', 'nonselect', 'controlNS', 'controlS'].sum().reset_index()        
        dms_gene_raw_fitness_groupby = dms_gene_raw_fitness.groupby(['wt_aa', 'mut_aa', 'pos', 'annotation'])

        dms_gene_raw_fitness_controls = dms_gene_raw_fitness_groupby[['controlS', 'controlNS']].agg([np.nanmean, std_notnans,count_notnans]).reset_index()
        dms_gene_raw_fitness_controls.columns = ['wt_aa', 'mut_aa', 'pos', 'annotation','controlS_mean','controlS_sd','controlS_count','controlNS_mean','controlNS_sd','controlNS_count']
       
        cur_data_idx = self.data_names.index(data_name)
        proper_replicate_counts = self.proper_num_replicates[cur_data_idx]                      
        for value_name in ['controlS', 'controlNS']:
            value_mean = value_name + '_mean'
            value_sd = value_name + '_sd'
            value_count = value_name +'_count'
            value_pseudo_count = value_name + '_pseudo_count'
            value_mean_3sd_plus = value_name + '_mean+3sd'
            value_mean_3sd_minus = value_name + '_mean-3sd'
            
            dms_gene_raw_fitness_controls[value_pseudo_count] = proper_replicate_counts - dms_gene_raw_fitness_controls[value_count]
            dms_gene_raw_fitness_controls[value_mean_3sd_plus] = dms_gene_raw_fitness_controls[value_mean] + 3 * dms_gene_raw_fitness_controls[value_sd]
            dms_gene_raw_fitness_controls[value_mean_3sd_minus] = dms_gene_raw_fitness_controls[value_mean] - 3 * dms_gene_raw_fitness_controls[value_sd]
            
        dms_gene_raw_fitness = pd.merge(dms_gene_raw_fitness,dms_gene_raw_fitness_controls,how = 'left')  
        
        
        # filter out the replicates when there are relative high sequencing error, nonselect      
        dms_gene_raw_fitness.loc[dms_gene_raw_fitness['nonselect'] <= dms_gene_raw_fitness['controlNS_mean+3sd'], 'nonselect'] = np.nan
        # filter out the replicates when there are relative high sequencing error, select
        dms_gene_raw_fitness.loc[dms_gene_raw_fitness['select'] <= dms_gene_raw_fitness['controlS_mean-3sd'],'select'] = np.nan
     
                     
#         # filter out the replicates when there are relative high sequencing error, nonselect      
#         dms_gene_raw_fitness = dms_gene_raw_fitness.loc[dms_gene_raw_fitness['nonselect'] > dms_gene_raw_fitness['controlNS_mean+3sd'],:]        
#         # filter out the replicates when there are relative high sequencing error, select
#         dms_gene_raw_fitness = dms_gene_raw_fitness.loc[dms_gene_raw_fitness['select'] > dms_gene_raw_fitness['controlS_mean-3sd'],:]
#         
        dms_gene_raw_fitness_groupby = dms_gene_raw_fitness.groupby(['wt_aa', 'mut_aa', 'pos', 'annotation'])
        dms_gene_raw_fitness_values = dms_gene_raw_fitness_groupby[['select', 'nonselect']].agg([np.nanmean, std_notnans,count_notnans]).reset_index()
        dms_gene_raw_fitness_values.columns = ['wt_aa', 'mut_aa', 'pos', 'annotation','select_mean','select_sd','select_count','nonselect_mean','nonselect_sd','nonselect_count']
                                    
        for value_name in ['select', 'nonselect']:
            value_mean = value_name + '_mean'
            value_sd = value_name + '_sd'
            value_count = value_name + '_count'
            value_pseudo_count = value_name + '_pseudo_count'
            # in the case value_count = 1 , we set std = 0
            dms_gene_raw_fitness_values.loc[dms_gene_raw_fitness_values[value_count] == 1, value_sd] = 0                         
            dms_gene_raw_fitness_values[value_pseudo_count] = proper_replicate_counts - dms_gene_raw_fitness_values[value_count]

        dms_gene_raw_fitness = pd.merge(dms_gene_raw_fitness_values,dms_gene_raw_fitness_controls,how = 'left')

        # B&L regularization on wt, nonselect and select condition               
        regression_df = dms_gene_raw_fitness.loc[dms_gene_raw_fitness['controlS_mean'].notnull() & dms_gene_raw_fitness['controlNS_mean'].notnull() & dms_gene_raw_fitness['select_mean'].notnull() & dms_gene_raw_fitness['nonselect_mean'].notnull()]
#         regression_df.to_csv(self.project_path + 'output/' + data_name + '_raw_processed_regression_df.csv') 
        
        n = regression_df.shape[0] 
        regressor = xgb.XGBRegressor(**{'subsample': 0.8, 'colsample_bytree': 1, 'max_depth': 3, 'n_estimators': 100, 'learning_rate': 0.08})        
        for value_name in ['controlS', 'controlNS', 'select', 'nonselect']:
            value_mean = value_name + '_mean'
            value_sd = value_name + '_sd'
            value_sd_prior = value_name + '_sd_prior'
            value_sd_reg = value_name + '_sd_reg'
            value_se_reg = value_name + '_se_reg'
            value_count = value_name + '_count'
            value_pseudo_count = value_name + '_pseudo_count'
            notnull_idx = dms_gene_raw_fitness.index[dms_gene_raw_fitness[value_mean].notnull()]  
            regressor.fit(np.array(regression_df[value_mean]).reshape(n, 1), np.array(regression_df[value_sd]))
            dms_gene_raw_fitness.loc[notnull_idx, value_sd_prior] = regressor.predict(np.array(dms_gene_raw_fitness.loc[notnull_idx, [value_mean]]))
            dms_gene_raw_fitness.loc[notnull_idx, value_sd_reg] = np.sqrt((dms_gene_raw_fitness[value_pseudo_count] * (dms_gene_raw_fitness[value_sd_prior] ** 2) + (dms_gene_raw_fitness[value_count] - 1) * (dms_gene_raw_fitness[value_sd] ** 2)) / (dms_gene_raw_fitness[value_pseudo_count] + dms_gene_raw_fitness[value_count] - 1))
            dms_gene_raw_fitness.loc[notnull_idx, value_se_reg] = dms_gene_raw_fitness.loc[notnull_idx, value_sd_reg] / np.sqrt(proper_replicate_counts)
            
        # Error propagation  
        dms_gene_raw_fitness['select-control_mean'] = np.nan
        dms_gene_raw_fitness['select-control_se'] = np.nan
        dms_gene_raw_fitness['nonselect-control_mean'] = np.nan
        dms_gene_raw_fitness['nonselect-control_se'] = np.nan
        dms_gene_raw_fitness['foldchange'] = np.nan
        dms_gene_raw_fitness['foldchange_se'] = np.nan
        dms_gene_raw_fitness['logfoldchange'] = np.nan
        dms_gene_raw_fitness['logfoldchange_se'] = np.nan                          
        dms_gene_raw_fitness[['select-control_mean', 'select-control_se']] = dms_gene_raw_fitness[['select_mean', 'select_se_reg', 'controlS_mean', 'controlS_se_reg']].apply(lambda x: alm_fun.error_propagation_operation(x['select_mean'], x['select_se_reg'], x['controlS_mean'], x['controlS_se_reg'], 'subtraction'), axis=1)
        # add pseudocount to the select-control_mean if it is 0
        dms_gene_raw_fitness.loc[dms_gene_raw_fitness['select-control_mean'] <= 0 , 'select-control_mean'] = 1                
        dms_gene_raw_fitness[['nonselect-control_mean', 'nonselect-control_se']] = dms_gene_raw_fitness[['nonselect_mean', 'nonselect_se_reg', 'controlNS_mean', 'controlNS_se_reg']].apply(lambda x: alm_fun.error_propagation_operation(x['nonselect_mean'], x['nonselect_se_reg'], x['controlNS_mean'], x['controlNS_se_reg'], 'subtraction'), axis=1)                
        dms_gene_raw_fitness[['foldchange', 'foldchange_se']] = dms_gene_raw_fitness[['select-control_mean', 'select-control_se', 'nonselect-control_mean', 'nonselect-control_se']].apply(lambda x: alm_fun.error_propagation_operation(x['select-control_mean'], x['select-control_se'], x['nonselect-control_mean'], x['nonselect-control_se'], 'division'), axis=1)
        dms_gene_raw_fitness[['logfoldchange', 'logfoldchange_se']] = dms_gene_raw_fitness[['foldchange', 'foldchange_se']].apply(lambda x: alm_fun.error_propagation_fun(x['foldchange'], x['foldchange_se'], 'log'), axis=1)
        dms_gene_raw_fitness['logfoldchange_sd'] = dms_gene_raw_fitness['logfoldchange_se'] * np.sqrt(proper_replicate_counts)   
                   
        dms_gene_raw_fitness['fitness_input'] = dms_gene_raw_fitness['logfoldchange'] 
        dms_gene_raw_fitness['fitness_input_se'] = dms_gene_raw_fitness['logfoldchange_se']
        dms_gene_raw_fitness['fitness_input_sd'] = dms_gene_raw_fitness['logfoldchange_sd']
  
        dms_gene_raw_fitness.rename(columns={'wt_aa':'aa_ref', 'mut_aa':'aa_alt', 'pos':'aa_pos'}, inplace=True)        
        dms_gene_raw_fitness['num_replicates'] = proper_replicate_counts
        dms_gene_raw_fitness['quality_score'] = dms_gene_raw_fitness['nonselect-control_mean']
        dms_gene_raw_fitness.to_csv(self.project_path + 'output/' + data_name + '_raw_processed.csv')   
        dms_gene_fitness = dms_gene_raw_fitness.copy()
        return(dms_gene_fitness)
        


    def data_preprocess(self, dms_landscape_file, dms_fasta_file, dms_protein_id, data_name, raw_processed, normalized, proper_num_replicates, reverse_flag, floor_flag, quality_cutoff):
        
        def aa_encode_notnull(x):
            if x not in self.dict_aaencode.keys():
                return -1
            else:
                return self.dict_aaencode[x]
            
        alm_fun.show_msg (self.log,self.verbose,'Class: [imputation] Fun: [data_preprocess] starts @' + str(datetime.now()))
        alm_fun.show_msg (self.log,self.verbose,'Processing data [' + data_name +']......')
        
        ####*************************************************************************************************************************************************************
        # Process the raw data and save to files
        ####*************************************************************************************************************************************************************
        dms_feature_file = self.humandb_path + 'dms/features/' + dms_protein_id + '_features.csv'
#         dms_psipred_file = self.humandb_path + 'psipred/' + dms_protein_id + '.seq'
#         dms_pfam_file = self.humandb_path + 'pfam/' + dms_protein_id + '.pfam'        
        dms_seq_file = open(self.project_path + 'upload/' + dms_fasta_file, "r")
        dms_protein_aa = '' 
        for line in dms_seq_file: 
            line = line.replace('\n', '')
            if not re.match('>', line):
                dms_protein_aa += line
        if int(raw_processed) == 0:
            dms_gene_raw_fitness = pd.read_csv(self.project_path + 'upload/' + dms_landscape_file, sep='\t')
            dms_gene_fitness = self.imputation_rawdata_process(dms_gene_raw_fitness, data_name)
        else:   
#             dms_gene_fitness = pd.read_csv(self.project_path + 'upload/' + dms_landscape_file, sep='\t')[['wt', 'mut', 'pos', 'numberOfReplicate', 'averageFitnessScore', 'sdFitnessScore', 'averageNonselect']]
#             dms_gene_fitness.columns = ['aa_ref', 'aa_alt', 'aa_pos', 'num_replicates', 'fitness_input_filtered', 'fitness_input_filtered_sd', 'quality_score']
            dms_gene_fitness = pd.read_csv(self.project_path + 'upload/' + dms_landscape_file, sep='\t')
            dms_gene_fitness.loc[dms_gene_fitness['fitness_input_sd'].isnull(), 'fitness_input_sd'] = 0
            dms_gene_fitness.fillna(0)
            dms_gene_fitness['fitness_input_se'] = dms_gene_fitness['fitness_input_sd'] / np.sqrt(dms_gene_fitness['num_replicates'])   
              
        dms_gene_matrix = np.full((21, len(dms_protein_aa)), np.nan)                                                           
        dms_fasta_df = pd.DataFrame(dms_gene_matrix, columns=range(1, len(dms_protein_aa) + 1), index=self.lst_aa_21)
        dms_fasta_df['aa_alt'] = dms_fasta_df.index
        dms_fasta_df = pd.melt(dms_fasta_df, ['aa_alt'])
        dms_fasta_df = dms_fasta_df.rename(columns={'variable': 'aa_pos', 'value': 'aa_ref'})        
        dms_fasta_df['aa_ref'] = dms_fasta_df['aa_pos'].apply(lambda x: list(dms_protein_aa)[x - 1])
        dms_fasta_df['aa_pos'] = dms_fasta_df['aa_pos'].astype(int)
        dms_gene_fitness.loc[dms_gene_fitness['aa_alt'] == '_', 'aa_alt'] = '*'
#         self.imputation_web_log.write('dms_fasta_df: ' + str(dms_fasta_df.dtypes) + '\n')
#         self.imputation_web_log.write('dms_gene_fitness: ' + str(dms_gene_fitness.dtypes) + '\n')
        dms_gene_fitness = pd.merge(dms_fasta_df, dms_gene_fitness, how='left')
        
        #to save the feature space, process blosum, funsum and aa properties here 
        dms_gene_features = pd.read_csv(dms_feature_file)  
         ####***************************************************************************************************************************************************************
        #### aa_ref and aa_alt AA properties
        ####***************************************************************************************************************************************************************
        aa_properties_df = pd.read_csv(self.humandb_path + "dms/other_features/aa_properties.csv")          
        aa_properties_features = aa_properties_df.columns        
        aa_properties_ref_features = [x + '_ref' for x in aa_properties_features]
        aa_properties_alt_features = [x + '_alt' for x in aa_properties_features]   
        aa_properties_ref =aa_properties_df.copy()
        aa_properties_ref.columns = aa_properties_ref_features
        aa_properties_alt = aa_properties_df.copy()
        aa_properties_alt.columns = aa_properties_alt_features                
        dms_gene_features = pd.merge(dms_gene_features, aa_properties_ref, how='left')
        dms_gene_features = pd.merge(dms_gene_features, aa_properties_alt, how='left')
        
        ####***************************************************************************************************************************************************************
        #### merge with the blosum properties
        ####***************************************************************************************************************************************************************
        df_blosums = pd.read_csv(self.humandb_path + "dms/other_features/blosums.csv")         
        dms_gene_features = pd.merge(dms_gene_features, df_blosums, how='left')
        
        ####***************************************************************************************************************************************************************
        #### merge with the funsum properties
        ####***************************************************************************************************************************************************************
        funsum_df = pd.read_csv(self.humandb_path + "dms/other_features/funsum.csv")  
        dms_gene_features = pd.merge(dms_gene_features, funsum_df, how='left')
        ####*************************************************************************************************************************************************************
        #### Encode name features
        ####*************************************************************************************************************************************************************        
        dms_gene_features['aa_ref_encode'] = dms_gene_features['aa_ref'].apply(lambda x: aa_encode_notnull(x))
        dms_gene_features['aa_alt_encode'] = dms_gene_features['aa_alt'].apply(lambda x: aa_encode_notnull(x))
            
        dms_gene_df = pd.merge(dms_gene_fitness, dms_gene_features, how='left')
        dms_gene_df['pseudo_count'] = proper_num_replicates - dms_gene_df['num_replicates']
        dms_gene_df.loc[dms_gene_df.aa_ref == dms_gene_df.aa_alt, 'annotation'] = 'SYN'
        dms_gene_df.loc[dms_gene_df.aa_ref != dms_gene_df.aa_alt, 'annotation'] = 'NONSYN'
        dms_gene_df.loc[dms_gene_df.aa_alt == '*', 'annotation'] = 'STOP'      
        dms_gene_df['gene_name'] = data_name
         
        ####*************************************************************************************************************************************************************
        # set the fitness to nan when it is lower than the quality cutoff 
        ####*************************************************************************************************************************************************************
        dms_gene_df['fitness_input_filtered'] = dms_gene_df['fitness_input']
        dms_gene_df['fitness_input_filtered_sd'] = dms_gene_df['fitness_input_sd']
        dms_gene_df['fitness_input_filtered_se'] = dms_gene_df['fitness_input_se']
          
        dms_gene_df.loc[dms_gene_df['quality_score'] < quality_cutoff, 'fitness_input_filtered'] = np.nan
        dms_gene_df.loc[dms_gene_df['quality_score'] < quality_cutoff, 'fitness_input_filtered_sd'] = np.nan
        dms_gene_df.loc[dms_gene_df['quality_score'] < quality_cutoff, 'fitness_input_filtered_se'] = np.nan 
         
        # set the fitness to nan when it is lower than the sd cutoff 
#         dms_gene_df.loc[dms_gene_df['fitness_input_filtered_se'] > 4, 'fitness_input_filtered'] = np.nan
#         dms_gene_df.loc[dms_gene_df['fitness_input_filtered_se'] > 4, 'fitness_input_filtered_sd'] = np.nan
#         dms_gene_df.loc[dms_gene_df['fitness_input_filtered_se'] > 4, 'fitness_input_filtered_se'] = np.nan 
        ####*************************************************************************************************************************************************************
        # step2: fitness normalization
        ####*************************************************************************************************************************************************************
        if int(normalized) == 0:            
            stop_coordinates = []
            for region in self.stop_exclusion[self.data_names.index(data_name)].split(","):
                if ':' in region:
                    start = int(region.split(':')[0])
                    end = int(region.split(':')[1])
                    stop_coordinates = stop_coordinates + list(range(start,end+1))
                else:
                    stop_coordinates = stop_coordinates + [int(region)]
                   
            dms_gene_df['syn_filtered'] = 1
            syn_keep_index = (dms_gene_df['annotation'] == 'SYN') & dms_gene_df['fitness_input_filtered'].notnull() & (dms_gene_df['fitness_input_filtered_se'] < 1) & (dms_gene_df['quality_score'] > self.synstop_cutoffs[self.data_names.index(data_name)])
            dms_gene_df.loc[syn_keep_index,'syn_filtered'] = 0
            syn_median = np.median(dms_gene_df.loc[syn_keep_index,'fitness_input_filtered'])
                                
            dms_gene_df['stop_filtered'] = 1
            stop_keep_index = (dms_gene_df['annotation'] == 'STOP') & dms_gene_df['fitness_input_filtered'].notnull() & (dms_gene_df['fitness_input_filtered_se'] < 1) & (dms_gene_df['quality_score'] > self.synstop_cutoffs[self.data_names.index(data_name)]) & ~dms_gene_df['aa_pos'].isin(stop_coordinates)
            dms_gene_df.loc[stop_keep_index,'stop_filtered'] = 0
            stop_median = np.median(dms_gene_df.loc[stop_keep_index,'fitness_input_filtered'])
            
            dms_gene_df['fitness_org'] = (dms_gene_df['fitness_input_filtered'] - stop_median) / (syn_median - stop_median)
            dms_gene_df['fitness_sd_org'] = dms_gene_df['fitness_input_filtered_sd'] / (syn_median - stop_median) 
            dms_gene_df['fitness_se_org'] = dms_gene_df['fitness_input_filtered_se'] / (syn_median - stop_median)
                        
        else:
            dms_gene_df['fitness_org'] = dms_gene_df['fitness_input_filtered']
            dms_gene_df['fitness_sd_org'] = dms_gene_df['fitness_input_filtered_sd']
            dms_gene_df['fitness_se_org'] = dms_gene_df['fitness_input_filtered_se']
            
            dms_gene_df['syn_filtered'] = 1
            syn_keep_index = (dms_gene_df['annotation'] == 'SYN') & dms_gene_df['fitness_input_filtered'].notnull()
            dms_gene_df.loc[syn_keep_index,'syn_filtered'] = 0
            syn_median = np.median(dms_gene_df.loc[syn_keep_index,'fitness_input_filtered'])
                                
            dms_gene_df['stop_filtered'] = 1
            stop_keep_index = (dms_gene_df['annotation'] == 'STOP') & dms_gene_df['fitness_input_filtered'].notnull()
            dms_gene_df.loc[stop_keep_index,'stop_filtered'] = 0
                   
        ####*************************************************************************************************************************************************************
        # step3: fitness reverse and floor
        ####*************************************************************************************************************************************************************
        dms_gene_df['fitness_reverse'] = dms_gene_df['fitness_org']
        dms_gene_df['fitness_sd_reverse'] = dms_gene_df['fitness_sd_org']
        dms_gene_df['fitness_se_reverse'] = dms_gene_df['fitness_se_org']
        
        dms_gene_df.loc[dms_gene_df['fitness_reverse'] > 1 , 'fitness_sd_reverse'] = dms_gene_df.loc[dms_gene_df['fitness_reverse'] > 1 , 'fitness_sd_reverse'] / np.power(dms_gene_df.loc[dms_gene_df['fitness_reverse'] > 1 , 'fitness_reverse'],2)
        dms_gene_df.loc[dms_gene_df['fitness_reverse'] > 1 , 'fitness_se_reverse'] = dms_gene_df.loc[dms_gene_df['fitness_reverse'] > 1 , 'fitness_se_reverse'] / np.power(dms_gene_df.loc[dms_gene_df['fitness_reverse'] > 1 , 'fitness_reverse'],2)        
        dms_gene_df.loc[dms_gene_df['fitness_reverse'] > 1 , 'fitness_reverse'] = 1 / dms_gene_df.loc[dms_gene_df['fitness_reverse'] > 1 , 'fitness_reverse']

        if reverse_flag == 1:            
            dms_gene_df['fitness'] = dms_gene_df['fitness_reverse']
            dms_gene_df['fitness_sd'] = dms_gene_df['fitness_sd_reverse'] 
            dms_gene_df['fitness_se'] = dms_gene_df['fitness_se_reverse']
        else:
            dms_gene_df['fitness'] = dms_gene_df['fitness_org']
            dms_gene_df['fitness_sd'] = dms_gene_df['fitness_sd_org']
            dms_gene_df['fitness_se'] = dms_gene_df['fitness_se_org']
            
        # floor
        if floor_flag == 1:
            dms_gene_df.loc[dms_gene_df['fitness'] < 0, 'fitness'] = 0
        
        ####*************************************************************************************************************************************************************
        # step4: save train and test data sets
        ####*************************************************************************************************************************************************************         
#         dms_gene_df.to_csv(self.project_path + 'output/' + dms_landscape_file + '_train') 
#         dms_gene_df.to_csv(self.project_path + 'output/' + dms_landscape_file + '_target')
#         
#         alm_fun.show_msg (self.log,self.verbose,"Saving training data to " + self.project_path + 'output/' + dms_landscape_file + '_train') 
#         alm_fun.show_msg (self.log,self.verbose,"Saving target data to " + self.project_path + 'output/' + dms_landscape_file + '_target')
        
        ####*************************************************************************************************************************************************************
        # step5: save file for Jochen's pipeline
        ####*************************************************************************************************************************************************************
#         dms_gene_jochen_df = dms_gene_df[['annotation','quality_score','aa_ref', 'aa_alt', 'aa_pos','fitness','fitness_sd','num_replicates','fitness_se']]
#         dms_gene_jochen_df.columns = ['annotation','quality_score','aa_ref', 'aa_alt', 'aa_pos','score','sd','df','se']
#         dms_gene_jochen_df['mut'] = dms_gene_jochen_df.apply(lambda x: x['aa_ref'] + str(x['aa_pos']) + x['aa_alt'],axis = 1)
#         #save NONSYN, Valid fitness score and sd , also quality cutoff
#         jochen_idx = (dms_gene_jochen_df['score'].notnull()) & (dms_gene_jochen_df['sd'].notnull()) & (dms_gene_jochen_df['annotation']=='NONSYN') & (dms_gene_jochen_df['quality_score'] > quality_cutoff)
#         dms_gene_jochen_df.loc[jochen_idx,['mut','score','quality_score','sd','df','se']].to_csv(self.project_path + 'output/' + dms_landscape_file + '_jochen',index = False)
#         alm_fun.show_msg (self.log,self.verbose,"Saving data for Jochen's pipeline to " + self.project_path + 'output/' + dms_landscape_file + '_jochen')


        ####*************************************************************************************************************************************************************
        # step6:   Remove the columns that not in the range of the input positions 
        ####*************************************************************************************************************************************************************
        
        min_pos  = min(dms_gene_df.loc[dms_gene_df['fitness_org'].notnull(),'aa_pos'])
        max_pos  = max(dms_gene_df.loc[dms_gene_df['fitness_org'].notnull(),'aa_pos'])
        dms_gene_df = dms_gene_df.loc[(dms_gene_df['aa_pos']>= min_pos) & (dms_gene_df['aa_pos']<= max_pos),:]
        dms_gene_df['aa_pos_index'] = dms_gene_df[['aa_pos']] - min_pos + 1
        dms_gene_df['ss_end_pos_index'] = dms_gene_df[['ss_end_pos']] - min_pos + 1
        dms_gene_df['pfam_end_pos_index'] = dms_gene_df[['pfam_end_pos']] - min_pos + 1
        alm_fun.show_msg (self.log,self.verbose,'Class: [imputation] Fun: [data_preprocess] done @' + str(datetime.now()) + "\n")
        return ([dms_gene_df, dms_gene_df, None])
        
    def feature_engineer(self, train_df, target_df): 
        def mean_fitness(cur_fitness):
            mean_fitness = (sum_all_fitness - cur_fitness) / (n_train - 1)
            return (mean_fitness)
         
        def knn_aa_col(cur_aa_pos, cur_value, cur_alt_encode, k_col, orderby):
            # col average (aa_pos wise)
            # stime = time.time()
            return_list = []
            if cur_aa_pos in dict_aa_pos.keys():  
                cur_alt_encode_list = dict_aa_pos[cur_aa_pos]['aa_alt_encode']
                cur_value_array = np.array(dict_aa_pos[cur_aa_pos][orderby])
                cur_se_array = np.array(dict_aa_pos[cur_aa_pos]['fitness_se'])
                cur_fitness_array = np.array(dict_aa_pos[cur_aa_pos]['fitness'])
                cur_k_col = len(cur_alt_encode_list)       
                cur_distnace_list = [np.abs(x - cur_value) for x in cur_value_array]
    
                if cur_alt_encode in cur_alt_encode_list:
                    data_exist = 1
                    idstart = 1
                    if cur_k_col < k_col + 1:
                        idend = cur_k_col
                    else:
                        idend = k_col + 1
                           
                    idx = cur_alt_encode_list.index(cur_alt_encode)
                    cur_distnace_list[idx] = -1
                else:
                    data_exist = 0
                    idstart = 0
                    if cur_k_col < k_col:
                        idend = cur_k_col
                    else:
                        idend = k_col
                k_real = idend - idstart
                sorted_idx = np.argsort(cur_distnace_list) 
                
                for centrality in self.centrality_names:
                    if centrality == 'mean':                
                        return_list.append(cur_fitness_array[sorted_idx][idstart:idend].mean())
                    if centrality == 'median':
                        return_list.append(np.median(cur_fitness_array[sorted_idx][idstart:idend]))
                    if centrality == 't_mean':
                        # return_list.apeend(stats.trim_mean(cur_fitness_array[sorted_idx][idstart:idend], 0.1))
                        if k_real > 2: 
                            fitness_array_fortrim = np.sort(cur_fitness_array[sorted_idx][idstart:idend])
                            fitness_array_fortrim = fitness_array_fortrim[1:(k_real - 1)]
                            pos_trimmed_mean_fitness = fitness_array_fortrim.mean()
                        else:
                            pos_trimmed_mean_fitness = cur_fitness_array[sorted_idx][idstart:idend].mean()
                        return_list.append(pos_trimmed_mean_fitness)
                    if centrality == 'se_weighted_mean':
                        se = cur_se_array[sorted_idx][idstart:idend]
                        se_inverse = [1 / x for x in se]
                        se_inverse_weight = se_inverse / np.sum(se_inverse)
                        return_list.append(np.dot(cur_fitness_array[sorted_idx][idstart:idend], se_inverse_weight))
                    if centrality == 'count':
                        return_list.append(k_real)                
                    if centrality == 'se':
                        return_list.append(np.std(cur_fitness_array[sorted_idx][idstart:idend]) / np.sqrt(k_real))
            else:
                return_list = [np.nan] * len(self.centrality_names)
            # etime = time.time()   
#             alm_fun.show_msg (self.log,self.verbose,"running time was %g seconds" % (etime - stime)) 
            return pd.Series(return_list)
        
        def knn_aa_row(cur_aa_pos, cur_alt_encode, k_row):
            # row average  (aa_alt wise)  
            if cur_alt_encode in dict_aa_alt.keys():  
                cur_pos_list = dict_aa_alt[cur_alt_encode][0]
                cur_value_array = np.array(dict_aa_alt[cur_alt_encode][1])
                cur_se_array = np.array(dict_aa_alt[cur_alt_encode][2])
                cur_fitness_array = np.array(dict_aa_alt[cur_alt_encode][3])
                cur_k_row = len(cur_pos_list)     
                cur_distnace_list = [np.abs(x - cur_aa_pos) for x in cur_pos_list]
     
                if cur_aa_pos in cur_pos_list:
                    data_exist = 1
                    idstart = 1
                    if cur_k_row < k_row + 1:
                        idend = cur_k_row
                    else:
                        idend = k_row + 1
                           
                    idx = cur_alt_encode_list.index(cur_alt_encode)
                    cur_distnace_list[idx] = -1
                else:
                    data_exist = 0
                    idstart = 0
                    if cur_k_row < k_row:
                        idend = cur_k_row
                    else:
                        idend = k_row
     
                sorted_idx = np.argsort(cur_distnace_list)                
                alt_mean_fitness = cur_fitness_array[sorted_idx][idstart:idend].mean()
                alt_median_fitness = np.median(cur_fitness_array[sorted_idx][idstart:idend])
                alt_trimmed_mean_fitness = stats.trim_mean(cur_fitness_array[sorted_idx][idstart:idend], 0.1) 
                se = cur_se_array[sorted_idx][idstart:idend]
                se_inverse = [1 / x for x in se]
                se_inverse_weight = se_inverse / np.sum(se_inverse)
                alt_se_weighted_mean_fitness = np.dot(cur_fitness_array[sorted_idx][idstart:idend], se_inverse_weight)
                alt_mean_count = idend - idstart
                alt_mean_se = np.std(cur_fitness_array[sorted_idx][idstart:idend]) / np.sqrt(alt_mean_count)
                if alt_mean_se == 0 :
                    alt_mean_se = 10e-5
            else:
                alt_mean_fitness = np.nan
                alt_median_fitness = np.nan
                alt_trimmed_mean_fitness = np.nan
                alt_se_weighted_mean_fitness = np.nan
                alt_mean_count = np.nan
                alt_mean_se = np.nan
             
            # combined_mean_fitness = (alt_se_weighted_mean_fitness/alt_mean_se + pos_se_weighted_mean_fitness/pos_mean_se) / (1/alt_mean_se + 1/pos_mean_se)
            return pd.Series([alt_mean_fitness, alt_trimmed_mean_fitness, alt_median_fitness, alt_se_weighted_mean_fitness, alt_mean_count, alt_mean_se])
        
        stime = time.time() 
        self.counter = 0 
        if self.add_funsum_onfly == 1:
            for fun_sum in self.use_funsums:  
                train_df = pd.merge(train_df, self.project.gi.dict_sums['funsum'][fun_sum], how='left')
                target_df = pd.merge(target_df, self.project.gi.dict_sums['funsum'][fun_sum], how='left')
         
        groupby_df = train_df[['aa_pos', 'aa_alt_encode', 'quality_score', 'fitness_se', self.dependent_variable] + self.value_orderby]
        groupby_df = groupby_df.loc[~groupby_df.isnull().any(axis=1), :]
        
        n_train = groupby_df.shape[0]
        sum_all_fitness = groupby_df[self.dependent_variable].sum()
        mean_all_fitness = groupby_df[self.dependent_variable].mean()
        dict_aa_pos = {}
        dict_aa_alt = {}
        for x in groupby_df['aa_pos'].unique():
            dict_aa_pos[x] = {}
            dict_aa_pos[x]['aa_alt_encode'] = list(groupby_df.loc[groupby_df['aa_pos'] == x, 'aa_alt_encode'])
            for orderby in self.value_orderby:
                dict_aa_pos[x][orderby] = list(groupby_df.loc[groupby_df['aa_pos'] == x, orderby])
            dict_aa_pos[x]['fitness_se'] = list(groupby_df.loc[groupby_df['aa_pos'] == x, 'fitness_se'])
            dict_aa_pos[x]['fitness'] = list(groupby_df.loc[groupby_df['aa_pos'] == x, self.dependent_variable])
                  
        for x in groupby_df['aa_alt_encode'].unique():
            dict_aa_alt[x] = [list(groupby_df.loc[groupby_df['aa_alt_encode'] == x, 'aa_pos']), list(groupby_df.loc[groupby_df['aa_alt_encode'] == x, orderby]), list(groupby_df.loc[groupby_df['aa_alt_encode'] == x, 'fitness_se']), list(groupby_df.loc[groupby_df['aa_alt_encode'] == x, self.dependent_variable])]
        
 
    
        # col_feature_names = ['pos_mean_fitness','pos_trimmed_mean_fitness','pos_median_fitness','pos_se_weighted_mean_fitness','pos_mean_count','pos_mean_se']
    
        for k_col in self.k_range:
            for  i in range(len(self.value_orderby)):
                orderby_name = self.value_orderby_name[i]
                orderby = self.value_orderby[i]
                cur_col_feature_names = [x + '_' + str(k_col) + '_' + orderby_name for x in self.centrality_names]
                stime = time.time()
                train_engineered_features = train_df[['aa_pos', 'aa_alt_encode', orderby]].apply(lambda x: knn_aa_col(x['aa_pos'], x[orderby], x['aa_alt_encode'], k_col, orderby), axis=1)
                target_engineered_features = target_df[['aa_pos', 'aa_alt_encode', orderby]].apply(lambda x: knn_aa_col(x['aa_pos'], x[orderby], x['aa_alt_encode'], k_col, orderby), axis=1)
                etime = time.time()
                alm_fun.show_msg (self.log,self.verbose,"[K:" + str(k_col) + ' orderby: ' + orderby_name + "] running time was %g seconds" % (etime - stime))           
                train_engineered_features.columns = cur_col_feature_names
                target_engineered_features.columns = cur_col_feature_names
                train_df = pd.concat([train_df, train_engineered_features], axis=1)
                target_df = pd.concat([target_df, target_engineered_features], axis=1)
            
    #     for k_row in []:
    #         cur_row_feature_names =[x +'_' + str(k_row) for x in row_feature_names]
    #         train_engineered_features = train_df.apply(lambda x: knn_aa_row(x['aa_pos'],x['aa_alt_encode'],k_row),axis = 1)
    #         target_engineered_features = target_df.apply(lambda x: knn_aa_row(x['aa_pos'],x['aa_alt_encode'],k_row),axis = 1)
    #         train_engineered_features.columns = cur_row_feature_names
    #         target_engineered_features.columns = cur_row_feature_names
    #         train_df = pd.concat([train_df,train_engineered_features],axis = 1)
    #         target_df = pd.concat([target_df,target_engineered_features],axis = 1)
            
        train_df['mean_fitness'] = train_df[self.dependent_variable].apply(lambda x: mean_fitness(x))
        target_df['mean_fitness'] = mean_all_fitness

   
        return [train_df, target_df]
    
    def data_slice(self, data_name, target_df, train_df, test_df, extra_train_df):        
#         cutoff = self.regression_quality_cutoffs[self.data_names.index(data_name)]        
#         if cutoff == 'no gradient' :
#             cutoff = float('-inf')
#         else:
#             cutoff = float(cutoff)

#         train_df = train_df.loc[train_df['quality_score'] >= cutoff, :]
        
        train_df = train_df.loc[train_df['aa_alt'] != '*', :]
        train_df = train_df.loc[train_df['aa_ref'] != train_df['aa_alt'], :]
        train_df = train_df.loc[train_df['fitness_se'].notnull(), :]

        target_df = target_df.loc[target_df['aa_alt'] != '*', :]
        target_df = target_df.loc[target_df['aa_ref'] != target_df['aa_alt'], :]

        return [target_df, train_df, test_df, extra_train_df]
    
    def test_split(self, train_df):
        kf_list = []                                
        return (kf_list)
    
    def cv_split(self, train_df):
        quality_scores = list(train_df['quality_score'])
        quality_scores.sort()
        cutoff = quality_scores[np.int(len(quality_scores) * 0.8)]        
        validation_index = train_df.loc[train_df['quality_score'] > cutoff, :].index
        train_index = train_df.loc[train_df['quality_score'] <= cutoff, :].index
        
        validataion_kf_index = [list(train_df.index).index(x) for x in validation_index]
        train_kf_index = [list(train_df.index).index(x) for x in train_index]
                                
        return ([(train_kf_index, validataion_kf_index)])
    
    def setup_gradients(self,train_df):
        gradients = []
        quality_scores = list(train_df['quality_score'])
        quality_scores.sort()
        cutoff = quality_scores[np.int(len(quality_scores) * 0.8)]        
        train_df_for_graidents = train_df.loc[train_df['quality_score'] <= cutoff, :]        
        max_quality = max(train_df_for_graidents['quality_score'])
        min_quality = min(train_df_for_graidents['quality_score'])      
        for cur_quality in np.linspace(min_quality, max_quality, 10, endpoint=False):
            gradients.append(str(cur_quality))
        return(gradients)
    
    def gradient_reshape(self, train_df, gradients):  
        train_gradient_df = {}   
        for cur_quality in gradients:
            cur_train_df = train_df.loc[train_df['quality_score'] >= np.float(cur_quality), :]
            train_gradient_df[str(cur_quality)] = cur_train_df.index                       
        return(train_gradient_df)
    
    def imputation_run(self,data_name):        
        def get_error_fontsize(se): 
            if np.isnan(se): return np.nan
            if se > 1: 
                return 16
            else:
                return se * 12 + 4
        
        alm_fun.show_msg (self.log,self.verbose,'**Class: [imputation] Fun: [imputation_run] .... starts @' + str(datetime.now()))
        cur_data_idx = self.data_names.index(data_name)
        
        features = []
        for k in self.k_range:
            for orderby_name in self.value_orderby_name:
                cur_col_feature_names = [x + '_' + str(k) + '_' + orderby_name for x in self.centrality_names]
                features += cur_col_feature_names  
        features = features + ['polyphen_score', 'provean_score', 'blosum100']
        self.project.train_features = features
        self.project.run_data_names = [data_name]
        
        ####*************************************************************************************************************************************************************
        # optional step: auto quality cutoff 
        ####*************************************************************************************************************************************************************        
        refresh_data = 1
        if self.project.data[data_name].if_gradient == 1:
            self.project.modes = ['gradient_comparison']
            run_return = self.project.run(refresh_data = refresh_data)
            opt_gradient = run_return['gradient_comparison'][data_name][2]
            self.project.data[data_name].cur_gradient_key = opt_gradient
            
            if opt_gradient == 'no_gradient':
                self.regression_quality_cutoffs[cur_data_idx] = float('-inf')
            else:
                self.regression_quality_cutoffs[cur_data_idx] = float(opt_gradient) 
                           
            refresh_data = 0
        ####*************************************************************************************************************************************************************
        # step4: run imputation ML 
        ####*************************************************************************************************************************************************************        
        self.project.modes = ['prediction']
        fitness_predicted = self.project.run(refresh_data = refresh_data) ['prediction'][data_name]    
        dms_gene_df = self.project.data[data_name].target_data_original_df.copy()
        dms_gene_df['fitness_imputed'] = np.nan
        dms_gene_df['knn_count'] = np.nan
        dms_gene_df['knn_se'] = np.nan

        knn_se_name = 'se_' + str(self.k_range[0]) + '_fs'
        knn_count_name = 'count_' + str(self.k_range[0]) + '_fs'
        dms_gene_df.loc[self.project.data[data_name].target_data_df.index, 'fitness_imputed'] = fitness_predicted
        dms_gene_df.loc[self.project.data[data_name].target_data_df.index, 'knn_count'] = self.project.data[data_name].predicted_target_df[knn_count_name]
        dms_gene_df.loc[self.project.data[data_name].target_data_df.index, 'knn_se'] = self.project.data[data_name].predicted_target_df[knn_se_name] 
        
        regression_quality_cutoff = self.regression_quality_cutoffs[cur_data_idx]
         
        ####*************************************************************************************************************************************************************
        # step5: fitness standard deviation regularization
        ####*************************************************************************************************************************************************************
        if (self.regularization_flags[cur_data_idx] == 1) & (self.raw_processed[cur_data_idx] == 1) :
#             regression_experimental_error_df = dms_gene_df.loc[dms_gene_df['fitness'].notnull() & (dms_gene_df['quality_score'] > regression_quality_cutoff) & (dms_gene_df['fitness_sd'] < 1), :]      
            regression_experimental_error_df = dms_gene_df.loc[dms_gene_df['fitness'].notnull() & (dms_gene_df['quality_score'] > regression_quality_cutoff), :]
            regressor = xgb.XGBRegressor(**{'subsample': 0.8, 'colsample_bytree': 1, 'max_depth': 3, 'n_estimators': 100, 'learning_rate': 0.08})              
            regressor.fit(np.array(regression_experimental_error_df[['fitness', 'quality_score']]), np.array(regression_experimental_error_df['fitness_sd']))   
            notnull_idx = dms_gene_df.index[dms_gene_df['fitness'].notnull()]         
            dms_gene_df['fitness_sd_prior'] = np.nan
            dms_gene_df.loc[notnull_idx, 'fitness_sd_prior'] = regressor.predict(np.array(dms_gene_df.loc[notnull_idx, ['fitness', 'quality_score']]))
    
            # standard deviation regularization ( assume prior mean is the sample mean )         
            dms_gene_df['fitness_sd_reg'] = np.sqrt((dms_gene_df['pseudo_count'] * (dms_gene_df['fitness_sd_prior'] ** 2) + (dms_gene_df['num_replicates'] - 1) * (dms_gene_df['fitness_sd'] ** 2)) / (dms_gene_df['pseudo_count'] + dms_gene_df['num_replicates'] - 1))
            dms_gene_df['fitness_se_reg'] = dms_gene_df['fitness_sd_reg'] / np.sqrt(self.proper_num_replicates[cur_data_idx]) 
        else:
            dms_gene_df['fitness_sd_prior'] = np.nan
            dms_gene_df['fitness_sd_reg'] = dms_gene_df['fitness_sd']
            dms_gene_df['fitness_se_reg'] = dms_gene_df['fitness_se']
         
        ####*************************************************************************************************************************************************************
        # step6: Fitness Refinement
        ####*************************************************************************************************************************************************************
        dms_gene_df['fitness_imputed_se'] = abs(dms_gene_df['fitness_imputed'] - dms_gene_df['fitness'])
        regression_imputation_error_df = dms_gene_df.loc[dms_gene_df['fitness'].notnull() & dms_gene_df['fitness_imputed'].notnull() & dms_gene_df['knn_se'].notnull() & dms_gene_df['knn_count'].notnull() , :]
#         regression_imputation_error_df.to_csv(self.project_path + 'output/' + data_name + '_regression_imputation_error.csv')          
        regressor = xgb.XGBRegressor(**{'subsample': 0.8, 'colsample_bytree': 1, 'max_depth': 3, 'n_estimators': 100, 'learning_rate': 0.08})                
        regressor.fit(np.array(regression_imputation_error_df[['fitness_imputed', 'knn_se', 'knn_count']]), np.array(regression_imputation_error_df['fitness_imputed_se']))

        notnull_idx = dms_gene_df.index[dms_gene_df['fitness_imputed'].notnull()]
        dms_gene_df['fitness_imputed_se_prior'] = np.nan
        dms_gene_df.loc[notnull_idx, 'fitness_imputed_se_prior'] = regressor.predict(np.array(dms_gene_df.loc[notnull_idx, ['fitness_imputed', 'knn_se', 'knn_count']]))

        # fitness regularization (assume the variance is known (regularized sd), imputed fitness and it standard error as hyper-parameters of the prior distribution of the mean.  
        dms_gene_df['fitness_refine'] = ((dms_gene_df['fitness_imputed_se_prior'] ** 2) * dms_gene_df['fitness'] + (dms_gene_df['fitness_se_reg'] ** 2) * dms_gene_df['fitness_imputed']) / (dms_gene_df['fitness_imputed_se_prior'] ** 2 + dms_gene_df['fitness_se_reg'] ** 2)
        dms_gene_df['fitness_se_refine'] = np.sqrt(((dms_gene_df['fitness_imputed_se_prior'] ** 2) * (dms_gene_df['fitness_se_reg'] ** 2)) / (dms_gene_df['fitness_imputed_se_prior'] ** 2 + dms_gene_df['fitness_se_reg'] ** 2))  
         
        # when there is no original fitness value
        dms_gene_df.loc[dms_gene_df['fitness'].isnull(), 'fitness_refine'] = dms_gene_df.loc[dms_gene_df['fitness'].isnull(), 'fitness_imputed']                
        dms_gene_df.loc[dms_gene_df['fitness'].isnull(), 'fitness_se_refine'] = dms_gene_df.loc[dms_gene_df['fitness'].isnull(), 'fitness_imputed_se_prior']
         
        # when there is no imputed fitness value
        dms_gene_df.loc[dms_gene_df['fitness_imputed'].isnull(), 'fitness_refine'] = dms_gene_df.loc[dms_gene_df['fitness_imputed'].isnull(), 'fitness']        
        dms_gene_df.loc[dms_gene_df['fitness_imputed'].isnull(), 'fitness_se_refine'] = dms_gene_df.loc[dms_gene_df['fitness_imputed'].isnull(), 'fitness_se_reg']
        
        ####*************************************************************************************************************************************************************
        # step6.1: save files for Funregressor
        ####*************************************************************************************************************************************************************
#         dms_gene_df.to_csv(self.project_path + 'output/' + data_name + '_funregressor.csv',index = False)
        
        ####*************************************************************************************************************************************************************
        # step7: save files and draw figures
        ####*************************************************************************************************************************************************************
        dms_gene_csv_df = dms_gene_df[['aa_ref', 'aa_pos', 'aa_pos_index', 'aa_alt', 'annotation', 'aa_psipred','ss_end_pos','ss_end_pos_index','hmm_id','pfam_end_pos','pfam_end_pos_index','quality_score', 'num_replicates', 'fitness_org', 'fitness_sd_org', 'fitness_reverse', 'fitness_sd_reverse', 'fitness', 'fitness_sd', 'fitness_se', 'fitness_sd_prior', 'fitness_sd_reg', 'fitness_se_reg', 'fitness_imputed', 'knn_se', 'knn_count', 'fitness_imputed_se', 'fitness_imputed_se_prior', 'fitness_refine', 'fitness_se_refine', 'polyphen_score', 'sift_score', 'provean_score', 'funsum_fitness_mean', 'blosum62', 'gnomad_af', 'asa_mean', 'pseudo_count', 'fitness_input', 'fitness_input_sd','fitness_input_filtered', 'fitness_input_filtered_sd','syn_filtered','stop_filtered']]  

        # fitness colorcode     
        if self.remediation[cur_data_idx] == 1:
            v_max_fitness = 1
            v_center_fitness = 0
            v_min_fitness = -1
            v_max_fitness_color = '#008000'
            v_center_fitness_color = '#FFFFFF'
            v_min_fitness_color = '#800000'
            n_gradient_max_fitness = 10
            n_gradient_min_fitness = 10
        else:
            v_max_fitness = 2
            v_center_fitness = 1
            v_min_fitness = 0
            v_max_fitness_color = '#C6172B'
            v_center_fitness_color = '#FFFFFF'
            v_min_fitness_color = '#3155C6'
            n_gradient_max_fitness = 5
            n_gradient_min_fitness = 5
        
        img_fig_width = 5
        img_fig_height = 1.2
        
        [lst_max_colors_fitness, lst_min_colors_fitness] = alm_fun.create_color_gradients(v_max_fitness, v_min_fitness, v_center_fitness, v_max_fitness_color, v_min_fitness_color, v_center_fitness_color, n_gradient_max_fitness, n_gradient_min_fitness)
#         self.plot_color_gradients(v_max_fitness, v_min_fitness, v_center_fitness, lst_max_colors_fitness, lst_min_colors_fitness, img_fig_width, img_fig_height, 'H', 'Fitness Score', project_path + 'imputation_legend.png')                                            
        dms_gene_csv_df['fitness_org'] = np.round(dms_gene_csv_df['fitness_org'], 4)
        dms_gene_csv_df['fitness_org_colorcode'] = dms_gene_csv_df['fitness_org'].apply(lambda x: alm_fun.get_colorcode(x, v_max_fitness, v_min_fitness, v_center_fitness, n_gradient_max_fitness, n_gradient_min_fitness, lst_max_colors_fitness, lst_min_colors_fitness))
        dms_gene_csv_df['fitness_se_org'] = np.round(dms_gene_csv_df['fitness_sd_org'] / np.sqrt(dms_gene_csv_df['num_replicates']), 4)        
        dms_gene_csv_df['fitness_reverse'] = np.round(dms_gene_csv_df['fitness_reverse'], 4)        
        dms_gene_csv_df['fitness_reverse_colorcode'] = dms_gene_csv_df['fitness_reverse'].apply(lambda x: alm_fun.get_colorcode(x, v_max_fitness, v_min_fitness, v_center_fitness, n_gradient_max_fitness, n_gradient_min_fitness, lst_max_colors_fitness, lst_min_colors_fitness))
        dms_gene_csv_df['fitness_se_reverse'] = np.round(dms_gene_csv_df['fitness_sd_reverse'] / np.sqrt(dms_gene_csv_df['num_replicates']), 4)
        dms_gene_csv_df['fitness_imputed'] = np.round(dms_gene_csv_df['fitness_imputed'], 4)
        dms_gene_csv_df['fitness_imputed_colorcode'] = dms_gene_csv_df['fitness_imputed'].apply(lambda x: alm_fun.get_colorcode(x, v_max_fitness, v_min_fitness, v_center_fitness, n_gradient_max_fitness, n_gradient_min_fitness, lst_max_colors_fitness, lst_min_colors_fitness))
        dms_gene_csv_df['fitness_refine'] = np.round(dms_gene_csv_df['fitness_refine'], 4)      
        dms_gene_csv_df['fitness_refine_colorcode'] = dms_gene_csv_df['fitness_refine'].apply(lambda x: alm_fun.get_colorcode(x, v_max_fitness, v_min_fitness, v_center_fitness, n_gradient_max_fitness, n_gradient_min_fitness, lst_max_colors_fitness, lst_min_colors_fitness))
        dms_gene_csv_df['fitness_se_refine'] = np.round(dms_gene_csv_df['fitness_se_refine'] , 4)
        dms_gene_csv_df['quality_score'] = np.round(dms_gene_csv_df['quality_score'] , 4)
        dms_gene_csv_df['se_org_fontsize'] = dms_gene_csv_df['fitness_se_org'].apply(lambda x: get_error_fontsize(x))
        dms_gene_csv_df['se_refine_fontsize'] = dms_gene_csv_df['fitness_se_refine'].apply(lambda x: get_error_fontsize(x))
        
        # ass colorcode
        [lst_max_colors_asa, lst_min_colors_asa] = alm_fun.create_color_gradients(1, 0, 0, '#3155C6', '#FFFFFF', '#FFFFFF', 10, 10)
#         self.plot_color_gradients(1, 0, 0, lst_max_colors_asa, lst_min_colors_asa, img_fig_width, img_fig_height, 'H', 'ASA score', project_path + 'asa_legend.png')  
        dms_gene_csv_df['asa_mean_normalized'] = (dms_gene_csv_df['asa_mean'] - np.nanmin(dms_gene_csv_df['asa_mean'])) / (np.nanmax(dms_gene_csv_df['asa_mean']) - np.nanmin(dms_gene_csv_df['asa_mean']))
        dms_gene_csv_df['asa_colorcode'] = dms_gene_csv_df['asa_mean_normalized'].apply(lambda x: alm_fun.get_colorcode(x, 1, 0, 0, 10, 10, lst_max_colors_asa, lst_min_colors_asa))
        
        # sift colorcode
#         [lst_max_colors_sift,lst_min_colors_sift] = alm_fun.create_color_gradients(1, 0, 0.05,'#C6172B','#FFFFFF','#3155C6',10,10)
        [lst_max_colors_sift, lst_min_colors_sift] = alm_fun.create_color_gradients(1, 0, 0, '#FFFFFF', '#3155C6', '#3155C6', 10, 10)
#         self.plot_color_gradients(1, 0, 0, lst_max_colors_sift, lst_min_colors_sift, img_fig_width, img_fig_height, 'H', 'SIFT Score', project_path + 'sift_legend.png')
        dms_gene_csv_df['sift_colorcode'] = dms_gene_csv_df['sift_score'].apply(lambda x: alm_fun.get_colorcode(x, 1, 0, 0, 10, 10, lst_max_colors_sift, lst_min_colors_sift))
        
        [lst_max_colors_polyphen, lst_min_colors_polyphen] = alm_fun.create_color_gradients(1, 0, 0, '#3155C6', '#FFFFFF', '#FFFFFF', 10, 10)
#         self.plot_color_gradients(1, 0, 0, lst_max_colors_polyphen, lst_min_colors_polyphen, img_fig_width, img_fig_height, 'H', 'Polyphen Score', project_path + 'polyphen_legend.png')
        dms_gene_csv_df['polyphen_colorcode'] = dms_gene_csv_df['polyphen_score'].apply(lambda x: alm_fun.get_colorcode(x, 1, 0, 0, 10, 10, lst_max_colors_polyphen, lst_min_colors_polyphen))
        
        [lst_max_colors_gnomad, lst_min_colors_gnomad] = alm_fun.create_color_gradients(10, 0.3, 0.3, '#3155C6', '#FFFFFF', '#FFFFFF', 10, 10)
#         self.plot_color_gradients(10, 0.3, 0.3, lst_max_colors_gnomad, lst_min_colors_gnomad, img_fig_width, img_fig_height, 'H', '-Log10 GNOMAD Score', project_path + 'gnomad_legend.png')
        dms_gene_csv_df['gnomad_af_log10'] = 0 - np.log10(dms_gene_csv_df['gnomad_af'])
        dms_gene_csv_df['gnomad_colorcode'] = dms_gene_csv_df['gnomad_af_log10'].apply(lambda x: alm_fun.get_colorcode(x, 10, 0.3, 0.3, 10, 10, lst_max_colors_gnomad, lst_min_colors_gnomad))
        
        [lst_max_colors_provean, lst_min_colors_provean] = alm_fun.create_color_gradients(4, -13, -13, '#FFFFFF', '#3155C6', '#3155C6', 10, 10)
#         self.plot_color_gradients(4, -13, -13, lst_max_colors_provean, lst_min_colors_provean, img_fig_width, img_fig_height, 'H', 'Provean Score', project_path + 'provean_legend.png') 
        dms_gene_csv_df['provean_colorcode'] = dms_gene_csv_df['provean_score'].apply(lambda x: alm_fun.get_colorcode(x, 4, -13, -13, 10, 10, lst_max_colors_provean, lst_min_colors_provean))
         
        syn_color = '#fdf9ce'        
        dms_gene_csv_df.loc[dms_gene_csv_df['annotation'] == 'SYN', ['fitness_org_colorcode', 'fitness_reverse_colorcode', 'fitness_imputed_colorcode', 'fitness_refine_colorcode', 'sift_colorcode', 'polyphen_colorcode', 'gnomad_colorcode', 'provean_colorcode']] = syn_color
        
        #*************************************************************************
        #imputation result for downloading
        #*************************************************************************
        dms_imputation_csv_df = dms_gene_csv_df[['aa_ref', 'aa_pos', 'aa_alt', 'annotation', 'quality_score', 'num_replicates', 'pseudo_count', 'fitness_input', 'fitness_input_sd', 'fitness_org', 'fitness_sd_org', 'fitness', 'fitness_sd', 'fitness_sd_prior', 'fitness_sd_reg', 'fitness_se_reg', 'fitness_imputed', 'fitness_imputed_se', 'fitness_imputed_se_prior', 'fitness_refine', 'fitness_se_refine', 'polyphen_score', 'sift_score', 'provean_score', 'gnomad_af', 'asa_mean', 'aa_psipred', 'ss_end_pos', 'hmm_id', 'pfam_end_pos']]            
        dms_imputation_csv_df.to_csv(self.project_path + 'output/' + data_name + '_imputation.csv')
        
                #*************************************************************************
        #remove columns that has no oringial fitness values 
        #*************************************************************************
        
#         available_pos = dms_gene_csv_df.loc[dms_gene_csv_df['fitness_org'].notnull(),'aa_pos'].unique()
#         dms_gene_csv_df = dms_gene_csv_df.loc[dms_gene_csv_df['aa_pos'].isin(available_pos),:]
#         
#         
        #*************************************************************************
        #imputation plot
        #*************************************************************************
        self.imputation_plot(data_name, dms_gene_csv_df, lst_max_colors_fitness, lst_min_colors_fitness, lst_max_colors_asa, lst_min_colors_asa,1,0)
        
        #debug mpas
#         self.imputation_plot(data_name, dms_gene_csv_df, lst_max_colors_fitness, lst_min_colors_fitness, lst_max_colors_asa, lst_min_colors_asa,1,1)
        #*************************************************************************
        #imputation data for GWT JSON
        #*************************************************************************
        dms_gene_gwt_df = dms_gene_csv_df[["aa_ref", "aa_pos", "aa_pos_index", "aa_alt", "quality_score", "num_replicates", "fitness_org", "fitness_sd_org", "fitness_reverse", "fitness_sd_reverse", "fitness_refine", "fitness_sd_reg", "polyphen_score", "sift_score", "provean_score", "funsum_fitness_mean", "blosum62", "gnomad_af", "aa_psipred", "asa_mean_normalized", "ss_end_pos", "ss_end_pos_index", "hmm_id", "pfam_end_pos", "pfam_end_pos_index","sift_colorcode", "provean_colorcode", "polyphen_colorcode", "gnomad_colorcode", "fitness_org_colorcode", "fitness_se_org", "fitness_reverse_colorcode", "fitness_se_reverse", "fitness_refine_colorcode", "asa_colorcode", "fitness_se_refine", "se_org_fontsize","se_refine_fontsize"]]

                
        alm_fun.show_msg (self.log,self.verbose,'**Class: [imputation] Fun: [run_imputation] .... done @' + str(datetime.now()))
        return(dms_gene_gwt_df.to_json(orient='records'))
    
  
    def imputation_plot(self, data_name, dms_gene_df, lst_max_colors_fitness, lst_min_colors_fitness, lst_max_colors_asa, lst_min_colors_asa,plot_ve_map = 1,plot_internal_map = 0):     
        cur_data_idx = self.data_names.index(data_name)
        ####*************************************************************************************************************************************************************
        # Variant effect map        
        ####*************************************************************************************************************************************************************
        if plot_ve_map == 1:
    #         for fitness_name in ['fitness_org','fitness_reverse','fitness_imputed','fitness_refine']:
    
            min_pos = min(dms_gene_df['aa_pos'])
            max_pos = max(dms_gene_df['aa_pos'])
            
            for fitness_name in ['fitness_org', 'fitness_refine']:
                if fitness_name == 'fitness_org':
                    se_name = 'fitness_se_org'
                    plot_title = data_name + ' variant effect map - original'
                if fitness_name == 'fitness_reverse':
                    se_name = 'fitness_se_reverse'
                    plot_title = data_name + ' variant effect map - reversed'
                if fitness_name == 'fitness_imputed':
                    plot_title = data_name + ' variant effect map - imputed'
                    se_name = 'fitness_se_reg'
                if fitness_name == 'fitness_refine':         
                    plot_title = data_name + ' variant effect map - refined'
                    se_name = 'fitness_se_refine'   
                pass
                   
                dms_gene_figure_df = dms_gene_df.copy()
                # set all wild type fitness to null
                syn_color = '#fdf9ce'
                dms_gene_figure_df.loc[dms_gene_figure_df['annotation'] == 'SYN', fitness_name] = np.nan
                dms_gene_figure_df.loc[dms_gene_figure_df['annotation'] == 'SYN', fitness_name + '_colorcode'] = syn_color
                
                dms_gene_figure_df.loc[dms_gene_figure_df[fitness_name].isnull(), se_name] = np.nan 
                dms_gene_figure_df.loc[dms_gene_figure_df[se_name] > 1, se_name] = 1                      
                vmax = np.nanmax(list(dms_gene_figure_df[fitness_name]))
                vmin = np.nanmin(list(dms_gene_figure_df[fitness_name]))           
                landscape_fitness = dms_gene_figure_df.pivot(index='aa_alt', columns='aa_pos', values=fitness_name)
                landscape_asa = dms_gene_figure_df.loc[dms_gene_figure_df['aa_alt'] == '*', ['aa_pos', 'asa_colorcode']]
                landscape_ss = dms_gene_figure_df.loc[dms_gene_figure_df['ss_end_pos'].notnull(), ['aa_pos', 'aa_psipred', 'ss_end_pos']]
                landscape_pfam = dms_gene_figure_df.loc[dms_gene_figure_df['hmm_id'].notnull(), ['aa_pos', 'hmm_id', 'pfam_end_pos']]
                
                landscape_colorcode = dms_gene_figure_df.pivot(index='aa_alt', columns='aa_pos', values=fitness_name + '_colorcode')
                landscape_se = dms_gene_figure_df.pivot(index='aa_alt', columns='aa_pos', values=se_name)
                
                landscape_fitness_copy = landscape_fitness.copy()            
                landscape_fitness_copy.drop('*', axis=0, inplace=True)
    #             landscape_fitness_copy.fillna(np.nanmin(landscape_fitness_copy)-1,inplace = True)
                landscape_fitness_copy.fillna(np.nanmax(landscape_fitness_copy) + 1, inplace=True)
                landscape_fitness_orderindex = landscape_fitness_copy.apply(lambda x: np.argsort(x), axis=0)
                
                landscape_colorcode_copy = landscape_colorcode.copy()
                landscape_colorcode_copy.drop('*', axis=0, inplace=True)
                landscape_colorcode_copy.replace(syn_color, '#C0C0C0', inplace=True)
                landscape_colorcode_ordered = landscape_colorcode_copy.apply(lambda x: np.array(x)[landscape_fitness_orderindex[x.name]], axis=0)
                
    #             landscape_colorcode_copy.replace('#C0C0C0','#gggggg',inplace = True)            
    #             landscape_colorcode_ordered = landscape_colorcode_copy.apply(lambda x: np.sort(x),axis = 0)
    #             landscape_colorcode_ordered.replace('#gggggg','#C0C0C0',inplace = True)
                landscape_colorcode_ordered.index = range(20)
                
                fig_width = int(len(landscape_colorcode.columns) / 5)
                f = len(landscape_colorcode.columns) / 550             
                fig_width = 40
                aa_plot = ['*', 'P', 'C', 'G', 'Q', 'N', 'T', 'S', 'E', 'D', 'K', 'H', 'R', 'W', 'Y', 'F', 'M', 'I', 'L', 'V', 'A']            
                fig = plt.figure(figsize=(fig_width, 8)) 
                fig.patch.set_facecolor('white')
                gs = GridSpec(10, 1, hspace=0.6)
                ####************************************************************************************************************************************************************* 
                # other tracks (secondary structure, accessible surface area, pfam etc)
                ####*************************************************************************************************************************************************************
                stime = time.time()
                n_tracks = 3
                ax_1 = plt.subplot(gs[1:3, :])                                                 
                x = [1] * (n_tracks + 1) + list(range(1, landscape_colorcode.shape[1] + 1))
                y = list(range(1, n_tracks + 2)) + [1] * landscape_colorcode.shape[1]                        
                ax_1.plot(x, y, alpha=0)
                
                # asa tracks
                asa_patches = []
                for idx in landscape_asa.index:                
                    x = landscape_asa.loc[idx, 'aa_pos'] - min_pos +1
                    xy_color = landscape_asa.loc[idx, 'asa_colorcode']
                    rect = patches.Rectangle((x, 1), 1, 1, linewidth=1, edgecolor=xy_color, facecolor=xy_color, fill=True, clip_on=False)
                    asa_patches.append(rect)
                pass
                ax_1.add_collection(collections.PatchCollection(asa_patches, match_original=True))
                
                # ss tracks
                ss_patches = []
                for idx in landscape_ss.index:                
                    x = landscape_ss.loc[idx, 'aa_pos'] - min_pos +1
                    if landscape_ss.loc[idx, 'aa_psipred'] == 'C':
                        width = int(landscape_ss.loc[idx, 'ss_end_pos'] - landscape_ss.loc[idx, 'aa_pos'] + 1)
                        ss_patch_C = patches.Rectangle((x, 2), width, 1, linewidth=1, edgecolor='#FFFFFF', facecolor='#FFFFFF', fill=True, clip_on=False)
                        ss_patches.append(ss_patch_C)
                        
                    if landscape_ss.loc[idx, 'aa_psipred'] == 'E':  # beat sheets
                        width = int(landscape_ss.loc[idx, 'ss_end_pos'] - landscape_ss.loc[idx, 'aa_pos'] + 1)
                        ss_patch_E = patches.Arrow(x, 2.5, width, 0, linewidth=1, edgecolor='#000000', facecolor='#000000', fill=True, clip_on=False)
                        ss_patches.append(ss_patch_E)
                        
                    if landscape_ss.loc[idx, 'aa_psipred'] == 'H':  # alpha helix
                        for i in range(x, int(landscape_ss.loc[idx, 'ss_end_pos'] - min_pos + 2)):
                            cur_path = mpath.Path
                            cur_path_data = [(cur_path.MOVETO, (i, 2)), (cur_path.CURVE4, (i + 1, 2.5)), (cur_path.CURVE4, (i + 1, 2.9)), (cur_path.CURVE4, (i + 0.5, 2.9)), (cur_path.CURVE4, (i - 0.5, 2.9)), (cur_path.CURVE4, (i, 2.5)), (cur_path.CURVE4, (i + 1, 2))]
                            codes, verts = zip(*cur_path_data)
                            cur_path = mpath.Path(verts, codes)                        
                            ss_patch_H = patches.PathPatch(cur_path, linewidth=1, edgecolor='#4D4D4D', facecolor='#4D4D4D', fill=False, clip_on=False)
                            ss_patches.append(ss_patch_H)                     
                pass  
                ax_1.add_collection(collections.PatchCollection(ss_patches, match_original=True))
                
                # pfam track
    #             ax_1.text(-4, 1, "TEST", rotation = 90, fontsize=10)
                pfam_patches = []
                pfam_patch = patches.Rectangle((1, 3), landscape_colorcode.shape[1], 1, linewidth=1, edgecolor='#C0C0C0', facecolor='#C0C0C0', fill=True, clip_on=False)
                pfam_patches.append(pfam_patch)
                for idx in landscape_pfam.index:                
                    x = landscape_pfam.loc[idx, 'aa_pos'] - min_pos +1
                    x_end = int(landscape_pfam.loc[idx, 'pfam_end_pos']) - min_pos +1
                    width = x_end - x + 1
                    ax_1.text(x + width / 2, 3.3, landscape_pfam.loc[idx, 'hmm_id'], fontstyle='italic', fontweight='bold', fontsize=10,)                    
                    pfam_patch = patches.Rectangle((x, 3), width, 1, linewidth=1, edgecolor='#FFD700', facecolor='#FFD700', fill=True, clip_on=False)
                    pfam_patches.append(pfam_patch)
                pass
                ax_1.add_collection(collections.PatchCollection(pfam_patches, match_original=True))
                
                ax_1.yaxis.set_ticks([1, 2.5, 4])
                ax_1.yaxis.set_ticklabels(['Rel.ASA', 'Sec.Struc', 'Domain'])
                ax_1.patch.set_facecolor('white')
                ax_1.set_xbound(1, landscape_colorcode.shape[1] + 1)
                for loc, spine in ax_1.spines.items():
                    spine.set_linestyle('-')
                    spine.set_smart_bounds(True)
                    spine.set_linewidth(2.0)
                    spine.set_color('black')
                    spine.set_visible(False)
    #                 if loc in ['left']:
    #                     spine.set_position(('data', -20))  # outward by 10 points
    #                     spine.set_visible(True)
                pass 
                ax_1.tick_params(direction='out', length=6, width=2, colors='black', grid_color='b', grid_alpha=0.0)
                # legend
                ax_1.xaxis.set_visible(False)
                ax_1.yaxis.set_visible(False)
                
                # legend for the asa
                l = landscape_colorcode.shape[1]
                x2 = 33
    #             rectangle_legend_fitness = []
                lst_colors_asa_new = lst_max_colors_asa
                step = 2 / len(lst_colors_asa_new)
                for i in range(len(lst_colors_asa_new)):
                    y2 = step * i + 1.5
                    xy_color = lst_colors_asa_new[len(lst_colors_asa_new) - i - 1] 
                    rect = patches.Rectangle((x2 * f + l, y2), 10 * f, step, linewidth=1, edgecolor=xy_color, facecolor=xy_color, fill=True, clip_on=False)
                    ax_1.add_patch(rect)
    #                 rectangle_legend_fitness.append(rect)
                pass
    #             ax_1.add_collection(collections.PatchCollection(rectangle_legend_fitness,match_original = True))
                legend_fitness = []            
                legend_fitness.append((((x2 + 13) * f + l, 1.5), ((x2 + 13) * f + l, 3.5)))
                legend_fitness.append((((x2 + 13) * f + l, 1.5), ((x2 + 17) * f + l, 1.5)))
                legend_fitness.append((((x2 + 13) * f + l, 2.5), ((x2 + 17) * f + l, 2.5)))
                legend_fitness.append((((x2 + 13) * f + l, 3.5), ((x2 + 17) * f + l, 3.5)))
                lc = collections.LineCollection(legend_fitness, linewidth=2, color='black', clip_on=False)
                ax_1.add_collection(lc) 
                ax_1.text((x2 + 25) * f + l, 2.5, "relative ASA", rotation=90, fontsize=10, va='center', weight='bold')
                ax_1.text((x2 + 19) * f + l, 1.5, "0", rotation=90, fontsize=10, va='center', weight='bold')
                ax_1.text((x2 + 19) * f + l, 2.5, "0.5", rotation=90, fontsize=10, va='center', weight='bold')
                ax_1.text((x2 + 19) * f + l, 3.5, "1", rotation=90, fontsize=10, va='center', weight='bold')
                
                # legends for the left side       
                x = -40    
                ax_1.text((x - 5) * f, 1.5, "Rel.ASA", fontsize=10, va='center', weight='bold')
                ax_1.text((x - 5) * f, 2.5, "Sec.Struc.", fontsize=10, va='center', weight='bold')
                ax_1.text((x - 5) * f, 3.5, "Pfam", fontsize=10, va='center', weight='bold')
                x1 = x + 11
                left_legend_lines = []        
                left_legend_lines.append(((x1 * f, 1.5), ((x1 + 3) * f, 1.5)))
                left_legend_lines.append(((x1 * f, 2.5), ((x1 + 3) * f, 2.5)))
                left_legend_lines.append(((x1 * f, 3.5), ((x1 + 3) * f, 3.5)))
                left_legend_lines.append((((x1 + 3) * f, 1.5), ((x1 + 3) * f, 3.5)))            
                lc = collections.LineCollection(left_legend_lines, linewidth=2, color='black', clip_on=False)
                ax_1.add_collection(lc) 
                
                etime = time.time()    
                alm_fun.show_msg (self.log,self.verbose,"additional tracks running time was %g seconds" % (etime - stime))  
                
                ####************************************************************************************************************************************************************* 
                # column ordered genophenogram
                ####*************************************************************************************************************************************************************
                stime = time.time()
                ax_2 = plt.subplot(gs[3:5, :])                                                 
                x = [1] * landscape_colorcode_ordered.shape[0] + list(range(1, landscape_colorcode_ordered.shape[1] + 1))
                y = list(range(1, landscape_colorcode_ordered.shape[0] + 1)) + [1] * landscape_colorcode_ordered.shape[1]                        
                ax_2.plot(x, y, alpha=0)
                rectangle_patches = []
                for x_pos in landscape_colorcode_ordered.columns:
                    for y_pos in landscape_colorcode_ordered.index:
                        x = x_pos - min_pos +1 
                        y = y_pos + 1
                        xy_color = landscape_colorcode_ordered.loc[y_pos, x_pos]
                        rect = patches.Rectangle((x, y), 1, 1, linewidth=1, edgecolor=xy_color, facecolor=xy_color, fill=True, clip_on=False)
                        rectangle_patches.append(rect) 
                pass           
                ax_2.add_collection(collections.PatchCollection(rectangle_patches, match_original=True))
    
                ax_2.patch.set_facecolor('white')  
                ax_2.yaxis.set_ticks([1, 5, 10, 15, 20])
                ax_2.yaxis.set_ticklabels([0, 0.25, 0.5, 0.75, 1])
                ax_2.set_ybound(1, 21)
                ax_2.set_xbound(1, landscape_colorcode.shape[1] + 1)  
                ax_2.xaxis.set_visible(False)
                ax_2.yaxis.set_visible(False)
                for loc, spine in ax_2.spines.items():
                    spine.set_linestyle('-')
                    spine.set_smart_bounds(True)
                    spine.set_linewidth(2.0)
                    spine.set_color('black')
                    spine.set_visible(False)
    #                 if loc in ['left']:
    #                     spine.set_position(('data', -20))  # outward by 10 points
    #                     spine.set_visible(True)
                pass 
                ax_2.tick_params(direction='out', length=6, width=2, colors='black', grid_color='b', grid_alpha=0.0)
                
                # legend for the fitness
                x2 = 33
    #             rectangle_legend_fitness = []
    
                if (self.remediation[cur_data_idx] == 1) | (fitness_name == 'fitness_org'):
                    lst_colors_fitness_new = lst_max_colors_fitness + lst_min_colors_fitness
                else:
                    lst_colors_fitness_new = lst_min_colors_fitness
                step = 19 / len(lst_colors_fitness_new)
                for i in range(len(lst_colors_fitness_new)):
                    y2 = step * i + 1.5
                    xy_color = lst_colors_fitness_new[len(lst_colors_fitness_new) - i - 1] 
                    rect = patches.Rectangle((x2 * f + l, y2), 10 * f, step, linewidth=1, edgecolor=xy_color, facecolor=xy_color, fill=True, clip_on=False)
                    ax_2.add_patch(rect)
    #                 rectangle_legend_fitness.append(rect)
                pass
    #             ax_2.add_collection(collections.PatchCollection(rectangle_legend_fitness,match_original = True))
                se_legend_fitness = [] 
                
                if self.remediation[cur_data_idx] == 1:  
                    se_legend_fitness.append((((x2 + 13) * f + l, 1.5), ((x2 + 13) * f + l, 20.5)))
                    se_legend_fitness.append((((x2 + 13) * f + l, 1.5), ((x2 + 17) * f + l, 1.5)))
                    se_legend_fitness.append((((x2 + 13) * f + l, 11), ((x2 + 17) * f + l, 11)))
                    se_legend_fitness.append((((x2 + 13) * f + l, 20.5), ((x2 + 17) * f + l, 20.5)))
                    lc = collections.LineCollection(se_legend_fitness, linewidth=2, color='black', clip_on=False)
                    ax_2.add_collection(lc) 
                    ax_2.text((x2 + 25) * f + l, 11.5, "score", rotation=90, fontsize=10, va='center', weight='bold')
                    ax_2.text((x2 + 19) * f + l, 1.5, "-1", rotation=90, fontsize=10, va='center', weight='bold')
                    ax_2.text((x2 + 19) * f + l, 11, "0", rotation=90, fontsize=10, va='center', weight='bold')
                    ax_2.text((x2 + 19) * f + l, 20.5, "1", rotation=90, fontsize=10, va='center', weight='bold')
                else: 
                    if fitness_name == 'fitness_org':
                        se_legend_fitness.append((((x2 + 13) * f + l, 1.5), ((x2 + 13) * f + l, 20.5)))
                        se_legend_fitness.append((((x2 + 13) * f + l, 1.5), ((x2 + 17) * f + l, 1.5)))
                        se_legend_fitness.append((((x2 + 13) * f + l, 11), ((x2 + 17) * f + l, 11)))
                        se_legend_fitness.append((((x2 + 13) * f + l, 20.5), ((x2 + 17) * f + l, 20.5)))
                        lc = collections.LineCollection(se_legend_fitness, linewidth=2, color='black', clip_on=False)
                        ax_2.add_collection(lc) 
                        ax_2.text((x2 + 25) * f + l, 11.5, "score", rotation=90, fontsize=10, va='center', weight='bold')
                        ax_2.text((x2 + 19) * f + l, 1.5, "0", rotation=90, fontsize=10, va='center', weight='bold')
                        ax_2.text((x2 + 19) * f + l, 11, "1", rotation=90, fontsize=10, va='center', weight='bold')
                        ax_2.text((x2 + 19) * f + l, 20.5, "2", rotation=90, fontsize=10, va='center', weight='bold')  
                        
                    else:   
                        se_legend_fitness.append((((x2 + 13) * f + l, 1.5), ((x2 + 13) * f + l, 21.5)))
                        se_legend_fitness.append((((x2 + 13) * f + l, 1.5), ((x2 + 17) * f + l, 1.5)))
                        se_legend_fitness.append((((x2 + 13) * f + l, 21.5), ((x2 + 17) * f + l, 21.5)))
                        lc = collections.LineCollection(se_legend_fitness, linewidth=2, color='black', clip_on=False)
                        ax_2.add_collection(lc) 
                        ax_2.text((x2 + 25) * f + l, 11.5, "score", rotation=90, fontsize=10, va='center', weight='bold')
                        ax_2.text((x2 + 19) * f + l, 1.5, "0", rotation=90, fontsize=10, va='center', weight='bold')
                        ax_2.text((x2 + 19) * f + l, 21.5, "1", rotation=90, fontsize=10, va='center', weight='bold')
                   
                # legends for the left side       
                x = -40    
                ax_2.text((x - 5) * f, 12, "pos/neutral/neg", rotation=90, fontsize=10, va='center', weight='bold')
                left_legend_lines = []                        
                x1 = x + 11
                for i in range(6):                
                    y1 = i * (19 / 5) + 1.5
                    left_legend_lines.append(((x1 * f, y1), ((x1 + 3) * f, y1)))
                    if i % 2 == 0:
                        ax_2.text((x + 6) * f, y1, i * 0.2, fontsize=10, ha='center', weight='bold')
                pass 
                left_legend_lines.append((((x1 + 3) * f, 1.5), ((x1 + 3) * f, 20.5)))
                lc = collections.LineCollection(left_legend_lines, linewidth=2, color='black', clip_on=False)
                ax_2.add_collection(lc) 
                
                etime = time.time()   
                alm_fun.show_msg (self.log,self.verbose,"ordered genophenogram running time was %g seconds" % (etime - stime)) 
                ####************************************************************************************************************************************************************* 
                # main genophenogram
                ####*************************************************************************************************************************************************************
                stime = time.time()
                ax_3 = plt.subplot(gs[5:, :])                                                 
                x = [1] * landscape_colorcode.shape[0] + list(range(1, landscape_colorcode.shape[1] + 1))
                y = list(range(1, landscape_colorcode.shape[0] + 1)) + [1] * (landscape_colorcode.shape[1])                        
                ax_3.plot(x, y, alpha=0)
                ax_3.xaxis.set_visible(True)
                ax_3.yaxis.set_visible(False)
                ax_3.patch.set_facecolor('white')    
                
                rectangle_patches = []
                for x_pos in landscape_colorcode.columns:
                    for y_aa in landscape_colorcode.index:
                        x = x_pos - min_pos +1 
                        y = aa_plot.index(y_aa) + 1
                        xy_color = landscape_colorcode.loc[y_aa, x_pos]
                        rect = patches.Rectangle((x, y), 1, 1, linewidth=1, edgecolor=xy_color, facecolor=xy_color, fill=True, clip_on=False)
                        rectangle_patches.append(rect)
                pass
                ax_3.add_collection(collections.PatchCollection(rectangle_patches, match_original=True))
                                     
                # Draw the standard error
                stime = time.time()
                se_lines = []
                for x_pos in landscape_se.columns:
                    for y_aa in landscape_se.index:
                        x = x_pos
                        y = aa_plot.index(y_aa) + 1
                        se = landscape_se.loc[y_aa, x_pos]
                        se_lines.append(((x + (1 - se) / 2, y + (1 - se) / 2), (x + (1 + se) / 2, y + (1 + se) / 2)))  
                pass
    #             ax_3.add_collection(collections.LineCollection(se_lines, linewidth=1, color='black'))  
                
                # legends for the aa       
                x = -40   
                ax_3.text((x - 5) * f, 12, "AA residue", rotation=90, fontsize=10, va='center', weight='bold') 
                aa_legend_lines = []            
                x1 = x + 11
                for i in range(21):
                    y1 = i + 1.5
                    aa_legend_lines.append(((x1 * f, y1), ((x1 + 3) * f, y1)))  
                    ax_3.text((x + 6) * f, y1, aa_plot[i], fontsize=10, va='center', weight='bold')
                pass 
                aa_legend_lines.append((((x1 + 3) * f, 1.5), ((x1 + 3) * f, 21.5)))
                
                aa_legend_lines.append((((x1 + 7) * f, 11.5), ((x1 + 9) * f, 11.5)))
                aa_legend_lines.append((((x1 + 7) * f, 13.5), ((x1 + 9) * f, 13.5)))
                aa_legend_lines.append((((x1 + 8) * f, 11.5), ((x1 + 8) * f, 13.5)))
                ax_3.text((x1 + 10) * f, 12.5, '+', fontsize=10, va='center', weight='bold')
                
                aa_legend_lines.append((((x1 + 7) * f, 9.5), ((x1 + 9) * f, 9.5)))
                aa_legend_lines.append((((x1 + 7) * f, 10.5), ((x1 + 9) * f, 10.5)))
                aa_legend_lines.append((((x1 + 8) * f, 9.5), ((x1 + 8) * f, 10.5)))
                ax_3.text((x1 + 10) * f, 10, '-', fontsize=10, va='center', weight='bold')
                
                aa_legend_lines.append((((x1 + 13) * f, 14.5), ((x1 + 15) * f, 14.5)))
                aa_legend_lines.append((((x1 + 13) * f, 21.5), ((x1 + 15) * f, 21.5)))
                aa_legend_lines.append((((x1 + 14) * f, 14.5), ((x1 + 14) * f, 21.5)))
                ax_3.text((x1 + 17) * f, 18, 'hydrophobic', rotation=90, fontsize=10, va='center', weight='bold')
                
                aa_legend_lines.append((((x1 + 13) * f, 5.5), ((x1 + 15) * f, 5.5)))
                aa_legend_lines.append((((x1 + 13) * f, 13.5), ((x1 + 15) * f, 13.5)))
                aa_legend_lines.append((((x1 + 14) * f, 5.5), ((x1 + 14) * f, 13.5)))
                ax_3.text((x1 + 17) * f, 9, 'polar', rotation=90, fontsize=10, va='center', weight='bold')
                
                lc = collections.LineCollection(aa_legend_lines, linewidth=2, color='black', clip_on=False)
                ax_3.add_collection(lc) 
    
                # legends for the error            
                se_legend_lines = []
                x = 10
                ax_3.text(x * f + l, 12, "stderr", rotation=90, fontsize=10, va='center', weight='bold')
                ax_3.text((x + 6) * f + l, 1.5, "0", rotation=90, fontsize=10, va='center', weight='bold')
                ax_3.text((x + 6) * f + l, 11.5, "0.5", rotation=90, fontsize=10, va='center', weight='bold')
                ax_3.text((x + 6) * f + l, 21.5, "1", rotation=90, fontsize=10, va='center', weight='bold')
                
                x1 = 21
                for i in range(11):
                    se = (i) / 10  
                    y1 = i * 2 + 1.5
                    se_legend_lines.append(((x1 * f + l, y1), ((x1 + 3) * f + l, y1)))  
                    
                    y = i * 2 + 1              
                    se_legend_lines.append((((x1 + 7 + (1 - se) / 2) * f + l, y + (1 - se) / 2), ((x1 + 7 + (1 + se) / 2) * f + l, y + (1 + se) / 2)))
                pass 
                se_legend_lines.append((((x1 + 3) * f + l, 1.5), ((x1 + 3) * f + l, 21.5)))
                lc = collections.LineCollection(se_legend_lines, linewidth=2, color='black', clip_on=False)
                ax_3.add_collection(lc) 
    
                # legend for the fitness
                x2 = 33
    #             rectangle_legend_fitness = []
                if (self.remediation[cur_data_idx] == 1) | (fitness_name == 'fitness_org'):
                    lst_colors_fitness_new = [syn_color] + lst_max_colors_fitness + lst_min_colors_fitness
                else:
                    lst_colors_fitness_new = [syn_color] + lst_min_colors_fitness
                step = 20 / len(lst_colors_fitness_new)
                for i in range(len(lst_colors_fitness_new)):
                    y2 = step * i + 1.5
                    xy_color = lst_colors_fitness_new[len(lst_colors_fitness_new) - i - 1] 
                    rect = patches.Rectangle((x2 * f + l, y2), 10 * f, step, linewidth=1, edgecolor=xy_color, facecolor=xy_color, fill=True, clip_on=False)
                    ax_3.add_patch(rect)
    #                 rectangle_legend_fitness.append(rect)
                pass
    #             ax_3.add_collection(collections.PatchCollection(rectangle_legend_fitness,match_original = True))
                se_legend_fitness = []  
                
                if self.remediation[cur_data_idx] == 1:  
                    se_legend_fitness.append((((x2 + 13) * f + l, 1.5), ((x2 + 13) * f + l, 21.5)))
                    se_legend_fitness.append((((x2 + 13) * f + l, 1.5), ((x2 + 17) * f + l, 1.5)))
                    se_legend_fitness.append((((x2 + 13) * f + l, (20 - step) / 2 + 1.5), ((x2 + 17) * f + l, (20 - step) / 2 + 1.5)))
                    se_legend_fitness.append((((x2 + 13) * f + l, 21.5 - step), ((x2 + 17) * f + l, 21.5 - step)))
                    se_legend_fitness.append((((x2 + 13) * f + l, 21.5), ((x2 + 17) * f + l, 21.5)))
                    lc = collections.LineCollection(se_legend_fitness, linewidth=2, color='black', clip_on=False)
                    ax_3.add_collection(lc) 
                    ax_3.text((x2 + 25) * f + l, 11.5, "score", rotation=90, fontsize=10, va='center', weight='bold')
                    ax_3.text((x2 + 19) * f + l, 1.5, "-1", rotation=90, fontsize=10, va='center', weight='bold')
                    ax_3.text((x2 + 19) * f + l, (20 - step) / 2 + 1.5, "0", rotation=90, fontsize=10, va='center', weight='bold')
                    ax_3.text((x2 + 19) * f + l, 21.5 - step, "1", rotation=90, fontsize=10, va='center', weight='bold')  
                    ax_3.text((x2 + 19) * f + l, 21.5, "wt", rotation=90, fontsize=10, va='center', weight='bold') 
                else:       
                    if fitness_name == 'fitness_org':
                        se_legend_fitness.append((((x2 + 13) * f + l, 1.5), ((x2 + 13) * f + l, 21.5)))
                        se_legend_fitness.append((((x2 + 13) * f + l, 1.5), ((x2 + 17) * f + l, 1.5)))
                        se_legend_fitness.append((((x2 + 13) * f + l, (20 - step) / 2 + 1.5), ((x2 + 17) * f + l, (20 - step) / 2 + 1.5)))
                        se_legend_fitness.append((((x2 + 13) * f + l, 21.5 - step), ((x2 + 17) * f + l, 21.5 - step)))
                        se_legend_fitness.append((((x2 + 13) * f + l, 21.5), ((x2 + 17) * f + l, 21.5)))
                        lc = collections.LineCollection(se_legend_fitness, linewidth=2, color='black', clip_on=False)
                        ax_3.add_collection(lc) 
                        ax_3.text((x2 + 25) * f + l, 11.5, "score", rotation=90, fontsize=10, va='center', weight='bold')
                        ax_3.text((x2 + 19) * f + l, 1.5, "0", rotation=90, fontsize=10, va='center', weight='bold')
                        ax_3.text((x2 + 19) * f + l, (20 - step) / 2 + 1.5, "1", rotation=90, fontsize=10, va='center', weight='bold')
                        ax_3.text((x2 + 19) * f + l, 21.5 - step, "2", rotation=90, fontsize=10, va='center', weight='bold')  
                        ax_3.text((x2 + 19) * f + l, 21.5, "wt", rotation=90, fontsize=10, va='center', weight='bold') 
                    else:                                        
                        se_legend_fitness.append((((x2 + 13) * f + l, 1.5), ((x2 + 13) * f + l, 21.5)))
                        se_legend_fitness.append((((x2 + 13) * f + l, 1.5), ((x2 + 17) * f + l, 1.5)))
                        se_legend_fitness.append((((x2 + 13) * f + l, 18), ((x2 + 17) * f + l, 18)))
                        se_legend_fitness.append((((x2 + 13) * f + l, 21.5), ((x2 + 17) * f + l, 21.5)))
                        lc = collections.LineCollection(se_legend_fitness, linewidth=2, color='black', clip_on=False)
                        ax_3.add_collection(lc) 
                        ax_3.text((x2 + 25) * f + l, 11.5, "score", rotation=90, fontsize=10, va='center', weight='bold')
                        ax_3.text((x2 + 19) * f + l, 1.5, "stop", rotation=90, fontsize=10, va='center', weight='bold')
                        ax_3.text((x2 + 19) * f + l, 18, "syn", rotation=90, fontsize=10, va='center', weight='bold')
                        ax_3.text((x2 + 19) * f + l, 21.5, "wt", rotation=90, fontsize=10, va='center', weight='bold')
            
                ax_3.set_ybound(1, 22)    
                ax_3.yaxis.set_ticks(np.arange(1.5, 22.5, 1))
                ax_3.yaxis.set_ticklabels(aa_plot)  
                                
                lst_xaxis = list(range(0, landscape_colorcode.shape[1] + 1, 10))
                lst_xaxis[0] = 1
                
                lst_xaxis_lables = list(range(min_pos-1, max_pos + 1, 10))                
                lst_xaxis_lables[0] = min_pos
                
                ax_3.set_xbound(1, landscape_colorcode.shape[1] + 1)  
                ax_3.xaxis.set_ticks(lst_xaxis)
                ax_3.xaxis.set_ticklabels(lst_xaxis_lables)
    
                for loc, spine in ax_3.spines.items():
                    spine.set_linestyle('-')
    #                 spine.set_smart_bounds(True)
                    spine.set_linewidth(2.0)
                    spine.set_color('black')
                    spine.set_visible(False)
    #                 if loc in ['left']:
    #                     spine.set_position(('data', -20))  # outward by 10 points
    #                     spine.set_visible(True)
                    if loc in ['bottom']:
                        spine.set_position(('axes', -0.03))  # outward by 10 points
                        spine.set_visible(True)
                pass        
                ax_3.tick_params(direction='out', length=6, width=2, colors='black', grid_color='b', grid_alpha=0.0)
                etime = time.time()
                alm_fun.show_msg (self.log,self.verbose,"adding error lines running time was %g seconds" % (etime - stime))        
                alm_fun.show_msg (self.log,self.verbose,"size:" + str(len(se_lines)))            
                plt.suptitle(plot_title, fontsize=30)
    #             plt.savefig(self.project_path + 'output/' + data_name + '_' + fitness_name  + '.png',format = 'png',dpi= 300)
                plt.savefig(self.project_path + 'output/' + data_name + '_' + fitness_name + '.pdf')
    #           plt.savefig(self.project_path + 'output/' + data_name + '_landscape_' + fitness_name  + '.pdf')
                plt.close()

        ####*************************************************************************************************************************************************************
        # Other internal use maps        
        ####*************************************************************************************************************************************************************
        if plot_internal_map == 1:   
                  
            fitness_idx = dms_gene_df['fitness'].notnull()
            fitness_nonsyn_idx = dms_gene_df['fitness'].notnull() & (dms_gene_df['annotation'] == 'NONSYN')
            fitness_syn_idx = dms_gene_df['fitness'].notnull() & (dms_gene_df['annotation'] == 'SYN')
            fitness_stop_idx = dms_gene_df['fitness'].notnull() & (dms_gene_df['annotation'] == 'STOP')
            
            fitness_syn_filtered_idx = dms_gene_df['fitness'].notnull() & (dms_gene_df['annotation'] == 'SYN') & (dms_gene_df['syn_filtered'] == 0)
            fitness_stop_filtered_idx = dms_gene_df['fitness'].notnull() & (dms_gene_df['annotation'] == 'STOP') & (dms_gene_df['stop_filtered'] == 0)
           
#            
#             fitness_syn_filtered_idx = dms_gene_df['fitness'].notnull() & (dms_gene_df['annotation'] == 'SYN') & (dms_gene_df['fitness_input_sd'] < 1)
#             fitness_stop_filtered_idx = dms_gene_df['fitness'].notnull() & (dms_gene_df['annotation'] == 'STOP') & (dms_gene_df['fitness_input_sd'] < 1)
#            
                             
            #plot for the experimental fitness and imputation fitness error regression
            fig = plt.figure(figsize=(20,10))   
            ax = plt.subplot(2,2,1)   
            ax = plt.scatter(np.array(dms_gene_df.loc[fitness_idx,'fitness']), np.array(dms_gene_df.loc[fitness_idx,'fitness_sd']))
            plt.xlabel('Experimental Fitness',size = 12)
            plt.ylabel('Standard Derivation',size = 12)
            plt.title(data_name + ' - Experimental Fitness VS Standard Derivation',size = 15)
             
            ax = plt.subplot(2,2,2)   
            ax = plt.scatter(np.array(dms_gene_df.loc[fitness_idx,'fitness_sd_prior']),np.array(dms_gene_df.loc[fitness_idx,'fitness_sd']))
            plt.xlabel('SD Regressed',size = 12)
            plt.ylabel('SD Original',size = 12)
            plt.title(data_name + ' - Experimental fitness SD Regressed VS SD Original',size = 15)
              
            ax = plt.subplot(2,2,3)   
            ax = plt.scatter(np.array(dms_gene_df.loc[fitness_idx,'fitness_imputed']), np.array(dms_gene_df.loc[fitness_idx,'fitness_imputed_se']))
            plt.xlabel('Imputed Fitness',size = 12)
            plt.ylabel('Imputation Error',size = 12)
            plt.title(data_name + ' - Imputed fitness VS Imputation Error',size = 15)
             
            ax = plt.subplot(2,2,4) 
            ax = plt.scatter(np.array(dms_gene_df.loc[fitness_idx,'fitness_imputed_se_prior']),np.array(dms_gene_df.loc[fitness_idx,'fitness_imputed_se']))
            plt.xlabel('SE regressed',size = 12)
            plt.ylabel('SE original',size = 12)
            plt.title(data_name + ' - Imputation fitness SE regressed VS SE original',size = 15)
            plt.tight_layout()
            plt.savefig(self.project_path + 'output/' + data_name + '_err_regression.png')
             
             
            #plot the refined and experimental histogram plot
            bins = 20
            fig = plt.figure(figsize=(30,15))  
                          
            ax = plt.subplot(2,3,1)
            ax.hist([dms_gene_df.loc[fitness_syn_idx,'fitness_input'],dms_gene_df.loc[fitness_stop_idx,'fitness_input']],color = ['red','blue'],bins = bins)
            ax.set_title (data_name + ' - SYN/STOP distribution [before filtering]',size = 15)
            ax.set_xlabel('Experimental log foldchange',size = 10)
            
            ax = plt.subplot(2,3,2)
            ax.hist([dms_gene_df.loc[fitness_syn_filtered_idx,'fitness_input'],dms_gene_df.loc[fitness_stop_filtered_idx,'fitness_input']],color = ['red','blue'],bins = bins)
            ax.set_title (data_name + ' - SYN/STOP distribution [after filtering]',size = 15)
            ax.set_xlabel('Experimental log foldchange',size = 10)      
             
            
            ax = plt.subplot(2,3,3)
            ax.hist(dms_gene_df.loc[fitness_nonsyn_idx,'fitness_input'],bins = bins)
            ax.set_title(data_name + ' - MISSENSE distribution',size = 15)
            ax.set_xlabel('Experimental log foldchange',size = 10)
                          
                          
                          
            ax = plt.subplot(2,3,4)
            ax.hist([dms_gene_df.loc[fitness_syn_idx,'fitness_org'],dms_gene_df.loc[fitness_stop_idx,'fitness_org']],color = ['red','blue'],bins = bins)
            ax.set_title (data_name + ' - SYN/STOP distribution [before filtering]',size = 15)
            ax.set_xlabel('Experimental fitness data (normalized, NO reverse, NO floor)',size = 10)
            
            ax = plt.subplot(2,3,5)
            ax.hist([dms_gene_df.loc[fitness_syn_filtered_idx,'fitness_org'],dms_gene_df.loc[fitness_stop_filtered_idx,'fitness_org']],color = ['red','blue'],bins = bins)
            ax.set_title (data_name + ' - SYN/STOP distribution [after filtering]',size = 15)
            ax.set_xlabel('Experimental fitness data (normalized, NO reverse, NO floor)',size = 10)      
             
            
            ax = plt.subplot(2,3,6)
            ax.hist(dms_gene_df.loc[fitness_nonsyn_idx,'fitness_org'],bins = bins)
            ax.set_title(data_name + ' - MISSENSE distribution',size = 15)
            ax.set_xlabel('Experimental fitness data (normalized, NO reverse, NO floor)',size = 10)
            
            
            
            
            
            
            
              
              
    #         ax = plt.subplot(6,1,2)
    #         ax.hist([dms_gene_df.loc[fitness_syn_idx,'fitness_reverse'],dms_gene_df.loc[fitness_stop_idx,'fitness_reverse']],color = ['red','blue'],bins = bins)
    #         ax.set_title (data_name + ' - synonymous and nonsense variants fitness distribution',size = 20)
    #         ax.set_xlabel('Experimental fitness data (normalized, reversed, NO floor)',size = 15)      
    #          
    #         ax = plt.subplot(6,1,3)
    #         ax.hist([dms_gene_df.loc[fitness_syn_idx,'fitness'],dms_gene_df.loc[fitness_stop_idx,'fitness']],color = ['red','blue'],bins = bins)
    #         ax.set_title (data_name + ' - synonymous and nonsense variants fitness distribution',size = 20)
    #         ax.set_xlabel('Experimental fitness data (normalized, reversed, floored)',size = 15)                
    #         plt.xlim(-0.05,1.05)
    
    
#      
#             ax = plt.subplot(2,3,4)
#             ax.hist(dms_gene_df.loc[fitness_nonsyn_idx,'fitness'],bins = bins)
#             ax.set_title(data_name + ' - MISSENSE distribution',size = 15)
#             ax.set_xlabel('Experimental fitness data (normalized, reversed, floored)',size = 10)
#             plt.xlim(-0.05,1.05)
#          
#             ax = plt.subplot(2,3,5)
#             ax.hist(dms_gene_df.loc[fitness_nonsyn_idx,'fitness_imputed'],bins = bins)
#             ax.set_title(data_name + ' - MISSENSE distribution - imputed',size = 15)
#             ax.set_xlabel('Imputed fitness data',size = 10)
# 
#             plt.xlim(-0.05,1.05)
#          
#             ax = plt.subplot(2,3,6)
#             ax.hist(dms_gene_df.loc[fitness_nonsyn_idx,'fitness_refine'],bins = bins)
#             ax.set_title(data_name + ' - MISSENSE distribution - refined',size = 15)
#             ax.set_xlabel('Refined fitness data',size = 10)            
#             plt.xlim(-0.05,1.05)
             
#             plt.tight_layout()
            plt.savefig(self.project_path + 'output/' + data_name + '_fitness_histograms.png')
              
             
            #plot the refined and experimental scatter plot
            fig = plt.figure(figsize=(30,10)) 
                     
            ax = plt.subplot(2,2,1)
            ax  = plt.scatter(dms_gene_df.loc[fitness_nonsyn_idx,'fitness'],dms_gene_df.loc[fitness_nonsyn_idx,'fitness_org'])
            plt.xlabel('Experiment fitness',size = 12)
            plt.ylabel('Raw Experiment fitness',size = 12)
            plt.title(data_name + ' - Raw Experimental fitness VS Experimental fitness',size = 15)
             
            ax = plt.subplot(2,2,2)
            ax  = plt.scatter(dms_gene_df.loc[fitness_nonsyn_idx,'fitness'],dms_gene_df.loc[fitness_nonsyn_idx,'fitness_imputed'])
            plt.xlabel('Experiment fitness',size = 12)
            plt.ylabel('Imputed fitness',size = 12)
            plt.title(data_name + ' - Experimental fitness VS Imputed fitness',size = 15)
             
            ax = plt.subplot(2,2,3)
            ax  = plt.scatter(dms_gene_df.loc[fitness_nonsyn_idx,'fitness'],dms_gene_df.loc[fitness_nonsyn_idx,'fitness_refine'],c = np.log(dms_gene_df.loc[fitness_nonsyn_idx,'quality_score']))
            plt.xlabel('Experimental fitness',size = 12)
            plt.ylabel('Refined fitness',size = 12)
            plt.title(data_name + ' - Experimental fitness VS Refined fitness (color: log quality score)',size = 15)
     
            ax = plt.subplot(2,2,4)
            ax  = plt.scatter(dms_gene_df.loc[fitness_nonsyn_idx,'fitness'],np.log10(dms_gene_df.loc[fitness_nonsyn_idx,'quality_score']))
            plt.xlabel('Experiment fitness',size = 12)
            plt.ylabel('log10 quality score',size = 12)
            plt.title(data_name + ' - Experimental fitness VS Log10 quality score',size = 15)
     
            plt.tight_layout()
            plt.savefig(self.project_path + 'output/' + data_name + '_fitness_scatterplot.png')
                                        
    def feature_init(self):
        # blousm feature
        self.blosum_features = ['blosum30', 'blosum35', 'blosum40', 'blosum45', 'blosum50', 'blosum55', 'blosum60', 'blosum62', 'blosum65', 'blosum70', 'blosum75', 'blosum80', 'blosum85', 'blosum90', 'blosum95', 'blosum100']        
        # amino acid (ref and alt) features
        self.aa_properties = ['mw', 'pka', 'pkb', 'pi', 'hi', 'pbr', 'avbr', 'vadw', 'asa', 'pbr_10', 'avbr_100', 'vadw_100', 'asa_100', 'cyclic', 'charge', 'positive', 'negative', 'hydrophobic', 'polar', 'ionizable', 'aromatic', 'aliphatic', 'hbond', 'sulfur', 'essential', 'size']   
        self.aa_physical_ref_features = [x + '_ref' for x in self.aa_properties]
        self.aa_physical_alt_features = [x + '_alt' for x in self.aa_properties] 
        self.aa_name_features = ['aa_ref_encode', 'aa_alt_encode']        
        # flanking amino acid features
        self.flanking_k = 0
        self.kmer_physical_features = []
        self.kmer_name_features = []
        for i in range(self.flanking_k):
            self.kmer_physical_features = self.kmer_physical_features + [x + '_ref_' + str(i + 1) + '_r' for x in self.aa_properties] + [x + '_ref_' + str(i + 1) + '_l' for x in self.aa_properties]
            self.kmer_name_features = self.kmer_name_features + ['aa_ref_' + str(i + 1) + '_r_encode', 'aa_ref_' + str(i + 1) + '_l_encode']
        pass        
        # allele frequency feature
        self.af_features = ['gnomad_af']        
        # third party score features
        self.score_features = ['sift_score', 'provean_score', 'polyphen_score', 'vest_score']        
        # other features
        self.other_features = ['aa_pos', 'aa_len']
        # engineered feature
        self.engineered_features = ['relative_pos']
        # one hot feature
        self.onehot_features = ['aa_psipred']
