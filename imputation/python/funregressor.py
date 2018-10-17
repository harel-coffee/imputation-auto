#!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3
import sys
import numpy as np
import pandas as pd
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
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
from sklearn import feature_selection as fs
from sklearn import linear_model as lm
from datetime import datetime

python_path = '/usr/local/projects/ml/python/'
sys.path.append(python_path)
import alm_project
sns.set(rc={'axes.facecolor':'#C0C0C0'}) 
warnings.filterwarnings("ignore")

class funregressor:

    def __init__(self, funregressor_init_params):
        print ('**Class: [funregressor] [__init__]......starts @' + str(datetime.now()))
        for key in funregressor_init_params:
            setattr(self, key, funregressor_init_params[key])  


            
        if self.project_init_extra == 1:
            self.project_init_extra()

        self.create_time = time.time()
        self.feature_init() 
        self.data_init()
        self.project_init() 
        
        print ('**Class: [funregressor] [__init__]......done @' + str(datetime.now()) + "\n")
        
    def project_init(self, rawdata_init_params):  
        print ('**Class: [funregressor] Fun: [project_init] .... starts @' + str(datetime.now()))          
        self.project = alm_project.alm_project(self.project_init_params,self.data_init_params,self.ml_init_params,self.es_init_params)
        print ('**Class: [funregressor] Fun: [project_init] .... done @' + str(datetime.now()) + "\n")
    
    def data_init(self):
        data_init_params['feature_engineer'] = self.feature_engineer
        data_init_params['data_slice'] = self.data_slice
        data_init_params['cv_split'] = self.cv_split
        data_init_params['test_split'] = self.test_split
        data_init_params['gradient_reshape'] = self.gradient_reshape
        
        if self.run_data_preprocess == 1:
            self.data_preprocess()

    def data_preprocess(self):
         
        funregressor_test = pd.read_csv(self.humandb_path + 'funregressor/funregressor_test_final.csv')
        funregressor_test['fitness'] = funregressor_test['label']
        funregressor_test['fitness_org'] = funregressor_test['label']
        funregressor_test['fitness_refine'] = funregressor_test['label']
        funregressor_test.loc[funregressor_test['label'] == 0,'clinsig_level'] = 3
        funregressor_test = funregressor_test.replace('.',np.nan)
        funregressor_test = funregressor_test.replace('-',np.nan)
        score_cols = [x for x in funregressor_test.columns if '_score' in x]
        score_cols =  set(score_cols) - set(['SIFT_score','Polyphen2_HDIV_score','Polyphen2_HVAR_score','MutationTaster_score','FATHMM_score','PROVEAN_score','VEST3_score'])
        for score_col in score_cols:
            funregressor_test[score_col] = funregressor_test[score_col].astype(float)
        funregressor_test.to_csv(self.humandb_path + 'funregressor/funregressor_test_final_processed.csv', index=False)
    
    def project_run(self):
        print ('Class: [funregressor] Fun: [project_run]......starts @' + str(datetime.now()))
        return_objs = self.project.run()
        print ('Class: [funregressor] Fun: [project_run]......done @' + str(datetime.now()))
        return(return_objs)       

    def feature_engineer(self, train_df, target_df): 
#         stime = time.time() 
#         
#         train_df['aa_asa'] = np.nan
#         train_df.loc[train_df['asa_mean'] > 100, 'aa_asa'] = 0
#         train_df.loc[train_df['asa_mean'] <= 100, 'aa_asa'] = 1
#          
#         target_df['aa_asa'] = np.nan
#         target_df.loc[target_df['asa_mean'] > 100, 'aa_asa'] = 0
#         target_df.loc[target_df['asa_mean'] <= 100, 'aa_asa'] = 1
#          
#         # add funsum  
#         for x in self.project.gi.dict_sums['funsum'].keys():
#             cur_funsum = self.project.gi.dict_sums['funsum'][x].copy()
#             # cur_funsum = cur_funsum.loc[cur_funsum[x+'_ste'] <0.1,:]
#             train_df = pd.merge(train_df, cur_funsum, how='left')
#             target_df = pd.merge(target_df, cur_funsum, how='left')            
#         
#         # add patsum
# #         train_df = pd.merge(train_df, self.project.gi.dict_sums['patsum_clinvar']['patsum_clinvar_label_logodds'], how='left')
# #         target_df = pd.merge(target_df, self.project.gi.dict_sums['patsum_clinvar']['patsum_clinvar_label_logodds'], how='left')
# 
#         #add delta features 
#         for x in self.aa_properties:
#             train_df[x+'_delta'] = train_df[x+'_ref'] - train_df[x+'_alt']
#             target_df[x+'_delta'] = target_df[x+'_ref'] - target_df[x+'_alt']
#             
# 
#         etime = time.time()
#         print("Elapsed time for feature engineering was %g seconds" % (etime - stime))    
        return [train_df, target_df]
     
    def feature_engineer_backup(self, data_name, train_df, target_df, extra_train_df): 
        stime = time.time() 
        
        train_df['aa_asa'] = np.nan
        train_df.loc[train_df['asa_mean'] > 100, 'aa_asa'] = 0
        train_df.loc[train_df['asa_mean'] <= 100, 'aa_asa'] = 1
         
        target_df['aa_asa'] = np.nan
        target_df.loc[target_df['asa_mean'] > 100, 'aa_asa'] = 0
        target_df.loc[target_df['asa_mean'] <= 100, 'aa_asa'] = 1
         
        # add funsum  
        for x in self.project.gi.dict_sums['funsum'].keys():
            cur_funsum = self.project.gi.dict_sums['funsum'][x].copy()
            # cur_funsum = cur_funsum.loc[cur_funsum[x+'_ste'] <0.1,:]
            train_df = pd.merge(train_df, cur_funsum, how='left')
            target_df = pd.merge(target_df, cur_funsum, how='left')            
        
        # add patsum
#         train_df = pd.merge(train_df, self.project.gi.dict_sums['patsum_clinvar']['patsum_clinvar_label_logodds'], how='left')
#         target_df = pd.merge(target_df, self.project.gi.dict_sums['patsum_clinvar']['patsum_clinvar_label_logodds'], how='left')

        #add delta features 
        for x in self.aa_properties:
            train_df[x+'_delta'] = train_df[x+'_ref'] - train_df[x+'_alt']
            target_df[x+'_delta'] = target_df[x+'_ref'] - target_df[x+'_alt']
            

        etime = time.time()
        print("Elapsed time for feature engineering was %g seconds" % (etime - stime))    
        return [train_df, target_df, extra_train_df]
    
    def data_slice(self, data_name, target_df, train_df, test_df, extra_train_df):
        ####***************************************************************************************************************************************************************
        # Slice the data
        ####***************************************************************************************************************************************************************                       
#         train_df = train_df.loc[train_df[self.dependent_variable] <0.8 , :]
#         train_df = train_df.loc[train_df['accessibility'] != 0, :]
#         train_df = train_df.loc[train_df['p_vid'] == 'P35520',:]  
        train_df.loc[train_df['quality_score'].isnull(),'quality_score'] = 0                     
        train_df = train_df.loc[train_df['aa_ref'] != train_df['aa_alt'], :]
        train_df = train_df.loc[train_df['aa_alt'] != '*', :]
        train_df = train_df.loc[train_df['quality_score'] >= self.quality_cutoff, :]
        train_df = train_df.loc[train_df['pos_importance'] <= self.pos_importance, :]
         
         
        
        
        
        test_df.loc[test_df['gnomad_af'].isnull(),'gnomad_af'] = 0
        test_df.loc[test_df['gnomad_gc_homo_alt'].isnull(),'gnomad_gc_homo_alt'] = 0                
        
        print ("Test set before slice: " + "deleterious: [" + str(test_df.loc[test_df['label'] == 1,:].shape[0]) + "] benign: [" + str(test_df.loc[test_df['label'] == 0,:].shape[0]) + "]" )                
#         test_df = test_df.loc[(test_df['clinvar_gene'] == 1),:] #remove VUS and Non clinvar gene
#         print ("Test set after remove non-clinvar gene: " + "deleterious: [" + str(test_df.loc[test_df['label'] == 1,:].shape[0]) + "] benign: [" + str(test_df.loc[test_df['label'] == 0,:].shape[0]) + "]" )
        test_df = test_df.loc[~test_df['p_vid'].isin(['P63279','P63165','P62166','Q9H3S4','P0DP23','Q9NZ01','P31150','P42898','P35520']),:] #remove DMS genes
        print ("Test set after remove dms genes: " + "deleterious: [" + str(test_df.loc[test_df['label'] == 1,:].shape[0]) + "] benign: [" + str(test_df.loc[test_df['label'] == 0,:].shape[0]) + "]" )
        test_df = test_df.loc[(test_df['review_star'] >= 2) & (test_df['clinsig_level'] == 3), :]
        print ("Test set after remove low review_star and low clinsig_level: " + "deleterious: [" + str(test_df.loc[test_df['label'] == 1,:].shape[0]) + "] benign: [" + str(test_df.loc[test_df['label'] == 0,:].shape[0]) + "]" )
        test_df = test_df.loc[~((test_df['gnomad_af'] > 0.0001) & (test_df['label'] == 1)),:] #remove high allele frequency
        print ("Test set after remove high allele frequency deleterious variants: " + "deleterious: [" + str(test_df.loc[test_df['label'] == 1,:].shape[0]) + "] benign: [" + str(test_df.loc[test_df['label'] == 0,:].shape[0]) + "]" )
        test_df = test_df.loc[~((test_df['gnomad_gc_homo_alt'] > 0) & (test_df['label'] == 1)),:] #remove high allele frequency
        print ("Test set after remove high homozgyotes counts delterious variants: " + "deleterious: [" + str(test_df.loc[test_df['label'] == 1,:].shape[0]) + "] benign: [" + str(test_df.loc[test_df['label'] == 0,:].shape[0]) + "]" )
        test_df = test_df.loc[~((test_df['label'] == 0) & (test_df['data_source'] == 'clinvar')),:] #remove clinvar benign variants
        print ("Test set after remove clinvar benign variants: " + "deleterious: [" + str(test_df.loc[test_df['label'] == 1,:].shape[0]) + "] benign: [" + str(test_df.loc[test_df['label'] == 0,:].shape[0]) + "]" )
        test_df = test_df.loc[~((test_df['label'] == 0) & (test_df['gnomad_gc_homo_alt']<=1)),:]
        print ("Test set after remove low homozgyotes counts benign variants: " + "deleterious: [" + str(test_df.loc[test_df['label'] == 1,:].shape[0]) + "] benign: [" + str(test_df.loc[test_df['label'] == 0,:].shape[0]) + "]" )
        test_df = test_df.loc[test_df['provean_new_score'].notnull() & test_df['polyphen_new_score'].notnull() & test_df['sift_new_score'].notnull(),:]
        print ("Test set after remove missing provean/polyphen/sift scores: " + "deleterious: [" + str(test_df.loc[test_df['label'] == 1,:].shape[0]) + "] benign: [" + str(test_df.loc[test_df['label'] == 0,:].shape[0]) + "]" )
#         
        test_df = test_df.loc[test_df['evm_epistatic_score'].notnull(),:]
        print ("Test set after remove EVMutation scores: " + "deleterious: [" + str(test_df.loc[test_df['label'] == 1,:].shape[0]) + "] benign: [" + str(test_df.loc[test_df['label'] == 0,:].shape[0]) + "]" )
         
        
#         test_df = test_df.loc[test_df['envi_Envision_predictions'].notnull() & test_df['envi_delta_psic'].notnull(),:]
#         print ("Test set after remove missing envision and psic scores: " + "deleterious: [" + str(test_df.loc[test_df['label'] == 1,:].shape[0]) + "] benign: [" + str(test_df.loc[test_df['label'] == 0,:].shape[0]) + "]" )
#          
        test_df = test_df.loc[test_df['CADD_raw'].notnull(),:]
        print ("Test set after remove missing CADD scores: " + "deleterious: [" + str(test_df.loc[test_df['label'] == 1,:].shape[0]) + "] benign: [" + str(test_df.loc[test_df['label'] == 0,:].shape[0]) + "]" )
        
        test_df = test_df.loc[test_df['M-CAP_score'].notnull(),:]
        print ("Test set after remove missing M-CAP scores: " + "deleterious: [" + str(test_df.loc[test_df['label'] == 1,:].shape[0]) + "] benign: [" + str(test_df.loc[test_df['label'] == 0,:].shape[0]) + "]" )        

        test_df = test_df.loc[test_df['primateai_score'].notnull(),:]
        print ("Test set after remove missing PrimateAI scores: " + "deleterious: [" + str(test_df.loc[test_df['label'] == 1,:].shape[0]) + "] benign: [" + str(test_df.loc[test_df['label'] == 0,:].shape[0]) + "]" )        

        test_df = test_df.loc[test_df['MutationTaster_selected_score'].notnull(),:]
        print ("Test set after remove missing MutationTaster scores: " + "deleterious: [" + str(test_df.loc[test_df['label'] == 1,:].shape[0]) + "] benign: [" + str(test_df.loc[test_df['label'] == 0,:].shape[0]) + "]" )        

#         test_df = test_df.loc[test_df['Eigen-raw'].notnull(),:]
#         print ("Test set after remove missing Eigen scores: " + "deleterious: [" + str(test_df.loc[test_df['label'] == 1,:].shape[0]) + "] benign: [" + str(test_df.loc[test_df['label'] == 0,:].shape[0]) + "]" )        
#   
#         test_df = test_df.loc[test_df['integrated_fitCons_score'].notnull(),:]
#         print ("Test set after remove missing fitCons scores: " + "deleterious: [" + str(test_df.loc[test_df['label'] == 1,:].shape[0]) + "] benign: [" + str(test_df.loc[test_df['label'] == 0,:].shape[0]) + "]" )        

#         test_df = test_df.loc[test_df['evm_epistatic_score'].notnull(),:]
#         print ("Test set after remove missing evmutation scores: " + "deleterious: [" + str(test_df.loc[test_df['label'] == 1,:].shape[0]) + "] benign: [" + str(test_df.loc[test_df['label'] == 0,:].shape[0]) + "]" )        

        test_df = test_df.loc[test_df['asa_mean'].notnull(),:]
        print ("Test set after remove missing asa scores: " + "deleterious: [" + str(test_df.loc[test_df['label'] == 1,:].shape[0]) + "] benign: [" + str(test_df.loc[test_df['label'] == 0,:].shape[0]) + "]" )        
         
#         test_df = test_df.loc[test_df['asa_mean'].notnull() & test_df['asa_mean_l'].notnull() & test_df['asa_mean_r'].notnull(),:]
#         print ("Test set after remove missing asa scores: " + "deleterious: [" + str(test_df.loc[test_df['label'] == 1,:].shape[0]) + "] benign: [" + str(test_df.loc[test_df['label'] == 0,:].shape[0]) + "]" )        
#           
        ####***************************************************************************************************************************************************************
        # Adding  other features
        ####***************************************************************************************************************************************************************         
        train_df['aa_asa'] = np.nan
        train_df.loc[train_df['asa_mean'] > 100, 'aa_asa'] = 0
        train_df.loc[train_df['asa_mean'] <= 100, 'aa_asa'] = 1
         
        target_df['aa_asa'] = np.nan
        target_df.loc[target_df['asa_mean'] > 100, 'aa_asa'] = 0
        target_df.loc[target_df['asa_mean'] <= 100, 'aa_asa'] = 1
        
        test_df['aa_asa'] = np.nan
        test_df.loc[test_df['asa_mean'] > 100, 'aa_asa'] = 0
        test_df.loc[test_df['asa_mean'] <= 100, 'aa_asa'] = 1
         
#         # add funsum  
#         for x in self.project.gi.dict_sums['funsum'].keys():
#             cur_funsum = self.project.gi.dict_sums['funsum'][x].copy()
#             # cur_funsum = cur_funsum.loc[cur_funsum[x+'_ste'] <0.1,:]
#             train_df = pd.merge(train_df, cur_funsum, how='left')
#             target_df = pd.merge(target_df, cur_funsum, how='left')    
#             test_df = pd.merge(test_df, cur_funsum, how='left')
        
        # add patsum
#         train_df = pd.merge(train_df, self.project.gi.dict_sums['patsum_clinvar']['patsum_clinvar_label_logodds'], how='left')
#         target_df = pd.merge(target_df, self.project.gi.dict_sums['patsum_clinvar']['patsum_clinvar_label_logodds'], how='left')

        #add delta features 
#         for x in self.aa_properties:
#             train_df[x+'_delta'] = train_df[x+'_ref'] - train_df[x+'_alt']
#             target_df[x+'_delta'] = target_df[x+'_ref'] - target_df[x+'_alt']
#             test_df[x+'_delta'] = test_df[x+'_ref'] - test_df[x+'_alt']
             
        ####***************************************************************************************************************************************************************
        # Balance the test set ?
        ####***************************************************************************************************************************************************************
        positive_idx = test_df.loc[test_df['fitness'] == 1].index
        negative_idx = test_df.loc[test_df['fitness'] == 0].index
             
#         balance new target
        diff = len(negative_idx) - len(positive_idx)        
        if diff > 0  :
            cur_negative_idx = set(np.random.permutation(list(negative_idx))[0:0 - diff])            
            cur_positive_idx = positive_idx
                 
        if diff < 0:
            cur_positive_idx = set(np.random.permutation(list(positive_idx))[0:diff])
            cur_negative_idx = negative_idx
                 
        if diff == 0:
            cur_negative_idx = negative_idx
            cur_positive_idx = positive_idx
              
        test_df = test_df.loc[cur_positive_idx.union(cur_negative_idx), :]
        return [target_df, train_df, test_df, extra_train_df]
    
    def cv_split(self, test_df):        
        ratio = 0.1    
        positive_idx = test_df.loc[test_df['fitness'] == 1].index
        validation_positive_idx = set(np.random.permutation(positive_idx)[0:int(np.floor(len(positive_idx) * ratio))])
        target_positive_idx = set(positive_idx) - validation_positive_idx
        negative_idx = test_df.loc[test_df['fitness'] == 0].index
        validation_negative_idx = set(np.random.permutation(negative_idx)[0:int(np.floor(len(negative_idx) * ratio))])
        target_negative_idx = set(negative_idx) - validation_negative_idx
        
        new_validation_df = test_df.loc[validation_positive_idx.union(validation_negative_idx), :]
        
#         balance new target
        diff = len(target_negative_idx) - len(target_positive_idx)        
        if diff > 0  :
            cur_target_negative_idx = set(np.random.permutation(list(target_negative_idx))[0:0 - diff])            
#             cur_target_negative_idx = test_df.loc[target_negative_idx,:].sort_values(['gnomad_af']).index[0:0 - diff]            
            cur_target_positive_idx = target_positive_idx
        if diff < 0:
            cur_target_negative_idx = target_negative_idx
            cur_target_positive_idx = set(np.random.permutation(list(target_positive_idx))[0:diff])
        if diff == 0:
            cur_target_negative_idx = target_negative_idx
            cur_target_positive_idx = target_positive_idx
         
#         cur_target_negative_idx = target_negative_idx
#         cur_target_positive_idx = target_positive_idx
         
        new_test_df = test_df.loc[cur_target_positive_idx.union(cur_target_negative_idx), :]

        return ([new_validation_df, new_test_df])
#         return ([new_validation_df, new_test_df])

    def test_split(self, train_df, test_df):
        return[train_df, test_df]
        
    def gradient_reshape(self,train_df): 
        train_gradient_df = {}
        
        #gradient by protein_ids
#         dict_dms_uniportid['P63279'] = 'UBE2I'
#         dict_dms_uniportid['P63165'] = 'SUMO1'
#         dict_dms_uniportid['Q9H3S4'] = 'TPK1'
#         dict_dms_uniportid['P0DP23'] = 'CALM1'
#         dict_dms_uniportid['P62166'] = 'NCS1'        
#         dict_dms_uniportid['P35520'] = 'CBS'
#         dict_dms_uniportid['Q9NZ01'] = 'TECR'
#         dict_dms_uniportid['P31150'] = 'GDI1'
#         dict_dms_uniportid['P42898'] = 'MTHFR'
#         dict_dms_uniportid['P04035'] = 'HMGCR'
#         gradients = ['UBE2I', 'SUMO1', 'NCS1', 'TPK1', 'CALM1', 'CBS', 'TECR', 'GDI1', 'MTHFR','CBS+MTHFR', 'ALL']
#         gradients_id = [['P63279'], ['P63165'], ['P62166'], ['Q9H3S4'], ['P0DP23'], ['P35520'], ['Q9NZ01'], ['P31150'], ['P42898'],['P35520','P42898'],['P63279','P63165','P62166','Q9H3S4','P0DP23','Q9NZ01','P31150','P42898','P35520']]
#         for protein_id in gradients_id:
#             cur_train_df = train_df.loc[train_df['p_vid'].isin(protein_id), :]  
#             train_gradient_df.append(cur_train_df)
#             validation_gradient_df.append(test_df)
            
#         gradient by quality cutoff
#         gradients = range(0, 550, 50)
#         for quality_cutoff in gradients:
#             cur_train_df = train_df.loc[train_df['quality_score'] >= quality_cutoff, :]  
#             train_gradient_df.append(cur_train_df)
#             validation_gradient_df.append(validation_cv_splits_df[0])
            
        #gradient by position importance
#         pos_importance_gradients = np.arange(0.5, 1.02, 0.02)
#         quality_cutoff_gradients = range(0, 550, 50)
        
        pos_importance_gradients = np.arange(0.7, 0.74, 0.02)
        quality_cutoff_gradients = range(200, 300, 50)
        
        gradients = []        
        for pos_importance in pos_importance_gradients:
            for quality_cutoff in quality_cutoff_gradients:
                gradients.append(str([pos_importance,quality_cutoff]))
                cur_train_df = train_df.loc[(train_df['pos_importance'] < pos_importance) & (train_df['quality_score'] >= quality_cutoff), :]  
                train_gradient_df[str([pos_importance,quality_cutoff])] = cur_train_df.index

        return([gradients,train_gradient_df])
        
    def gradient_reshape_backup(self, train_df,test_df,validation_cv_splits_df): 
        train_gradient_df = []
        validation_gradient_df = []
        
        #gradient by protein_ids
#         dict_dms_uniportid['P63279'] = 'UBE2I'
#         dict_dms_uniportid['P63165'] = 'SUMO1'
#         dict_dms_uniportid['Q9H3S4'] = 'TPK1'
#         dict_dms_uniportid['P0DP23'] = 'CALM1'
#         dict_dms_uniportid['P62166'] = 'NCS1'        
#         dict_dms_uniportid['P35520'] = 'CBS'
#         dict_dms_uniportid['Q9NZ01'] = 'TECR'
#         dict_dms_uniportid['P31150'] = 'GDI1'
#         dict_dms_uniportid['P42898'] = 'MTHFR'
#         dict_dms_uniportid['P04035'] = 'HMGCR'
#         gradients = ['UBE2I', 'SUMO1', 'NCS1', 'TPK1', 'CALM1', 'CBS', 'TECR', 'GDI1', 'MTHFR','CBS+MTHFR', 'ALL']
#         gradients_id = [['P63279'], ['P63165'], ['P62166'], ['Q9H3S4'], ['P0DP23'], ['P35520'], ['Q9NZ01'], ['P31150'], ['P42898'],['P35520','P42898'],['P63279','P63165','P62166','Q9H3S4','P0DP23','Q9NZ01','P31150','P42898','P35520']]
#         for protein_id in gradients_id:
#             cur_train_df = train_df.loc[train_df['p_vid'].isin(protein_id), :]  
#             train_gradient_df.append(cur_train_df)
#             validation_gradient_df.append(test_df)
            
#         gradient by quality cutoff
#         gradients = range(0, 550, 50)
#         for quality_cutoff in gradients:
#             cur_train_df = train_df.loc[train_df['quality_score'] >= quality_cutoff, :]  
#             train_gradient_df.append(cur_train_df)
#             validation_gradient_df.append(validation_cv_splits_df[0])
            
        #gradient by position importance
        pos_importance_gradients = np.arange(0.5, 1.02, 0.02)
        quality_cutoff_gradients = range(0, 550, 50)
        
#         pos_importance_gradients = np.arange(0.5, 0.62, 0.02)
#         quality_cutoff_gradients = range(0, 150, 50)
        
        gradients = []
        
        for pos_importance in pos_importance_gradients:
            for quality_cutoff in quality_cutoff_gradients:
                cur_train_df = train_df.loc[(train_df['pos_importance'] < pos_importance) & (train_df['quality_score'] >= quality_cutoff), :]  
                train_gradient_df.append(cur_train_df)
                validation_gradient_df.append(validation_cv_splits_df[0])   
                gradients.append([pos_importance,quality_cutoff])
            
                #gradient by position importance
#         gradients = [[1,1],[1,2],[1,3],[2,1],[2,2],[2,3]]
#         for gradient in gradients:              
#             train_gradient_df.append(train_df)
#             cur_test_df = test_df.loc[(test_df['review_star'] >= gradient[0]) & (test_df['clinsig_level'] <= gradient[1]), :]
#             validation_gradient_df.append(cur_test_df)   
                        
        return([train_gradient_df, validation_gradient_df, gradients])
        
    def feature_init(self):
        # blousm feature
        self.blosum_features = ['blosum30', 'blosum35', 'blosum40', 'blosum45', 'blosum50', 'blosum55', 'blosum60', 'blosum62', 'blosum65', 'blosum70', 'blosum75', 'blosum80', 'blosum85', 'blosum90', 'blosum95', 'blosum100']        
        # amino acid (ref and alt) features
        self.aa_properties = ['mw', 'pka', 'pkb', 'pi', 'hi', 'pbr', 'avbr', 'vadw', 'asa', 'pbr_10', 'avbr_100', 'vadw_100', 'asa_100', 'cyclic', 'charge', 'positive', 'negative', 'hydrophobic', 'polar', 'ionizable', 'aromatic', 'aliphatic', 'hbond', 'sulfur', 'essential', 'size']   
        self.aa_physical_ref_features = [x + '_ref' for x in self.aa_properties]
        self.aa_physical_alt_features = [x + '_alt' for x in self.aa_properties] 
        self.aa_physical_delta_features = [x + '_delta' for x in self.aa_properties]
        
        lst_aa_20 = ["S", "A", "V", "R", "D", "F", "T", "I", "L", "K", "G", "Y", "N", "C", "P", "E", "M", "W", "H", "Q"]        
#         self.aa_name_features = ['aa_ref_encode', 'aa_alt_encode']        
        # flanking amino acid features
        self.flanking_k = 2
        self.aa_name_features = []
        for aa in lst_aa_20:
            self.aa_name_features = self.aa_name_features + ['aa_ref_' + aa]
            self.aa_name_features = self.aa_name_features + ['aa_alt_' + aa]
            

        self.kmer_physical_features = []
        self.kmer_name_features = []
        for i in range(self.flanking_k):
            self.kmer_physical_features = self.kmer_physical_features + [x + '_ref_' + str(i + 1) + '_r' for x in self.aa_properties] + [x + '_ref_' + str(i + 1) + '_l' for x in self.aa_properties]
            
#             self.kmer_name_features = self.kmer_name_features + ['aa_ref_' + str(i + 1) + '_r_encode', 'aa_ref_' + str(i + 1) + '_l_encode']
            self.kmer_name_features = []
            for aa in lst_aa_20:
                self.aa_name_features = self.aa_name_features + ['aa_ref_' + str(i + 1) + '_l_' +  aa]
                self.aa_name_features = self.aa_name_features + ['aa_ref_' + str(i + 1) + '_r_' +  aa]
            
            
            
        pass   
    
        # psipred features
        self.aa_psipred_features = ['aa_psipred_E','aa_psipred_H','aa_psipred_C']     
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
        # envision fetures
        self.envision_features = ['envi_AA1_PI','envi_AA2_PI','envi_deltaPI','envi_AA1_weight',
                                  'envi_AA2_weight','envi_deltaWeight','envi_AA1vol','envi_AA2vol',
                                  'envi_deltavolume','envi_AA1_psic','envi_AA2_psic','envi_delta_psic',
                                  'envi_accessibility','envi_delta_solvent_accessibility',
                                  'envi_b_factor','envi_mut_msa_congruency','envi_mut_mut_msa_congruency',
                                  'envi_seq_ind_closest_mut','envi_evolutionary_coupling_avg',
                                  'envi_evolutionary_coupling_prop','envi_evolutionary_coupling_avg_norm']

#         self.envision_features = ['envi_AA1_PI', 'envi_AA2_PI', 'envi_deltaPI', 'envi_AA1_weight',
#                                   'envi_AA2_weight', 'envi_deltaWeight', 'envi_AA1vol', 'envi_AA2vol',
#                                   'envi_deltavolume', 'envi_AA1_psic', 'envi_AA2_psic', 'envi_delta_psic']     
        
