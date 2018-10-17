import sys
import numpy as np
import pandas as pd
import random
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt 
import matplotlib.path as mpath
import matplotlib.patches as patches  
from matplotlib.lines import Line2D  
from matplotlib.gridspec import GridSpec
import matplotlib.collections as collections
import codecs
import re
import datetime
import time
import warnings
import pickle
import glob
import os
import mysql.connector
from scipy import stats
from sklearn import preprocessing
python_path = '/usr/local/projects/ml/python/'
humandb_path = '/usr/local/database/humandb/'
sys.path.append(python_path)
import alphame_ml
import alphame_gi
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
warnings.filterwarnings("ignore")
python_path = '/usr/local/projects/ml/python/'
project_path = '/usr/local/projects/imputation/gwt/www/'
humandb_path = '/usr/local/database/humandb/'

#***************************************************************************************************************************************************************
# gi initialization
#***************************************************************************************************************************************************************
assembly = 'GRCh37'
ncbi_ftp = 'ftp.ncbi.nlm.nih.gov'
flanking_k = 2
gi = alphame_gi.alm_gi(humandb_path, assembly, flanking_k, slow_version=0, upgrade=0)

####***************************************************************************************************************************************************************
#Time: 2018-08-22 8:18AM
#Purpose: Get UCSC ID mapping 
####***************************************************************************************************************************************************************
# ucsc2uniprot_dict = gi.get_ucsc_idmapping()
# print ('OK')

####***************************************************************************************************************************************************************
#Time: 2018-08-16 7:01AM
#Purpose: revisit the Clinvar record check the review status 
####***************************************************************************************************************************************************************
# gi.get_gnomad_af()
#gi.update_ftp_files()
# gi.get_refseq_seqdict()
# gi.get_refseq_idmapping()
# gi.prepare_funregressor_test_data()
# gi.prepare_funregressor_training_data()
# gi.variant_process_clinvar()
# gi.prepare_eigen_data()
# gi.prepare_evmutation_data()

gi.prepare_psipred_data()
# gi.prepare_psipred_region_data()



        
        
        
#check the difference between the old data
cols = ['chr', 'nt_pos', 'nt_ref', 'nt_alt','p_vid','aa_pos','aa_ref','aa_alt','gnomad_gc_homo_alt','gnomad_af','clin_sig','clinsig_level','review_star','evaluate_time','data_source','label']
clinvar_plus_gnomad_srv_old = pd.read_csv(humandb_path + 'funregressor/clinvar_plus_gnomad_final_funregressor.csv')
clinvar_plus_gnomad_srv_old = clinvar_plus_gnomad_srv_old.loc[(clinvar_plus_gnomad_srv_old['label'] == 1) & (clinvar_plus_gnomad_srv_old['review_star'] >0),cols]
funregressor_test = pd.read_csv(humandb_path+ 'funregressor/funregressor_test_final_processed.csv')  
funregressor_test = funregressor_test.loc[funregressor_test['label'] == 1,cols]


funregressor_test.loc[(funregressor_test['review_star']>=2) & (funregressor_test['clinsig_level'] == 3) ]



clinvar_plus_gnomad_srv_groupby  = clinvar_plus_gnomad_srv.groupby(['chr','nt_pos','nt_ref','nt_alt']).size().reset_index()
clinvar_plus_gnomad_srv_groupby = clinvar_plus_gnomad_srv_groupby.rename(columns = {0:'srv_counts'})
clinvar_plus_gnomad_srv = pd.merge(clinvar_plus_gnomad_srv,clinvar_plus_gnomad_srv_groupby)        
clinvar_plus_gnomad_srv.loc[clinvar_plus_gnomad_srv['srv_counts']>1,:] 
        
clinvar_plus_gnomad_srv.loc[clinvar_plus_gnomad_srv['nt_pos'] == 44474003,:]  


         
        #save srv and vcf file
   


# #check dbnsfp
# uniprot_accs = "Q9Y6K9-3;Q9Y6K9-2;F8W9A9;Q9Y6K9"
# uniprot_aaposs = "157" 
# uniprot_acc = "Q9Y6K9"
# 
# 
# def retrieve_aapos(uniprot_accs, uniprot_acc, uniprot_aaposs):
#     try:        
#         uniprot_accs_list = uniprot_accs.split(";")
#         uniprot_poss_list = uniprot_aaposs.split(";")
#         
#         if len(uniprot_poss_list) == 1:
#             uniprot_aa_pos = uniprot_poss_list[0]
#         else:
#             unprot_accs_dict = {uniprot_accs_list[x]:x for x in range(len(uniprot_accs_list))}        
#             uniprot_aa_pos = uniprot_poss_list[unprot_accs_dict.get(uniprot_acc,np.nan)]
#         if not chk_int(uniprot_aa_pos):
#             uniprot_aa_pos = np.nan
#         else:
#             uniprot_aa_pos = int(uniprot_aa_pos)
#             
#     except:
#         uniprot_aa_pos = np.nan
#     return uniprot_aa_pos
# 
# def retrieve_ensembl_pos_index(uniprot_aa_pos,ensembl_aaposs):
#     try:
#         ensembl_poss_list = ensembl_aaposs.split(";")
#         ensembl_poss_dict = {ensembl_poss_list[x]:x for x in range(len(ensembl_poss_list))}
#         ensembl_pos_index = ensembl_poss_dict.get(uniprot_aa_pos,np.nan)
#         
#     except:
#         ensembl_pos_index  = np.nan
#     return ensembl_pos_index
# 
# 
# def chk_int(str):
#     try:
#         x = int(str)        
#         return True
#     except:
#         return False
# 
# def chk_float(str):
#     try:
#         x = float(str)        
#         return True
#     except:
#         return False
#     
# def get_value_byfun(values, fun):
#     try:
#         value_list = values.split(";")
# #         value_list = list(filter(lambda a: a.isnumeric(), value_list))
#         value_list = [float(x) for x in value_list if chk_float(x)]
#         if fun == 'min':
#             value = min(value_list)
#         if fun == 'max':
#             value = max(value_list)
#     except:
#         value = np.nan
#     return value
#                   
#     
#      
# dbnsfp_out = pd.read_csv(humandb_path + 'clinvar/csv/clinvar_srv_dbnsfp.out',sep = '\t')
# basic_cols = ['clinvar_id','rs_dbSNP150','genename','Gene_full_name','chr','nt_pos','nt_ref','nt_alt','Uniprot_acc','Uniprot_acc_Polyphen2','uniprot_aa_pos','aapos','aaref','aaalt','clinsig_level','review_star','evaluate_time','data_source','label']
# score_cols = ['SIFT_score','Polyphen2_HDIV_score','Polyphen2_HVAR_score','LRT_score','MutationTaster_score','MutationAssessor_score','FATHMM_score','PROVEAN_score','VEST3_score'] + \
#              ['MetaSVM_score','MetaLR_score','M-CAP_score','REVEL_score','MutPred_score','CADD_raw','DANN_score','fathmm-MKL_coding_score','Eigen-raw','GenoCanyon_score'] + \
#              ['integrated_fitCons_score','GERP++_RS','phyloP100way_vertebrate','phyloP20way_mammalian','phastCons100way_vertebrate','phastCons20way_mammalian','SiPhy_29way_logOdds'] + \
#              ['SIFT_selected_score','Polyphen2_selected_HDIV_score','Polyphen2_selected_HVAR_score','MutationTaster_selected_score','FATHMM_selected_score','PROVEAN_selected_score','VEST3_selected_score']
# 
# dbnsfp_out['uniprot_aa_pos'] = dbnsfp_out.apply(lambda x: retrieve_aapos(x['Uniprot_acc_Polyphen2'],x['Uniprot_acc'],x['Uniprot_aapos_Polyphen2']),axis = 1 )
# dbnsfp_out['SIFT_selected_score'] = dbnsfp_out.apply(lambda x: get_value_byfun(x['SIFT_score'],'min'),axis = 1)
# dbnsfp_out['Polyphen2_selected_HDIV_score'] = dbnsfp_out.apply(lambda x: get_value_byfun(x['Polyphen2_HDIV_score'],'max'),axis = 1)
# dbnsfp_out['Polyphen2_selected_HVAR_score'] = dbnsfp_out.apply(lambda x: get_value_byfun(x['Polyphen2_HVAR_score'],'max'),axis = 1)
# dbnsfp_out['MutationTaster_selected_score'] = dbnsfp_out.apply(lambda x: get_value_byfun(x['Polyphen2_HVAR_score'],'max'),axis = 1)
# dbnsfp_out['FATHMM_selected_score'] = dbnsfp_out.apply(lambda x: get_value_byfun(x['Polyphen2_HVAR_score'],'min'),axis = 1)
# dbnsfp_out['PROVEAN_selected_score'] = dbnsfp_out.apply(lambda x: get_value_byfun(x['Polyphen2_HVAR_score'],'min'),axis = 1)
# dbnsfp_out['VEST3_selected_score'] = dbnsfp_out.apply(lambda x: get_value_byfun(x['Polyphen2_HVAR_score'],'max'),axis = 1)
# 
# 
# # dbnsfp_out['uniprot_longest_aa_pos'] = dbnsfp_out.apply(lambda x: get_value_byfun(x['Uniprot_aapos_Polyphen2'],'max'),axis = 1)
# # dbnsfp_out['ensembl_longest_aa_pos'] = dbnsfp_out.apply(lambda x: get_value_byfun(x['aapos'],'max'),axis = 1)
# # dbnsfp_out.loc[dbnsfp_out['uniprot_aa_pos'] != dbnsfp_out['uniprot_longest_aa_pos'],['rs_dbSNP150','genename','Uniprot_acc_Polyphen2','Uniprot_acc','Uniprot_aapos_Polyphen2','aapos','uniprot_aa_pos','uniprot_longest_aa_pos','ensembl_longest_aa_pos']]
# 
# dbnsfp_out_uniprot_filtered = dbnsfp_out.loc[dbnsfp_out['uniprot_aa_pos'].notnull() & (dbnsfp_out['Uniprot_acc'].str.len() == 6) & (dbnsfp_out['aaref'] != 'X') & (dbnsfp_out['aaalt'] != 'X') ,basic_cols + score_cols ]
# 
# dbnsfp_out[['Polyphen2_HDIV_score','VEST3_score']]
# 
# print(str(dbnsfp_out.shape))




####***************************************************************************************************************************************************************
#Time: 2018-07-21 8:03AM , 2018-07-24 5:37PM
#Purpose: Try new fucntion to pre-compute dms gene features 
####***************************************************************************************************************************************************************
# gi.prepare_dms_features('P63165',0,0)

#gi.prepare_dms_features(dms_gene_id,0)
# clinvar_srv = pd.read_csv(humandb_path + 'clinvar/csv/clinvar_srv.csv')        
# clinvar_ids = clinvar_srv['p_vid'].unique()
#  
# for clinvar_id in clinvar_ids:
#     gi.prepare_dms_features(clinvar_id,0,1)
#     
# for key in gi.dict_dms_genes.keys():
#         gi.prepare_dms_features(gi.dict_dms_genes[key],0,1)


####***************************************************************************************************************************************************************
#Time: 2018-07-24 5:02PM
#Purpose: Create BLOSUM , FUNSUM and AA properties file (separate them from the feature table)
####***************************************************************************************************************************************************************
# dms_feature_df = pd.read_csv(humandb_path + "dms/P63279_features.csv")
# gi.aa_properties.to_csv(humandb_path + "dms/features/aa_properties.csv",index = False)
# gi.df_blosums.to_csv(humandb_path + "dms/features/blosums.csv",index = False)
# gi.dict_sums['funsum']['funsum_fitness_mean'].to_csv(humandb_path + "dms/features/funsum.csv",index = False)

####***************************************************************************************************************************************************************
#### aa_ref and aa_alt AA properties
####***************************************************************************************************************************************************************
# aa_properties_df = pd.read_csv(humandb_path + "dms/features/aa_properties.csv")          
# aa_properties_features = aa_properties_df.columns        
# aa_properties_ref_features = [x + '_ref' for x in aa_properties_features]
# aa_properties_alt_features = [x + '_alt' for x in aa_properties_features]   
# aa_properties_ref =aa_properties_df.copy()
# aa_properties_ref.columns = aa_properties_ref_features
# aa_properties_alt = aa_properties_df.copy()
# aa_properties_alt.columns = aa_properties_alt_features                
# dms_feature_df = pd.merge(dms_feature_df, aa_properties_ref, how='left')
# dms_feature_df = pd.merge(dms_feature_df, aa_properties_alt, how='left')

####***************************************************************************************************************************************************************
#### merge with the blosum properties
####***************************************************************************************************************************************************************
# df_blosums = pd.read_csv(humandb_path + "dms/features/blosums.csv")         
# dms_feature_df = pd.merge(dms_feature_df, df_blosums, how='left')

####***************************************************************************************************************************************************************
#### merge with the funsum properties
####***************************************************************************************************************************************************************
# funsum_df = pd.read_csv(humandb_path + "dms/features/funsum.csv")  
# dms_feature_df = pd.merge(dms_feature_df, funsum_df, how='left')

####*************************************************************************************************************************************************************
#### Encode name features
####*************************************************************************************************************************************************************        
# dms_feature_df['aa_ref_encode'] = dms_feature_df['aa_ref'].apply(lambda x: gi.aa_encode_notnull(x))
# dms_feature_df['aa_alt_encode'] = dms_feature_df['aa_alt'].apply(lambda x: gi.aa_encode_notnull(x))

####***************************************************************************************************************************************************************
#Time: 2018-07-22 9:44AM
#Purpose: Create test imputation input file  
####***************************************************************************************************************************************************************
# gi.prepare_dms_input(dms_gene_id)
# syn_ratio = 0.6
# stop_ratio = 0.6
# missense_ratio = 0.6
# dms_input = pd.read_csv(humandb_path + 'dms/for_score/' + dms_gene_id + '_for_score.txt',sep = ' ')
# dms_input.columns = ['p_vid','aa_pos','aa_ref','aa_alt']
# dms_input['quality_score'] = np.random.random(dms_input.shape[0])*2000
# dms_input['num_replicates'] = 2
# dms_input['fitness_input'] = np.nan
# dms_input['fitness_input_sd'] = np.random.random(dms_input.shape[0])*0.2
# 
# #syn
# syn_index = dms_input.loc[dms_input['aa_ref'] == dms_input['aa_alt'],:].index
# syn_left_index = np.random.random_integers(0,len(syn_index),int(len(syn_index)*syn_ratio))
# dms_input.loc[syn_left_index,'fitness_input'] = np.random.random(len(syn_left_index))*0.1 + 0.9
# 
# #stop
# stop_index = dms_input.loc[dms_input['aa_ref'] == dms_input['aa_alt'],:].index
# stop_left_index = np.random.random_integers(0,len(stop_index),int(len(stop_index)*stop_ratio))
# dms_input.loc[stop_left_index,'fitness_input'] = np.random.random(len(stop_left_index))*0.1
# 
# #missen
# missense_index = dms_input.loc[dms_input['aa_ref'] == dms_input['aa_alt'],:].index
# missense_left_index = np.random.random_integers(0,len(missense_index),int(len(missense_index)*missense_ratio))
# dms_input.loc[missense_left_index,'fitness_input'] = np.random.random(len(missense_left_index))
# 
# 
# dms_input = dms_input.loc[~dms_input['fitness_input'].isnull(),:][['aa_ref','aa_pos','aa_alt','quality_score','num_replicates','fitness_input','fitness_input_sd']]
# dms_input.to_csv(humandb_path + 'dms/dms_input/' + dms_gene_id + '_input.txt',sep = '\t',index = False)
# with open(humandb_path + 'dms/dms_input/' + dms_gene_id + '.fasta') as f
#     f.write('>' + dms_gene_id + '\n')
#     f.write()

####***************************************************************************************************************************************************************
#Time: 2018-07-22 9:30AM
#Purpose: Check log of DMS feature processing 
####***************************************************************************************************************************************************************
# feature_log = pd.read_csv(humandb_path + 'dms/feature_log.txt',sep = '\t')
# feature_log.columns = ['p_vid','asa','psipred','pfam','polyphen','sift','provean','provean_sift']      
# 
# feature_log.loc[feature_log['asa'] == 0,:]
# feature_log.loc[feature_log['psipred'] == 0,:]
# feature_log.loc[feature_log['pfam'] == 0,:].shape
# feature_log.loc[feature_log['polyphen'] == 0,:]
# feature_log.loc[feature_log['provean'] == 0,:]

####***************************************************************************************************************************************************************
#Time: 2018-07-21 8:18AM
#Purpose: psipred data, create individual psipred file for each uniprot id  
####***************************************************************************************************************************************************************
# psipred_path = humandb_path + 'psipred/' 
# for psipred_file in glob.glob(os.path.join(psipred_path, '*.seq')):
#     cur_psipred_df = pd.read_csv(psipred_file, header=None, sep='\t')
#     cur_psipred_df.columns = ['aa_psipred', 'aa_pos', 'ss_end_pos']
#     # convert to uniprot id 
#     p_id = psipred_file[len(psipred_path):-4]
#     if 'NP' in p_id:
# #         p_vid = self.refseq2uniprot_dict.get(p_id, None)
# #         if p_vid is not None:
# #             cur_psipred_df.to_csv(self.db_path + 'psipred/dms_feature/' + p_vid + '_psipred.csv',index = False)
#         a=1
#     else:
#         cur_psipred_df.to_csv(humandb_path + 'psipred/dms_feature/' + p_id + '_psipred.csv',index = False)


####***************************************************************************************************************************************************************
#Time: 2018-07-20 10:23PM
#Purpose: Imputation pipeline pre-computed features for clinvar related uniprot IDs 
####***************************************************************************************************************************************************************
# clinvar_srv = pd.read_csv(humandb_path + 'clinvar/csv/clinvar_srv.csv')        
# clinvar_ids = clinvar_srv['p_vid'].unique()

####***************************************************************************************************************************************************************
#Time: 2018-07-20 10:23PM
#Purpose: Prepare gene features for dms genes in Roth Lab
####***************************************************************************************************************************************************************
# dms_gene_ids = ['P63279','P63165','P62166','Q9H3S4','P0DP23','Q9NZ01','P31150','P42898','P35520']
# dms_gene_ids = ['P0DP23']
# for dms_gene_id in dms_gene_ids:
#     gi.prepare_dms_features(dms_gene_id,0)

####***************************************************************************************************************************************************************
#Time: 2018-07-20 9:27PM
#Purpose: polyphen data, create individual polyphen file for each uniprot id 
####***************************************************************************************************************************************************************
# polyphen_df = pd.read_csv(humandb_path + 'polyphen/uniprot_polyphen.csv',header = None)
# polyphen_df.columns = ['p_vid', 'aa_pos', 'aa_ref', 'aa_alt', 'polyphen_score']
# gene_ids = polyphen_df['p_vid'].unique()
# for gene_id in gene_ids:
#     cur_polyphen = polyphen_df.loc[polyphen_df['p_vid'] == gene_id,:]
#     cur_polyphen.to_csv(humandb_path + 'polyphen/dms_feature/' + gene_id + '_polyphen.csv',index = False) 

# cur_pid = ''
# with open(humandb_path + 'polyphen/uniprot_polyphen.csv') as infile:
#     for line in infile:
#         new_pid =  line.split(',')[0]
#         if (new_pid != cur_pid):
#             if cur_pid != '':
#                 f.close()
#             f = open(humandb_path + 'polyphen/dms_feature/' + new_pid + '_polyphen.csv','w')
#             f.write(line)
#             cur_pid = new_pid
#         else:
#             f.write(line)          

####***************************************************************************************************************************************************************
#Time: 2018-07-20 10:05PM
#Purpose: PROVEAN data, create individual PROVEAN file for each uniprot id 
####***************************************************************************************************************************************************************
# cur_pid = ''
# with open(humandb_path + 'provean/PROVEAN_scores_ensembl66_human.tsv') as infile:
#     for line in infile:
#         new_pid =  line.split('\t')[0]
#         if (new_pid != cur_pid):
#             if cur_pid != '':
#                 f.close()
#             f = open(humandb_path + 'provean/dms_feature/' + new_pid + '_provean.tsv','w')
#             f.write(line)
#             cur_pid = new_pid
#         else:
#             f.write(line)       

            
####***************************************************************************************************************************************************************
#Time: 2018-07-20 10:05PM
#Purpose: SIFT data, create individual SIFT file for each uniprot id 
####***************************************************************************************************************************************************************
# cur_pid = ''
# with open(humandb_path + 'sift/SIFT_scores_and_info_ensembl66_human.tsv') as infile:
#     for line in infile:
#         new_pid =  line.split('\t')[0]
#         if (new_pid != cur_pid):
#             if cur_pid != '':
#                 f.close()
#             f = open(humandb_path + 'sift/dms_feature/' + new_pid + '_sift.tsv','w')
#             f.write(line)
#             cur_pid = new_pid
#         else:
#             f.write(line)       
    
####***************************************************************************************************************************************************************
#Time: 2018-07-20 9:09PM
#Purpose: pdb asa data, create individual asa file for each uniprot id 
####***************************************************************************************************************************************************************
# cur_pid = ''
# with open(humandb_path + 'pfam/9606.tsv') as infile:
#     next(infile)
#     next(infile)
#     next(infile)
#     for line in infile:
#         new_pid =  line.split('\t')[0]
#         if (new_pid != cur_pid):
#             if cur_pid != '':
#                 f.close()
#             f = open(humandb_path + 'pfam/dms_feature/' + new_pid + '_pfam.tsv','w')
#             f.write(line)
#             cur_pid = new_pid
#         else:
#             f.write(line)   

####***************************************************************************************************************************************************************
#Time: 2018-07-20 11:03PM
#Purpose: pfam data, create individual pfam file for each uniprot id 
####***************************************************************************************************************************************************************
# asa_df = pd.read_csv(humandb_path + 'pfam/9606.tsv',header = None)
# asa_df.columns = ['aa_pos', 'aa_ref', 'asa_mean', 'asa_std', 'asa_count','p_vid']
# gene_ids = asa_df['p_vid'].unique()
# 
# for gene_id in gene_ids:
#     cur_asa = asa_df.loc[asa_df['p_vid'] == gene_id,:]
#     cur_asa.to_csv(humandb_path + 'pdb/dms_feature/' + gene_id + '_asa.csv',index = False) 

####***************************************************************************************************************************************************************
#Time: 2018-07-20 11:26PM
#Purpose: gnoamd data, create individual gnomad file for each uniprot id 
####***************************************************************************************************************************************************************
# cur_pid = ''
# with open(humandb_path + 'gnomad/gnomad_aa.txt') as infile:
#     for line in infile:
#         new_pid =  line.split('\t')[0]
#         if (new_pid != cur_pid):
#             if cur_pid != '':
#                 f.close()
#             f = open(humandb_path + 'gnomad/dms_feature/' + new_pid + '_gnomad.txt','w')
#             f.write(line)
#             cur_pid = new_pid
#         else:
#             f.write(line)   



print ("OK")
