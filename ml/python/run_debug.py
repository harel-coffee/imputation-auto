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
# import mysql.connector
from scipy import stats
from sklearn import preprocessing
python_path = '/usr/local/projects/ml/python/'
humandb_path = '/usr/local/database/humandb/'
sys.path.append(python_path)
import alm_ml
import alm_gi
import alm_fun
import alm_humandb
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
warnings.filterwarnings("ignore")
python_path = '/usr/local/projects/ml/python/'
project_path = '/usr/local/projects/manuscript/funregressor/'
humandb_path = '/usr/local/database/humandb/'



#***************************************************************************************************************************************************************
# humandb initialization
#***************************************************************************************************************************************************************
assembly = 'GRCh37'
db_path = '/usr/local/database/humandb_new/'
db = alm_humandb.alm_humandb(db_path, assembly, new_db = 1)



#***************************************************************************************************************************************************************
# gi initialization
#***************************************************************************************************************************************************************
assembly = 'GRCh37'
ncbi_ftp = 'ftp.ncbi.nlm.nih.gov'
flanking_k = 2
gi = alm_gi.alm_gi(humandb_path, assembly, flanking_k, slow_version=0, upgrade=0)

####***************************************************************************************************************************************************************
#Time: 2019-01-26 10:45AM
#Purpose: Compare old and new test set 
####***************************************************************************************************************************************************************
# test_old = pd.read_csv(project_path + 'data/funregressor_test_final_processed_0112.csv')
# test_new = pd.read_csv(project_path + 'data/funregressor_test_final_processed_0126.csv')
# 
# test_old = test_old[['p_vid','aa_pos','aa_ref','aa_alt']]
# test_old['old_flag'] = 1
# test_new = test_new[['p_vid','aa_pos','aa_ref','aa_alt']]
# test_new['new_flag'] = 1
# 
# test_compare = pd.merge(test_old,test_new)


clinvar_gnomad_old = pd.read_csv(project_path + 'data/clinvar_plus_gnomad_srv_0927.csv')
clinvar_gnomad_new = pd.read_csv(project_path + 'data/clinvar_plus_gnomad_srv_0127-1.csv')

clinvar_gnomad_old = clinvar_gnomad_old[['chr','nt_pos','nt_ref','nt_alt','data_source','review_star','clinsig_level','gnomad_af','gnomad_gc_homo_alt']]
clinvar_gnomad_old.columns = ['chr','nt_pos','nt_ref','nt_alt','data_source','review_star_old','clinsig_level_old','gnomad_af_old','gnomad_gc_homo_alt_old']
clinvar_gnomad_old['old_flag'] = 1
clinvar_gnomad_new = clinvar_gnomad_new[['chr','nt_pos','nt_ref','nt_alt','data_source','review_star','clinsig_level','gnomad_af','gnomad_gc_homo_alt']]
clinvar_gnomad_new.columns = ['chr','nt_pos','nt_ref','nt_alt','data_source','review_star_new','clinsig_level_new','gnomad_af_new','gnomad_gc_homo_alt_new']
clinvar_gnomad_new['new_flag'] = 1

clinvar_gnomad_compare = pd.merge(clinvar_gnomad_old,clinvar_gnomad_new,how = 'outer')

clinvar_gnomad_compare.loc[(clinvar_gnomad_compare['old_flag'] == 1) & clinvar_gnomad_compare['new_flag'].isnull(),:].to_csv(project_path + 'data/clinvar_plus_gnomad_diff.csv' )


####***************************************************************************************************************************************************************
#Time: 2019-01-17 09:24AM
#Purpose: Revisist PDB PISA to add more structural related features   
####***************************************************************************************************************************************************************
# gi.prepare_pisa_data()
gi.retrieve_pisa_by_uniprotid('Q9GZX7')

# pvids_download_error = ['P31150','Q9NZ01','P42898','P35520','Q9H3S4','P38398','P31327','P28331','Q9GZX7','P07225']
# for id in pvids_download_error:
#     gi.retrieve_pisa_by_uniprotid(id)


# pvids_2652 = ['P62873', 'P12755', 'O00329', 'Q9HAN9', 'P42345', 'P29317',
#        'P60953', 'O60341', 'Q14376', 'P35914', 'P54577', 'P54819',
#        'P50897', 'P11166', 'P06132', 'Q9Y4U1', 'Q15468', 'Q13415',
#        'Q8NBP7', 'P23458', 'P49591', 'P08754', 'P01111', 'Q9UGM6',
#        'O43175', 'P54868', 'O43395', 'P43235', 'P04062', 'P30613',
#        'Q92963', 'P02545', 'P04629', 'P02549', 'Q92542', 'P50336',
#        'O75306', 'P25189', 'Q99972', 'Q6PI48', 'P15529', 'P35609',
#        'Q92736', 'P07954', 'Q96P20', 'O14832', 'Q5T5U3', 'P07949',
#        'P19367', 'P18206', 'Q00266', 'P60484', 'Q00653', 'P05093',
#        'P21802', 'Q92743', 'P04181', 'P30084', 'P01112', 'P37837',
#        'P07101', 'P51787', 'Q13586', 'O14773', 'Q9Y6N9', 'Q14896',
#        'O94766', 'Q8TDP1', 'P11498', 'P49821', 'P38935', 'Q9ULV1',
#        'Q8NCM8', 'P24752', 'Q03393', 'Q03164', 'P08397', 'Q9H3H5',
#        'P22681', 'Q01543', 'Q9GZV9', 'P04275', 'P09871', 'P50542',
#        'Q9GZX7', 'O75581', 'O00429', 'Q9Y2Z4', 'Q99959', 'Q5S007',
#        'O43323', 'Q71U36', 'P41181', 'P13647', 'P04264', 'P21860',
#        'P11802', 'Q9UHD2', 'P00439', 'Q96EY8', 'Q03426', 'Q9HBA0',
#        'Q99593', 'P20823', 'Q9Y2M0', 'Q06609', 'Q9NR61', 'P20807',
#        'Q9UGJ1', 'P35555', 'Q02750', 'P06865', 'P13804', 'P12271',
#        'P54098', 'P48735', 'P24468', 'Q9UNE7', 'P55789', 'O15553',
#        'Q92793', 'O15305', 'P22695', 'Q14807', 'P35637', 'Q9C0B1',
#        'P08253', 'P49588', 'P17735', 'Q15046', 'P34059', 'Q6P2Q9',
#        'P07359', 'P49748', 'P04637', 'P51648', 'P21359', 'P02533',
#        'P14923', 'P54802', 'P38398', 'P02730', 'Q15029', 'Q9NVS9',
#        'P10644', 'P48436', 'P16144', 'P51570', 'P10253', 'P51688',
#        'P32322', 'Q86YT6', 'O15118', 'Q719H9', 'P02766', 'Q13485',
#        'P08246', 'Q15831', 'O75251', 'Q14353', 'P36507', 'Q9GZU1',
#        'Q15833', 'P26358', 'P01130', 'Q92947', 'O00459', 'P49747',
#        'P12955', 'P39019', 'O95571', 'P18074', 'O00204', 'Q7Z406',
#        'P30153', 'P19429', 'Q9Y6K1', 'P27708', 'Q14397', 'Q9UBP0',
#        'Q16678', 'Q07889', 'P43246', 'P52701', 'Q8TCS8', 'O75923',
#        'O95630', 'Q16854', 'Q14203', 'Q6P4Q7', 'P43403', 'P14868',
#        'O43929', 'Q04771', 'Q9BYX4', 'Q99250', 'P15882', 'P02461',
#        'Q6NVY1', 'P42224', 'Q13873', 'P28331', 'P07320', 'P31327',
#        'P02751', 'Q9NZC9', 'Q13618', 'Q12756', 'P21549', 'P68400',
#        'Q9BZ23', 'P78504', 'P23526', 'P48637', 'P00813', 'O43526',
#        'Q01196', 'Q13627', 'P43320', 'O96017', 'P35240', 'P30566',
#        'Q09472', 'P17050', 'P19971', 'P15289', 'P04049', 'P10828',
#        'P37173', 'P16278', 'P40692', 'Q14524', 'P08590', 'P63316',
#        'O75369', 'Q04446', 'P07225', 'Q93099', 'P08100', 'Q9H244',
#        'Q99574', 'Q9BZK7', 'P42336', 'Q9H3D4', 'P61328', 'P35475',
#        'Q13363', 'P22607', 'P10721', 'P37840', 'Q8TBZ6', 'Q16836',
#        'P05156', 'Q8IVH4', 'P03952', 'P03951', 'P48643', 'P16871',
#        'O15520', 'O60741', 'P27986', 'P07686', 'Q9UBT6', 'Q9UI17',
#        'Q13426', 'P25054', 'O14949', 'Q86WV6', 'P12081', 'Q9P2J5',
#        'P07333', 'P54136', 'P52952', 'Q9UBV7', 'P51649', 'P22033',
#        'O43318', 'P05089', 'P21580', 'Q9Y233', 'Q14004', 'P35557',
#        'P08236', 'P04424', 'Q9Y3A5', 'P40926', 'P04792', 'P16671',
#        'O00522', 'O00189', 'P08581', 'P13569', 'P20839', 'Q14315',
#        'Q9Y5L0', 'P15056', 'Q12809', 'Q16555', 'P49675', 'P11362',
#        'P49638', 'Q7LG56', 'Q9Y6M9', 'O94761', 'O60674', 'P17643',
#        'P07902', 'Q9Y223', 'Q13825', 'P09467', 'P36897', 'O95477',
#        'Q13285', 'P17813', 'Q05193', 'P00966', 'Q9NRR6', 'P46531',
#        'Q8N0W4', 'P08842', 'Q8N5Y2', 'O76039', 'O15537', 'P08559',
#        'P51812', 'P52788', 'P51843', 'P04839', 'Q92834', 'P00480',
#        'Q6W2J9', 'Q93008', 'O00571', 'O14936', 'P21397', 'Q00604',
#        'O15550', 'O75695', 'Q9NX14', 'P51784', 'P42768', 'O60828',
#        'Q9BZS1', 'P51795', 'P41229', 'Q99714', 'Q7Z6Z7', 'Q9UHD9',
#        'P10275', 'Q92796', 'P31785', 'P08034', 'P21675', 'Q9BY41',
#        'P46100', 'Q04656', 'P00558', 'Q06187', 'P06280', 'P60891',
#        'O75914', 'O43602', 'Q9BZI7', 'O15239', 'Q13620', 'O60880',
#        'Q01968', 'O95831', 'Q8IWS0', 'P00492', 'P29965', 'Q06787',
#        'P22304', 'P51610', 'P51608', 'P21333', 'P50402', 'P27635',
#        'P11413', 'Q9Y6K9', 'P00451'] + ['P63279','P63165','P62166','Q9H3S4','P0DP23','Q9NZ01','P31150','P42898','P35520'] 
#  
# for id in pvids_2652:
#     gi.retrieve_pisa_by_uniprotid(id)




####***************************************************************************************************************************************************************
#Time: 2019-01-15 10:18PM
#Purpose: Extract aaindex   
####***************************************************************************************************************************************************************
# gi.get_aa_index()

performance_df = pd.read_csv(project_path + 'output/csv/performance_test_df.csv')
aa_index2_desc = pd.read_csv(humandb_path + 'aaindex/csv/aa_index2_desc.csv')

cols = list(performance_df.columns) 
cols = [x if x != 'Unnamed: 0' else 'aaindex_id' for x in cols]
performance_df.columns = cols

aa_index2_desc_with_performance = aa_index2_desc.merge(performance_df,how = 'left')
aa_index2_desc_with_performance.to_csv(humandb_path + 'aaindex/csv/aa_index2_desc_with_performance.csv')




####***************************************************************************************************************************************************************
#Time: 2019-01-11 08:14AM
#Purpose: Generate fasta file for each uniprot id has no pispred yet and qsub batch files for Guru grid engine   
####***************************************************************************************************************************************************************
# qsub_psipred_756 = open(humandb_path + 'psipred/qsub_psipred_758.bat','w')
# uncovered_ids_756  = ['P11802', 'P15289', 'Q13363', 'P17813', 'Q8N0W4', 'Q8N5Y2','Q92834', 'Q6W2J9', 'O14936', 'Q9NX14', 'Q92796', 'P21675','O75914', 'P27635', 'P11413']
# for id in uncovered_ids_756:
#     gi.generate_uniprot_fasta([id],id)
#     qsub_psipred_756.write('qsub -V -cwd runpsipred ~/humandb/uniprot/fasta/' + id + '.fasta' + '\n')
# qsub_psipred_756.close()

# qsub_psipred_5620 = open(humandb_path + 'psipred/qsub_psipred_5620.bat','w')
# uncovered_ids_5620  = ['Q86SQ9', 'P54819', 'Q9UIF7', 'Q6UX65', 'Q9UGM6', 'P54868',
#        'P53621', 'Q5T1V6', 'P45379', 'P61812', 'Q9ULD0', 'P07602',
#        'P62847', 'Q7KZN9', 'Q02962', 'P17405', 'Q86WG5', 'Q14654',
#        'P48547', 'Q92989', 'Q96KG9', 'Q13936', 'Q8NDX2', 'Q06124',
#        'P26440', 'Q99081', 'Q02241', 'P24468', 'P49411', 'P55017',
#        'Q15046', 'Q9HD42', 'Q15131', 'Q719H9', 'P15884', 'O00555',
#        'Q969Y2', 'O00204', 'Q7Z406', 'Q8WX94', 'Q9H902', 'O43929',
#        'Q15858', 'P02751', 'Q15147', 'Q9NUV7', 'P23526', 'P10619',
#        'O43426', 'Q13627', 'Q8NCE0', 'P10826', 'Q96H96', 'O43424',
#        'Q9UBK8', 'P48643', 'Q8N183', 'O43678', 'Q13428', 'Q5ST30',
#        'P26640', 'O60673', 'P08922', 'Q8NF91', 'Q8TDX9', 'P20839',
#        'Q15910', 'Q16555', 'Q8N6M0', 'Q7Z2E3', 'O75899', 'P06396',
#        'O60663', 'P61764', 'P00519', 'Q5JUK3', 'Q96NR3', 'P78381',
#        'Q9Y4W2', 'Q9NZ94', 'O15541', 'Q92581', 'P04000']
# 
# for id in uncovered_ids_5620:
#     gi.generate_uniprot_fasta([id],id)
#     qsub_psipred_5620.write('qsub -V -cwd runpsipred ~/humandb/uniprot/fasta/' + id + '.fasta' + '\n')
# qsub_psipred_5620.close()

####***************************************************************************************************************************************************************
#Time: 2019-01-10 11:58AM
#Purpose: convert refseq psipred to uniprot psipred, and run the uniprot ids that have no psipred yet   
####***************************************************************************************************************************************************************

# mave_data = pd.read_csv(humandb_path + 'funregressor/funregressor_training_final.csv')
# gi.prepare_psipred_data()

clinical_data = pd.read_csv(humandb_path + 'funregressor/funregressor_test_final_processed_0101_backup.csv')
clinical_data.drop(columns = [x for x in clinical_data.columns if 'psipred' in x],inplace = True)

psipred = pd.read_csv(humandb_path + 'psipred/psipred_df.csv')
clinical_data = pd.merge(clinical_data, psipred, how='left')
psipred_lst = ['E','H','C']
for ss in psipred_lst:
    clinical_data['aa_psipred' + '_' + ss] = clinical_data['aa_psipred'].apply(lambda x: int(x == ss))
pass 

clinical_data.loc[clinical_data['aa_psipred'].isnull(),:].shape

uncovered_ids = set(clinical_data.loc[clinical_data['aa_psipred'].isnull(),'p_vid'].unique())
gi.generate_uniprot_fasta(uncovered_ids, 'uniprot_uncovered_ids')

clinical_data.to_csv(humandb_path + 'funregressor/funregressor_test_final_processed.csv',index = False)

print ('OK')

####***************************************************************************************************************************************************************
#Time: 2018-11-28 10:00AM
#Purpose: retrieve the uniprot Ids from Clinvar database (Imputation paper supported ids)  
####***************************************************************************************************************************************************************
# download the most up to date clinvar and uniport id mapping 
# gi.update_ftp_files()
# gi.get_refseq_idmapping1()


# gi.get_clinvar_uniprot_ids()




####***************************************************************************************************************************************************************
#Time: 2018-08-22 8:18AM
#Purpose: Get UCSC ID mapping 
####***************************************************************************************************************************************************************
# ucsc2uniprot_dict = gi.get_ucsc_idmapping()
# print ('OK')
    
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
