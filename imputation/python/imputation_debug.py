#!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3
import sys
import numpy as np
import pandas as pd
import random
import codecs
import re
import datetime
import gensim
import time
import warnings
import pickle
from scipy import stats
from sklearn import preprocessing
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt 
python_path = '/usr/local/projects/ml/python/'
project_path = '/usr/local/projects/imputation/project/'
humandb_path = '/usr/local/database/humandb/'
sys.path.append(python_path)
import alm_gi
import imputation
import imputation_web 
import alm_fun
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
warnings.filterwarnings("ignore")

#--------------------------------------------------------------------------------------- Project main scripts ---------------------------------------------------------------------------------------
server_address = 'smtp-relay.gmail.com'
server_port = 587
login_user = 'noreply@varianteffect.org'
login_password = 'EnlightenedLlamaInNirvana'
from_address = 'noreply@varianteffect.org'
subject = 'No Reply'


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

def create_email_msg(session_id):      
    msg = "*** This is an automatically generated e-mail, please do not reply!\n" + \
    "*** Send your questions and comments to: joe.wu@varianteffect.org\n" + \
    "Your imputation session " + "'" + session_id + "'" + " has completed at " + time.ctime() + "(Eastern Time). " + \
    "You can revisit your imputed map by entering your session ID in 'View Landscape' section at impute.varianteffect.org. " + \
    "You also can access the imputed results directly from the following links:\n" + \
    "(1) Original variant effect map: http://impute.varianteffect.org/output/" + session_id + "_fitness_refine.pdf\n" + \
    "(2) Refined variant effect map: http://impute.varianteffect.org/output/" + session_id + "_fitness_org.pdf\n" + \
    "(3) Imputation results: http://impute.varianteffect.org/output/" + session_id + "_imputation.csv\n\n" + \
    "Regards,\nRoth Lab"            
    return(msg)

def create_email_error_msg(user_email,session_id):
    msg = "Imputation error happened on user " +"[" + user_email + "] session " + "'" + session_id  + "' at " + time.ctime() + "(Eastern Time). " + \
          "Error Log: http://impute.varianteffect.org/log/" + session_id + ".log\n" + \
          "Regards,\nRoth Lab"  
    return(msg)  
    
def create_email_notification_msg(session_id):
    msg = "Imputation happened with session " + "[" +  session_id  + "] at " + time.ctime() + "(Eastern Time). " + \
          "Check Log at http://impute.varianteffect.org/log/" + session_id + ".log\n" + \
          "Regards,\nRoth Lab"  
    return(msg)



#***************************************************************************************************************************************************************
# Create data for BRAC1
#***************************************************************************************************************************************************************
# brca1_org = pd.read_csv(project_path + 'downloads/imputation_example_BRCA1/41586_2018_461_MOESM3_ESM.csv', skiprows =2)
# brca1_raw = brca1_org.loc[brca1_org['consequence'].isin(['Missense','Synonymous','Nonsense']), ['aa_pos','aa_ref','aa_alt','reference','alt','consequence','d5.r1','d11.r1','d5.r2','d11.r2','negative','negative','negative','negative']]
# brca1_raw.columns = ['aa_pos','aa_ref','aa_alt','nt_ref','nt_alt','annotation','nonselect1','select1','nonselect2','select2','controlS1','controlS2','controlNS1','contorlNS2']
#  
#  
# brca1_processed = brca1_org.loc[brca1_org['consequence'].isin(['Missense','Synonymous','Nonsense']), ['aa_pos','aa_ref','aa_alt','library','function.score.r1','function.score.r2']]
# brca1_processed.columns = ['aa_pos','aa_ref','aa_alt','quality_score','fitness1','fitness2']
# 
# brca1_processed = brca1_processed.groupby(['aa_pos','aa_ref','aa_alt'])['quality_score','fitness1','fitness2'].mean().reset_index()
# 
#  
# brca1_processed['num_replicates'] = 2
# brca1_processed['fitness_input'] = brca1_processed[['fitness1','fitness2']].mean(axis = 1)
# brca1_processed['fitness_input_sd'] = brca1_processed[['fitness1','fitness2']].std(axis = 1)
# brca1_processed = brca1_processed[['aa_pos','aa_ref','aa_alt','quality_score','num_replicates','fitness_input','fitness_input_sd']]
#  
# brca1_processed.to_csv(project_path + 'downloads/imputation_example_BRCA1/processedData_BRCA1.txt',sep = '\t',index = False)
#  
# print ('OK')




#***************************************************************************************************************************************************************
# Create maps for the supported Uniport ID 
#***************************************************************************************************************************************************************
# uniprot_id = 'P63279'
# imputation_web_output = open(project_path + 'output/*' + uniprot_id + '.out', 'w')
# dms_gene_csv_df = pd.read_csv(humandb_path + 'dms/features/'+ uniprot_id + '_features.csv')
# 
# # ass colorcode
# [lst_max_colors_asa, lst_min_colors_asa] = alm_fun.create_color_gradients(1, 0, 0, '#3155C6', '#FFFFFF', '#FFFFFF', 10, 10)
# #         self.plot_color_gradients(1, 0, 0, lst_max_colors_asa, lst_min_colors_asa, img_fig_width, img_fig_height, 'H', 'ASA score', project_path + 'asa_legend.png')  
# dms_gene_csv_df['asa_mean_normalized'] = (dms_gene_csv_df['asa_mean'] - np.nanmin(dms_gene_csv_df['asa_mean'])) / (np.nanmax(dms_gene_csv_df['asa_mean']) - np.nanmin(dms_gene_csv_df['asa_mean']))
# dms_gene_csv_df['asa_colorcode'] = dms_gene_csv_df['asa_mean_normalized'].apply(lambda x: alm_fun.get_colorcode(x, 1, 0, 0, 10, 10, lst_max_colors_asa, lst_min_colors_asa))
# 
# # sift colorcode
# #         [lst_max_colors_sift,lst_min_colors_sift] = alm_fun.create_color_gradients(1, 0, 0.05,'#C6172B','#FFFFFF','#3155C6',10,10)
# [lst_max_colors_sift, lst_min_colors_sift] = alm_fun.create_color_gradients(1, 0, 0, '#FFFFFF', '#3155C6', '#3155C6', 10, 10)
# #         self.plot_color_gradients(1, 0, 0, lst_max_colors_sift, lst_min_colors_sift, img_fig_width, img_fig_height, 'H', 'SIFT Score', project_path + 'sift_legend.png')
# dms_gene_csv_df['sift_colorcode'] = dms_gene_csv_df['sift_score'].apply(lambda x: alm_fun.get_colorcode(x, 1, 0, 0, 10, 10, lst_max_colors_sift, lst_min_colors_sift))
# 
# [lst_max_colors_polyphen, lst_min_colors_polyphen] = alm_fun.create_color_gradients(1, 0, 0, '#3155C6', '#FFFFFF', '#FFFFFF', 10, 10)
# #         self.plot_color_gradients(1, 0, 0, lst_max_colors_polyphen, lst_min_colors_polyphen, img_fig_width, img_fig_height, 'H', 'Polyphen Score', project_path + 'polyphen_legend.png')
# dms_gene_csv_df['polyphen_colorcode'] = dms_gene_csv_df['polyphen_score'].apply(lambda x: alm_fun.get_colorcode(x, 1, 0, 0, 10, 10, lst_max_colors_polyphen, lst_min_colors_polyphen))
# 
# [lst_max_colors_gnomad, lst_min_colors_gnomad] = alm_fun.create_color_gradients(10, 0.3, 0.3, '#3155C6', '#FFFFFF', '#FFFFFF', 10, 10)
# #         self.plot_color_gradients(10, 0.3, 0.3, lst_max_colors_gnomad, lst_min_colors_gnomad, img_fig_width, img_fig_height, 'H', '-Log10 GNOMAD Score', project_path + 'gnomad_legend.png')
# dms_gene_csv_df['gnomad_af_log10'] = 0 - np.log10(dms_gene_csv_df['gnomad_af'])
# dms_gene_csv_df['gnomad_colorcode'] = dms_gene_csv_df['gnomad_af_log10'].apply(lambda x: alm_fun.get_colorcode(x, 10, 0.3, 0.3, 10, 10, lst_max_colors_gnomad, lst_min_colors_gnomad))
# 
# [lst_max_colors_provean, lst_min_colors_provean] = alm_fun.create_color_gradients(4, -13, -13, '#FFFFFF', '#3155C6', '#3155C6', 10, 10)
# #         self.plot_color_gradients(4, -13, -13, lst_max_colors_provean, lst_min_colors_provean, img_fig_width, img_fig_height, 'H', 'Provean Score', project_path + 'provean_legend.png') 
# dms_gene_csv_df['provean_colorcode'] = dms_gene_csv_df['provean_score'].apply(lambda x: alm_fun.get_colorcode(x, 4, -13, -13, 10, 10, lst_max_colors_provean, lst_min_colors_provean))
#  
# 
# out_file =  dms_gene_csv_df.to_json(orient='records')
# imputation_web_output.write(out_file)
# imputation_web_output.close()
# 
# 






#***************************************************************************************************************************************************************
# Create new raw data 
#***************************************************************************************************************************************************************
# old_data_files = ['rawData_UBE2I_solid.txt', 'rawData_SUMO1_solid.txt', 'rawData_NCS1_2017Q20.txt', 'rawData_TPK1.txt', 'rawData_CALM1.txt', 'rawData_B6-400_Q5_new.txt', 'rawData_TECR_2017Q20.txt', 'rawData_GDI1_2016Q20.txt', 'rawData_WT_Q20_fol100.txt']
#  
#  
# for old_data_file in old_data_files: 
#     raw_data = pd.read_csv(project_path + 'upload/' + old_data_file, sep = '\t')
#     raw_data_r1 = raw_data[['wt_aa','pos','mut_aa','wt_codon','mut_codon','annotation','nonselect1','select1','controlNS1','controlS1']]
#     raw_data_r1.columns = ['wt_aa','pos','mut_aa','wt_codon','mut_codon','annotation','nonselect','select','controlNS','controlS']
#     raw_data_r1['replicate_id'] = 1
#     raw_data_r2 = raw_data[['wt_aa','pos','mut_aa','wt_codon','mut_codon','annotation','nonselect2','select2','controlNS2','controlS2']]
#     raw_data_r2.columns = ['wt_aa','pos','mut_aa','wt_codon','mut_codon','annotation','nonselect','select','controlNS','controlS']
#     raw_data_r2['replicate_id'] = 2
#     raw_data_new = pd.concat([raw_data_r1,raw_data_r2])     
#     raw_data_new.to_csv(project_path + 'upload/' + old_data_file,sep = '\t',index = False)
#      

#***************************************************************************************************************************************************************
# debug for the web application 
#***************************************************************************************************************************************************************
dict_arg = pickle.load(open(project_path + "output/P38398[BRCA1-1]_1.pickle", "rb"))
JSON_Return = imputation_web.run_imputation(1,dict_arg) 
print (JSON_Return)   

#***************************************************************************************************************************************************************
# Imputation Project Parameters (Debug)
#***************************************************************************************************************************************************************
imputation_debug_log = open(project_path + 'log/imputation_debug.log', 'a') 

 #alm_project class parameters
project_params = {}

project_params['project_name'] = 'imputation'
project_params['project_path'] = project_path
project_params['humandb_path'] = humandb_path
project_params['log'] = imputation_debug_log 
project_params['verbose'] = 1

    
#the reason the following parameters don't belong to data class is we may want to create multiple data instance in one project instance 
project_params['data_names'] = [] 
project_params['train_data'] = []        
project_params['test_data'] = []
project_params['target_data'] = []
project_params['extra_train_data'] = []
project_params['use_extra_train_data'] = []
project_params['input_data_type'] = []
 
project_params['run_data_names'] = None
project_params['run_estimator_name'] = 'xgb_r'
project_params['run_estimator_scorename'] = 'rmse'
project_params['grid_search_on'] = 0

project_params['modes'] = None
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
    
#alm_data class parameters
data_params = {}
data_params['path'] = project_path 
data_params['log'] = imputation_debug_log  
data_params['verbose'] = 1

data_params['name'] = None 
data_params['target_data_original_df'] = None
data_params['train_data_original_df'] = None
data_params['test_data_original_df'] = None                
data_params['extra_train_data_original_df'] = None 
data_params['use_extra_train_data'] =  None
data_params['predicted_target_df'] = None

data_params['independent_testset'] = 0

data_params['test_split_method'] = 0
data_params['test_split_folds'] = 1
data_params['test_split_ratio'] = 0.1
data_params['cv_split_method'] = 0
data_params['cv_split_folds'] = 10
data_params['cv_split_ratio'] = 0.1
data_params['validation_from_testset'] = False
data_params['percent_min_feature'] = 1

data_params['dependent_variable'] = 'fitness'
data_params['filter_target'] = 0
data_params['filter_test'] = 0
data_params['filter_train'] = 0
data_params['filter_validation'] = 0
data_params['prediction_bootstrapping'] = 0
data_params['bootstrapping_num'] = 3

data_params['if_gradient'] = 1
data_params['if_engineer'] = 1
data_params['load_from_disk'] = 0
data_params['save_to_disk'] = 1
data_params['cur_test_split_fold'] = 0
data_params['cur_gradient_key'] = 'no_gradient'

data_params['onehot_features'] = []
    
#alm_ml class parameters
ml_params = {}
ml_params['log'] = imputation_debug_log 
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
es_params['if_feature_engineer'] = 0
es_params['feature_engineer'] = None
 
#data preprocess and update data_params                  
imputation_params = {} 
imputation_params['log'] = imputation_debug_log 
imputation_params['verbose'] = 1
imputation_params['project_path'] = project_path
imputation_params['humandb_path'] = humandb_path

imputation_params['project_params'] = project_params
imputation_params['data_params'] = data_params
imputation_params['ml_params'] = ml_params
imputation_params['es_params'] = es_params
   
#imputation class: parameters for data preprocessing
imputation_params['run_data_preprocess'] = 1 

#**************FOR Multiple DMS Data Imputations********************************
# imputation_params['dms_landscape_files'] = ['rawData_UBE2I_solid.txt', 'rawData_SUMO1_solid.txt', 'rawData_NCS1_2017Q20.txt', 'rawData_TPK1.txt', 'rawData_CALM1.txt', 'rawData_B6-400_Q5_new.txt', 'rawData_TECR_2017Q20.txt', 'rawData_GDI1_2016Q20.txt', 'rawData_WT_Q20_fol100.txt']
# imputation_params['dms_fasta_files'] = ['P63279.fasta','P63165.fasta', 'P62166.fasta', 'Q9H3S4.fasta', 'P0DP23.fasta', 'P35520.fasta', 'Q9NZ01.fasta', 'P31150.fasta', 'P42898.fasta']
# imputation_params['dms_protein_ids'] = ['P63279', 'P63165', 'P62166', 'Q9H3S4', 'P0DP23', 'P35520', 'Q9NZ01', 'P31150', 'P42898']
# imputation_params['data_names'] = ['UBE2I', 'SUMO1', 'NCS1', 'TPK1', 'CALM1', 'CBS', 'TECR', 'GDI1', 'MTHFR']
# imputation_params['quality_cutoffs'] = [0, 0, 0, 0, 0, 0, 0, 0, 0]
# imputation_params['regression_quality_cutoffs'] = imputation_params['quality_cutoffs']
# imputation_params['proper_num_replicates'] = [8, 8, 8, 8, 8, 8, 8, 8, 8]
# imputation_params['raw_processed'] = [0, 0, 0, 0, 0, 0, 0, 0, 0]
# imputation_params['normalized_flags'] = [0, 0, 0, 0, 0, 0, 0, 0, 0]
# imputation_params['regularization_flags'] = [1, 1, 1, 1, 1, 1, 1, 1, 1]
# imputation_params['reverse_flags'] = [1, 1, 1, 1, 1, 1, 1, 1, 1]
# imputation_params['combine_flags'] = [1, 1, 1, 1, 1, 1, 1, 1, 1]
# imputation_params['synstop_cutoffs'] = [0, 0, 0, 0, 0, 0, 0, 0, 0]
# imputation_params['stop_exclusion'] = ["0", "0", "0", "0", "0", "0", "0", "0", "0"]
# imputation_params['remediation'] = [0, 0, 0, 0, 0, 0, 0, 0, 0]
# imputation_params['pre_process'] = 1
# imputation_params['combine_dms'] = 1

#**************FOR Individual DMS Data Imputation********************************
imputation_params['dms_landscape_files'] = ['rawData_SUMO1_solid.txt']
imputation_params['dms_fasta_files'] = ['P63165.fasta']
imputation_params['dms_protein_ids'] = ['P63165']
imputation_params['data_names'] = ['SUMO1']
 
# imputation_params['dms_landscape_files'] = ['rawData_UBE2I_new.txt']
# imputation_params['dms_fasta_files'] = ['P63279.fasta']
# imputation_params['dms_protein_ids'] = ['P63279']
# imputation_params['data_names'] = ['UBE2I']
 
imputation_params['quality_cutoffs'] = [0]
imputation_params['regression_quality_cutoffs'] = imputation_params['quality_cutoffs']
imputation_params['proper_num_replicates'] = [8]
imputation_params['raw_processed'] = [0]
imputation_params['normalized_flags'] = [0]
imputation_params['regularization_flags'] = [1]
imputation_params['reverse_flags'] = [1]
imputation_params['combine_flags'] = [1]
imputation_params['synstop_cutoffs'] = [0]
imputation_params['stop_exclusion'] = ["0"]
imputation_params['remediation'] = [0]
imputation_params['pre_process'] = 1
imputation_params['combine_dms'] = 0
    
#imputation class: parameters for feature engineering
imputation_params['k_range'] = range(3, 4)
imputation_params['use_funsums'] = ['funsum_fitness_mean']
imputation_params['use_funsums_name'] = ['fs']  
imputation_params['value_orderby'] = imputation_params['use_funsums']
imputation_params['value_orderby_name'] = imputation_params['use_funsums_name']
imputation_params['centrality_names'] = ['mean', 'se', 'count']
imputation_params['dependent_variable'] = 'fitness'
imputation_params['add_funsum_onfly'] = 0 

#imputation class: Jochen's R script related parameters
imputation_params['if_runR'] = 0
imputation_params['R_command'] = '/Library/Frameworks/R.framework/Versions/3.3/Resources/Rscript'
imputation_params['R_wd'] = '/Users/joewu/Google_Drive/Business/AlphaMe/Source_Code/R/R_GI/Jochen/dmspipeline'
imputation_params['R_script_path'] = '/Users/joewu/Google_Drive/Business/AlphaMe/Source_Code/R/R_GI/Jochen/dmspipeline/bin/simpleImpute.R'
imputation_params['R_bend'] = '0'   

#imputation class: email related parameters
imputation_params['email_server_address'] = server_address
imputation_params['email_server_port'] = server_port
imputation_params['email_login_user'] = login_user
imputation_params['email_login_password'] = login_password
imputation_params['email_from_address'] = from_address
imputation_params['email_msg_content'] = create_email_msg('DEBUG')
imputation_params['email_error_content'] = create_email_error_msg('joe.wu.ca@gmail.com','DEBUG')
imputation_params['email_notification_content'] = create_email_notification_msg('DEBUG')
    
im_proj = imputation.imputation(imputation_params) 
   

#**************************************************************************************************#
# training features
#**************************************************************************************************#   
features = []
for k in im_proj.k_range:
    for orderby_name in im_proj.value_orderby_name:
        cur_col_feature_names = [x + '_' + str(k) + '_' + orderby_name for x in im_proj.centrality_names]
        features += cur_col_feature_names  
features = features + ['polyphen_score', 'provean_score', 'blosum100'] + im_proj.aa_name_features + im_proj.aa_physical_ref_features + im_proj.aa_physical_alt_features
im_proj.project.train_features = features
im_proj.project.train_features_name = features

#*************************************************************
# show methods comparison
#*************************************************************
# for gene_name in ['UBE2I']:
#     im_proj.project.run_data_names = [gene_name] 
#     im_proj.project.data[gene_name].load_from_disk = 1
#     im_proj.project.modes = ['method_comparison']
#     im_proj.project.compare_methods = ['xgb_r','rf_r','en_r','ada_en_r']
#     run_return = im_proj.project.run(refresh_data = 1)
#     

#*************************************************************
# Cross-validation and feature importance
#*************************************************************   
# for gene_name in ['SUMO1']:
#     im_proj.project.run_data_names = [gene_name] 
#     im_proj.project.data[gene_name].load_from_disk = 1
#     im_proj.project.modes = ['cross_validation']
#     run_return = im_proj.project.run(refresh_data = 1)
#     
#     
    
#*************************************************************
# gradient comparsion
#*************************************************************   
for gene_name in ['SUMO1']:
    im_proj.project.run_data_names = [gene_name] 
    im_proj.project.data[gene_name].load_from_disk = 1
    im_proj.project.modes = ['gradient_comparison']
    run_return = im_proj.project.run(refresh_data = 1)
    
#**************************************************************************************************#
# Run individual dms gene
#**************************************************************************************************#
im_proj.project.run_data_names = ['NCS1']   
im_proj.run_imputation('NCS1', feature_engineer_params, data_slice_params, data_split_params, project_run_params)


#**************************************************************************************************#
# Run all imputations to create FunRgerssor training data
#**************************************************************************************************#
funregerssor_train_df = None
for dms_name in rawdata_init_params['data_names']:
#     project_run_params['run_data_names'] = [dms_name]
#     project_run_params['train_features'] = im_proj.create_train_features(feature_engineer_params)
#     project_run_params['train_features_name'] = project_run_params['train_features']
#     im_proj.run_imputation(dms_name, feature_engineer_params, data_slice_params, data_split_params, project_run_params)
    
    cur_train_dms_df = pd.read_csv(project_path + 'output/' + dms_name +'_funregressor.csv') 
    if funregerssor_train_df is None:
        funregerssor_train_df = cur_train_dms_df
    else:
        funregerssor_train_df = pd.concat([funregerssor_train_df, cur_train_dms_df])
        
if funregerssor_train_df is not None:
#     funregerssor_train_df = funregerssor_train_df.reset_index()
    funregerssor_train_df['clinvar_gene'] = 0
    funregerssor_train_df.to_csv(humandb_path +'dms/funregressor_training_from_imputation.csv',index = False)      

#***************************************************************************************************************************************************************
# debug imputation ML  
#***************************************************************************************************************************************************************    
# project_run_params['train_features'] = im_proj.create_train_features(feature_engineer_params)
# project_run_params['train_features_name'] = project_run_params['train_features']

#*************************************************************
# compare Joe's pipeline with Jochen's pipeline
#*************************************************************
# project_run_params['train_features'] = ['mean_3_bs100', 'mean_3_fs', 'mean_3_fs_d', 'polyphen_score', 'provean_score', 'blosum100']
# project_run_params['train_features_name'] = ['knn_blousm', 'knn_funsum', 'knn_funsum_domain', 'polyphen_score', 'provean_score', 'blosum100']
# project_run_params['modes'] = ['prediction']
# JSON_return = im_proj.run_imputation('UBE2I', feature_engineer_params, data_slice_params, data_split_params, project_run_params)
# print (JSON_return)
# im_proj.run_imputation_R('UBE2I')

#*************************************************************
# show the best k used in kNN
#*************************************************************
# project_run_params['train_features'] = [['mean_2_fs'],['polyphen_score'],im_proj.aa_name_features,im_proj.aa_physical_ref_features + im_proj.aa_physical_alt_features]
# project_run_params['train_features'] = []
# project_run_params['start_features'] = []
# project_run_params['start_features_name'] = 'N/A'
# project_run_params['compare_features'] = ['mean_fitness'] + ['mean_' + str(k) + '_fs' for k in feature_engineer_params['k_range']]
# project_run_params['compare_features_name'] = ['All_Mean'] + ['Mean_' + str(k) for k in feature_engineer_params['k_range']]
# project_run_params['feature_compare_direction'] = 0
# project_run_params['modes'] = ['feature_comparison']
# run_return = im_proj.project_run(feature_engineer_params, data_slice_params, data_split_params, project_run_params)

#*************************************************************
# show methods comparison
#*************************************************************
# project_run_params['train_features'] = [['mean_2_fs'],['polyphen_score'],im_proj.aa_name_features,im_proj.aa_physical_ref_features + im_proj.aa_physical_alt_features]
# project_run_params['modes'] = ['method_comparison']
# run_return = im_proj.project_run(feature_engineer_params, data_slice_params, data_split_params, project_run_params)

# *************************************************************
#  Single feature (group) performance
# *************************************************************
# project_run_params['train_features'] = [['mean_3_fs'],['polyphen_score'],im_proj.aa_name_features,im_proj.aa_physical_ref_features + im_proj.aa_physical_alt_features]
# project_run_params['train_features_name'] = 'All features'
# project_run_params['start_features'] = []
# project_run_params['start_features_name'] = ''
# project_run_params['compare_features'] = [['mean_3_fs'],['polyphen_score'],im_proj.aa_name_features,im_proj.aa_physical_ref_features + im_proj.aa_physical_alt_features]
# project_run_params['compare_features_name'] = ['knn_FunSUM','Polyphen_score','aa names','aa properties']
# project_run_params['feature_compare_direction'] = 0
# project_run_params['modes'] = ['feature_comparison']
# run_return = im_proj.project_run(feature_engineer_params, data_slice_params, data_split_params, project_run_params)
# print (run_return)

# *************************************************************
#  Single feature (group) removal performance 
# *************************************************************
# project_run_params['train_features'] = []
# project_run_params['train_features_name'] = ''
# project_run_params['start_features'] = [['mean_3_fs'],['polyphen_score'],im_proj.aa_name_features,im_proj.aa_physical_ref_features + im_proj.aa_physical_alt_features]
# project_run_params['start_features_name'] = 'All features'
# project_run_params['compare_features'] = [['mean_3_fs'],['polyphen_score'],im_proj.aa_name_features,im_proj.aa_physical_ref_features + im_proj.aa_physical_alt_features]
# project_run_params['compare_features_name'] = ['-knn_FunSUM','-Polyphen_score','-aa names','-aa properties']
# project_run_params['feature_compare_direction'] = 1
# project_run_params['modes'] = ['feature_comparison']
# run_return = im_proj.project_run(feature_engineer_params, data_slice_params, data_split_params, project_run_params)
# print (run_return)

# *************************************************************
#  show knn_funsum_indomain > knn_funsum > knn_blousm 
# *************************************************************
# project_run_params['train_features'] = [['mean_2_fs'],['polyphen_score'],im_proj.aa_name_features,im_proj.aa_physical_ref_features + im_proj.aa_physical_alt_features]
# project_run_params['start_features'] = []
# project_run_params['start_features_name'] = ''
# project_run_params['compare_features'] = [['mean_2_fs'],['polyphen_score'],im_proj.aa_name_features,im_proj.aa_physical_ref_features + im_proj.aa_physical_alt_features]
# project_run_params['compare_features_name'] = ['knn_FunSUM','Polyphen_score','aa names','aa properties']
# project_run_params['feature_compare_direction'] = 0
# project_run_params['modes'] = ['feature_comparison']
# run_return = im_proj.project_run(feature_engineer_params, data_slice_params, data_split_params, project_run_params)
# print (run_return)





#***************************************************************************************************************************************************************
# Compare new and old raw data processing  
#***************************************************************************************************************************************************************
# imputation_old = pd.read_csv(project_path + 'output/P63279[UBE2I_Example[old]]_imputation.csv')
# imputation_new = pd.read_csv(project_path + 'output/P63279[UBE2I_Example[new]]_imputation.csv')
# 
# imputation_new1 = imputation_new[['aa_ref','aa_pos','aa_alt','fitness_input','fitness_input_sd','fitness_imputed']]
# imputation_new1.columns = ['aa_ref','aa_pos','aa_alt','fitness_input_new','fitness_input_sd_new','fitness_imputed_new']
#     
# imputation_old1 = imputation_old[['aa_ref','aa_pos','aa_alt','fitness_input','fitness_input_sd','fitness_imputed']]
# imputation_old1.columns = ['aa_ref','aa_pos','aa_alt','fitness_input_old','fitness_input_sd_old','fitness_imputed_old']
#     
# imputation_compare = pd.merge(imputation_old1,imputation_new1,how = 'left')
# imputation_compare.loc[imputation_compare['fitness_input_new'] == imputation_compare['fitness_input_old'],['aa_ref','aa_pos','aa_alt','fitness_input_new','fitness_input_old','fitness_imputed_new','fitness_imputed_old']]
#  
# #  
# imputation_old = pd.read_csv(project_path + 'output/P63279[UBE2I_Example[old]]_raw_processed_regression_df.csv')
# imputation_new = pd.read_csv(project_path + 'output/P63279[UBE2I_Example[new]]_raw_processed_regression_df.csv')
#    
# imputation_new1 = imputation_new[['wt_aa','pos','mut_aa','select_mean']]
# imputation_new1.columns = ['aa_ref','aa_pos','aa_alt','select_mean_new']
#    
# imputation_old1 = imputation_old[['wt_aa','pos','mut_aa','select_mean']]
# imputation_old1.columns = ['aa_ref','aa_pos','aa_alt','select_mean_old']
#    
# imputation_compare = pd.merge(imputation_old1,imputation_new1,how = 'left')
# imputation_compare.loc[imputation_compare['select_mean_new'] == imputation_compare['select_mean_old'],:]
# 
#    
# # imputation_old.loc[(imputation_old['wt_aa'] == 'A') & (imputation_old['mut_aa'] == 'D') & (imputation_old['pos'] == 10),:]
# # imputation_new.loc[(imputation_new['wt_aa'] == 'A') & (imputation_new['mut_aa'] == 'D') & (imputation_new['pos'] == 10),:]
# 
# print ('OK')



 
#***************************************************************************************************************************************************************
# Combine CBS raw data
# ***************************************************************************************************************************************************************   
# CBS_B0_old = pd.read_csv(project_path + 'rawdata/rawData_B6-0_Q5_old.txt')
# CBS_B1_old = pd.read_csv(project_path + 'rawdata/rawData_B6-1_Q5_old.txt')
# CBS_B400_old = pd.read_csv(project_path + 'rawdata/rawData_B6-400_Q5_old.txt')
# CBS_B0_new = pd.read_csv(project_path + 'rawdata/rawData_B6-0_Q5_new.txt')
# CBS_B1_new = pd.read_csv(project_path + 'rawdata/rawData_B6-1_Q5_new.txt')
# CBS_B400_new = pd.read_csv(project_path + 'rawdata/rawData_B6-400_Q5_new.txt')
# 
# CBS_B0 = pd.concat([CBS_B0_old,CBS_B0_new])
# CBS_B1 = pd.concat([CBS_B1_old,CBS_B1_new])
# CBS_B400 = pd.concat([CBS_B400_old,CBS_B400_new])
# 
# CBS_B01 = pd.concat([CBS_B0_old,CBS_B0_new,CBS_B1_old,CBS_B1_new])
# 
# CBS_B0.to_csv(project_path + 'rawdata/rawData_B6-0_Q5.txt')
# CBS_B1.to_csv(project_path + 'rawdata/rawData_B6-1_Q5.txt')
# CBS_B400.to_csv(project_path + 'rawdata/rawData_B6-400_Q5.txt')
# CBS_B01.to_csv(project_path + 'rawdata/rawData_B6-01_Q5.txt')
# 
# print('OK')

#***************************************************************************************************************************************************************
 # Compare to Jochen's published result
# ***************************************************************************************************************************************************************
# calm1_Jochen = pd.read_csv(project_path + 'imputed_regularized_calm1_flipped_scores.csv')
# calm1_Jochen['aa_ref'] = calm1_Jochen['mut'].apply(lambda x: x[:1])
# calm1_Jochen['aa_alt'] = calm1_Jochen['mut'].apply(lambda x: x[-1:])
# calm1_Jochen['aa_pos'] = calm1_Jochen['mut'].apply(lambda x: x[1:-1])
# calm1_Jochen['aa_pos'] = calm1_Jochen['aa_pos'].astype(int)
# calm1_Joe = pd.read_csv(project_path + 'CALM1_imputation.csv')
#          
# calm1 = pd.merge(calm1_Joe,calm1_Jochen,how = 'left')
# calm1 = calm1.loc[calm1['annotation'] == 'NONSYN']
# calm1['label'] = np.nan
# calm1.loc[calm1['mut'].isin(['N54I', 'F90L', 'N98S', 'D130G', 'F142L', 'Q136P', 'D134H', 'D132E', 'N98I']),'label'] = 1
# calm1.loc[calm1['mut'].isin(['I10V', 'A16T', 'P67S', 'R107C', 'V109I', 'V143L', 'Q144E', 'I131T', 'I131L', 'R127G', 'Q50R', 'T35I', 'T27S', 'G24A', 'S18L', 'K14R', 'T6S', 'I10T', 'I28V', 'G62R', 'M77I', 'R87Q', 'A89V', 'A103T', 'A104T', 'R107H', 'N138S', 'M146V']),'label'] = 0
#      
# calm1_ref = calm1.loc[calm1['label'].notnull(),['aa_ref','aa_alt','aa_pos','quality_score','sift_score','provean_score','evm_epistatic_score','polyphen_score','joint.score','fitness_org','fitness','fitness_refine','label']]
#      
# # calm1_ref = calm1_ref.loc[calm1_ref['screen.score'].notnull(),:]
# plot_file = project_path + 'calm1_auroc.png' 
# alphame_ml.plot_prc(calm1_ref['label'],calm1_ref[['joint.score','fitness_org','fitness','fitness_refine','polyphen_score','provean_score']],plot_file,20,10)
# print('OK')

# calm1['filter_flag'] = np.nan
# calm1.loc[calm1['fitness'].notnull() & calm1['screen.score'].isnull(),'filter_flag'] = 0 #picked only Joe'
# calm1.loc[calm1['fitness'].isnull() & calm1['screen.score'].notnull(),'filter_flag'] = 1 #picked only Jochen'
# calm1.loc[calm1['fitness'].notnull() & calm1['screen.score'].notnull(),'filter_flag'] = 2 #picked by both'
# calm1.loc[calm1['fitness'].isnull() & calm1['screen.score'].isnull(),'filter_flag'] = 3 #picked by nobody'
# calm1.loc[calm1['filter_flag'] == 0,['aa_ref','aa_alt','aa_pos','quality_score','fitness_org','fitness_sd_org','num_replicates']].to_csv(project_path + dms_name + '_filter_diff.csv',index = False)
# 

# fitness_idx = calm1.index
# dms_name = 'calm1'
#  
# fig = plt.figure(figsize=(20,10))   
# ax = plt.subplot() 
# ax = plt.scatter(np.array(calm1.loc[calm1['filter_flag'] == 0 ,'fitness_org']), np.log10(np.array(calm1.loc[calm1['filter_flag'] == 0,'quality_score'])), color = 'red',label = 'only picked by Joe')
# ax = plt.scatter(np.array(calm1.loc[calm1['filter_flag'] == 1 ,'fitness_org']), np.log10(np.array(calm1.loc[calm1['filter_flag'] == 1,'quality_score'])), color = 'blue',label = 'only picked by Jochen')
# ax = plt.scatter(np.array(calm1.loc[calm1['filter_flag'] == 2 ,'fitness_org']), np.log10(np.array(calm1.loc[calm1['filter_flag'] == 2,'quality_score'])), color = 'green',label = 'picked by both')
# ax = plt.scatter(np.array(calm1.loc[calm1['filter_flag'] == 3 ,'fitness_org']), np.log10(np.array(calm1.loc[calm1['filter_flag'] == 3,'quality_score'])), color = 'grey',label = 'picked by nobody')
# #plt.legend(loc = 2,fontsize=15)
#  
# plt.xlabel('Raw experimental fitness',size = 15)
# plt.ylabel('Log10 qaulity_score',size = 15)
# plt.title(dms_name + ' - Raw experimental fitness VS Log10 quality_score',size = 20)
# plt.tight_layout()
# plt.savefig(project_path + dms_name + '_raw_fitness_filter_comparison.png')
#  
#  
# fig = plt.figure(figsize=(20,10))   
# ax = plt.subplot(2,2,1)   
# ax = plt.scatter(np.array(calm1.loc[fitness_idx,'fitness_reverse']), np.array(calm1.loc[fitness_idx,'screen.score']))
# plt.xlabel('Experimental Fitness - Joe',size = 12)
# plt.ylabel('Experimental Fitness - Jochen',size = 12)
# plt.title(dms_name + ' - Experimental Fitness (Joe) VS Experimental Fitness (Jochen)',size = 15)
# plt.xlim(-1,1.05)
# plt.ylim(-1,1.05)
#  
# ax = plt.subplot(2,2,2)   
# ax = plt.scatter(np.array(calm1.loc[fitness_idx,'fitness_sd']),np.array(calm1.loc[fitness_idx,'screen.sd']))
# plt.xlabel('Experimental Fitness SD - Joe',size = 12)
# plt.ylabel('Experimental Fitness SD - Jochen',size = 12)
# plt.title(dms_name + ' - Experimental Fitness SD (Joe) VS Experimental Fitness SD (Jochen)',size = 15)
#  
#  
# ax = plt.subplot(2,2,4) 
# ax = plt.scatter(np.array(calm1.loc[fitness_idx,'fitness_refine']),np.array(calm1.loc[fitness_idx,'joint.score']))
# plt.xlabel('Refined Fitness - Joe',size = 12)
# plt.ylabel('Refined Fitness - Jochen',size = 12)
# plt.title(dms_name + ' - Refined Fitness (Joe) VS Refined Fitness (Jochen)',size = 15)
# plt.xlim(-1,1.05)
# plt.ylim(-1,1.05)
#   
# ax = plt.subplot(2,2,4) 
# ax = plt.scatter(np.array(calm1.loc[fitness_idx,'fitness_se_refine']),np.array(calm1.loc[fitness_idx,'joint.se']))
# plt.xlabel('Refined Fitness SE - Joe',size = 12)
# plt.ylabel('Refined Fitness SE - Jochen',size = 12)
# plt.title(dms_name + ' - Refined Fitness SE (Joe) VS Refined Fitness SE (Jochen)',size = 15)
#   
#  
# ax = plt.subplot(2,2,3)   
# ax = plt.scatter(np.array(calm1.loc[fitness_idx,'fitness_imputed']), np.array(calm1.loc[fitness_idx,'predicted.score']))
# plt.xlabel('Imputed Fitness SD - Joe',size = 12)
# plt.ylabel('Imputed Fitness SD - Jochen',size = 12)
# plt.title(dms_name + ' - Imputed Fitness SD (Joe) VS Imputed Fitness SD (Jochen)',size = 15)
# plt.xlim(-1,1.05)
# plt.ylim(-1,1.05)
#  
# plt.tight_layout()
# plt.savefig(self.project_path + 'output/' + dms_name + '_err_regression.png')
#  
#***************************************************************************************************************************************************************
# Check the imputation result
#*************************************************************************************************************************************************************** 
#  #Q9NZ01
# dms_map = pd.read_csv(project_path + 'P62166_imputation.csv')
# #dms_map = pd.read_csv(project_path + 'Q9NZ01_imputation.csv')
# dms_name = 'NCS1'
# dms_map = dms_map.loc[(dms_map['annotation'] == 'NONSYN') & (dms_map['quality_score'] >40)]
# dms_map.loc[dms_map['fitness'] == 0].shape
#  
# #plot the refined and experimental scatter plot
# fig = plt.figure(figsize=(20,10))
# ax = plt.subplot(2,1,1)
# ax = plt.hist(dms_map['fitness'],bins = 10)
# plt.title('NCS1 Experimental fitness VS Refined fitness',size = 25)
# plt.xlabel('Experimental fitness data (floored, reversed and quality score filtered)',size = 20)
# plt.xlim(-0.05,1.05)
# ax = plt.subplot(2,1,2)
# ax = plt.hist(dms_map['fitness_refine'],bins = 10)
# plt.xlabel('Refined fitness data',size = 20)
# plt.xlim(-0.05,1.05)
# plt.savefig(project_path + dms_name + '_experimental_refined_comparison.png')
# 
# fig = plt.figure(figsize=(20,10))  
# ax = plt.subplot()
# ax  = scatter_plots(ax,dms_map['fitness'],dms_map['fitness_refine'],'Experiment value','Refined value',None,"",dms_name,50)
# plt.title('NCS1 Experimental fitness VS Refined fitness',size = 25)
# fig.tight_layout()
# plt.subplots_adjust(top=0.9)
# plt.savefig(project_path + dms_name + '_scatter.png')

#***************************************************************************************************************************************************************
# debug for HMGCR
#***************************************************************************************************************************************************************

# dms_name = "HMGCR"
# 
# raw_data = pd.read_csv(project_path + "output/P04035[HMGCR2]_raw_processed.csv")
# 
# raw_data_plot =  raw_data.loc[(raw_data['aa_pos'] < 1000) & raw_data['select_mean'].notnull() & raw_data['nonselect_mean'].notnull() & raw_data['controlS_mean'].notnull(),:]
# 
# x = pd.DataFrame(raw_data_plot['select_mean']).values #returns a numpy array
# min_max_scaler = preprocessing.MinMaxScaler()
# x_scaled = min_max_scaler.fit_transform(x)
# raw_data_plot['select_mean_normalized'] = np.log10(x_scaled)
# 
# x = pd.DataFrame(raw_data_plot['nonselect_mean']).values #returns a numpy array
# min_max_scaler = preprocessing.MinMaxScaler()
# x_scaled = min_max_scaler.fit_transform(x)
# raw_data_plot['nonselect_mean_normalized'] = np.log10(x_scaled)
# 
# x = pd.DataFrame(raw_data_plot['controlS_mean']).values #returns a numpy array
# min_max_scaler = preprocessing.MinMaxScaler()
# x_scaled = min_max_scaler.fit_transform(x)
# raw_data_plot['controlS_mean_normalized'] = np.log10(x_scaled)
# 
# 
# hist(raw_data_plot['nonselect_sd'])
# 
# fig = plt.figure(figsize=(30,10))  
# 
# ax = plt.subplot(4,1,1)
# ax  = scatter_plots(ax,raw_data_plot['aa_pos'],raw_data_plot['nonselect_mean'],'Position','Quality [NonSelect Mean]',None,"",dms_name,20)
# # plt.title('Select SD VS Position',size = 25)
# fig.tight_layout()
# plt.subplots_adjust(top=0.9)
# 
# ax = plt.subplot(4,1,2)
# ax  = scatter_plots(ax,raw_data_plot['aa_pos'],raw_data_plot['select_sd'],'Position','Select SD',None,"",dms_name,20)
# # plt.title('Select SD VS Position',size = 25)
# fig.tight_layout()
# plt.subplots_adjust(top=0.9)
#  
# ax = plt.subplot(4,1,3)
# ax  = scatter_plots(ax,raw_data_plot['aa_pos'],raw_data_plot['nonselect_sd'],'Position','NonSelect SD',None,"",dms_name,20)
# # plt.title('NoneSelect SD VS Position',size = 25)
# fig.tight_layout()
# plt.subplots_adjust(top=0.9)
# 
# ax = plt.subplot(4,1,4)
# ax  = scatter_plots(ax,raw_data_plot['aa_pos'],raw_data_plot['controlS_sd'],'Position','Control SD',None,"",dms_name,20)
# # plt.title('Control SD VS Position',size = 25)
# fig.tight_layout()
# plt.subplots_adjust(top=0.9)
#  
# plt.savefig(project_path + dms_name + '_scatter.png')
# 
# 
# fig = plt.figure(figsize=(30,10))  
# 
# ax = plt.subplot()
# ax  = scatter_plots(ax,raw_data_plot['nonselect_mean'],raw_data_plot['fitness_input_se'],'NonSelect_Mean','fitness_input_se',None,"",dms_name,20)
# # plt.title('Select SD VS Position',size = 25)
# fig.tight_layout()
# plt.subplots_adjust(top=0.9)
# 
# 
# 
# 
# print ("OK")
#***************************************************************************************************************************************************************
# calculate the remidiation 
# **************************************************************************************************************************************************************
# cbs_low = pd.read_csv(g_path + 'rawdata/CBS_01Combined_oldnewCombined_raw_processed.txt',sep = '\t')
# cbs_high = pd.read_csv(g_path + 'rawdata/CBS_400_oldnewCombined_raw_processed.txt',sep = '\t')
#  
# cbs_low.loc[cbs_low['fitness_org'] > 1,'fitness_org'] = 1/cbs_low.loc[cbs_low['fitness_org'] > 1,'fitness_org']
# cbs_low.loc[cbs_low['fitness_org'] < 0,'fitness_org'] = 0
# cbs_high.loc[cbs_high['fitness_org'] > 1,'fitness_org'] = 1/cbs_high.loc[cbs_high['fitness_org'] > 1,'fitness_org']
# cbs_high.loc[cbs_high['fitness_org'] < 0,'fitness_org'] = 0
#  
# cbs_low.loc[cbs_low['averageNonselect'] <=20,'fitness_org'] = np.nan
# cbs_low.loc[cbs_low['fitness_se_org'] >=0.2,'fitness_org'] = np.nan
# cbs_high.loc[cbs_high['averageNonselect'] <=20,'fitness_org'] = np.nan
# cbs_high.loc[cbs_high['fitness_se_org'] >=0.2,'fitness_org'] = np.nan
 
# cbs_low.rename(columns={'fitness_org':'fitness_org_low'}, inplace=True)
# cbs_low.rename(columns={'fitness_se_org':'fitness_se_org_low'}, inplace=True)
# cbs_low.rename(columns={'averageNonselect':'averageNonselect_low'}, inplace=True)
# cbs_high.rename(columns={'fitness_org':'fitness_org_high'}, inplace=True)
# cbs_high.rename(columns={'fitness_se_org':'fitness_se_org_high'}, inplace=True)
# cbs_high.rename(columns={'averageNonselect':'averageNonselect_high'}, inplace=True)
#  
# cbs_diff = pd.merge(cbs_low,cbs_high,how = 'left')
# cbs_diff['fitness_org_diff'] = cbs_diff['fitness_org_high'] - cbs_diff['fitness_org_low']
# cbs_diff['fitness_org_diff_se'] = np.sqrt(cbs_diff['fitness_se_org_high']**2 + cbs_diff['fitness_se_org_low']**2)

# cbs_remediation = cbs_diff[['AAchange','wt_aa','pos','mut_aa','annotation','averageNonselect_low','fitness_org_diff','fitness_org_diff_se']]
# cbs_remediation['averageNonselect_low'] = np.inf
#  
#  
# cbs_low.to_csv(g_path + 'rawdata/CBS_01Combined_oldnewCombined_raw_processed1.txt',sep = '\t',index = False)
# cbs_high.to_csv(g_path + 'rawdata/CBS_400_oldnewCombined_raw_processed1.txt',sep = '\t',index = False)
# cbs_diff.to_csv(g_path + 'rawdata/CBS_diff.txt',sep = '\t',index = False)
# cbs_remediation.to_csv(g_path + 'rawdata/CBS_remediation.txt',sep = '\t',index = False)
#  
# cbs_low_refine = pd.read_csv(g_path + 'output/CBS_low_imputation.csv')
# cbs_low_refine = cbs_low_refine[['aa_ref','aa_pos','aa_alt','annotation','fitness_refine','fitness_se_refine']]
# cbs_low_refine.rename(columns={'fitness_refine':'fitness_refine_low'}, inplace=True)
# cbs_low_refine.rename(columns={'fitness_se_refine':'fitness_se_refine_low'}, inplace=True)
# cbs_high_refine = pd.read_csv(g_path + 'output/CBS_high_imputation.csv')
# cbs_high_refine = cbs_high_refine[['aa_ref','aa_pos','aa_alt','annotation','fitness_refine','fitness_se_refine']]
# cbs_high_refine.rename(columns={'fitness_refine':'fitness_refine_high'}, inplace=True)
# cbs_high_refine.rename(columns={'fitness_se_refine':'fitness_se_refine_high'}, inplace=True)
# cbs_refine_diff = pd.merge(cbs_low_refine,cbs_high_refine,how = 'left')
# cbs_refine_diff['fitness_refine_diff'] = cbs_refine_diff['fitness_refine_high'] - cbs_refine_diff['fitness_refine_low']
# cbs_refine_diff['fitness_refine_diff_se'] = np.sqrt(cbs_refine_diff['fitness_se_refine_high']**2 + cbs_refine_diff['fitness_se_refine_low']**2)
# cbs_refine_remediation = cbs_refine_diff[['aa_ref','aa_pos','aa_alt','annotation','fitness_refine_diff','fitness_refine_diff_se']]
# cbs_refine_remediation['AAchange'] = 'NA'
# cbs_refine_remediation['averageNonselect'] = 1000
# cbs_refine_remediation = cbs_refine_remediation[['AAchange','aa_ref','aa_pos','aa_alt','annotation','averageNonselect','fitness_refine_diff','fitness_refine_diff_se']]
# cbs_refine_remediation.to_csv(g_path + 'rawdata/CBS_refine_remediation.txt',sep = '\t',index = False)
# cbs_refine_diff.to_csv(g_path + 'rawdata/CBS_refine_diff.txt',sep = '\t',index = False) 
# # print('OK')

# 
# cbs_low = pd.read_csv(g_path + 'rawdata/CBS_01Combined_oldnewCombined_raw_processed.txt',sep = '\t')
# cbs_high = pd.read_csv(g_path + 'rawdata/CBS_400_oldnewCombined_raw_processed.txt',sep = '\t')
#   
# cbs_low.loc[cbs_low['fitness_org'] > 2,'fitness_org'] = 2
# cbs_low.loc[cbs_low['fitness_org'] < 0,'fitness_org'] = 0
# cbs_high.loc[cbs_high['fitness_org'] > 2,'fitness_org'] = 2
# cbs_high.loc[cbs_high['fitness_org'] < 0,'fitness_org'] = 0
#   
# cbs_low.loc[cbs_low['averageNonselect'] <=20,'fitness_org'] = np.nan
# cbs_low.loc[cbs_low['fitness_se_org'] >=0.2,'fitness_org'] = np.nan
# cbs_high.loc[cbs_high['averageNonselect'] <=20,'fitness_org'] = np.nan
# cbs_high.loc[cbs_high['fitness_se_org'] >=0.2,'fitness_org'] = np.nan
#   
# cbs_low.rename(columns={'fitness_org':'fitness_org_low'}, inplace=True)
# cbs_low.rename(columns={'fitness_se_org':'fitness_se_org_low'}, inplace=True)
# cbs_low.rename(columns={'averageNonselect':'averageNonselect_low'}, inplace=True)
# cbs_high.rename(columns={'fitness_org':'fitness_org_high'}, inplace=True)
# cbs_high.rename(columns={'fitness_se_org':'fitness_se_org_high'}, inplace=True)
# cbs_high.rename(columns={'averageNonselect':'averageNonselect_high'}, inplace=True)
#   
# cbs_low.to_csv(g_path + 'rawdata/CBS_01Combined_oldnewCombined_raw_processed2.txt',sep = '\t',index = False)
# cbs_high.to_csv(g_path + 'rawdata/CBS_400_oldnewCombined_raw_processed2.txt',sep = '\t',index = False)

