#!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3
# -*- coding: UTF-8 -*-
import sys
import os
import traceback
import time
import glob
import pandas as pd
import numpy as np
from datetime import datetime
python_path = '/usr/local/projects/ml/python/'
project_path = '/usr/local/projects/imputation/gwt/www/'
humandb_path = '/usr/local/database/humandb/'
sys.path.append(python_path)
import alm_fun
import imputation

#email server related parametes
server_address = 'smtp-relay.gmail.com'
server_port = 587
login_user = 'noreply@varianteffect.org'
login_password = 'WoolyMammothInThePorcellanShop'
from_address = 'noreply@varianteffect.org'
subject = 'No Reply'

def run_imputation(run_mode, arguments):
    www_output_path = project_path + 'output/' 
    www_log_path = project_path + 'log/'
    www_upload_path = project_path + 'upload/'
     
    ####*************************************************************************************************************************************************************
    # run for local debug
    ####*************************************************************************************************************************************************************
    if run_mode == 0:
        JSON_imputation = run_imputation()
        print (JSON_imputation)  
        sys.exit()    
             
    ####*************************************************************************************************************************************************************
    # run for web 
    ####*************************************************************************************************************************************************************
    if run_mode == 1:
        queryflag = arguments['queryflag']
        session_id = arguments.get('sessionid',None)
        imputation_web_log = open(www_log_path + 'imputation_web.log', 'a') 
        imputation_web_log.write('\n' + str(datetime.now()) + '\n') 
        imputation_web_log.write('queryflag: ' + str(queryflag) + '\n') 
        if session_id is not None:
            imputation_web_session_log_file = www_log_path + session_id + '.log'
            imputation_web_session_log = open(imputation_web_session_log_file, 'a')  
            imputation_web_session_log.write('\n' + str(datetime.now()) + '\n') 
            imputation_web_session_log.write('queryflag: ' + str(queryflag) + '\n')

    try:

        
    ####*************************************************************************************************************************************************************
    # queryflag -1 : Get the resolution of the user screen  
    ####*************************************************************************************************************************************************************    
        if int(queryflag) == -1:
            image_width = arguments['imagewidth']
            image_height = arguments['imageheight']
            callback = arguments['callback']
            if (int(image_width) < 160) or (int(image_height) < 820):  
                error_msg = "Your browser window size is " + str(image_width) + "*" + str(image_height) + " which is smaller than the minimum browser window size requirement 1600*820 for this web application. " + \
                            "To change your browser window size you could press [CTRL or COMMAND] & [-] in your browser or change screen resolution. Please reload the website after browser window size adjustment."
                imputation_web_log.write(error_msg + '\n')                                              
                JSON_Return = str(callback) + '([{"content":"' +  error_msg + '"}])'
            else:
                JSON_Return = str(callback) + '([{"content":"OK"}])' 
    ####*************************************************************************************************************************************************************
    # queryflag 0 : upload user files for imputation 
    ####*************************************************************************************************************************************************************      
        if int(queryflag) == 0:
            image_width = arguments['imagewidth']
            image_height = arguments['imageheight']
            session_id = arguments['sessionid']    
        
            dms_landscape_file = arguments['upload_landscape_filename']
            dms_fasta_file = arguments['upload_fasta_filename']
            dms_landscape_file_dest = www_upload_path + session_id + '.txt'
            dms_landscape_content = arguments['upload_landscape']
            dms_fasta_file_dest = www_upload_path + session_id + '.fasta'
            dms_fasta_content = arguments['upload_fasta']
                     
            imputation_web_session_log.write('image_width: ' + str(image_width) + '\n')
            imputation_web_session_log.write('image_height: ' + str(image_height) + '\n')
            imputation_web_session_log.write('landscape file name: ' + str(dms_landscape_file) + '\n')
            imputation_web_session_log.write('landscape file dest name: ' + str(dms_landscape_file_dest) + '\n')
            imputation_web_session_log.write('fasta file name: ' + str(dms_fasta_file) + '\n')
            imputation_web_session_log.write('fasta file dest name: ' + str(dms_fasta_file_dest) + '\n')
        
            uploaded_landscape_file = open(dms_landscape_file_dest, 'wb')
            uploaded_landscape_file.write(dms_landscape_content)
            uploaded_landscape_file.close() 
            uploaded_fasta_file = open(dms_fasta_file_dest, 'wb')
            uploaded_fasta_file.write(dms_fasta_content)
            uploaded_fasta_file.close()
            JSON_Return = "N/A"
            alm_fun.send_email(server_address,server_port,login_user,login_password,from_address,'joe.wu.ca@gmail.com', 'Imputation Notification', create_email_notification_msg(session_id))

        ####*************************************************************************************************************************************************************
        # queryflag 1 : Run imputation and save the result in JSON format to [sessionid].out and send it back to the broswer 
        ####*************************************************************************************************************************************************************      
        if int(queryflag) == 1: 
            email_address = arguments.get('email_address','')                
            callback = arguments['callback']
            session_id = arguments['sessionid']   
            protein_id = arguments['proteinid']             
            data_name = session_id                
            imputation_web_session_log.write(str(arguments) + '\n')                  
            
            if os.path.isfile((humandb_path + 'dms/features/' + protein_id + '_features.csv')  ):
                im_proj = create_imputation_instance(arguments,imputation_web_session_log)
        #                     imputation_web_session_log.write(str(rawdata_init_params) + '\n')
                sessionid_JSON = im_proj.imputation_run(session_id)                          
                imputation_web_output = open(project_path + 'output/' + session_id + '.out', 'w')
                imputation_web_output.write(sessionid_JSON)
                imputation_web_output.close()
                JSON_Return = str(callback) + '(' + sessionid_JSON + ')'                       
            else:
                error_msg = "The input Uniprot ID " + protein_id + " is not supported yet , please leave your email address in the box below. We will notify your once this ID is supported."
        #                     error_msg = protein_id                    
                imputation_web_session_log.write(error_msg + '\n')                        
                JSON_Return = str(callback) + '([{"content":"' +  error_msg + '"}])'
                return (JSON_Return)
            
            if "@" in email_address:
                alm_fun.send_email(server_address,server_port,login_user,login_password,from_address,email_address, subject, create_email_msg(session_id))

        ####*************************************************************************************************************************************************************
        # queryflag 2 : Read a list of available landscapes
        ####*************************************************************************************************************************************************************      
        if int(queryflag) == 2: 
            file_list = []
            callback = arguments['callback']
            
            #       imputation_web_log.write('callback: ' + str(callback) + '\n')
            imputation_web_log.write(str(arguments))  
            for file in glob.glob(project_path + 'output/*.out'):
                file_name = os.path.basename(file)[:-4]
                if (file_name[0] == '*'):
                    file_list.append(file_name)
            file_list.sort();
            landscapes_JSON = '['
            for file_name in file_list:     
                    landscapes_JSON += '{"landscape_name":"' + file_name + '"},'            
            landscapes_JSON += ']'
            JSON_Return = str(callback) + '(' + landscapes_JSON + ')'
            imputation_web_log.write(JSON_Return + '\n')
             
        ####*************************************************************************************************************************************************************
        # queryflag 3 : Read JSON from [sessionid].out file and send it back to the broswer 
        ####*************************************************************************************************************************************************************      
        if int(queryflag) == 3: 
            callback = arguments['callback']
            session_id = arguments['sessionid']
        #                 imputation_web_log.write('callback: ' + str(callback) + '\n')
            imputation_web_session_log.write(str(arguments) + '\n')
            
            map_file =  project_path + 'output/' + session_id + '.out'   
            
            if (session_id[0] == "!") & (not os.path.isfile(map_file)) :
                create_uniprotid_map(session_id[1:])

            if os.path.isfile(map_file):
                with open(map_file, 'r') as myfile:
                    landscape_JSON = myfile.read()
                JSON_Return = str(callback) + '(' + landscape_JSON + ')'
            else:
                error_msg = "The map requested is not avalible!"                                       
                imputation_web_session_log.write(error_msg + '\n')                        
                JSON_Return = str(callback) + '([{"error":"' +  error_msg + '"}])'
                
            
        ####*************************************************************************************************************************************************************
        # queryflag 4 : Return pubmed link of the session id  
        ####*************************************************************************************************************************************************************                     
        if int(queryflag) == 4:     
            callback = arguments['callback']            
            session_id = arguments['sessionid']
            imputation_web_session_log.write(str(arguments) + '\n') 
            pubmed_file =  project_path + 'output/' +   session_id + '_pubmed.txt'             
            if os.path.isfile(pubmed_file):
                with open(pubmed_file , 'r') as myfile:
                    pubmed_link = myfile.read().replace('\r', '').replace('\n', '')
                JSON_Return =  str(callback) + '([{"content":' + '"' + pubmed_link  + '"'+ '}])'                  
            else:
                error_msg = "Error! The pubmed link for the variant effect map doesn't exist!"                                       
                imputation_web_session_log.write(error_msg + '\n')                        
                JSON_Return = str(callback) + '([{"error":"' +  error_msg + '"}])'
        
        ####*************************************************************************************************************************************************************
        # queryflag 5 : send error email  
        ####*************************************************************************************************************************************************************                     
        if int(queryflag) == 5:     
            callback = arguments['callback']            
            session_id = arguments['sessionid']
            error_email = arguments['email_address']
            imputation_web_session_log.write(str(arguments) + '\n')  
            if '@' in error_email:               
                 alm_fun.send_email(server_address,server_port,login_user,login_password,from_address,error_email, subject, create_email_error_msg(error_email,session_id)) 
            JSON_Return = str(callback) + '([{"content":"OK"}])' 
            
        ####*************************************************************************************************************************************************************
        # queryflag 6 : View option async  
        ####*************************************************************************************************************************************************************                     
        if int(queryflag) == 6:     
            callback = arguments['callback']            
            session_id = arguments['sessionid']
            view_option = arguments['view_option']
            imputation_web_session_log.write(str(arguments) + '\n')   
            JSON_Return = str(callback) + '([{"content":"' + view_option +'"}])' 
#             imputation_web_session_log.write(JSON_Return + '\n')  
        imputation_web_log.close()
        if session_id is not None:    
            imputation_web_session_log.close()
        
        return (JSON_Return)
    except:
        err_msg = traceback.format_exc()
        err_msg = err_msg.replace('\"',' ');
        err_msg = err_msg.replace('\'',' ');
        print (err_msg + '\n')
        callback = arguments.get('callback','no callback')
        if arguments.get("sessionid",None) is None:
            imputation_web_log.write(err_msg + '\n')
            imputation_web_log.close()
        else:
            imputation_web_session_log.write(err_msg + '\n')
            imputation_web_session_log.close()
        JSON_Return = str(callback) + '([{"error":"Imputation Error! Please check your inputs or leave your email address on the box below, we will notify you once the problem is found."}])'
        return (JSON_Return)
          
def create_imputation_instance(arguments,imputation_web_session_log):     
    protein_id = arguments['proteinid']
    session_id = arguments.get('sessionid','')
    email_address = arguments.get('email_address','')      
        
    dms_landscape_file = session_id + '.txt'
    dms_fasta_file = session_id + '.fasta'            
    regression_cutoff = float(arguments.get('regression_cutoff','-inf'))
    data_cutoff = float(arguments.get('data_cutoff','-inf'))        
    auto_regression_cutoff = int(arguments['if_auto_cutoff'])
    data_cutoff_flag = int(arguments['if_data_cutoff'])
    normalized_flag = 1 - int(arguments['if_normalization'])   
    regularization_flag = int(arguments['if_regularization'])
    rawprocessed_flag = 1 - int(arguments['if_rawprocessing'])         
    proper_count = int(arguments.get('proper_count',8))  
    synstop_cutoff = float(arguments.get('synstop_cutoff','-inf'))
    stop_exclusion = arguments.get('stop_exclusion','0')  
    
    #alm_project class parameters
    project_params = {}

    
    project_params['project_name'] = 'imputation'
    project_params['project_path'] = project_path
    project_params['humandb_path'] = humandb_path
    project_params['log'] = imputation_web_session_log 
    project_params['verbose'] = 1

        
    #the reason the following parameters don't belong to data class is we may want to create multiple data instance in one project instance 
    project_params['data_names'] = [] 
    project_params['train_data'] = []        
    project_params['test_data'] = []
    project_params['target_data'] = []
    project_params['extra_train_data'] = []
    project_params['use_extra_train_data'] = []
    project_params['input_data_type'] = []
     
    project_params['run_data_names'] = [session_id]
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
    data_params['log'] = imputation_web_session_log  
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
    data_params['test_split_ratio'] = 0
    data_params['cv_split_method'] = 2
    data_params['cv_split_folds'] = 1
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
    
    data_params['if_gradient'] = auto_regression_cutoff
    data_params['if_engineer'] = 0
    data_params['load_from_disk'] = 0
    data_params['save_to_disk'] = 1
    data_params['cur_test_split_fold'] = 0
    data_params['cur_gradient_key'] = 'no_gradient'
    data_params['innerloop_cv_fit_once'] = 0

    data_params['onehot_features'] = []
    data_params['cv_fitonce'] = 0
        
    #alm_ml class parameters
    ml_params = {}
    ml_params['log'] = imputation_web_session_log 
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
    es_params['if_feature_engineer'] = 1
    es_params['feature_engineer'] = None
     
    #data preprocess and update data_params                  
    imputation_params = {} 
    imputation_params['log'] = imputation_web_session_log 
    imputation_params['verbose'] = 1
    imputation_params['project_path'] = project_path
    imputation_params['humandb_path'] = humandb_path
    
    imputation_params['project_params'] = project_params
    imputation_params['data_params'] = data_params
    imputation_params['ml_params'] = ml_params
    imputation_params['es_params'] = es_params
       
    #imputation class: parameters for data preprocessing
    imputation_params['run_data_preprocess'] = 1 
    imputation_params['dms_landscape_files'] = [dms_landscape_file]
    imputation_params['dms_fasta_files'] = [dms_fasta_file]
    imputation_params['dms_protein_ids'] = [protein_id]
    imputation_params['data_names'] = [session_id]
    imputation_params['remediation'] = [0]
    
    if normalized_flag == 1:
        imputation_params['synstop_cutoffs'] = [float("-inf")]
        imputation_params['stop_exclusion'] = ["0"]
    else:
        imputation_params['synstop_cutoffs'] = [synstop_cutoff]
        imputation_params['stop_exclusion'] = [stop_exclusion]
        
    if data_cutoff_flag == 1:
        imputation_params['quality_cutoffs'] = [data_cutoff]
    else:
        imputation_params['quality_cutoffs'] = [float("-inf")]
    
    if auto_regression_cutoff == 1:
        imputation_params['regression_quality_cutoffs'] = [float("-inf")]
    else:
        imputation_params['regression_quality_cutoffs'] = [regression_cutoff]
                    
    imputation_params['proper_num_replicates'] = [proper_count]
    imputation_params['raw_processed'] = [rawprocessed_flag]
    imputation_params['normalized_flags'] = [normalized_flag]
    imputation_params['regularization_flags'] = [regularization_flag]
    imputation_params['reverse_flags'] = [1]
    imputation_params['floor_flags'] = [1]
    imputation_params['combine_flags'] = [0]

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
    imputation_params['email_msg_content'] = create_email_msg(session_id)
    imputation_params['email_error_content'] = create_email_error_msg(email_address,session_id)
    imputation_params['email_notification_content'] = create_email_notification_msg(session_id)
        
    im_proj = imputation.imputation(imputation_params)    
    return (im_proj)
    
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

def create_uniprotid_map(uniprot_id):
    uniprot_out_file = project_path + 'output/!' + uniprot_id + '.out'
    feature_file = humandb_path + 'dms/features/'+ uniprot_id + '_features.csv'
    
    if (not os.path.isfile(uniprot_out_file)) and os.path.isfile(feature_file):
        uniprot_out = open(uniprot_out_file,'w')
        
        dms_gene_csv_df = pd.read_csv(humandb_path + 'dms/features/'+ uniprot_id + '_features.csv')
         
        if str(dms_gene_csv_df['polyphen_score'].dtype) != 'float64':
            dms_gene_csv_df.loc[dms_gene_csv_df['polyphen_score'].str.contains("\?").fillna(False),'polyphen_score'] = np.nan
            dms_gene_csv_df['polyphen_score'] = dms_gene_csv_df['polyphen_score'].astype(float)
             
        dms_gene_csv_df['aa_pos_index'] = dms_gene_csv_df['aa_pos']
        dms_gene_csv_df['ss_end_pos_index'] = dms_gene_csv_df['ss_end_pos']
        dms_gene_csv_df['pfam_end_pos_index'] = dms_gene_csv_df['pfam_end_pos']
        #      ass colorcode
        [lst_max_colors_asa, lst_min_colors_asa] = alm_fun.create_color_gradients(1, 0, 0, '#3155C6', '#FFFFFF', '#FFFFFF', 10, 10)  
        dms_gene_csv_df['asa_mean_normalized'] = (dms_gene_csv_df['asa_mean'] - np.nanmin(dms_gene_csv_df['asa_mean'])) / (np.nanmax(dms_gene_csv_df['asa_mean']) - np.nanmin(dms_gene_csv_df['asa_mean']))
        dms_gene_csv_df['asa_colorcode'] = dms_gene_csv_df['asa_mean_normalized'].apply(lambda x: alm_fun.get_colorcode(x, 1, 0, 0, 10, 10, lst_max_colors_asa, lst_min_colors_asa))
         
        if  'sift_score' in dms_gene_csv_df.columns:  
        #      sift colorcode
            [lst_max_colors_sift,lst_min_colors_sift] = alm_fun.create_color_gradients(1, 0, 0.05,'#C6172B','#FFFFFF','#3155C6',10,10)
            [lst_max_colors_sift, lst_min_colors_sift] = alm_fun.create_color_gradients(1, 0, 0, '#FFFFFF', '#3155C6', '#3155C6', 10, 10)
            dms_gene_csv_df['sift_colorcode'] = dms_gene_csv_df['sift_score'].apply(lambda x: alm_fun.get_colorcode(x, 1, 0, 0, 10, 10, lst_max_colors_sift, lst_min_colors_sift))
               
        [lst_max_colors_polyphen, lst_min_colors_polyphen] = alm_fun.create_color_gradients(1, 0, 0, '#3155C6', '#FFFFFF', '#FFFFFF', 10, 10)
        dms_gene_csv_df['polyphen_colorcode'] = dms_gene_csv_df['polyphen_score'].apply(lambda x: alm_fun.get_colorcode(x, 1, 0, 0, 10, 10, lst_max_colors_polyphen, lst_min_colors_polyphen))
           
        [lst_max_colors_gnomad, lst_min_colors_gnomad] = alm_fun.create_color_gradients(10, 0.3, 0.3, '#3155C6', '#FFFFFF', '#FFFFFF', 10, 10)            
        dms_gene_csv_df['gnomad_af_log10'] = 0 - np.log10(dms_gene_csv_df['gnomad_af'])
        dms_gene_csv_df['gnomad_colorcode'] = dms_gene_csv_df['gnomad_af_log10'].apply(lambda x: alm_fun.get_colorcode(x, 10, 0.3, 0.3, 10, 10, lst_max_colors_gnomad, lst_min_colors_gnomad))
           
        [lst_max_colors_provean, lst_min_colors_provean] = alm_fun.create_color_gradients(4, -13, -13, '#FFFFFF', '#3155C6', '#3155C6', 10, 10)
        dms_gene_csv_df['provean_colorcode'] = dms_gene_csv_df['provean_score'].apply(lambda x: alm_fun.get_colorcode(x, 4, -13, -13, 10, 10, lst_max_colors_provean, lst_min_colors_provean))
            
        out_file =  dms_gene_csv_df.to_json(orient='records')
        uniprot_out.write(out_file)
        uniprot_out.close()
        
        