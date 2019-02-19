#!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3
import numpy as np
import pandas as pd
import matplotlib
from seaborn.palettes import color_palette
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import seaborn as sns
import sys
import csv
import os
import glob
import operator
import itertools
import time
import math
import random
import traceback
import re
import gzip
import urllib
from subprocess import call
from ftplib import FTP
from scipy import stats
from operator import itemgetter 
from functools import partial
from datetime import datetime
from numpy import inf
from cgi import log
from sklearn.metrics.ranking import roc_auc_score
from posix import lstat
import xml.etree.ElementTree as ET
from io import StringIO

import alm_fun
              
#*****************************************************************************************************************************
#file based human database 
#*****************************************************************************************************************************

class alm_humandb:
        
    def __init__(self, project_path, db_path, assembly):
        stime = time.time()  
        self.assembly = assembly
        self.db_path = db_path
        self.project_path = project_path
        self.log = self.project_path + 'log/humandb_log.txt'
        self.humandb_object_logs = {}
        self.verbose = 1






        ####***************************************************************************************************************************************************************
        # Nucleotide and Amino Acids related
        ####***************************************************************************************************************************************************************
        self.lst_nt = ['A', 'T', 'C', 'G']
        self.lst_aa = ["S", "A", "V", "R", "D", "F", "T", "I", "L", "K", "G", "Y", "N", "C", "P", "E", "M", "W", "H", "Q", "U", "*", '_']
        self.lst_aa_21 = ["S", "A", "V", "R", "D", "F", "T", "I", "L", "K", "G", "Y", "N", "C", "P", "E", "M", "W", "H", "Q", "*"]
        self.lst_aa_20 = ["S", "A", "V", "R", "D", "F", "T", "I", "L", "K", "G", "Y", "N", "C", "P", "E", "M", "W", "H", "Q"]
        self.lst_aa3 = ["Ser", "Ala", "Val", "Arg", "Asp", "Phe", "Thr", "Ile", "Leu", "Lys", "Gly", "Tyr", "Asn", "Cys", "Pro", "Glu", "Met", "Trp", "His", "Gln", "Sec", "Ter", 'Unk']
        self.lst_aaname = ["Serine", "Alanine", "Valine", "Arginine", "Asparitic Acid", "Phenylalanine", "Threonine", "Isoleucine", "Leucine", "Lysine", "Glycine", "Tyrosine", "Asparagine", "Cysteine", "Proline", "Glutamic Acid", "Methionine", "Tryptophan", "Histidine", "Glutamine", "Selenocysteine", "Stop", "Unknown"]
        self.lst_chr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13''14', '15', '16', '17', '18', '19', '20', '21', '22', 'X', 'Y', 'MT']

        self.dict_aa3 = {}
        for i in range(len(self.lst_aa3)):
            self.dict_aa3[self.lst_aa3[i]] = self.lst_aa[i]
            
        self.dict_aa3_upper = {}
        for i in range(len(self.lst_aa3)):
            self.dict_aa3_upper[self.lst_aa3[i].upper()] = self.lst_aa[i]
            
        self.dict_aaname = {}
        for i in range(len(self.lst_aa)):
            self.dict_aaname[self.lst_aa[i]] = self.lst_aaname[i]

        self.dict_aaencode = {}
        for i in range(len(self.lst_aa)):
            self.dict_aaencode[self.lst_aa[i]] = i

                      
#         ####***************************************************************************************************************************************************************
#         # Uniprot ids and sequences
#         ####***************************************************************************************************************************************************************
#         self.uniprot_seqdict = np.load(self.db_path + 'uniprot/npy/uniprot_seq_dict.npy').item()
#         self.uniprot_human_reviewed_ids = np.load(self.db_path + 'uniprot/npy/uniprot_human_reviewed_ids.npy')
#         self.uniprot_human_reviewed_isoform_ids = np.load(self.db_path + 'uniprot/npy/uniprot_human_reviewed_isoform_ids.npy')
#         
#         ####***************************************************************************************************************************************************************
#         # hgnc to uniprot id mapping 
#         ####***************************************************************************************************************************************************************
#         self.hgnc2uniprot_dict = np.load(self.db_path + 'hgnc/npy/hgnc2uniprot_dict.npy').item()
         
    def get_gene_data(self,table_name,gene_name):
        alm_fun.show_msg(self.log,self.verbose,'OK')
        
    def init_humamdb_object(self,obj_name):        
        if not os.path.isdir(self.db_path + obj_name):
            os.mkdir(self.db_path + obj_name)
            os.mkdir(self.db_path + obj_name + '/org')
            os.mkdir(self.db_path + obj_name + '/all')
            os.mkdir(self.db_path + obj_name + '/bygene')
            os.mkdir(self.db_path + obj_name + '/npy') 
            os.mkdir(self.db_path + obj_name + '/log')
            alm_fun.show_msg(self.log,self.verbose, 'Created folder ' + obj_name + ' and its subfolders.' )
        self.humandb_object_logs['obj_name'] = self.db_path + obj_name + '/log/' + obj_name + '_data.log'   
        
    def create_new_humandb(self, db_object = 'all', parallel_id = 0, parallel_num = 1):
                
        if db_object == 'all':        
            self.create_hgnc_data()
            self.create_uniprot_data()
            self.create_ensembl66_data()
            self.create_matched_uniprot_mapping()
            self.create_pisa_data()
            self.create_pfam_data()
            self.create_sift_data()
            self.create_provean_data()
            self.create_clinvar_data()

        if db_object  == 'hgnc':
            self.create_hgnc_data()
        if db_object  == 'uniprot':
            self.create_uniprot_data()   
        if db_object  == 'ensembl66':
            self.create_ensembl66_data()
        if db_object  == 'matched_uniprot_mapping':            
            self.create_matched_uniprot_mapping()
        if db_object  == 'pisa':
            self.create_pisa_data()            
        if db_object  == 'pisa_parallel':
            self.retrieve_pisa_data(parallel_id, parallel_num)             
        if db_object  == 'pfam':
            self.create_pfam_data()
        if db_object  == 'sift':
            self.create_sift_data()
        if db_object  == 'provean':
            self.create_provean_data()                    
        if db_object  == 'clinvar':
            self.create_clinvar_data()                    

    def create_hgnc_data(self):
        def fill_dict(hngc_id,input,in_dict):
            lst_input = input.split('|')
            in_dict.update({x:hngc_id for x in lst_input})
                        
        ####***************************************************************************************************************************************************************    
        # EBI FTP
        ####***************************************************************************************************************************************************************
        # HGNC compplete set 
        # ftp://ftp.ebi.ac.uk/pub/databases/genenames/new/tsv/hgnc_complete_set.txt          
        ####***************************************************************************************************************************************************************
        self.init_humamdb_object('hgnc') 
       
        self.ebi_ftp_obj = alm_fun.create_ftp_object('ftp.ebi.ac.uk')
        return_info = alm_fun.download_ftp(self.ebi_ftp_obj, '/pub/databases/genenames/new/tsv/', 'hgnc_complete_set.txt', self.db_path + 'hgnc/org/hgnc_complete_set.txt')        
        hgnc = pd.read_csv(self.db_path + 'hgnc/org/hgnc_complete_set.txt',sep ='\t')        
        
        #******************************************************
        # hgnc to Uniprot ID
        #******************************************************
        hgnc_uniprot_ids = hgnc.loc[hgnc['uniprot_ids'].notnull(),['hgnc_id','uniprot_ids']]
        hgnc2uniprot_dict = {hgnc_uniprot_ids.loc[i,'hgnc_id']:hgnc_uniprot_ids.loc[i,'uniprot_ids'] for i in hgnc_uniprot_ids.index}
        np.save(self.db_path + 'hgnc/npy/hgnc2uniprot_dict.npy', hgnc2uniprot_dict)  
        
        #******************************************************
        # IDs to hgnc
        #******************************************************
        id2hgnc_dict = {}
        id_lst = ['symbol','ensembl_gene_id','refseq_accession','uniprot_ids','ucsc_id']
        
        for id in id_lst:
            cur_hgnc_ids = hgnc.loc[hgnc[id].notnull(),['hgnc_id',id]] 
            cur_id_dict = {}
            cur_hgnc_ids.apply(lambda x: fill_dict(x['hgnc_id'],x[id],cur_id_dict),axis = 1)
            id2hgnc_dict[id] = cur_id_dict
        pass
        np.save(self.db_path + 'hgnc/npy/id2hgnc_dict.npy', id2hgnc_dict)
        
        #******************************************************
        # hgnc to IDS
        #******************************************************
        hgnc2id_dict = {}
        id_lst = ['symbol','ensembl_gene_id','refseq_accession','uniprot_ids','ucsc_id']
        
        for id in id_lst:
            cur_id_dict = {hgnc.loc[i,'hgnc_id']:hgnc.loc[i,id] for i in hgnc.index }             
            hgnc2id_dict[id] = cur_id_dict
        pass
        np.save(self.db_path + 'hgnc/npy/hgnc2id_dict.npy', hgnc2id_dict)

      
        alm_fun.show_msg(self.log,self.verbose,'Created hgnc data.')

    def create_uniprot_data(self):
        def fill_dict(uniprot_id,input,in_dict):
            lst_input = input.split(';')
            in_dict.update({x:uniprot_id for x in lst_input})
        ####***************************************************************************************************************************************************************    
        # UniProt FTP
        ####***************************************************************************************************************************************************************
        # Reviewed uniprot fasta file
        # ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz      
        # Reviewd uniprot isoforms fasta fil e
        # ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot_varsplic.fasta.gz
        # Uniprot ID mapping
        # ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/idmapping/by_organism/HUMAN_9606_idmapping_selected.tab.gz
        ####***************************************************************************************************************************************************************        
        self.init_humamdb_object('uniprot')        

        self.uniprot_ftp_obj = alm_fun.create_ftp_object('ftp.uniprot.org')
        return_info = alm_fun.download_ftp(self.uniprot_ftp_obj, '/pub/databases/uniprot/current_release/knowledgebase/complete/', 'uniprot_sprot.fasta.gz', self.db_path + 'uniprot/org/uniprot_sprot.fasta.gz')
        if (return_info == 'updated') | (return_info == 'downloaded'):
            alm_fun.gzip_decompress(self.db_path + 'uniprot/org/uniprot_sprot.fasta.gz', self.db_path + 'uniprot/org/uniprot_sprot.fasta')  
        
        return_info = alm_fun.download_ftp(self.uniprot_ftp_obj, '/pub/databases/uniprot/current_release/knowledgebase/complete/', 'uniprot_sprot_varsplic.fasta.gz', self.db_path + 'uniprot/org/uniprot_sprot_varsplic.fasta.gz')
        if (return_info == 'updated') | (return_info == 'downloaded'):
            alm_fun.gzip_decompress(self.db_path + 'uniprot/org/uniprot_sprot_varsplic.fasta.gz', self.db_path + 'uniprot/org/uniprot_sprot_varsplic.fasta')  


        return_info = alm_fun.download_ftp(self.uniprot_ftp_obj, '/pub/databases/uniprot/current_release/knowledgebase/idmapping/by_organism/', 'HUMAN_9606_idmapping_selected.tab.gz', self.db_path + 'uniprot/org/HUMAN_9606_idmapping_selected.tab.gz')
        if (return_info == 'updated') | (return_info == 'downloaded'):
            alm_fun.gzip_decompress(self.db_path + 'uniprot/org/HUMAN_9606_idmapping_selected.tab.gz', self.db_path + 'uniprot/org/HUMAN_9606_idmapping_selected.tab')  

        #******************************************************
        # Uniprot sequence dictionary and reviewed ids
        #******************************************************
        p_fa_dict = {}
        uniprot_human_reviewed_isoform_ids = []
        uniprot_human_reviewed_ids = []
        p_fa = open(self.db_path + 'uniprot/org/uniprot_sprot.fasta', 'r') 
        for line in p_fa:
            if line[0] == ">" :
                cur_key = line.split('|')[1]
                if 'OS=Homo sapiens' in line :                    
                    uniprot_human_reviewed_ids.append(cur_key)
                p_fa_dict[cur_key] = ''
            else:
                p_fa_dict[cur_key] += line.strip()
                
        p_fa_isoform = open(self.db_path + 'uniprot/org/uniprot_sprot_varsplic.fasta', 'r') 
        for line in p_fa_isoform:
            if line[0] == ">" :
                cur_key = line.split('|')[1]
                if 'OS=Homo sapiens' in line : 
                    uniprot_human_reviewed_isoform_ids.append(cur_key)
                p_fa_dict[cur_key] = ''
            else:
                p_fa_dict[cur_key] += line.strip()       
                
        np.save(self.db_path + 'uniprot/npy/uniprot_seq_dict.npy', p_fa_dict)
        np.save(self.db_path + 'uniprot/npy/uniprot_human_reviewed_ids.npy', uniprot_human_reviewed_ids)
        np.save(self.db_path + 'uniprot/npy/uniprot_human_reviewed_isoform_ids.npy', uniprot_human_reviewed_isoform_ids)

        #******************************************************
        # All IDs maps to uniprot
        #******************************************************
        id_maps = pd.read_table(self.db_path + 'uniprot/org/HUMAN_9606_idmapping_selected.tab', sep='\t', header=None) 
        id_maps.columns = ['UniProtKB-AC', 'UniProtKB-ID', 'GeneID', 'RefSeq', 'GI', 'PDB', 'GO', 'UniRef100', 'UniRef90', 'UniRef50', 'UniParc', 'PIR', 'NCBI-taxon', 'MIM', 'UniGene', 'PubMed', 'EMBL', 'EMBL-CDS', 'Ensembl', 'Ensembl_TRS', 'Ensembl_PRO', 'Additional PubMed']
        id2uniprot_dict = {}
        id_lst = ['RefSeq','Ensembl_PRO']        
        for id in id_lst:
            cur_id_maps = id_maps.loc[id_maps['UniProtKB-AC'].isin(uniprot_human_reviewed_ids) & id_maps[id].notnull(), ['UniProtKB-AC',id]] 
            cur_id_dict = {}
            cur_id_maps.apply(lambda x: fill_dict(x['UniProtKB-AC'],x[id],cur_id_dict),axis = 1)
            id2uniprot_dict[id] = cur_id_dict
        pass
        np.save(self.db_path + 'uniprot/npy/id2uniprot_dict.npy', id2uniprot_dict)
        alm_fun.show_msg(self.log,self.verbose,'Created uniprot data.')
        
    def create_clinvar_data(self):
        
        def get_aa_by_pos(seq,pos):
            try:
                y = seq[pos-1]
                return y
            except:
                return(np.nan)
        
        def get_gene_symbol_hgvs(x):
            if '(p.' not in x:
                return np.nan
            else:
                try:
                    if '(' not in x.split(':')[0]:
                        y = x.split(':')[0]
                    else:
                        y = x.split(':')[0].split('(')[1].split(')')[0]                    
                    return y                                       
                except:
                    return '?'
        
        def get_aa_ref_hgvs(x):
            if '(p.' not in x:
                return np.nan
            else:
                y = x.split(':')[1].split('(')[1][2:5]
                if y in self.dict_aa3.keys():
                    return self.dict_aa3[y]
                else:
                    return '?'            
    
        def get_aa_pos_hgvs(x):
            if '(p.' not in x:
                return -1
            else:
                if '=' in x:
                    y = x.split(':')[1].split('(')[1][5:-2]
                    if y.isdigit():
                        return y
                    else:
                        return -1    
                else:            
                    y = x.split(':')[1].split('(')[1][5:-4]
                    if y.isdigit():
                        return y
                    else:
                        return -1    
            
        def get_aa_alt_hgvs(x):
            if '(p.' not in x:
                return np.nan
            else:
                if '=' in x:
                    return '*'
                else:  
                    y = x.split(':')[1].split('(')[1][-4:-1]
                    if y in self.dict_aa3.keys():
                        return self.dict_aa3[y]
                    else:
                        return '?' 
        
        ####***************************************************************************************************************************************************************    
        # NCBI FTP
        ####***************************************************************************************************************************************************************
        # Clinvar 
        # ftp://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz
        ####***************************************************************************************************************************************************************        
        self.init_humamdb_object ('clinvar')        
        return_info = alm_fun.download_ftp(self.ncbi_ftp_obj, '/pub/clinvar/tab_delimited/', 'variant_summary.txt.gz', self.db_path + 'clinvar/org/variant_summary.txt.gz')
        if (return_info == 'updated') | (return_info == 'downloaded'):
            alm_fun.gzip_decompress(self.db_path + 'clinvar/org/variant_summary.txt.gz', self.db_path + 'clinvar/org/variant_summary.txt')             

        #load clinvar rawdata
        self.clinvar_raw_file = 'clinvar/org/variant_summary.txt'        
        clinvar_raw = pd.read_table(self.db_path + self.clinvar_raw_file, sep='\t')
        alm_fun.show_msg(self.log,self.verbose,'clinVAR all records : ' + str(clinvar_raw.shape[0]))
            
        #create clinvar_snv
        clinvar_snv = clinvar_raw.loc[(clinvar_raw['Type'] == 'single nucleotide variant') & (clinvar_raw['Assembly'] == self.assembly) & (clinvar_raw['Chromosome'].isin(self.lst_chr)), ['GeneSymbol','HGNC_ID','Chromosome', 'Start', 'ReferenceAllele', 'AlternateAllele', 'ReviewStatus','LastEvaluated', 'ClinicalSignificance', 'Name', 'NumberSubmitters', 'RS# (dbSNP)', 'PhenotypeIDS', 'PhenotypeList','VariationID']]
        clinvar_snv.columns = ['gene_symbol','hgnc_id','chr', 'nt_pos', 'nt_ref', 'nt_alt', 'review_status','evaluate_time', 'clin_sig', 'hgvs', 'ev_num', 'rs', 'phenotype_id', 'phenotype_name','clinvar_id']
        clinvar_snv['chr'] = clinvar_snv['chr'].str.strip()
        clinvar_snv = clinvar_snv.loc[clinvar_snv['hgvs'].notnull(), :] 
        clinvar_snv['gene_symbol_from_hgvs'] = clinvar_snv['hgvs'].apply(lambda x: get_gene_symbol_hgvs(x))
        clinvar_snv['aa_ref_hgvs'] = clinvar_snv['hgvs'].apply(lambda x: get_aa_ref_hgvs(x))
        clinvar_snv['aa_pos_hgvs'] = clinvar_snv['hgvs'].apply(lambda x: get_aa_pos_hgvs(x))
        clinvar_snv['aa_pos_hgvs'] = clinvar_snv['aa_pos_hgvs'].astype(int)
        clinvar_snv['aa_alt_hgvs'] = clinvar_snv['hgvs'].apply(lambda x: get_aa_alt_hgvs(x))
        clinvar_snv['refseq_tvid'] = clinvar_snv['hgvs'].apply(lambda x: x.split(':')[0].split('(')[0])
        clinvar_snv['refseq_tid'] = clinvar_snv['refseq_tvid'].apply(lambda x: x.split('.')[0])
        clinvar_snv['data_source'] = 'clinvar'        
        clinvar_snv['review_star'] = 0
        clinvar_snv.loc[clinvar_snv['review_status'] == 'practice guideline','review_star'] = 4
        clinvar_snv.loc[clinvar_snv['review_status'] == 'reviewed by expert panel','review_star'] = 3
        clinvar_snv.loc[clinvar_snv['review_status'] == 'criteria provided, multiple submitters, no conflicts','review_star'] = 2
        clinvar_snv.loc[clinvar_snv['review_status'] == 'criteria provided, single submitter','review_star'] = 1    
        clinvar_snv.loc[clinvar_snv['review_status'] == 'criteria provided, conflicting interpretations','review_star'] = 1
                       
        clinvar_snv['clinsig_level'] = 0
        clinvar_snv.loc[(clinvar_snv['clin_sig'] == 'Pathogenic'), 'clinsig_level'] = 3
        clinvar_snv.loc[(clinvar_snv['clin_sig'] == 'Benign'), 'clinsig_level'] = 3                        
        clinvar_snv.loc[(clinvar_snv['clin_sig'] == 'Pathogenic/Likely pathogenic'), 'clinsig_level'] = 2
        clinvar_snv.loc[(clinvar_snv['clin_sig'] == 'Benign/Likely benign'), 'clinsig_level'] = 2
        clinvar_snv.loc[(clinvar_snv['clin_sig'] == 'Likely pathogenic'), 'clinsig_level'] = 1
        clinvar_snv.loc[(clinvar_snv['clin_sig'] == 'Likely benign'), 'clinsig_level'] = 1  
             
        clinvar_snv['label'] = -1
        clinvar_snv.loc[(clinvar_snv['clin_sig'] == 'Pathogenic'), 'label'] = 1
        clinvar_snv.loc[(clinvar_snv['clin_sig'] == 'Benign'), 'label'] = 0                        
        clinvar_snv.loc[(clinvar_snv['clin_sig'] == 'Pathogenic/Likely pathogenic'), 'label'] = 1
        clinvar_snv.loc[(clinvar_snv['clin_sig'] == 'Benign/Likely benign'), 'label'] = 0
        clinvar_snv.loc[(clinvar_snv['clin_sig'] == 'Likely pathogenic'), 'label'] = 1
        clinvar_snv.loc[(clinvar_snv['clin_sig'] == 'Likely benign'), 'label'] = 0     
        alm_fun.show_msg(self.log,self.verbose,'clinVAR snv records : ' + str(clinvar_snv.shape[0]))        
                        
        # remove non-coding region, synonymous, invalid records 
        clinvar_snv = clinvar_snv.loc[(clinvar_snv['aa_pos_hgvs'] != -1) & (clinvar_snv['aa_ref_hgvs'] != '*') & (clinvar_snv['aa_ref_hgvs'] != '?') & (clinvar_snv['aa_alt_hgvs'] != '?') & (clinvar_snv['aa_alt_hgvs'] != '_') & (clinvar_snv['aa_alt_hgvs'] != '*') & (clinvar_snv['aa_ref_hgvs'] != clinvar_snv['aa_alt_hgvs']), :]
        alm_fun.show_msg(self.log,self.verbose,'clinVAR exome missense snv records : ' + str(clinvar_snv.shape[0]))  
                          
        # remove 0 reveiw star, 0 clinical significance value 
        clinvar_snv = clinvar_snv.loc[(clinvar_snv['review_star'] > 0 ) & (clinvar_snv['clinsig_level'] >0) ,:]
        alm_fun.show_msg(self.log,self.verbose,'clinVAR review_star > 0 , clinsig_level > 0 records : ' + str(clinvar_snv.shape[0]))
        
        hgnc2uniprot_dict = np.load(self.db_path + 'hgnc/npy/hgnc2uniprot_dict.npy').item()
        id2hgnc_dict = np.load(self.db_path + 'hgnc/npy/id2hgnc_dict.npy').item()
        uniprot_seq_dict = np.load(self.db_path +'uniprot/npy/uniprot_seq_dict.npy').item()
        
        clinvar_snv['hgnc_id_from_hgvs'] = clinvar_snv['gene_symbol_from_hgvs'].apply(lambda x: id2hgnc_dict['symbol'].get(x,np.nan))        
        clinvar_snv['p_vid'] = clinvar_snv['hgnc_id_from_hgvs'].apply(lambda x: hgnc2uniprot_dict.get(x,np.nan))
        
        clinvar_snv['aa_ref_uniprot'] = clinvar_snv.apply(lambda x: get_aa_by_pos(uniprot_seq_dict.get(x['p_vid'],np.nan),x['aa_pos_hgvs']), axis = 1)
        
        clinvar_snv.loc[clinvar_snv['aa_ref_hgvs']!= clinvar_snv['aa_ref_uniprot'],:]
            
            
        clinvar_snv.to_csv(self.db_path + 'clinvar/all/clinvar_snv.csv')
        alm_fun.show_msg(self.log,self.verbose,'Created clinVAR data.')
 
    def create_aa_property_data(self):
        alm_fun.show_msg(self.log,self.verbose,'OK')   
   
    def create_blosum_data(self):
        alm_fun.show_msg(self.log,self.verbose,'OK')
        
    def create_funsum_data(self):
        alm_fun.show_msg(self.log,self.verbose,'OK')

    def create_pisa_data(self,parallel_id,parallel_num):
        ####***************************************************************************************************************************************************************    
        # EBI FTP
        ####***************************************************************************************************************************************************************
        # Uniprot ID to PDB ids 
        # (ftp://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/csv/pdb_chain_uniprot.csv.gz)       
        ####***************************************************************************************************************************************************************
        self.init_humamdb_object('pisa') 
        
        self.ebi_ftp_obj = alm_fun.create_ftp_object('ftp.ebi.ac.uk')
        return_info = alm_fun.download_ftp(self.ebi_ftp_obj, '/pub/databases/msd/sifts/flatfiles/csv/', 'pdb_chain_uniprot.csv.gz', self.db_path + 'pisa/org/pdb_chain_uniprot.csv.gz')
        if (return_info == 'updated') | (return_info == 'downloaded'):        
            alm_fun.gzip_decompress(self.db_path + 'pisa/org/pdb_chain_uniprot.csv.gz', self.db_path + 'pisa/org/pdb_chain_uniprot.csv')  

    def create_psipred_data(self):
        alm_fun.show_msg(self.log,self.verbose,'OK')
        
    def create_pfam_data(self):
        ####***************************************************************************************************************************************************************    
        # EBI FTP
        ####***************************************************************************************************************************************************************
        # Pfam
        # ftp://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/proteomes/9606.tsv.gz       
        ####***************************************************************************************************************************************************************
        self.init_humamdb_object('pfam') 
        self.ebi_ftp_obj = alm_fun.create_ftp_object('ftp.ebi.ac.uk')
        return_info = alm_fun.download_ftp(self.ebi_ftp_obj, '/pub/databases/Pfam/current_release/proteomes/', '9606.tsv.gz', self.db_path + 'pfam/org/9606.tsv.gz')
        if (return_info == 'updated') | (return_info == 'downloaded'):        
            alm_fun.gzip_decompress(self.db_path + 'pfam/org/9606.tsv.gz', self.db_path + 'pfam/org/9606.tsv')  
        
        pfam = pd.read_csv(self.db_path + 'pfam/org/9606.tsv', header=None, skiprows=3, sep='\t')
        pfam.columns = ['p_vid', 'a_start', 'a_end', 'e_start', 'e_end', 'hmm_id', 'hmm_name', 'type', 'hmm_start', 'hmm_end', 'hmm_length', 'bit_score', 'e_value', 'clan']
        p_vids = pfam.p_vid.unique()        
        for p_vid in p_vids:
            pfam.loc[(pfam.p_vid == p_vid), :].to_csv(self.db_path + 'pfam/bygene/' + p_vid + '_pfam.csv', index=None)
    
    def create_ensembl66_data(self):
        ####***************************************************************************************************************************************************************    
        # ensembl FTP
        ####***************************************************************************************************************************************************************
        # ensembl protein sequences 66 , because provean and sift was using ensembl release 66
        # ftp://ftp.ensembl.org/pub/release-66/fasta/homo_sapiens/pep/Homo_sapiens.GRCh37.66.pep.all.fa.gz     
        ####***************************************************************************************************************************************************************
        self.init_humamdb_object('ensembl66')  
        self.ensembl_ftp_obj = alm_fun.create_ftp_object('ftp.ensembl.org')      
        return_info = alm_fun.download_ftp(self.ensembl_ftp_obj, '/pub/release-66/fasta/homo_sapiens/pep/', 'Homo_sapiens.GRCh37.66.pep.all.fa.gz', self.db_path + 'ensembl66/org/Homo_sapiens.GRCh37.66.pep.all.fa.gz')
        if (return_info == 'updated') | (return_info == 'downloaded'):
            alm_fun.gzip_decompress(self.db_path + 'ensembl66/org/Homo_sapiens.GRCh37.66.pep.all.fa.gz', self.db_path + 'ensembl66/org/Homo_sapiens.GRCh37.66.pep.all.fa')  
        
        
        ensg2ensp66_dict = {}
        ensp2ensg66_dict = {}
        p_fa_dict = {}        
        cur_file = self.db_path + 'ensembl66/org/Homo_sapiens.GRCh37.66.pep.all.fa'
        if os.path.isfile(cur_file):
            p_fa = open(cur_file, 'r') 
            for line in p_fa:
                if line[0] == ">" :
                    cur_ensp = line.split(' ')[0][1:]
                    cur_ensg = line.split(' ')[3].split(':')[1]
                    ensp2ensg66_dict[cur_ensp] = cur_ensg
                    if cur_ensg in ensg2ensp66_dict.keys():
                        ensg2ensp66_dict[cur_ensg].append(cur_ensp)
                    else:
                        ensg2ensp66_dict[cur_ensg] = [cur_ensp]                    
                    p_fa_dict[cur_ensp] = ''
                else:
                    p_fa_dict[cur_ensp] += line.strip() 
        np.save(self.db_path + 'ensembl66/npy/ensembl66_seq_dict.npy', p_fa_dict) 
        np.save(self.db_path + 'ensembl66/npy/ensg2ensp66_dict.npy', ensg2ensp66_dict)
        np.save(self.db_path + 'ensembl66/npy/ensp2ensg66_dict.npy', ensp2ensg66_dict)
            
        return (p_fa_dict)  
    
    def create_sift_data(self):
        ####***************************************************************************************************************************************************************    
        # jcvi FTP
        ####***************************************************************************************************************************************************************
        # SIFT scores
        # ftp://ftp.jcvi.org/pub/data/provean/precomputed_scores/SIFT_scores_and_info_ensembl66_human.tsv.gz    
        ####***************************************************************************************************************************************************************
        self.init_humamdb_object('sift')  
        self.jcvi_ftp_obj = alm_fun.create_ftp_object('ftp.jcvi.org')
        return_info = alm_fun.download_ftp(self.jcvi_ftp_obj, '/pub/data/provean/precomputed_scores/', 'SIFT_scores_and_info_ensembl66_human.tsv.gz', self.db_path + 'sift/org/SIFT_scores_and_info_ensembl66_human.tsv.gz') 
        if (return_info == 'updated') | (return_info == 'downloaded'):
            alm_fun.gzip_decompress(self.db_path + 'sift/org/SIFT_scores_and_info_ensembl66_human.tsv.gz', self.db_path + 'sift/org/SIFT_scores_and_info_ensembl66_human.tsv')  
               
               
        id2uniprot_matched_dict = np.load(self.db_path + 'uniprot/npy/id2uniprot_matched_dict.npy').item()       
        sift_scores = open(self.db_path + 'sift/org/SIFT_scores_and_info_ensembl66_human.tsv', 'r')
        cur_enspid = None 
        for line in sift_scores:
            if not re.match('#',line):
                lst_line = line.split('\t')
                if cur_enspid != lst_line[0]:
                    if cur_enspid is not None: # close the old file 
                        cur_ensp_file.close()
                        uniprot_id = id2uniprot_matched_dict['ensembl66'].get(cur_enspid,np.nan)
                        if str(uniprot_id) != 'nan':
                            sift = pd.read_csv(self.db_path + 'sift/bygene/' + cur_enspid + '.tsv',header = None,sep = '\t')    
                            sift.columns = ['ensp_id','aa_pos','A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','median','n_seq'] 
                            sift['p_vid'] = uniprot_id
                            sift.drop(['median','n_seq'],axis = 1,inplace = True)
                            sift_melt = pd.melt(sift,id_vars=['p_vid','aa_pos'],value_vars = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'])
                            sift_melt.columns = ['p_vid','aa_pos','aa_alt','sift_score']
                            sift_melt.to_csv(self.db_path + 'sift/bygene/' + uniprot_id + '_sift.csv',index = None)                        
                    cur_enspid = lst_line[0]
                    cur_ensp_file = open(self.db_path + 'sift/bygene/' + cur_enspid + '.tsv','w')
                    cur_ensp_file.write(line)
                else:
                    cur_ensp_file.write(line)
        
    def create_provean_data(self):
        ####***************************************************************************************************************************************************************    
        # jcvi FTP
        ####***************************************************************************************************************************************************************
        # provean scores
        # ftp://ftp.jcvi.org/pub/data/provean/precomputed_scores/PROVEAN_scores_ensembl66_human.tsv.gz   
        ####***************************************************************************************************************************************************************
        self.init_humamdb_object('provean')  
        self.jcvi_ftp_obj = alm_fun.create_ftp_object('ftp.jcvi.org')
        return_info = alm_fun.download_ftp(self.jcvi_ftp_obj, '/pub/data/provean/precomputed_scores/', 'PROVEAN_scores_ensembl66_human.tsv.gz', self.db_path + 'provean/org/PROVEAN_scores_ensembl66_human.tsv.gz') 
        if (return_info == 'updated') | (return_info == 'downloaded'):
            alm_fun.gzip_decompress(self.db_path + 'provean/org/PROVEAN_scores_ensembl66_human.tsv.gz', self.db_path + 'provean/org/PROVEAN_scores_ensembl66_human.tsv')  
                              
        id2uniprot_matched_dict = np.load(self.db_path + 'uniprot/npy/id2uniprot_matched_dict.npy').item()       
        provean_scores = open(self.db_path + 'provean/org/provean_scores_and_info_ensembl66_human.tsv', 'r')
        cur_enspid = None 
        for line in provean_scores:
            if not re.match('#',line):
                lst_line = line.split('\t')
                if cur_enspid != lst_line[0]:
                    if cur_enspid is not None: # close the old file 
                        cur_ensp_file.close()
                        uniprot_id = id2uniprot_matched_dict['ensembl66'].get(cur_enspid,np.nan)
                        if str(uniprot_id) != 'nan':
                            provean = pd.read_csv(self.db_path + 'provean/bygene/' + cur_enspid + '.tsv',header = None,sep = '\t')    
                            provean.columns = ['ensp_id','aa_pos','A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','Del'] 
                            provean['p_vid'] = uniprot_id
                            provean_melt = pd.melt(provean,id_vars=['p_vid','aa_pos'],value_vars = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','Del'])
                            provean_melt.columns = ['p_vid','aa_pos','aa_alt','provean_score']
                            provean_melt.loc[provean_melt['aa_alt'] == 'Del','aa_alt'] = '*'
                            provean_melt.to_csv(self.db_path + 'provean/bygene/' + uniprot_id + '_provean.csv',index = None)                        
                    cur_enspid = lst_line[0]
                    cur_ensp_file = open(self.db_path + 'provean/bygene/' + cur_enspid + '.tsv','w')
                    cur_ensp_file.write(line)
                else:
                    cur_ensp_file.write(line)  
            
    def create_polyphen_data(self):
        alm_fun.show_msg(self.log,self.verbose,'OK')   
            
    def create_evmutation_data(self):
        alm_fun.show_msg(self.log,self.verbose,'OK')    
        
    def create_cadd_data(self):
        alm_fun.show_msg(self.log,self.verbose,'OK')    
    
    def retrieve_pisa_data(self,parallel_id,parallel_num):
        
        self.pdb_to_uniprot = pd.read_csv(self.db_path + 'pisa/org/pdb_chain_uniprot.csv', skiprows = 1 ,dtype={"PDB": str})          
        uniprot_human_reviewed_ids = list(np.load(self.db_path + 'uniprot/npy/uniprot_human_reviewed_ids.npy'))        
        total_gene_num = len(uniprot_human_reviewed_ids)        
        gene_index_array = np.linspace(0,total_gene_num-1,parallel_num+1 , dtype = int)        
        cur_parallel_indices = list(range(gene_index_array[parallel_id],gene_index_array[parallel_id+1]+1))
        cur_gene_ids = [uniprot_human_reviewed_ids[i] for i in cur_parallel_indices]        
        cur_log = self.db_path + 'pisa/log/pisa_data_parallel_' + str(parallel_id) + '.log'          
        for uniprot_id in cur_gene_ids:
            self.retrieve_pisa_by_uniprotid(uniprot_id,cur_log)    
            
    def retrieve_pisa_by_uniprotid(self,uniprot_id,cur_log):
        try: 
            if os.path.isfile(self.db_path + 'pisa/bygene/' + uniprot_id + '_pisa.csv'):
                return(0)
            interface_url = 'http://www.ebi.ac.uk/pdbe/pisa/cgi-bin/interfaces.pisa?'
#             multimers_url = 'http://www.ebi.ac.uk/pdbe/pisa/cgi-bin/multimers.pisa?'
            interface_xml_file_path  = self.db_path + 'pisa/org/' + uniprot_id + '_interface.xml'            
            interface_xml_file = open(interface_xml_file_path,'w')
#             multimers_xml_file = open(self.db_path + 'pdb/org/' + uniprot_id + '_multimers.xml', 'w')
#             multimers_asa_file = open(self.db_path + 'pdb/org/' + uniprot_id + '_multimers.txt', 'w')                         
            cur_gene_pdb = self.pdb_to_uniprot.loc[self.pdb_to_uniprot['SP_PRIMARY'] == uniprot_id, :]
            if cur_gene_pdb.shape[0] > 0 :                   
                #******************************************************
                # 1) Download the pisa xml file 
                #******************************************************
                pdb_lst = ','.join(cur_gene_pdb['PDB'].unique())                                                                                
                cur_interface_url = interface_url + pdb_lst
                response = urllib.request.urlopen(cur_interface_url) 
                interface_r = response.read().decode('utf-8')
                interface_xml_file.write(interface_r)
                interface_xml_file.close()
                alm_fun.show_msg(cur_log, 1, uniprot_id + ' pisa xml file is downloaded.')
                
                #******************************************************
                # 2) Generate molecule and bond file from xml 
                #******************************************************
                interface_molecule_file_path = self.db_path + 'pisa/bygene/' + uniprot_id + '_molecule.txt'
                interface_bond_file_path = self.db_path + 'pisa/bygene/' + uniprot_id + '_bond.txt'                               
                if (not os.path.isfile(interface_molecule_file_path)) | (not os.path.isfile(interface_bond_file_path)):            
                    interface_molecule_file = open(interface_molecule_file_path, 'w')
                    interface_bond_file = open(interface_bond_file_path, 'w')
                    with open(interface_xml_file_path) as infile:
                        infile_str = infile.read()
                        if len(infile_str) == 0:
                            return(0)
                        interface_tree = ET.fromstring(infile_str)
                        for pdb_entry in interface_tree.iter('pdb_entry'):
                            # print (pdb_entry[0].tag + ':' + pdb_entry[0].text)   #pdb_code
                            for interface in pdb_entry.iter("interface"):
                                # print (interface[0].tag + ':' + interface[0].text) #interface id 
                                for h_bonds in interface.iter("h-bonds"):
                                    for bond in h_bonds.iter("bond"):
                                        interface_bond_file.write('H\t' + str(bond[0].text) + '\t' + str(bond[1].text) + '\t' + \
                                                                          str(bond[2].text) + '\t' + str(bond[3].text) + '\t' + \
                                                                          str(bond[4].text) + '\t' + str(pdb_entry[0].text) + '\t' + str(interface[0].text) + '\n') 
                                        interface_bond_file.write('H\t' + str(bond[5].text) + '\t' + str(bond[6].text) + '\t' + \
                                                                          str(bond[7].text) + '\t' + str(bond[8].text) + '\t' + \
                                                                          str(bond[9].text) + '\t' + str(pdb_entry[0].text) + '\t' + str(interface[0].text) + '\n') 
                                for salt_bridges in interface.iter("salt-bridges"):
                                    for bond in salt_bridges.iter("bond"):
                                        interface_bond_file.write('S\t' + str(bond[0].text) + '\t' + str(bond[1].text) + '\t' + \
                                                                          str(bond[2].text) + '\t' + str(bond[3].text) + '\t' + \
                                                                          str(bond[4].text) + '\t' + str(pdb_entry[0].text) + '\t' + str(interface[0].text) + '\n')
                                        interface_bond_file.write('S\t' + str(bond[5].text) + '\t' + str(bond[6].text) + '\t' + \
                                                                          str(bond[7].text) + '\t' + str(bond[8].text) + '\t' + \
                                                                          str(bond[9].text) + '\t' + str(pdb_entry[0].text) + '\t' + str(interface[0].text) + '\n')                                   
                                for ss_bonds in interface.iter("ss-bonds"):
                                    for bond in ss_bonds.iter("bond"):
                                        interface_bond_file.write('D\t' + str(bond[0].text) + '\t' + str(bond[1].text) + '\t' + \
                                                                          str(bond[2].text) + '\t' + str(bond[3].text) + '\t' + \
                                                                          str(bond[4].text) + '\t' + str(pdb_entry[0].text) + '\t' + str(interface[0].text) + '\n')
                                        interface_bond_file.write('D\t' + str(bond[5].text) + '\t' + str(bond[6].text) + '\t' + \
                                                                          str(bond[7].text) + '\t' + str(bond[8].text) + '\t' + \
                                                                          str(bond[9].text) + '\t' + str(pdb_entry[0].text) + '\t' + str(interface[0].text) + '\n')                                                  
                                for cov_bonds in interface.iter("cov-bonds"):
                                    for bond in cov_bonds.iter("bond"):
                                        interface_bond_file.write('C\t' + str(bond[0].text) + '\t' + str(bond[1].text) + '\t' + \
                                                                          str(bond[2].text) + '\t' + str(bond[3].text) + '\t' + \
                                                                          str(bond[4].text) + '\t' + str(pdb_entry[0].text) + '\t' + str(interface[0].text) + '\n')
                                        interface_bond_file.write('C\t' + str(bond[5].text) + '\t' + str(bond[6].text) + '\t' + \
                                                                          str(bond[7].text) + '\t' + str(bond[8].text) + '\t' + \
                                                                          str(bond[9].text) + '\t' + str(pdb_entry[0].text) + '\t' + str(interface[0].text) + '\n')                                    
                                for molecule in interface.iter("molecule"):
                                    # print(molecule[1].tag +':' + molecule[1].text) #chain_id
                                    for residue in molecule.iter("residue"):
                                        # print (residue[0].tag + ':' + residue[0].text + '|' + residue[1].tag + ':' + residue[1].text +'|' + residue[5].tag + ':' + residue[5].text)
                                        interface_molecule_file.write(str(pdb_entry[0].text) + '\t' + str(interface[0].text) + '\t' + str(molecule[1].text) + '\t' + str(residue[0].text) + '\t' + str(residue[1].text) + '\t' + str(residue[2].text) + '\t' + str(residue[3].text) + '\t' + str(residue[4].text) + '\t' + str(residue[5].text) + '\t' + str(residue[6].text) + '\t' + str(residue[7].text) + '\n')
                    interface_bond_file.close()                    
                    interface_molecule_file.close()
                    alm_fun.show_msg(cur_log, 1, uniprot_id + ' pisa xml file is processed.')
                    #******************************************************
                    # 3) Generate pisa file for each gene  
                    #******************************************************                        
                    if (os.stat(interface_bond_file_path).st_size != 0) & (os.stat(interface_molecule_file_path).st_size != 0):     
                        cur_bond_df = pd.read_csv(self.db_path + 'pisa/bygene/' + uniprot_id + '_bond.txt', sep='\t', header=None, dtype={6: str})
                        cur_bond_df.columns = ['bond','CHAIN', 'residue', 'aa_pos', 'ins_code','atom','PDB','interface']                                    
                        cur_bond_df = cur_bond_df.groupby(['bond','CHAIN','residue','aa_pos','ins_code','PDB','interface'])['atom'].agg('count').reset_index()
                        cur_bond_df['h_bond']  = cur_bond_df.apply(lambda x: x['atom'] if x['bond'] == 'H' else 0, axis = 1)
                        cur_bond_df['salt_bridge']  = cur_bond_df.apply(lambda x: x['atom'] if x['bond'] == 'S' else 0, axis = 1)
                        cur_bond_df['disulfide_bond']  = cur_bond_df.apply(lambda x: x['atom'] if x['bond'] == 'D' else 0, axis = 1)
                        cur_bond_df['covelent_bond']  = cur_bond_df.apply(lambda x: x['atom'] if x['bond'] == 'C' else 0, axis = 1)                                        
                        cur_bond_df.drop(columns = ['atom','bond'],inplace = True)
                        cur_bond_df = cur_bond_df.groupby(['CHAIN','residue','aa_pos','ins_code','PDB','interface'])['h_bond','salt_bridge','disulfide_bond','covelent_bond'].agg('sum').reset_index()                
                        cur_molecule_df = pd.read_csv(self.db_path + 'pisa/bygene/' + uniprot_id + '_molecule.txt', sep='\t', header=None, dtype={0: str})
                        cur_molecule_df.columns = ['PDB', 'interface', 'CHAIN', 'ser_no', 'residue', 'aa_pos', 'ins_code','bonds','asa','bsa','solv_ne']                    
                        
                        cur_molecule_df = cur_molecule_df.merge(cur_bond_df,how = 'left')
                        cur_molecule_df['bsa_ratio'] = 0
                        cur_molecule_df.loc[cur_molecule_df['asa'] !=0, 'bsa_ratio'] = cur_molecule_df.loc[cur_molecule_df['asa'] !=0, 'bsa'] / cur_molecule_df.loc[cur_molecule_df['asa'] != 0, 'asa']
                                                  
                        cur_molecule_df_groupby = cur_molecule_df.groupby(['residue', 'aa_pos'])
                        cur_pisa_value_df1 = cur_molecule_df_groupby['asa'].agg(['mean', 'std', 'count']).reset_index().sort_values(['aa_pos'])
                        cur_pisa_value_df2 = cur_molecule_df_groupby['bsa','bsa_ratio','solv_ne','h_bond','salt_bridge','disulfide_bond','covelent_bond'].agg('max').reset_index().sort_values(['aa_pos'])
                        cur_pisa_value_df3 = cur_molecule_df_groupby['solv_ne'].agg('min').reset_index().sort_values(['aa_pos'])
                            
                        cur_pisa_value_df1.columns = ['residue', 'aa_pos', 'asa_mean', 'asa_std', 'asa_count']   
                        cur_pisa_value_df2.columns = ['residue', 'aa_pos', 'bsa_max', 'solv_ne_max','bsa_ratio_max', 'h_bond_max','salt_bridge_max','disulfide_bond_max','covelent_bond_max']
                        cur_pisa_value_df3.columns = ['residue', 'aa_pos', 'solv_ne_min']
                        
                        cur_pisa_df = cur_pisa_value_df1.merge(cur_pisa_value_df2,how = 'left')
                        cur_pisa_df = cur_pisa_df.merge(cur_pisa_value_df3,how = 'left')
                         
                        cur_pisa_df['aa_ref'] = cur_pisa_df['residue'].apply(lambda x: self.dict_aa3_upper.get(x, np.nan))
                        cur_pisa_df = cur_pisa_df.loc[cur_pisa_df['aa_ref'].notnull(), ]
                        cur_pisa_df['p_vid'] = uniprot_id
                        cur_pisa_df.drop(['residue'],axis = 1,inplace = True)
                        cur_pisa_df = cur_pisa_df.fillna(0)
        
                        cur_pisa_df.columns = ['aa_pos','asa_mean','asa_std','asa_count','bsa_max','solv_ne_max','bsa_ratio_max','h_bond_max','salt_bridge_max','disulfide_bond_max','covelent_bond_max','solv_ne_min','aa_ref','p_vid']    
                        cur_pisa_df.to_csv(self.db_path + 'pisa/bygene/' + uniprot_id + '_pisa.csv', index=False)
                        alm_fun.show_msg(cur_log, 1, uniprot_id + ' pisa csv file is generated.')                
            else:
                return(0)    
        except:
            alm_fun.show_msg(cur_log, 1, uniprot_id + traceback.format_exc() + '\n')            
            return(0)     
    
    def create_matched_uniprot_mapping(self):
        
        #*************************************************************************************
        # Create mapping from other protein ids to uniprot ids
        # 1) Other ID - > HGNC id   (1 to 1 relationship) 
        # 2) HGNC id  - > Uniprot Id ( 1 to many relationship)
        # 3) Compare the protein sequence of other id and each uniprot id 
        # 4) Other ID  - > Last Uniprot ID that has matched sequence
        #*************************************************************************************
        
        id2uniprot_matched_dict = {}
        
        hgnc2id_dict = np.load(self.db_path + 'hgnc/npy/hgnc2id_dict.npy').item()
        id2hgnc_dict = np.load(self.db_path + 'hgnc/npy/id2hgnc_dict.npy').item()    
        uniprot_seq_dict = np.load(self.db_path + 'uniprot/npy/uniprot_seq_dict.npy').item()
        ensembl66_seq_dict = np.load(self.db_path + 'ensembl66/npy/ensembl66_seq_dict.npy').item()
        ensp2ensg66_dict = np.load(self.db_path + 'ensembl66/npy/ensp2ensg66_dict.npy').item()
                
        #matched ensembl66 to uniprot
        id2uniprot_matched_dict['ensembl66']={}
        for cur_ensp in ensp2ensg66_dict.keys():
            cur_uniprot_ids = hgnc2id_dict['uniprot_ids'].get(id2hgnc_dict['ensembl_gene_id'].get(ensp2ensg66_dict[cur_ensp],np.nan),np.nan)
            if str(cur_uniprot_ids) != 'nan':
                for cur_uniprot_id in cur_uniprot_ids.split('|'):
                    if ensembl66_seq_dict.get(cur_ensp,np.nan) == uniprot_seq_dict.get(cur_uniprot_id,np.nan):            
                        id2uniprot_matched_dict['ensembl66'][cur_ensp] = cur_uniprot_id    
        np.save(self.db_path + 'uniprot/npy/id2uniprot_matched_dict.npy', id2uniprot_matched_dict) 
    
    
    
    
    
    
    
    
    
    
    
    
    
       