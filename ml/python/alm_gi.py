#!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3
import numpy as np
import pandas as pd
import matplotlib
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
import pydotplus 
import traceback
import re
import gzip
import urllib
import subprocess

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
from lxml import etree
import xml.etree.ElementTree as ET
from io import StringIO
import mysql.connector
# from IPython.core.tests.test_inputsplitter import line

# pd.set_option('display.height', 1000)
# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)


def range_list(start, end, folds):
    step = int((end - start) / folds)
    l = list(range(start, end + 1, step))
    l.remove(l[-1])
    l.append(end)
    return l


def hamming_distance(s1, s2):
        assert len(s1) == len(s2)
        return sum(ch1 != ch2 for ch1, ch2 in zip(s1, s2))   

    
def show_msg(infile, verbose, msg):
    with open(infile, 'a') as f:
        f.write(msg)
    if (verbose == 1): print (msg)   

              
class alm_gi:

    def __init__(self, db_path, assembly, flanking_k, slow_version=0, upgrade=0):
        stime = time.time()  
        self.upgrade = upgrade  # if 1, all the relevant files will be refreshed from FTP server
        self.assembly = assembly
        self.db_path = db_path
        self.flanking_k = flanking_k
        sns.set(rc={'axes.facecolor':'#C0C0C0'})
        
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
            
        self.aa_properties = self.get_aa_properties()
        
        ####***************************************************************************************************************************************************************
        # dms
        ####***************************************************************************************************************************************************************
        self.dict_dms_genes = {}
        self.dict_dms_genes['UBE2I'] = 'P63279'
        self.dict_dms_genes['SUMO1'] = 'P63165'
        self.dict_dms_genes['TPK1'] = 'Q9H3S4'
        self.dict_dms_genes['CALM1'] = 'P0DP23'
        self.dict_dms_genes['NCS1'] = 'P62166'        
        self.dict_dms_genes['CBS_0'] = 'P35520'
        self.dict_dms_genes['CBS_1'] = 'P35520'
        self.dict_dms_genes['CBS_400'] = 'P35520'
        self.dict_dms_genes['TECR'] = 'Q9NZ01'
        self.dict_dms_genes['GDI1'] = 'P31150'
        self.dict_dms_genes['MTHFR'] = 'P42898'
        self.dict_dms_genes['HMGCR'] = 'P04035'

        self.dms_seqdict = self.get_dms_seqdict()
        
        ####***************************************************************************************************************************************************************
        # blosum,funsum,mafsum,mutsum,selsum,clinvarsum,humsavarsum
        ####***************************************************************************************************************************************************************
        self.dict_sums = {}
        
        #funsums
        self.funsum_genes = ['P63279','P63165','P62166','Q9H3S4','P0DP23','Q9NZ01','P31150','P42898','P35520']     
        self.funsum_scores = ['fitness']
        self.funsum_centralities = ['mean']
        self.funsum_properties = ['in_domain']
        self.funsum_dmsfiles = self.db_path + 'dms/funregressor_training_final.csv'
        self.funsum_weightedby_columns = ['']
        self.funsum_weightedby_columns_inverse = [0]
        self.create_funsums(quality_cutoff=0)
        #mafsum and patsum
        self.mafsum_scores = ['gnomad_af']
        self.mafsum_centralities = ['normalization']
        self.mafsum_properties = []
        self.patsum_centralities = ['mean']
        
        #accsum (accessibility)
        self.get_human_codon_usage()
        self.create_accsums()
                
        #blosums
        [self.df_blosums, self.dict_blosums] = self.get_blosums()
        
        ####***************************************************************************************************************************************************************
        # pdb to uniprot (be aware there are cases that PDB id is 'XEXX', the pandas will read it as float
        ####***************************************************************************************************************************************************************
        self.pdb_to_uniprot = pd.read_csv(self.db_path + 'pdb/csv/pdb_chain_uniprot.csv', dtype={"PDB": str})
        ####***************************************************************************************************************************************************************
        # ensembl
        ####***************************************************************************************************************************************************************
        self.ensembl_seqdict = np.load(self.db_path + 'ensembl/npy/ensembl_seq_dict.npy').item()
        ####***************************************************************************************************************************************************************
        # uniprot
        ####***************************************************************************************************************************************************************
        self.uniprot_seqdict = np.load(self.db_path + 'uniprot/npy/uniprot_seq_dict.npy').item()
        self.uniprot2ensembl_dict = np.load(self.db_path + 'uniprot/npy/uniprot2ensembl_dict.npy').item()
        self.uniprot_reviewed_ids = np.load(self.db_path + 'uniprot/npy/uniprot_reviewed_ids.npy')
        self.uniprot_isoform_ids = np.load(self.db_path + 'uniprot/npy/uniprot_isoform_ids.npy')
        self.uniprot_df = pd.read_csv(self.db_path + 'uniprot/csv/uniprot_df.csv')        
        self.dict_uniprot_ac2seq = np.load(self.db_path + 'uniprot/npy/dict_uniprot_ac2seq.npy')
        self.dict_uniprot_gene2ac = np.load(self.db_path + 'uniprot/npy/dict_uniprot_gene2ac.npy')
        self.dict_uniprot_ac2gname_gsyn = np.load(self.db_path + 'uniprot/npy/dict_uniprot_ac2gname_gsyn.npy')  
              
        ####***************************************************************************************************************************************************************
        # refseq
        ####***************************************************************************************************************************************************************           
        self.refseq_vids = pd.read_csv(self.db_path + 'refseq/csv/refseq_vids.csv')    
        self.refseq2uniprot_isoform_dict = np.load(self.db_path + 'refseq/npy/refseq2uniprot_isoform_dict.npy').item()
        self.refseq2uniprot_dict = np.load(self.db_path + 'refseq/npy/refseq2uniprot_dict.npy').item()
        self.refseq_seqdict = np.load(self.db_path + 'refseq/npy/refseq_seq_dict.npy').item() 
        
        ####***************************************************************************************************************************************************************
        # ucsc
        ####***************************************************************************************************************************************************************           
#         self.ucsc2uniprot_dict = np.load(self.db_path + 'ucsc/npy/ucsc2uniprot_dict.npy').item() 
        
#         self.gnomad_aa = pd.read_csv(self.db_path + 'gnomad/gnomad_output_snp_aa_uniprot.txt', sep='\t', dtype={"chr": str})
        ####***************************************************************************************************************************************************************
        # database upgrade 
        ####***************************************************************************************************************************************************************
        if slow_version == 1:
            if upgrade == 1:
                # self.update_ftp_files()
                self.polyphen_train = self.get_polyphen_train_data()
                [self.uniprot_seqdict, self.uniprot_reviewed_ids, self.uniprot_isoform_ids] = self.get_uniprot_seqdict()
                [self.uniprot_df, self.dict_uniprot_ac2seq, self.dict_uniprot_gene2ac, self.dict_uniprot_ac2gname_gsyn] = self.get_uniprot_info()
                self.refseq_seqdict = self.get_refseq_seqdict()
                [self.refseq_vids, self.refseq2uniprot_isoform_dict, self.refseq2uniprot_dict] = self.get_refseq_idmapping()
                # self.clinvar_snv = self.prepare_clinvar_data()
            else:                   
#                 ####***************************************************************************************************************************************************************
#                 #polyphen training data
#                 ####***************************************************************************************************************************************************************
#                 self.polyphen_train = pd.read_csv(self.db_path + 'polyphen/csv/polyphen_train.csv')
#                  

                ####***************************************************************************************************************************************************************
                # gnomad
                ####***************************************************************************************************************************************************************
#                 self.gnomad_nt_aa = pd.read_csv(self.db_path+'gnomad/gnomad_output_snp_processed.txt',sep = '\t',dtype={"chr": str})
#                 self.gnomad_nt_aa.loc[(self.gnomad_nt_aa['chr'] == '15') & (self.gnomad_nt_aa['nt_pos'] == 85342440),:]         
#                 self.gnomad_nt = pd.read_csv(self.db_path+'gnomad/gnomad_output_snp_af.txt',sep = '\t')
#                 self.gnomad_nt.columns = ['chr','nt_pos','rs_id','nt_ref','nt_alt','quality_score','filter','gnomad_ac','gnomad_an','gnomad_af','gnomad_esa_af','gnoamd_gc','gnomad_gc_homo_ref','gnomad_gc_hetero','gnomad_gc_homo_alt']
#                 self.gnomad_aa = pd.read_csv(self.db_path + 'gnomad/gnomad_output_snp_aa_uniprot.txt', sep='\t', dtype={"chr": str})
#                                 
                ####***************************************************************************************************************************************************************
                # pdb asa 
                ####***************************************************************************************************************************************************************
                self.pdb_asa = pd.read_csv(self.db_path + 'pdb/csv/asa_df.csv', header=None)
                self.pdb_asa.columns = ['aa_pos', 'aa_ref', 'asa_mean', 'asa_std', 'asa_count', 'p_vid']
                     
                ####***************************************************************************************************************************************************************
                # psipred 
                ####***************************************************************************************************************************************************************
                self.psipred = pd.read_csv(self.db_path + 'psipred/psipred_df.csv')
                self.psipred = self.psipred.drop_duplicates()

                ####***************************************************************************************************************************************************************
                # pdb ss
                ####***************************************************************************************************************************************************************
                self.pdbss = pd.read_csv(self.db_path + 'pdbss/pdbss_final.csv')

                ####***************************************************************************************************************************************************************
                #### pfam
                ####***************************************************************************************************************************************************************  
                self.pfam = pd.read_csv(self.db_path + 'pfam/9606.tsv', header=None, skiprows=3, sep='\t')
                self.pfam.columns = ['p_vid', 'a_start', 'a_end', 'e_start', 'e_end', 'hmm_id', 'hmm_name', 'type', 'hmm_start', 'hmm_end', 'hmm_length', 'bit_score', 'e_value', 'clan']
                
                ####***************************************************************************************************************************************************************
                #### evmutation
                ####***************************************************************************************************************************************************************                
#                 self.evm_score = pd.read_csv(self.db_path + 'evmutation/csv/evmutation_df_org.csv')
                
                ####***************************************************************************************************************************************************************
                #### envision
                ####***************************************************************************************************************************************************************
#                 self.envision_score = pd.read_csv(self.db_path + 'envision/csv/envision_score_for_extrapolation_processed.csv')

                ####***************************************************************************************************************************************************************
                #### primateAI
                ####***************************************************************************************************************************************************************
#                 self.primateai_score = pd.read_csv(self.db_path + 'primateai/PrimateAI_scores_v0.2.tsv', sep = '\t')
#                 primateai_score.column = ['chr_org','nt_pos','nt_ref','nt_alt','aa_ref','aa_alt','codon','ucsc_id','exac_af','primateai_score']
#                 primateai_score['chr'] = primateai_score['chr_org'].apply(lambda x:  x[3:-1])
#                         

            pass
        etime = time.time()
        print("Class: [alphame_gi] Fun: [__init__] -- alphame_gi initiation took %g seconds" % (etime - stime))  
        
    def get_polyphen_train_data(self):
        polyphen_train_deleterious_file = 'polyphen/org/humdiv-2011_12.deleterious.pph.input'
        polyphen_train_neutral_file = 'polyphen/org/humdiv-2011_12.neutral.pph.input'
        polyphen_train0 = pd.read_table(self.db_path + polyphen_train_neutral_file, header=None, names=['p_vid', 'aa_pos', 'aa_ref', 'aa_alt'])        
        polyphen_train1 = pd.read_table(self.db_path + polyphen_train_deleterious_file, header=None, names=['p_vid', 'aa_pos', 'aa_ref', 'aa_alt'])
        polyphen_train = pd.concat([polyphen_train0, polyphen_train1])
        polyphen_train['polyphen_train'] = 1
        polyphen_train.to_csv(self.db_path + 'polyphen/csv/polyphen_train.csv', index=False)
        return (polyphen_train)
               
    def get_human_codon_usage(self):
        self.codon_usage_df = pd.read_csv(self.db_path + 'other/codon_usage_human.txt', sep='\t', header=None)
        self.codon_usage_df.columns = ['codon_r', 'codon', 'aa', 'freq_aa', 'freq_all', 'count']
        self.codon_usage_df.loc[self.codon_usage_df['aa'] == '_', 'aa'] = '*'
        
        self.dict_aa_codon = {}
        self.dict_codon = {}
        self.dict_codon_freq_aa = {}
        self.dict_codon_freq_all = {}
        for i in range(self.codon_usage_df.shape[0]):
            cur_aa = self.codon_usage_df.loc[i, 'aa']
            cur_codon = self.codon_usage_df.loc[i, 'codon']
            
            if self.dict_aa_codon.get(cur_aa, '') == '':
                self.dict_aa_codon[cur_aa] = [cur_codon]
            else:
                self.dict_aa_codon[cur_aa] = self.dict_aa_codon[cur_aa] + [cur_codon]
            self.dict_codon[cur_codon] = cur_aa
            self.dict_codon_freq_aa[cur_codon] = self.codon_usage_df.loc[i, 'freq_aa']
            self.dict_codon_freq_all[cur_codon] = self.codon_usage_df.loc[i, 'freq_all']
    
    def get_codon_usage(self):
        self.codon_usage_df = pd.read_csv(self.db_path + 'other/codon_usage_human.txt', sep='\t', header=None)
        self.codon_usage_df.columns = ['codon_r', 'codon', 'aa', 'freq_aa', 'freq_all', 'count']
        self.aa_codon_df = self.codon_usage_df[['codon', 'aa']]
           
        self.codonusage_species_df = pd.read_csv(self.db_path + 'codon/codonusage_species.txt', sep='|')
        lst_all_species = list(self.codonusage_species_df.id)
        self.codonusage_species_df['cds_num'] = self.codonusage_species_df.cds_num.apply(lambda x: x.strip(' '))
        self.codonusage_species_df['numeric_cds_num'] = self.codonusage_species_df.cds_num.apply(lambda x:  x.isnumeric())
        
        self.codonusage_species_df = self.codonusage_species_df.loc[self.codonusage_species_df['numeric_cds_num'] == True , :]
        self.codonusage_species_df['cds_num'] = self.codonusage_species_df['cds_num'].astype(int)
        lst_target_species = list(self.codonusage_species_df.loc[self.codonusage_species_df.cds_num > 10000, 'id'])
        
        self.codonusage_df = pd.read_csv(self.db_path + 'codon/codonusage.txt', sep=' ', header=None)
        self.codonusage_df = self.codonusage_df.transpose()
        self.codonusage_df.columns = ['codon_r'] + lst_all_species
        self.codonusage_df = self.codonusage_df[['codon_r'] + lst_target_species]
        
        self.codonusage_df['codon'] = self.codonusage_df['codon_r'].apply(lambda x: x.replace('U', 'T'))
        self.codonusage_df = pd.merge(self.codonusage_df, self.aa_codon_df, how='left')

        self.dict_codon = {}
        for i in range(self.codonusage_df.shape[0]):
            self.dict_codon[self.codonusage_df.loc[i, 'codon']] = self.codonusage_df.loc[i, 'aa']
        
        self.dict_codon_freq_all = {}
        for id in list(self.codonusage_species_df.id):
            try:
                cur_id_dict = {}
                cur_species = self.codonusage_df[str(id)]
                cur_species = cur_species / sum(cur_species)
                for i in range(self.codonusage_df.shape[0]):
                    cur_id_dict[self.codon_usage_df.loc[i, 'codon']] = cur_species[i]
                self.dict_codon_freq_all[str(id)] = cur_id_dict
            except:
                continue
              
    def get_aa_editdistance(self, aa_ref, aa_alt):        
        aa_ref_codons = self.dict_codon[aa_ref]
        aa_alt_codons = self.dict_codon[aa_alt]
        avg_dist = 0
        for ref_codon in aa_ref_codons:
            for alt_codon in aa_alt_codons:
                avg_dist += hamming_distance(ref_codon, alt_codon) * self.dict_codon_freq_aa[ref_codon] * self.dict_codon_freq_aa[alt_codon]            
        return (avg_dist)
    
    def get_aa_accessibility(self, aa_ref, aa_alt, titv_ratio=2):        
        aa_ref_codons = self.dict_aa_codon[aa_ref]
        aa_alt_codons = self.dict_aa_codon[aa_alt]
        access = 0
        
        for ref_codon in aa_ref_codons:
            for alt_codon in aa_alt_codons:
                if hamming_distance(ref_codon, alt_codon) == 1:
                    if self.get_codon_titv(ref_codon, alt_codon) == 'ti':
                        access += self.dict_codon_freq_aa[ref_codon] * self.dict_codon_freq_aa[alt_codon]
                    if self.get_codon_titv(ref_codon, alt_codon) == 'tv':
                        access += self.dict_codon_freq_aa[ref_codon] * self.dict_codon_freq_aa[alt_codon] / titv_ratio
#         if access == 0 :
#             access = np.nan
        return (access)
    
    def get_codon_titv(self, ref_codon, alt_codon):
        ref_codon_list = list(ref_codon)
        alt_codon_list = list(alt_codon)
        ref_nt = ''
        alt_nt = ''
        for i in range(len(ref_codon_list)):
            if ref_codon_list[i] != alt_codon_list[i]:
                ref_nt = ref_codon_list[i]
                alt_nt = alt_codon_list[i]
                break
        if ((ref_nt == 'A') & (alt_nt == 'G')) | ((ref_nt == 'G') & (alt_nt == 'A')) | ((ref_nt == 'C') & (alt_nt == 'T')) | ((ref_nt == 'T') & (alt_nt == 'C')):
            return ('ti')
        else:
            return ('tv')
    
    def get_nt_titv(self, ref_nt, alt_nt):
        if ((ref_nt == 'A') & (alt_nt == 'G')) | ((ref_nt == 'G') & (alt_nt == 'A')) | ((ref_nt == 'C') & (alt_nt == 'T')) | ((ref_nt == 'T') & (alt_nt == 'C')):
            return ('ti')
        else:
            return ('tv')    
    
    def create_dstsum(self):        
        self.dstsum_df = pd.DataFrame(columns=['aa_ref', 'aa_alt', 'avg_dist'])        
        for aa_ref in self.lst_aa:
            for aa_alt in self.lst_aa:
                cur_df = pd.DataFrame.from_records([(aa_ref, aa_alt, self.get_aa_editdistance(aa_ref, aa_alt))], columns=['aa_ref', 'aa_alt', 'avg_dist'])
                self.dstsum_df = self.dstsum_df.append(cur_df)
        self.dstsum_df.to_csv(self.db_path + 'dstsum/' + 'dstsum.csv')
        
    def create_mut_transmatrix(self, titv_ratio=2):
        self.codon_transition_df = pd.DataFrame(columns=['codon_ref', 'codon_alt', 'aa_ref', 'aa_alt', 'accessibility'])   
        for codon_ref in self.dict_codon:
            # the probability of a random mutation happen in first, second or third position of the target codon
            p_codon_position = 1 / 3
            # the probability of substituted nucleotide (two possible transversion and one possible transition) 
            p_transition = titv_ratio / (titv_ratio + 2)
            p_transversion = 1 / (titv_ratio + 2)
            
            lst_codon_ref = list(codon_ref)
            for i in range(3):
                nt_ref = lst_codon_ref[i]
                lst_nt_alt = self.lst_nt.copy()
                lst_nt_alt.remove(nt_ref)
                for nt_alt in lst_nt_alt:
                    lst_codon_alt = lst_codon_ref.copy()
                    lst_codon_alt[i] = nt_alt
                    codon_alt = ''.join(lst_codon_alt)
                    aa_ref = self.dict_codon[codon_ref]
                    aa_alt = self.dict_codon[codon_alt]
                    
                    if self.get_nt_titv(nt_ref, nt_alt) == 'ti':
                        accessibility = p_codon_position * p_transition
                    if self.get_nt_titv(nt_ref, nt_alt) == 'tv':
                        accessibility = p_codon_position * p_transversion
                        
                    cur_df = pd.DataFrame.from_records([(codon_ref, codon_alt, aa_ref, aa_alt, accessibility)], columns=['codon_ref', 'codon_alt', 'aa_ref', 'aa_alt', 'accessibility'])
                    self.codon_transition_df = self.codon_transition_df.append(cur_df) 
        self.codon_transition_df.to_csv(self.db_path + 'mutsum/' + 'codon_transition.csv', index=False)
        
        self.codon_transition_matrix = self.codon_transition_df.pivot(index='codon_ref', columns='codon_alt', values='accessibility')
        self.codon_transition_matrix.fillna(0, inplace=True)
        self.codon_transition_matrix.to_csv(self.db_path + 'mutsum/' + 'codon_transition_matrix.csv', index=True)
        
    def create_mutsums(self, titv_ratio=2):        
        self.codon_transition_matrix = pd.read_csv(self.db_path + 'mutsum/' + 'codon_transition_matrix.csv', index_col=0)          
        # for species in self.dict_codon_freq_all:
        for species in self.mutsum_species:
            cur_codon_usage = [] 
            cur_codons = []
            for codon in self.codon_transition_matrix.columns:
                cur_codon_usage.append(self.dict_codon_freq_all[species][codon])
                cur_codons.append(codon)
            pass   
            start_codon_usage_diag = np.diag(cur_codon_usage)
            mutsum_codon_matrix = start_codon_usage_diag * np.matrix(self.codon_transition_matrix)            
            mutsum_codon_matrix_df = pd.DataFrame(mutsum_codon_matrix, index=cur_codons, columns=cur_codons)
            mutsum_codon_matrix_df['codon_ref'] = mutsum_codon_matrix_df.index
            mutsum_codon_matrix_df = pd.melt(mutsum_codon_matrix_df, 'codon_ref')
            mutsum_codon_matrix_df.columns = ['codon_ref', 'codon_alt', 'mutsum']
            mutsum_codon_matrix_df['mutsum'] = mutsum_codon_matrix_df['mutsum'] / mutsum_codon_matrix_df['mutsum'].sum()
            mutsum_codon_matrix_df['aa_ref'] = mutsum_codon_matrix_df['codon_ref'].apply(lambda x: self.dict_codon[x])
            mutsum_codon_matrix_df['aa_alt'] = mutsum_codon_matrix_df['codon_alt'].apply(lambda x: self.dict_codon[x])
            mutsum_aa_matrix_df = mutsum_codon_matrix_df.groupby(['aa_ref', 'aa_alt'])['mutsum'].sum().reset_index()    
            mutsum_codon_matrix_df['mutsum'] = mutsum_codon_matrix_df['mutsum']
            mutsum_aa_matrix_df['mutsum'] = mutsum_aa_matrix_df['mutsum']
            
            mutsum_codon_matrix_df = mutsum_codon_matrix_df.replace(0, np.nan)      
            mutsum_aa_matrix_df = mutsum_aa_matrix_df.replace(0, np.nan)
            
            # mutsum_codon_matrix_df
                  
            mutsum_codon_matrix_df.to_csv(self.db_path + 'mutsum/' + 'mutsum_codon_' + species + '.csv', index=False)
            mutsum_aa_matrix_df.to_csv(self.db_path + 'mutsum/' + 'mutsum_' + species + '.csv', index=False)
    
    def create_accsums(self, titv_ratio=2):        
        self.accsum_df = pd.DataFrame(columns=['aa_ref', 'aa_alt', 'accessibility'])        
        for aa_ref in self.lst_aa_21:
            for aa_alt in self.lst_aa_21:
                cur_df = pd.DataFrame.from_records([(aa_ref, aa_alt, self.get_aa_accessibility(aa_ref, aa_alt, titv_ratio))], columns=['aa_ref', 'aa_alt', 'accessibility'])
                self.accsum_df = self.accsum_df.append(cur_df)
        self.accsum_df.to_csv(self.db_path + 'accsum/' + 'accsum.csv', index=False)
        
    def create_mutsums_backup(self, generations, titv_ratio=2):        
#         self.mutsum_df = pd.DataFrame(columns = ['aa_ref','aa_alt','accessibility'])        
#         for aa_ref in self.lst_aa:
#             for aa_alt in self.lst_aa:
#                 cur_df = pd.DataFrame.from_records([(aa_ref,aa_alt,self.get_aa_accessibility(aa_ref,aa_alt,titv_ratio))],columns = ['aa_ref','aa_alt','accessibility'])
#                 self.mutsum_df = self.mutsum_df.append(cur_df)
#         self.mutsum_df.to_csv(self.db_path +'mutsum/' + 'mutsum.csv',index = False)

        self.codon_transition_matrix = pd.read_csv(self.db_path + 'mutsum/' + 'codon_transition_matrix.csv', index_col=0)          
        # for species in self.dict_codon_freq_all:
        for species in self.mutsum_species:
            cur_codon_usage = [] 
            cur_codons = []
            for codon in self.codon_transition_matrix.columns:
                cur_codon_usage.append(self.dict_codon_freq_all[species][codon])
                cur_codons.append(codon)
            pass   
            start_codon_usage = np.matrix(cur_codon_usage)
            end_condon_usage = np.matrix(cur_codon_usage)
            start_codon_usage_diag = np.diag(cur_codon_usage)
            variation_codon_matrix = np.zeros(self.codon_transition_matrix.shape)
            for i in range(generations):
                cur_variation_codon_matrix = start_codon_usage_diag * np.matrix(self.codon_transition_matrix) * (1.1e-08)
                codon_ref_from = cur_variation_codon_matrix.sum(axis=1).reshape(1, 64)
                codon_alt_to = cur_variation_codon_matrix.sum(axis=0)
                end_condon_usage = end_condon_usage - codon_ref_from + codon_alt_to
                start_codon_usage_diag = np.diag(np.array(end_condon_usage).squeeze())
                variation_codon_matrix += cur_variation_codon_matrix
            
            variation_codon_matrix_df = pd.DataFrame(variation_codon_matrix, index=cur_codons, columns=cur_codons)
            variation_codon_matrix_df['codon_ref'] = variation_codon_matrix_df.index
            variation_codon_matrix_df = pd.melt(variation_codon_matrix_df, 'codon_ref')
            variation_codon_matrix_df.columns = ['codon_ref', 'codon_alt', 'variation']
            variation_codon_matrix_df['variation'] = variation_codon_matrix_df['variation'] / variation_codon_matrix_df['variation'].sum()
            variation_codon_matrix_df['aa_ref'] = variation_codon_matrix_df['codon_ref'].apply(lambda x: self.dict_codon[x])
            variation_codon_matrix_df['aa_alt'] = variation_codon_matrix_df['codon_alt'].apply(lambda x: self.dict_codon[x])
            variation_aa_matrix_df = variation_codon_matrix_df.groupby(['aa_ref', 'aa_alt'])['variation'].sum().reset_index()    
            variation_codon_matrix_df['variation'] = np.log2(variation_codon_matrix_df['variation'])
            variation_aa_matrix_df['variation'] = np.log2(variation_aa_matrix_df['variation'])
            variation_codon_matrix_df.to_csv(self.db_path + 'mutsum/' + 'variation_codon_matrix_' + species + '.csv', index=True)
            variation_aa_matrix_df.to_csv(self.db_path + 'mutsum/' + 'mutsum_' + species + '.csv', index=True)
    
    def codon_to_aa(self, x):
        return self.dict_codon[x]
    
    def get_mutsums(self):
        self.dict_mutsums = {}        
        for species in self.mutsum_species:
            self.dict_mutsums[species] = pd.read_csv(self.db_path + 'mutsum/' + 'mutsum_' + species + '.csv')
            
    def create_mafsums(self, maf_cutoff=1): 
        gnomad_missense_aa_freqflipped = pd.read_csv(self.db_path + 'gnomad/gnomad_output_snp_aa_freqflipped.txt', sep='\t')
        gnomad_missense_aa_freqflipped = gnomad_missense_aa_freqflipped.loc[(gnomad_missense_aa_freqflipped.aa_ref != 'U') & (gnomad_missense_aa_freqflipped.gnomad_af < maf_cutoff), :]
        self.create_aasum(self.mafsum_centralities, self.mafsum_properties, gnomad_missense_aa_freqflipped, self.mafsum_scores, 'mafsum_', 'mafsum')  
              
    def create_mafsums_blosum(self, maf_cutoff=1):
        gnomad_missense_aa = pd.read_csv(self.db_path + 'gnomad/gnomad_output_snp_aa.txt', sep='\t')
        gnomad_missense_aa = gnomad_missense_aa.loc[(gnomad_missense_aa.aa_ref != 'U') & (gnomad_missense_aa.gnomad_af < maf_cutoff), :]
        gnomad_missense_aa = gnomad_missense_aa.loc[gnomad_missense_aa.aa_ref != gnomad_missense_aa.aa_alt, :]    
        gnomad_alt_freq = gnomad_missense_aa[['uniprot_id', 'aa_ref', 'aa_pos', 'aa_alt', 'gnomad_af']]
        
        gnomad_ref_freq = gnomad_missense_aa.groupby(['aa_ref', 'aa_pos', 'uniprot_id'])['gnomad_af'].sum().reset_index()
        gnomad_ref_freq['gnomad_af'] = 1 - gnomad_ref_freq['gnomad_af']
        gnomad_ref_freq['aa_alt'] = gnomad_ref_freq['aa_ref']
        
        gnomad_all_freq = pd.concat([gnomad_alt_freq, gnomad_ref_freq]).reset_index()
        gnomad_all_freq.drop(['index'], axis=1, inplace=True)

        gnomad_all_freq['new_index'] = gnomad_all_freq['uniprot_id'] + '_' + gnomad_all_freq['aa_ref'] + '_' + gnomad_all_freq['aa_pos'].astype(str) 
        gnomad_all_freq_pivot = gnomad_all_freq.pivot(index='new_index', columns='aa_alt', values='gnomad_af').reset_index()
        gnomad_all_freq_pivot.fillna(0, inplace=True)
        gnomad_all_freq_pivot['U'] = 0
        gnomad_all_freq_pivot['_'] = 0
        
        self.lst_aa_reverse = list(reversed(np.array(self.lst_aa)))
        empty_matrix = pd.DataFrame(np.zeros([len(self.lst_aa), len(self.lst_aa)]), columns=self.lst_aa, index=self.lst_aa)    
        
        observation_matrix_df = empty_matrix.copy()
        expectation_vector_df = pd.DataFrame(np.zeros([1, len(self.lst_aa)]), columns=self.lst_aa)
        sample_size = 10e+06
        for aa1 in self.lst_aa:
            for aa2 in self.lst_aa:
                if observation_matrix_df.loc[aa1, aa2] == 0:
                    if aa1 == aa2:                        
                        counts = ((sample_size ** 2) * np.dot(np.array(gnomad_all_freq_pivot[aa1]), np.array(gnomad_all_freq_pivot[aa1])) - sample_size * gnomad_all_freq_pivot[aa1].sum()) / 2
                        observation_matrix_df.loc[aa1, aa2] = counts
                        expectation_vector_df[aa1] += counts
                    if aa1 != aa2:
                        counts = (sample_size ** 2) * np.dot(np.array(gnomad_all_freq_pivot[aa1]), np.array(gnomad_all_freq_pivot[aa2]))
                        observation_matrix_df.loc[aa1, aa2] = counts
                        observation_matrix_df.loc[aa2, aa1] = counts
                        expectation_vector_df[aa1] += counts / 2
                        expectation_vector_df[aa2] += counts / 2
                        
        observation_matrix = np.matrix(observation_matrix_df)
        total_pairs_count = (observation_matrix.sum() + np.diag(observation_matrix).sum()) / 2
        expectation_matrix = np.array(expectation_vector_df).transpose() * np.array(expectation_vector_df)
        expectation_matrix_df = pd.DataFrame(expectation_matrix, columns=self.lst_aa, index=self.lst_aa)
        mafsum_matrix = np.log2((observation_matrix * total_pairs_count) / expectation_matrix)
        mafsum_matrix_df = pd.DataFrame(mafsum_matrix, columns=self.lst_aa, index=self.lst_aa)
        mafsum_matrix_df.to_csv(self.db_path + 'mafsum/mafsum_' + str(maf_cutoff) + '.csv')
        mafsum_matrix_df['aa'] = mafsum_matrix_df.index
        mafsum_matrix_df_melt = pd.melt(mafsum_matrix_df, id_vars=['aa'])
        mafsum_matrix_df_melt.columns = ['aa_ref', 'aa_alt', 'mafsum_fitness']
        mafsum_matrix_df_melt.to_csv(self.db_path + 'mafsum/mafsum_melt_' + str(maf_cutoff) + '.csv')
        # take the accessibility into account
#         gnomad_missense_aa_freqflipped = pd.merge(gnomad_missense_aa_freqflipped,self.mutsum_df)
#         for mafsum_score in self.mafsum_scores:
#             gnomad_missense_aa_freqflipped[mafsum_score] = gnomad_missense_aa_freqflipped[mafsum_score]-gnomad_missense_aa_freqflipped['accessibility']
        # self.create_aasum(self.aasum_centralities,self.mafsum_properties,gnomad_missense_aa,self.mafsum_scores,'mafsum_','mafsum')   
        
    def get_selsum(self, generations, cutoff):
        self.dict_selsums = {}
        for species in self.mutsum_species:        
            selsum_df = pd.merge(self.dict_mafsums['gnomad_af_normalization'], self.dict_mutsums[species], how='left')
            selsum_df['selsum_fitness'] = selsum_df['mafsum_gnomad_af_normalization'] - selsum_df['variation']
            selsum_df = selsum_df[['aa_ref', 'aa_alt', 'selsum_fitness']]
            self.dict_selsums[species] = selsum_df
            
    def create_clinvarsum(self):
        clinvar_df = pd.read_csv(self.db_path + 'clinvar/csv/clinvar_final.csv')
        clinvar_df = clinvar_df[['aa_ref', 'aa_alt', 'label']]
        self.create_aasum(self.patsum_centralities, [], clinvar_df, ['label'], 'clinvarsum_', 'clinvarsum')
        
    def get_clinvarsum(self):
        try:
            self.clinvarsum_df = pd.read_csv(self.db_path + 'clinvarsum/' + 'clinvarsum_label_mean.csv')
        except:
            self.create_clinvarsum()
        total_count = self.clinvarsum_df['clinvarsum_label_mean_count'].sum()
        self.clinvarsum_df['positive_count'] = self.clinvarsum_df['clinvarsum_label_mean_count'] * self.clinvarsum_df['clinvarsum_label_mean']
        positive_count = self.clinvarsum_df['positive_count'].sum()
        prior = positive_count / total_count
        self.clinvarsum_df['clinvarsum_label_mean'] = -np.log2(self.clinvarsum_df['clinvarsum_label_mean'] / prior)
#         self.clinvarsum_df['clinvarsum_label_mean'] = 1-self.clinvarsum_df['clinvarsum_label_mean']      
    
    def create_humsavarsum(self):
        humsavar_df = pd.read_csv(self.db_path + 'humsavar/humsavar_final.csv')
        humsavar_df = humsavar_df[['aa_ref', 'aa_alt', 'label']]
        self.create_aasum(self.patsum_centralities, [], humsavar_df, ['label'], 'humsavarsum_', 'humsavarsum', quality_cutoff=None)
        
    def get_humsavarsum(self):
        try:
            self.humsavarsum_df = pd.read_csv(self.db_path + 'humsavarsum/' + 'humsavarsum_label_mean.csv')
        except:
            self.create_humsavarsum()
        total_count = self.humsavarsum_df['humsavarsum_label_mean_count'].sum()
        self.humsavarsum_df['positive_count'] = self.humsavarsum_df['humsavarsum_label_mean_count'] * self.humsavarsum_df['humsavarsum_label_mean']
        positive_count = self.humsavarsum_df['positive_count'].sum()
        prior = positive_count / total_count
        self.humsavarsum_df['humsavarsum_label_mean'] = -np.log2(self.humsavarsum_df['humsavarsum_label_mean'] / prior)
        # self.humsavarsum_df['humsavarsum_label_mean'] = 1-self.humsavarsum_df['humsavarsum_label_mean']
    
    def create_aasum(self, sum_name, centralities, properties, value_df, value_score_names, aasum_prefix, aasum_folder, quality_cutoff=None, weightedby_columns=[''], weightedby_columns_inverse=[0]):
        self.dict_sums[sum_name] = {}

        def cal_weighted_average(value_df, groupby, value_score, weighted_by, inverse):

            def single_group_by(x):
                groupby_cols = list(x[0])
                groupby_df = x[1]
                groupby_df = groupby_df.loc[groupby_df[value_score].notnull() & groupby_df[weighted_by].notnull()]
                
                weighted_by_cols = groupby_df[weighted_by]
                value_cols = groupby_df[value_score]
                if inverse == 1:
                    weighted_by_cols = 1 / weighted_by_cols
                weighted_by_cols = weighted_by_cols / np.sum(weighted_by_cols)
                weighted_mean = np.dot(value_cols, weighted_by_cols)
                cur_row = groupby_cols + [weighted_mean]
                return cur_row            
            
            if groupby != None:
                group_obj = value_df.groupby(groupby)   
                group_list = list(group_obj)     
                return_list = [single_group_by(x) for x in group_list]       
                return_obj = pd.DataFrame(return_list, columns=groupby + [value_score])
            else:
                weighted_by_cols = value_df[weighted_by]
                value_cols = value_df[value_score]
                if inverse == 1:
                    weighted_by_cols = 1 / weighted_by_cols
                weighted_by_cols = weighted_by_cols / np.sum(weighted_by_cols)
                weighted_mean = np.dot(value_cols, weighted_by_cols)
                return_obj = weighted_mean
            return (return_obj)
        
        def create_aasum_sub(centrality, value_df, value_score, aasum_groupby, aasum_name, aasum_folder, weighted_by, weighted_by_inverse):                         
            if centrality == 'mean':    
                if weighted_by == '':
                    aasum_value = value_df.groupby(aasum_groupby)[value_score].mean().reset_index()
                else:
                    aasum_value = cal_weighted_average(value_df, aasum_groupby, value_score, weighted_by, weighted_by_inverse)           
            if centrality == 'median':
                aasum_value = value_df.groupby(aasum_groupby)[value_score].median().reset_index()
            if centrality == 'normalization':                    
                aasum_value = value_df.groupby(aasum_groupby)[value_score].sum().reset_index()
                total_value = aasum_value[value_score].sum()                                        
                aasum_value[value_score] = aasum_value[value_score] / total_value
            if centrality == 'logodds':            
                if weighted_by == '':
                    aasum_value = value_df.groupby(aasum_groupby)[value_score].mean().reset_index()
                    total_mean_value = aasum_value[value_score].mean()                                        
                    aasum_value[value_score] = -np.log2((aasum_value[value_score] / total_mean_value))
                    # aasum_value[value_score] = aasum_value[value_score] / total_mean_value
                else:
                    aasum_value = cal_weighted_average(value_df, aasum_groupby, value_score, weighted_by, weighted_by_inverse)
                    total_mean_value = cal_weighted_average[value_df, None, value_score, weighted_by, weighted_by_inverse]                                        
                    aasum_value[value_score] = -np.log2((aasum_value[value_score] / total_mean_value))                    
                    # aasum_value[value_score] = aasum_value[value_score] / total_mean_value
                
            aasum_value.rename(columns={value_score: aasum_name}, inplace=True)
            aasum_std = value_df.groupby(aasum_groupby)[value_score].std().reset_index()
            aasum_std.rename(columns={value_score: aasum_name + '_std'}, inplace=True)
            aasum_count = value_df.groupby(aasum_groupby)[value_score].count().reset_index()
            aasum_count.rename(columns={value_score: aasum_name + '_count'}, inplace=True)
            
            aasum = pd.merge(aasum_value, aasum_std, how='left')
            aasum = pd.merge(aasum, aasum_count, how='left')
            aasum[aasum_name + '_ste'] = aasum[aasum_name + '_std'] / np.sqrt(aasum[aasum_name + '_count'])            
            aasum.to_csv(aasum_folder + '/' + aasum_name + '.csv', index=False)
            return(aasum)
                    
        # print ('create ' + aasum_prefix + '......')
        aasum_groupby = ['aa_ref', 'aa_alt']
        for value_score in value_score_names:
            cur_value_df = value_df.copy()
            if quality_cutoff != None:
                cur_value_df = cur_value_df.loc[cur_value_df[value_score].notnull() & (cur_value_df['quality_score'] > quality_cutoff), :]
            else:
                cur_value_df = cur_value_df.loc[cur_value_df[value_score].notnull() , :]
            for centrality in centralities:
                for i in range(len(weightedby_columns)):
                    weighted_by = weightedby_columns[i]
                    weighted_by_inverse = weightedby_columns_inverse[i]
                    if weighted_by == '':  
                        aasum_name = aasum_prefix + value_score + '_' + centrality
                    else:
                        aasum_name = aasum_prefix + value_score + '_' + centrality + '_' + weighted_by 
                    self.dict_sums[sum_name][aasum_name] = create_aasum_sub(centrality, cur_value_df, value_score, aasum_groupby, aasum_name, aasum_folder, weighted_by, weighted_by_inverse)
                    for property in properties:
                        lst_property = property.split(',')
                        property_aasum_groupby = aasum_groupby + lst_property
                        property_aasum_name = aasum_name + '_' + property.replace(',', '_')                         
                        # create funsum for each property value
                        k = 0;
                        for property_groupby in list(cur_value_df.groupby(lst_property)):
                            k += 1;
                            cur_property_value = str(property_groupby[0])
                            cur_property_value_df = property_groupby[1]
                            cur_property_aasum_name = property_aasum_name + '_' + str(cur_property_value)   
                            cur_property_aasum_df = create_aasum_sub(centrality, cur_property_value_df, value_score, property_aasum_groupby, cur_property_aasum_name, aasum_folder, weighted_by, weighted_by_inverse)
                            # self.dict_sums[sum_name][cur_property_aasum_name] = cur_property_aasum_df
                            
                            cur_property_aasum_df.columns = aasum_groupby + lst_property + [property_aasum_name, property_aasum_name + '_std', property_aasum_name + '_count', property_aasum_name + '_ste']
                            
                            if k == 1:
                                all_property_aasum_df = cur_property_aasum_df
                            else:
                                all_property_aasum_df = pd.concat([all_property_aasum_df, cur_property_aasum_df], axis=0)
                        all_property_aasum_df.to_csv(aasum_folder + '/' + property_aasum_name + '.csv', index=False)    
                        self.dict_sums[sum_name][property_aasum_name] = all_property_aasum_df    
    
    def create_funsums(self, quality_cutoff):
        sum_name = 'funsum'
        aasum_prefix = 'funsum_'
        aasum_folder = self.db_path + 'funsum/'
        value_df = pd.read_csv(self.funsum_dmsfiles)
        value_df = value_df.loc[(value_df['p_vid'].isin(self.funsum_genes)) & (value_df['annotation'] == 'NONSYN'), :]  
        self.create_aasum(sum_name, self.funsum_centralities, self.funsum_properties, value_df, self.funsum_scores, aasum_prefix, aasum_folder, quality_cutoff, self.funsum_weightedby_columns, self.funsum_weightedby_columns_inverse)
           
    def prepare_humsavar_data(self):
        # pre_proccessing
        humsavar_snv = pd.read_table(self.db_path + 'humsavar/humsavar1.txt')
        humsavar_snv.columns = ['info']
        humsavar_snv['p_vid'] = humsavar_snv['info'].apply(lambda x: x[10:16])
        humsavar_snv['aa_ref'] = humsavar_snv['info'].apply(lambda x: self.get_aa_ref_humsavar(x))
        humsavar_snv['aa_pos'] = humsavar_snv['info'].apply(lambda x: self.get_aa_pos_humsavar(x))
        humsavar_snv['aa_pos'] = humsavar_snv['aa_pos'].astype(int)
        humsavar_snv['aa_alt'] = humsavar_snv['info'].apply(lambda x: self.get_aa_alt_humsavar(x))
        humsavar_snv['clin_sig'] = humsavar_snv['info'].apply(lambda x: x[48:61].strip())
        humsavar_snv['chr'] = -1  # for empty merge purpose

        # remove polyphen training data
        humsavar_snv = pd.merge(humsavar_snv, self.polyphen_train, how='left')
        humsavar_snv = humsavar_snv.loc[humsavar_snv['polyphen_train'].isnull(), :]
        
        # variants process
        self.humsavar_processed_df = self.variants_process('humsavar', humsavar_snv, self.uniprot_seqdict, self.flanking_k, 'uniprot_id')
        self.humsavar_processed_df.to_csv(self.db_path + 'humsavar/humsavar_processed_df.csv', index=False)
        
        # self.humsavar_processed_df = pd.read_csv(self.db_path + 'humsavar/humsavar_processed_df.csv',dtype={"chr": str})
        self.humsavar_final = self.humsavar_processed_df.copy()
        self.humsavar_final['chr'] = self.humsavar_final['chr'].astype(str)
        
        remove_features = [s for s in self.humsavar_final.columns.get_values() if "Unnamed" in s]
        self.humsavar_final['label'] = -1
        self.humsavar_final.loc[(self.humsavar_final['clin_sig'] == 'Disease'), 'label'] = 1
        self.humsavar_final.loc[(self.humsavar_final['clin_sig'] == 'Polymorphism'), 'label'] = 0
        self.humsavar_final = self.humsavar_final.loc[self.humsavar_final.label != -1, list(set(self.humsavar_final.columns.get_values()) - set(remove_features))]
        self.humsavar_final.drop_duplicates(inplace=True)
        self.humsavar_final.to_csv(self.db_path + 'humsavar/humsavar_final.csv', index=False)
         
        self.humsavar_final_nolabel = self.humsavar_final.loc[self.humsavar_final.label == -1, list(set(self.humsavar_final.columns.get_values()) - set(remove_features))]
        self.humsavar_final_nolabel.drop_duplicates(inplace=True)
        self.humsavar_final_nolabel.to_csv(self.db_path + 'humsavar/humsavar_final_nolabel.csv', index=False)
        
        # create the data set one mutation per protein 
        # self.humsavar_final = pd.read_csv(self.db_path + 'humsavar/humsavar_final.csv')
        unique_proteins = self.humsavar_final.p_vid.unique()
        all_proteins = self.humsavar_final.p_vid 
        single_p_idxs = []
        count = 0
        for p in unique_proteins:
            count += 1
            # print (count)
            p_idx = random.choice(np.where(np.array(all_proteins == p))[0]) 
            single_p_idxs.append(p_idx)
        self.humsavar_final_singles = self.humsavar_final.loc[single_p_idxs, :]
        self.humsavar_final_singles.to_csv(self.db_path + 'humsavar/humsavar_final_singles.csv', index=False)

    def prepare_clinvar_data_backup(self):
        #************************************************************************************************************************************************************************
        ##  Important Notes  ##
        #1) The CLinVar data is downloaded from 3. variant_summary.txt.gz (ftp://ftp.ncbi.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz) 
        #2) Each ClinVar missense record asscoiates with a HGVS expression explicitely provide the RefSeq cDNA either from a submittre or automaticly determined using reference standard transcript. check https://www.ncbi.nlm.nih.gov/clinvar/docs/hgvs_types/
        #************************************************************************************************************************************************************************
        
        self.gnomad_nt = pd.read_csv(self.db_path+'gnomad/gnomad_output_snp_af.txt',sep = '\t',header = None)
        self.gnomad_nt.columns = ['chr','nt_pos','rs_id','nt_ref','nt_alt','quality_score','filter','gnomad_ac','gnomad_an','gnomad_af','gnomad_esa_af','gnoamd_gc','gnomad_gc_homo_ref','gnomad_gc_hetero','gnomad_gc_homo_alt']
        self.gnomad_nt['chr'] = self.gnomad_nt['chr'].astype(str) 
#         self.gnomad_nt_aa = pd.read_csv(self.db_path+'gnomad/gnomad_output_snp_aa.txt',sep = '\t')
#         self.gnomad_nt_aa['chr'] = self.gnomad_nt_aa['chr'].astype(str) 
#         #************************************************************************************
#         # clinvar pathogenic data set 
#         #************************************************************************************        
        self.clinvar_raw_file = 'clinvar/org/variant_summary.txt'        
        clinvar_raw = pd.read_table(self.db_path + self.clinvar_raw_file, sep='\t')
        print ('Clinvar all records : ' + str(clinvar_raw.shape[0]))
 
        clinvar_snv = clinvar_raw.loc[(clinvar_raw['Type'] == 'single nucleotide variant') & (clinvar_raw['Assembly'] == self.assembly) & (clinvar_raw['Chromosome'].isin(self.lst_chr)), ['GeneSymbol','HGNC_ID','Chromosome', 'Start', 'ReferenceAllele', 'AlternateAllele', 'ReviewStatus','LastEvaluated', 'ClinicalSignificance', 'Name', 'NumberSubmitters', 'RS# (dbSNP)', 'PhenotypeIDS', 'PhenotypeList']]
        clinvar_snv.columns = ['gene_symbol','hgnc_id','chr', 'nt_pos', 'nt_ref', 'nt_alt', 'review_status','evaluate_time', 'clin_sig', 'hgvs', 'ev_num', 'rs', 'phenotype_id', 'phenotype_name']
        print ('Clinvar snv records : ' + str(clinvar_snv.shape[0]))        
                 
        clinvar_snv['review_star'] = 0
        clinvar_snv.loc[clinvar_snv['review_status'] == 'practice guideline','review_star'] = 4
        clinvar_snv.loc[clinvar_snv['review_status'] == 'reviewed by expert panel','review_star'] = 3
        clinvar_snv.loc[clinvar_snv['review_status'] == 'criteria provided, multiple submitters, no conflicts','review_star'] = 2
        clinvar_snv.loc[clinvar_snv['review_status'] == 'criteria provided, single submitter','review_star'] = 1        
                   
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
 
        # process the relevant columns
        clinvar_snv['chr'] = clinvar_snv['chr'].str.strip()
        clinvar_snv = clinvar_snv.loc[clinvar_snv['hgvs'].notnull(), :]
        
#         clinvar_snv['aa_ref'] = clinvar_snv['hgvs'].apply(lambda x: self.get_aa_ref_clinvar(x))
#         clinvar_snv['aa_pos'] = clinvar_snv['hgvs'].apply(lambda x: self.get_aa_pos_clinvar(x))
#         clinvar_snv['aa_pos'] = clinvar_snv['aa_pos'].astype(int)
#         clinvar_snv['aa_alt'] = clinvar_snv['hgvs'].apply(lambda x: self.get_aa_alt_clinvar(x))
#         clinvar_snv['t_vid'] = clinvar_snv['hgvs'].apply(lambda x: x.split(':')[0].split('(')[0])
#         clinvar_snv['t_id'] = clinvar_snv['t_vid'].apply(lambda x: x.split('.')[0])
                            
        # remove non-coding region, synonymous, invalid records 
        clinvar_snv = clinvar_snv.loc[(clinvar_snv['aa_pos'] != -1) & (clinvar_snv['aa_ref'] != '?') & (clinvar_snv['aa_alt'] != '?') & (clinvar_snv['aa_alt'] != '_') & (clinvar_snv['aa_alt'] != '*') & (clinvar_snv['aa_ref'] != clinvar_snv['aa_alt']), :]
        print ('Clinvar exome missense snv records : ' + str(clinvar_snv.shape[0]))                
        # 0 reveiw star, 0 clinical significance value 
        clinvar_snv = clinvar_snv.loc[(clinvar_snv['review_star'] > 0 ) & (clinvar_snv['clinsig_level'] >0) ,:]
        print ('Clinvar review_star > 0 , clinsig_level > 0 records : ' + str(clinvar_snv.shape[0]))
#                  
#         # Map NM_id to NP_id
#         clinvar_snv = pd.merge(clinvar_snv, self.refseq_vids, how='left')
         
        # set data_source and clinvar_gene                
#         clinvar_snv['clinvar_gene'] = 1
        clinvar_snv['data_source'] = 'clinvar'
         
        # merge with Gnomad
#         self.gnomad_nt['p_vid'] = self.gnomad_nt['uniprot_id']     
        clinvar_snv = pd.merge(clinvar_snv,self.gnomad_nt[['chr','nt_pos','nt_ref','nt_alt','gnomad_af','gnomad_gc_homo_alt']],how = 'left')
#         clinvar_snv.to_csv(self.db_path + 'clinvar/csv/clinvar_snv.csv', index=False)


#         #remove ids without refseq protein id or not sequence information available
#         clinvar_snv = clinvar_snv.loc[clinvar_snv['refseq_vid'].notnull()  & (clinvar_snv['seq_available']  == 1) ,:]
#         print ('Clinvar valid refseq protein id, refseq protein sequence available  : ' + str(clinvar_snv.shape[0]))
#          
#         #Still decide to use Uniport as Gnomad only provide ENSP or Uniprot
#         #Make sure all uniport id is working and sequence is matching 
#         clinvar_snv = clinvar_snv.loc[clinvar_snv['uniprot_id'].notnull()  & (clinvar_snv['seq_match']  == 1) ,:]
#         print ('Clinvar uniport sequence match  : ' + str(clinvar_snv.shape[0]))
#                  
#         clinvar_snv = clinvar_snv.loc[clinvar_snv['isoform'] == 0,:]
#         print ('Clinvar uniport non-isoform only  : ' + str(clinvar_snv.shape[0]))
                                       
        # save clinvar snv data 
        clinvar_snv['p_vid'] = clinvar_snv['uniprot_id']       
        clinvar_ids = clinvar_snv['p_vid'].unique()                   
        clinvar_snv.to_csv(self.db_path + 'clinvar/csv/clinvar_snv.csv', index=False)
 
        # select columns to form srv records        
        clinvar_srv = clinvar_snv[['chr', 'nt_pos', 'nt_ref', 'nt_alt','p_vid','uniprot_id','aa_pos', 'aa_ref', 'aa_alt','gnomad_gc_homo_alt','gnomad_af','clin_sig','clinsig_level','review_star','evaluate_time','data_source','label','clinvar_gene']]
        clinvar_srv.to_csv(self.db_path + 'clinvar/csv/clinvar_srv.csv', index=False)
        
        

        #************************************************************************************
        # Gnomad benign data set 
        #************************************************************************************
        gnomad_nt_low_af_idx = self.gnomad_nt.loc[(self.gnomad_nt_aa['gnomad_gc_homo_alt'] > 0) & (self.gnomad_nt_aa['gnomad_af'] < 0.001), :].index                                                                                                                                                                                                                                                          
        gnomad_srv = self.gnomad_nt.loc[gnomad_nt_low_af_idx, ['chr', 'nt_pos', 'nt_ref', 'nt_alt','uniprot_id', 'aa_pos', 'aa_ref', 'aa_alt','gnomad_gc_homo_alt','gnomad_af']]
        gnomad_srv['clin_sig'] = 'Benign'
        gnomad_srv['clinsig_level'] = 1
        gnomad_srv['review_star'] = 5  
        gnomad_srv['evaluate_time'] = '1900-01-01'
        gnomad_srv['data_source'] = 'gnomad'
        gnomad_srv['label'] = 0
        gnomad_srv['clinvar_gene'] = 0
        gnomad_srv.loc[gnomad_srv['uniprot_id'].isin(clinvar_ids), 'clinvar_gene'] = 1
        gnomad_srv = gnomad_srv.loc[gnomad_srv['uniprot_id'].notnull() & gnomad_srv['clinvar_gene'] == 1,:]
        gnomad_srv['p_vid'] = gnomad_srv['uniprot_id']       
        gnomad_srv.to_csv(self.db_path + 'clinvar/csv/gnomad_srv.csv', index=False)









#         gnomad_nt_low_af_idx = self.gnomad_nt_aa.loc[(self.gnomad_nt_aa['gnomad_gc_homo_alt'] > 0) & (self.gnomad_nt_aa['gnomad_af'] < 0.001), :].index                                                                                                                                                                                                                                                          
#         gnomad_srv = self.gnomad_nt_aa.loc[gnomad_nt_low_af_idx, ['chr', 'nt_pos', 'nt_ref', 'nt_alt','uniprot_id', 'aa_pos', 'aa_ref', 'aa_alt','gnomad_gc_homo_alt','gnomad_af']]
#         gnomad_srv['clin_sig'] = 'Benign'
#         gnomad_srv['clinsig_level'] = 1
#         gnomad_srv['review_star'] = 5  
#         gnomad_srv['evaluate_time'] = '1900-01-01'
#         gnomad_srv['data_source'] = 'gnomad'
#         gnomad_srv['label'] = 0
#         gnomad_srv['clinvar_gene'] = 0
#         gnomad_srv.loc[gnomad_srv['uniprot_id'].isin(clinvar_ids), 'clinvar_gene'] = 1
#         gnomad_srv = gnomad_srv.loc[gnomad_srv['uniprot_id'].notnull() & gnomad_srv['clinvar_gene'] == 1,:]
#         gnomad_srv['p_vid'] = gnomad_srv['uniprot_id']       
#         gnomad_srv.to_csv(self.db_path + 'clinvar/csv/gnomad_srv.csv', index=False)
        
        print ('Gnomad Homozoytes individual > 0, MAF < 0.001 , Uniprot ID Available, Clinvar Gene Only: ' + str(gnomad_srv.shape[0]))
        
        #************************************************************************************
        # Merge pathogenic and benign set 
        #************************************************************************************     
        clinvar_plus_gnomad_srv = pd.concat([clinvar_srv, gnomad_srv])
            
        #mark the polyphen training record
        self.polyphen_train = self.get_polyphen_train_data()
        clinvar_plus_gnomad_srv = pd.merge(clinvar_plus_gnomad_srv, self.polyphen_train, how='left')
        
#         #merge with hngc    
#         hngc = pd.read_csv(self.db_path + 'hgnc/org/hgnc_complete_set.txt',sep ='\t')
#         hngc_map = hngc.loc[hngc['refseq_accession'].notnull() & hngc['uniprot_ids'].notnull(),['hgnc_id','refseq_accession','uniprot_ids']]
#         hngc_map.columns = ['hgnc_id','hgnc_tid','hgnc_uniprot_id']
#         clinvar_plus_gnomad_srv = pd.merge(clinvar_plus_gnomad_srv, hngc_map, how='left')  
        
        #save merged records and output for score 
        clinvar_plus_gnomad_srv.drop_duplicates(inplace=True)
        clinvar_plus_gnomad_srv.reset_index(drop=True, inplace=True)    
        clinvar_plus_gnomad_srv['aa_pos'] = clinvar_plus_gnomad_srv['aa_pos'].astype(int)
        clinvar_plus_gnomad_srv['chr'] = clinvar_plus_gnomad_srv['chr'].astype(str)   
        
        
        
        clinvar_plus_gnomad_srv_nt_counts = clinvar_plus_gnomad_srv.groupby(['chr','nt_pos','nt_ref','nt_alt']).size()
        clinvar_plus_gnomad_srv_nt_counts = clinvar_plus_gnomad_srv_nt_counts.reset_index()
        clinvar_plus_gnomad_srv_nt_counts.columns = ['chr','nt_pos','nt_ref','nt_alt','nt_counts']
        clinvar_plus_gnomad_srv = pd.merge(clinvar_plus_gnomad_srv,clinvar_plus_gnomad_srv_nt_counts,how = "left")
        
        clinvar_plus_gnomad_srv_aa_counts = clinvar_plus_gnomad_srv.groupby(['chr','nt_pos','nt_ref','nt_alt','aa_pos','aa_ref','aa_alt']).size()
        clinvar_plus_gnomad_srv_aa_counts = clinvar_plus_gnomad_srv_aa_counts.reset_index()
        clinvar_plus_gnomad_srv_aa_counts.columns = ['chr','nt_pos','nt_ref','nt_alt','aa_pos','aa_ref','aa_alt','aa_counts']
        clinvar_plus_gnomad_srv = pd.merge(clinvar_plus_gnomad_srv,clinvar_plus_gnomad_srv_aa_counts,how = "left")
 
        clinvar_plus_gnomad_srv.to_csv(self.db_path + 'clinvar/csv/clinvar_plus_gnomad_srv.csv',index = False)
        

        clinvar_plus_gnomad_srv_for_score = clinvar_plus_gnomad_srv[['p_vid', 'aa_pos', 'aa_ref', 'aa_alt']]
        clinvar_plus_gnomad_srv_for_score.drop_duplicates(inplace = True)        
        clinvar_plus_gnomad_srv_for_score.to_csv(self.db_path + 'clinvar/csv/clinvar_plus_gnomad_srv_for_score.txt', sep='\t', index=False, header=None)
        
        #for polyphen is ok to submit the for_score file but it is too big for provean
        indices = np.arange(0,clinvar_plus_gnomad_srv_for_score.shape[0], int(clinvar_plus_gnomad_srv_for_score.shape[0]/3))
        for i in range(len(indices)-1):
                clinvar_plus_gnomad_srv_for_score.loc[indices[i]:indices[i+1]-1,:].to_csv(self.db_path + 'clinvar/csv/clinvar_plus_gnomad_srv_for_score_' + str(i) + '.txt', sep='\t', index=False, header=None)
                if i == 2:
                    clinvar_plus_gnomad_srv_for_score.loc[indices[i]:clinvar_plus_gnomad_srv_for_score.shape[0]-1,['p_vid', 'aa_pos', 'aa_ref', 'aa_alt']].to_csv(self.db_path + 'clinvar/csv/clinvar_plus_gnomad_srv_for_score_' + str(i) + '.txt', sep='\t', index=False, header=None)
        
        #some other scores you need to submit vcf (CADD, MutationTester)
#         clinvar_plus_gnomad_srv_for_vcf = clinvar_plus_gnomad_srv.copy()
#         clinvar_plus_gnomad_srv_for_vcf['#CHROM'] = clinvar_plus_gnomad_srv_for_vcf['chr']
#         clinvar_plus_gnomad_srv_for_vcf['POS'] = clinvar_plus_gnomad_srv_for_vcf['nt_pos']
#         clinvar_plus_gnomad_srv_for_vcf['REF'] = clinvar_plus_gnomad_srv_for_vcf['nt_ref']
#         clinvar_plus_gnomad_srv_for_vcf['ALT'] = clinvar_plus_gnomad_srv_for_vcf['nt_alt']
#         clinvar_plus_gnomad_srv_for_vcf['ID'] = '.'
#                 
#         clinvar_plus_gnomad_srv_for_vcf[['#CHROM','POS','ID','REF','ALT'] + list(clinvar_plus_gnomad_srv.columns)].to_csv(self.db_path + 'clinvar/csv/clinvar_plus_gnomad_srv_for_vcf.txt', sep='\t', index=False)
        
#         clinvar_plus_gnomad_srv_for_vcf = clinvar_plus_gnomad_srv[['chr', 'nt_pos', 'nt_ref', 'nt_alt']]
#         clinvar_plus_gnomad_srv_for_vcf.columns = ['#CHROM','POS','REF','ALT']
#         clinvar_plus_gnomad_srv_for_vcf.drop_duplicates(inplace = True)        
#         clinvar_plus_gnomad_srv_for_vcf.to_csv(self.db_path + 'clinvar/csv/clinvar_plus_gnomad_srv_for_vcf.txt', sep='\t', index=False)
                
        print ('Clinvar plus gnoamd [low af + homozygotes available] aa change records : ' + str(clinvar_plus_gnomad_srv.shape[0]) + " saved.")
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
                
#         
#         # Map NM_id to uniport_id using hngc set 
#         clinvar_snv = pd.merge(clinvar_snv, hngc_map, how='left')  
#         # remove records have no protein id (some of refseq_id cannot map to reviewed uniprot id , we removed that as well)
#         clinvar_snv['p_vid'] = clinvar_snv['uniprot_id']
#         clinvar_snv = clinvar_snv.loc[clinvar_snv['p_vid'].notnull(), :]
#         print ('Clinvar exome snv records (reviewed uniport id available) : ' + str(clinvar_snv.shape[0]))
 
#         # Map NM_id to NP_id
#         clinvar_snv = pd.merge(clinvar_snv, self.refseq_vids, how='left')        
  
#         # remove records have no protein id (some of refseq_id cannot map to reviewed uniprot id , we removed that as well)
#         clinvar_snv['p_vid'] = clinvar_snv['uniprot_id']
#         clinvar_snv = clinvar_snv.loc[clinvar_snv['p_vid'].notnull(), :]
#         print ('Clinvar exome snv records (reviewed uniport id available) : ' + str(clinvar_snv.shape[0]))
         
#         clinvar_snv = clinvar_snv.loc[clinvar_snv['isoform'] != 1, :]
#         print ('Clinvar exome snv records (reviewed uniport id available, canonical only) : ' + str(clinvar_snv.shape[0]))
#  
#         #remove the variants that don't have a evaluate time 
#         clinvar_snv = clinvar_snv.loc[clinvar_snv['evaluate_time'].str.contains(','), :]
#         print ('Clinvar exome snv records (reviewed uniport id available, canonical only, valid evaluate time) : ' + str(clinvar_snv.shape[0]))
#          
#         clinvar_snv['evaluate_time'] = pd.to_datetime(clinvar_snv['evaluate_time'])
 
#         #************************************************************************************
#         # remove polyphen training data
#         #************************************************************************************
#         self.polyphen_train = self.get_polyphen_train_data()
#         clinvar_snv = pd.merge(clinvar_snv, self.polyphen_train, how='left')
# #         clinvar_snv = clinvar_snv.loc[clinvar_snv['polyphen_train'].isnull(), :]
         
        #************************************************************************************
        # save data
        #************************************************************************************
#         clinvar_snv.to_csv(self.db_path + 'clinvar/csv/clinvar_snv.csv', index=False)
# #         print ('Clinvar exome snv records (reviewed uniport id available, canonical only, valid evaluate time, polyphen training removed) : ' + str(clinvar_snv.shape[0]))
#          
        # remove the ones that have duplication in the aa change level
#         clinvar_snv = pd.read_csv(self.db_path + 'clinvar/csv/clinvar_snv.csv', dtype={"chr": str})   
#         clinvar_snv['num'] = 1   
#         groupby_df = clinvar_snv.groupby(['p_vid', 'aa_pos', 'aa_ref', 'aa_alt'])['num'].sum()
#         groupby_df = groupby_df.reset_index()
#         groupby_df.columns = ['p_vid', 'aa_pos', 'aa_ref', 'aa_alt', 'sum_num']
#         clinvar_srv = pd.merge(clinvar_snv, groupby_df, how='left')
#         clinvar_srv = clinvar_srv.loc[clinvar_srv['sum_num'] == 1, :]
                         
#         print ('Clinvar aa change records - duplicates removed (reviewed uniport id available, canonical only, valid evaluate time, polyphen training removed) : ' + str(clinvar_srv.shape[0]))
#         clinvar_srv = clinvar_snv
#         clinvar_ids = clinvar_srv['p_vid'].unique()                  
#         clinvar_srv['clinvar_gene'] = 1
#         clinvar_srv['data_source'] = 'clinvar'
#         
# #         self.gnomad_aa['p_vid'] = self.gnomad_aa['uniprot_id']     
# #         clinvar_srv = pd.merge(clinvar_srv,self.gnomad_aa[['p_vid', 'aa_pos', 'aa_ref', 'aa_alt','gnomad_af','gnomad_gc_homo_alt']],how = 'left')
#          
#         self.gnomad_nt['p_vid'] = self.gnomad_nt['uniprot_id']     
#         clinvar_srv = pd.merge(clinvar_srv,self.gnomad_nt[['p_vid', 'aa_pos', 'aa_ref', 'aa_alt','gnomad_af','gnomad_gc_homo_alt']],how = 'left')
#                 
        
#         clinvar_srv.to_csv(self.db_path + 'clinvar/csv/clinvar_srv.csv', index=False)
#         clinvar_srv_for_score = clinvar_srv[['p_vid', 'aa_pos', 'aa_ref', 'aa_alt']]
#         clinvar_srv_for_score = clinvar_srv_for_score.drop_duplicates()
#         clinvar_srv_for_score.to_csv(self.db_path + 'clinvar/csv/clinvar_srv_for_score.txt', sep=' ', index=False, header=None)
#         print ('Clinvar aa change records - saved.')
          

        return (clinvar_plus_gnomad_srv)
        
    def prepare_funregressor_training_data(self):
        funregressor_training = pd.read_csv(self.db_path + 'dms/funregressor_training_from_imputation.csv')        
        funregressor_training = funregressor_training[['p_vid', 'aa_pos', 'aa_ref', 'aa_alt','fitness','fitness_se_reg','fitness_refine','fitness_se_refine','quality_score']]
        
        #Add oosition importance
        pos_importance_groupby = funregressor_training.groupby(['p_vid','aa_pos'] )['fitness'].median().reset_index()
        pos_importance_groupby.columns = ['p_vid','aa_pos','pos_importance']
        funregressor_training = pd.merge(funregressor_training,pos_importance_groupby,how = 'left')
        
        #************************************************************************************
        # save data
        #************************************************************************************                
        funregressor_training.to_csv(self.db_path + 'funregressor/funregressor_training.csv',index = False)
                
        funregressor_training_for_score = funregressor_training[['p_vid', 'aa_pos', 'aa_ref', 'aa_alt']]
        funregressor_training_for_score.drop_duplicates(inplace = True)        
        funregressor_training_for_score.to_csv(self.db_path + 'funregressor/funregressor_training_for_score.txt', sep='\t', index=False, header=None)

        #for polyphen is ok to submit the for_score file but it is too big for provean
        indices = np.arange(0,funregressor_training_for_score.shape[0], int(funregressor_training_for_score.shape[0]/3)-1)
        for i in range(len(indices)-1):
                funregressor_training_for_score.loc[indices[i]:indices[i+1]-1,:].to_csv(self.db_path + 'funregressor/funregressor_training_for_score_' + str(i) + '.txt', sep='\t', index=False, header=None)
                if i == 2:
                    funregressor_training_for_score.loc[indices[i]:funregressor_training_for_score.shape[0]-1,['p_vid', 'aa_pos', 'aa_ref', 'aa_alt']].to_csv(self.db_path + 'funregressor/funregressor_training_for_score_' + str(i) + '.txt', sep='\t', index=False, header=None)

        #************************************************************************************
        # run variant processing
        #************************************************************************************     
        clinvar_plus_gnomad_final_df = self.variants_process('funregressor_training', funregressor_training, self.uniprot_seqdict, self.flanking_k, nt_input = 0, gnomad_available = 0, gnomad_merge_id = 'uniprot_id')                    
        clinvar_plus_gnomad_final_df.to_csv(self.db_path + 'funregressor/funregressor_training_final.csv', index=False)   
        
        
     
    def prepare_funregressor_test_data(self):
        #************************************************************************************************************************************************************************
        ##  Important Notes  ##
        #1) The CLinVar data is downloaded from 3. variant_summary.txt.gz (ftp://ftp.ncbi.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz) 
        #2) Each ClinVar missense record asscoiates with a HGVS expression explicitely provide the RefSeq cDNA either from a submittre or automaticly determined using reference standard transcript. check https://www.ncbi.nlm.nih.gov/clinvar/docs/hgvs_types/
        #************************************************************************************************************************************************************************         
        #load gnomad
        self.gnomad_nt = pd.read_csv(self.db_path+'gnomad/gnomad_output_snp_af.txt',sep = '\t',header = None)       
        self.gnomad_nt.columns = ['chr','nt_pos','rs_id','nt_ref','nt_alt','quality_score','filter','gnomad_ac','gnomad_an','gnomad_af','gnomad_esa_af','gnoamd_gc','gnomad_gc_homo_ref','gnomad_gc_hetero','gnomad_gc_homo_alt']
        self.gnomad_nt = self.gnomad_nt.loc[self.gnomad_nt['gnomad_af'] != '.',:] 
        self.gnomad_nt['chr'] = self.gnomad_nt['chr'].astype(str) 
        self.gnomad_nt['gnomad_af'] = self.gnomad_nt['gnomad_af'].astype(float)
            
        self.gnomad_nt_aa = pd.read_csv(self.db_path+'gnomad/gnomad_output_snp_aa.txt',sep = '\t')
        self.gnomad_nt_aa['chr'] = self.gnomad_nt_aa['chr'].astype(str) 
           
        #************************************************************************************************************************************************************************
        ##  Create the pathogenic postive and benign negative dataset from clinVAR
        #************************************************************************************************************************************************************************
        #load clinvar rawdata
        self.clinvar_raw_file = 'clinvar/org/variant_summary.txt'        
        clinvar_raw = pd.read_table(self.db_path + self.clinvar_raw_file, sep='\t')
        print ('clinVAR all records : ' + str(clinvar_raw.shape[0]))
            
        #create clinvar_snv
        clinvar_snv = clinvar_raw.loc[(clinvar_raw['Type'] == 'single nucleotide variant') & (clinvar_raw['Assembly'] == self.assembly) & (clinvar_raw['Chromosome'].isin(self.lst_chr)), ['GeneSymbol','HGNC_ID','Chromosome', 'Start', 'ReferenceAllele', 'AlternateAllele', 'ReviewStatus','LastEvaluated', 'ClinicalSignificance', 'Name', 'NumberSubmitters', 'RS# (dbSNP)', 'PhenotypeIDS', 'PhenotypeList','VariationID']]
        clinvar_snv.columns = ['gene_symbol','hgnc_id','chr', 'nt_pos', 'nt_ref', 'nt_alt', 'review_status','evaluate_time', 'clin_sig', 'hgvs', 'ev_num', 'rs', 'phenotype_id', 'phenotype_name','clinvar_id']
        clinvar_snv['chr'] = clinvar_snv['chr'].str.strip()
        clinvar_snv = clinvar_snv.loc[clinvar_snv['hgvs'].notnull(), :] 
        clinvar_snv['aa_ref_clinvar'] = clinvar_snv['hgvs'].apply(lambda x: self.get_aa_ref_clinvar(x))
        clinvar_snv['aa_pos_clinvar'] = clinvar_snv['hgvs'].apply(lambda x: self.get_aa_pos_clinvar(x))
        clinvar_snv['aa_pos_clinvar'] = clinvar_snv['aa_pos_clinvar'].astype(int)
        clinvar_snv['aa_alt_clinvar'] = clinvar_snv['hgvs'].apply(lambda x: self.get_aa_alt_clinvar(x))
        clinvar_snv['refseq_tvid'] = clinvar_snv['hgvs'].apply(lambda x: x.split(':')[0].split('(')[0])
        clinvar_snv['refseq_tid'] = clinvar_snv['refseq_tvid'].apply(lambda x: x.split('.')[0])
        clinvar_snv['data_source'] = 'clinvar'        
        clinvar_snv['review_star'] = 0
        clinvar_snv.loc[clinvar_snv['review_status'] == 'practice guideline','review_star'] = 4
        clinvar_snv.loc[clinvar_snv['review_status'] == 'reviewed by expert panel','review_star'] = 3
        clinvar_snv.loc[clinvar_snv['review_status'] == 'criteria provided, multiple submitters, no conflicts','review_star'] = 2
        clinvar_snv.loc[clinvar_snv['review_status'] == 'criteria provided, single submitter','review_star'] = 1        
                       
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
        print ('clinVAR snv records : ' + str(clinvar_snv.shape[0]))        
                        
        # remove non-coding region, synonymous, invalid records 
        clinvar_snv = clinvar_snv.loc[(clinvar_snv['aa_pos_clinvar'] != -1) & (clinvar_snv['aa_ref_clinvar'] != '?') & (clinvar_snv['aa_alt_clinvar'] != '?') & (clinvar_snv['aa_alt_clinvar'] != '_') & (clinvar_snv['aa_alt_clinvar'] != '*') & (clinvar_snv['aa_ref_clinvar'] != clinvar_snv['aa_alt_clinvar']), :]
        print ('clinVAR exome missense snv records : ' + str(clinvar_snv.shape[0]))  
                          
        # 0 reveiw star, 0 clinical significance value 
        clinvar_snv = clinvar_snv.loc[(clinvar_snv['review_star'] > 0 ) & (clinvar_snv['clinsig_level'] >0) ,:]
        print ('clinVAR review_star > 0 , clinsig_level > 0 records : ' + str(clinvar_snv.shape[0]))
            
        # merge with Gnomad
        clinvar_snv = pd.merge(clinvar_snv,self.gnomad_nt[['chr','nt_pos','nt_ref','nt_alt','gnomad_af','gnomad_gc_homo_alt']],how = 'left')
    
        #select columns for clinvar_srv
        clinvar_srv = clinvar_snv[['chr', 'nt_pos', 'nt_ref', 'nt_alt','gnomad_gc_homo_alt','gnomad_af','clin_sig','clinsig_level','review_star','evaluate_time','data_source','label']]
            
        #************************************************************************************************************************************************************************
        ## Add more benign negative dataset from Gnomad (low MAF)
        #************************************************************************************************************************************************************************
        gnomad_nt_low_af_idx = self.gnomad_nt_aa.loc[(self.gnomad_nt_aa['gnomad_gc_homo_alt'] > 0) & (self.gnomad_nt_aa['gnomad_af'] < 0.001), :].index                                                                                                                                                                                                                                                          
        gnomad_srv = self.gnomad_nt_aa.loc[gnomad_nt_low_af_idx, ['chr', 'nt_pos', 'nt_ref', 'nt_alt','gnomad_gc_homo_alt','gnomad_af']]
        #remove duplicated nucleotide coordinates
        gnomad_srv = gnomad_srv.drop_duplicates()
        #remove the overlap records with clinvar_snv
        gnomad_srv = pd.merge(gnomad_srv,clinvar_snv[['chr','nt_pos','nt_ref','nt_alt','data_source']],how = 'left')
        gnomad_srv = gnomad_srv.loc[gnomad_srv['data_source'] != 'clinvar',:]
        gnomad_srv['clin_sig'] = 'Benign'
        gnomad_srv['clinsig_level'] = 3
        gnomad_srv['review_star'] = 5  
        gnomad_srv['evaluate_time'] = '1900-01-01'
        gnomad_srv['data_source'] = 'gnomad'
        gnomad_srv['label'] = 0
           
        print ('Gnomad MAF < 0.001 , homozygous individual  > 0 records, non-overlap with clinVAR srv : ' + str(gnomad_srv.shape[0]))
            
        #************************************************************************************************************************************************************************
        ## Combine data from clinVAR and Gnomad , use this combined dataset serach on dbNFSP database (vcf format needed)
        #************************************************************************************************************************************************************************        
        clinvar_plus_gnomad_srv = pd.concat([clinvar_srv, gnomad_srv])
        clinvar_plus_gnomad_srv_vcf = clinvar_plus_gnomad_srv.copy()
        clinvar_plus_gnomad_srv_vcf['#CHROM'] = clinvar_plus_gnomad_srv_vcf['chr']
        clinvar_plus_gnomad_srv_vcf['POS'] = clinvar_plus_gnomad_srv_vcf['nt_pos']
        clinvar_plus_gnomad_srv_vcf['REF'] = clinvar_plus_gnomad_srv_vcf['nt_ref']
        clinvar_plus_gnomad_srv_vcf['ALT'] = clinvar_plus_gnomad_srv_vcf['nt_alt']
        clinvar_plus_gnomad_srv_vcf['ID'] = '.'
        clinvar_plus_gnomad_srv_vcf['SPLIT'] = '|'        
        print ('Gnomad and clinVAR combine records : ' + str(clinvar_plus_gnomad_srv.shape[0]))
           
#         #check if there is duplicated neucleotide coordinates
#         clinvar_plus_gnomad_srv_groupby  = clinvar_plus_gnomad_srv.groupby(['chr','nt_pos','nt_ref','nt_alt']).size().reset_index()
#         clinvar_plus_gnomad_srv_groupby = clinvar_plus_gnomad_srv_groupby.rename(columns = {0:'srv_counts'})
#         clinvar_plus_gnomad_srv = pd.merge(clinvar_plus_gnomad_srv,clinvar_plus_gnomad_srv_groupby)        
#         clinvar_plus_gnomad_srv.loc[clinvar_plus_gnomad_srv['srv_counts']==1,:].shape
#         
   
        #save srv and vcf file
        clinvar_plus_gnomad_srv.to_csv(self.db_path + 'clinvar/csv/clinvar_plus_gnomad_srv.csv', index=False)            
        clinvar_plus_gnomad_srv_vcf[['#CHROM','POS','ID','REF','ALT'] + list(clinvar_plus_gnomad_srv.columns) + ['SPLIT']].to_csv(self.db_path + 'clinvar/csv/clinvar_plus_gnomad_srv.vcf', sep = '\t', index=False)
 
         
        #************************************************************************************************************************************************************************
        ## Run searching dbNFSP
        #************************************************************************************************************************************************************************  
        dbnsfp_cmd = "java " + "search_dbNSFP35a -v hg19 -i " + self.db_path + "clinvar/csv/clinvar_plus_gnomad_srv.vcf -o " + self.db_path + "clinvar/csv/clinvar_plus_gnomad_srv_.out -p"
        subprocess.run(dbnsfp_cmd.split(" "), cwd = self.db_path + "dbNSFPv3.5a")
          
        #************************************************************************************************************************************************************************
        ## Define a few functions for processing dbNFSP result
        #************************************************************************************************************************************************************************
        def retrieve_aapos(uniprot_accs, uniprot_acc, uniprot_aaposs):
            try:        
                uniprot_accs_list = uniprot_accs.split(";")
                uniprot_poss_list = uniprot_aaposs.split(";")
                
                if len(uniprot_poss_list) == 1:
                    uniprot_aa_pos = uniprot_poss_list[0]
                else:
                    unprot_accs_dict = {uniprot_accs_list[x]:x for x in range(len(uniprot_accs_list))}        
                    uniprot_aa_pos = uniprot_poss_list[unprot_accs_dict.get(uniprot_acc,np.nan)]
                if not chk_int(uniprot_aa_pos):
                    uniprot_aa_pos = np.nan
                else:
                    uniprot_aa_pos = int(uniprot_aa_pos)
                    
            except:
                uniprot_aa_pos = np.nan
            return uniprot_aa_pos
            
        def chk_int(str):
            try:
                x = int(str)        
                return True
            except:
                return False
        
        def chk_float(str):
            try:
                x = float(str)        
                return True
            except:
                return False
            
        def get_value_byfun(values, fun):
            try:
                value_list = values.split(";")
                value_list = [float(x) for x in value_list if chk_float(x)]
                if fun == 'min':
                    value = min(value_list)
                if fun == 'max':
                    value = max(value_list)
            except:
                value = np.nan
            return value
        
        def get_residue_by_pos(seq,pos):
            try:
                residue = seq[pos-1]
            except:
                residue = np.nan
            return residue
        
        #************************************************************************************************************************************************************************
        ## Process dbNFSP result (fitler non-clinvar gene
        #************************************************************************************************************************************************************************        
        dbnsfp_out = pd.read_csv(self.db_path + 'clinvar/csv/clinvar_plus_gnomad_srv.out',sep = '\t')
        print ('Gnomad and clinVAR combine records after matching dbNSFP : ' + str(dbnsfp_out.shape[0]))
        
        
        basic_cols = ['#CHROM','POS','ID','REF','ALT','SPLIT'] + ['chr', 'nt_pos', 'nt_ref', 'nt_alt','gnomad_gc_homo_alt','gnomad_af','clin_sig','clinsig_level','review_star','evaluate_time','data_source','label'] + \
                     ['uniprot_aa_pos','aaref','aaalt','genename','Uniprot_acc']
        score_cols = ['SIFT_score','Polyphen2_HDIV_score','Polyphen2_HVAR_score','LRT_score','MutationTaster_score','MutationAssessor_score','FATHMM_score','PROVEAN_score','VEST3_score'] + \
                     ['MetaSVM_score','MetaLR_score','M-CAP_score','REVEL_score','MutPred_score','CADD_raw','DANN_score','fathmm-MKL_coding_score','Eigen-raw','GenoCanyon_score'] + \
                     ['integrated_fitCons_score','GERP++_RS','phyloP100way_vertebrate','phyloP20way_mammalian','phastCons100way_vertebrate','phastCons20way_mammalian','SiPhy_29way_logOdds']
#                      ['SIFT_selected_score','Polyphen2_selected_HDIV_score','Polyphen2_selected_HVAR_score','MutationTaster_selected_score','FATHMM_selected_score','PROVEAN_selected_score','VEST3_selected_score']
        
        #remove records that has multiple uniprot acc or invalid position
        clinvar_gnomad_dbnsfp_srv = dbnsfp_out.loc[(dbnsfp_out['Uniprot_acc'].str.len() == 6) & (dbnsfp_out['aaref'] != 'X') & (dbnsfp_out['aaalt'] != 'X') ,: ]                        
        print ('dbNSFP remove multiple or invalid uniprot acc for one variant: ' + str(clinvar_gnomad_dbnsfp_srv.shape[0]))
        
        #remove non-clinvar genes and make sure each gene has uniprot sequence
        clinvar_genes = clinvar_gnomad_dbnsfp_srv.loc[clinvar_gnomad_dbnsfp_srv['data_source'] == 'clinvar','Uniprot_acc'].unique()
        clinvar_genes = [x for x in clinvar_genes if x in self.uniprot_seqdict.keys()]
        clinvar_gnomad_dbnsfp_srv = clinvar_gnomad_dbnsfp_srv.loc[clinvar_gnomad_dbnsfp_srv['Uniprot_acc'].isin(clinvar_genes),:]                
        clinvar_gnomad_dbnsfp_srv['uniprot_aa_pos'] = clinvar_gnomad_dbnsfp_srv.apply(lambda x: retrieve_aapos(x['Uniprot_acc_Polyphen2'],x['Uniprot_acc'],x['Uniprot_aapos_Polyphen2']),axis = 1 )        
        
        clinvar_gnomad_dbnsfp_srv = clinvar_gnomad_dbnsfp_srv.loc[clinvar_gnomad_dbnsfp_srv['uniprot_aa_pos'].notnull(),basic_cols + score_cols]
        clinvar_gnomad_dbnsfp_srv['uniprot_aa_pos'] = clinvar_gnomad_dbnsfp_srv['uniprot_aa_pos'].astype(int)
        clinvar_gnomad_dbnsfp_srv = clinvar_gnomad_dbnsfp_srv.rename(columns = {'uniprot_aa_pos':'aa_pos','aaref':'aa_ref','aaalt':'aa_alt','Uniprot_acc':'p_vid'})
        print ('dbNSFP clinVAR genes only and also has seqeunce avaliable: ' + str(clinvar_gnomad_dbnsfp_srv.shape[0]))
        
        #remove records has unmatch aa_ref 
        clinvar_gnomad_dbnsfp_srv['aa_ref_uniprot'] = clinvar_gnomad_dbnsfp_srv.apply(lambda x: get_residue_by_pos(self.uniprot_seqdict[x['p_vid']], x['aa_pos']),axis = 1)
        clinvar_gnomad_dbnsfp_srv = clinvar_gnomad_dbnsfp_srv.loc[clinvar_gnomad_dbnsfp_srv['aa_ref_uniprot'] == clinvar_gnomad_dbnsfp_srv['aa_ref'],:]
        print ('dbNSFP after aa_ref check: ' + str(clinvar_gnomad_dbnsfp_srv.shape[0]))
                
        #remove records with duplicated coordinatees
        clinvar_gnomad_dbnsfp_srv_groupby = clinvar_gnomad_dbnsfp_srv.groupby(['p_vid','aa_pos','aa_ref','aa_alt']).size().reset_index()
        clinvar_gnomad_dbnsfp_srv_groupby = clinvar_gnomad_dbnsfp_srv_groupby.rename(columns = {0:'aa_counts'})
        clinvar_gnomad_dbnsfp_srv = pd.merge(clinvar_gnomad_dbnsfp_srv,clinvar_gnomad_dbnsfp_srv_groupby,how = 'left')  
        clinvar_gnomad_dbnsfp_srv = clinvar_gnomad_dbnsfp_srv.loc[clinvar_gnomad_dbnsfp_srv['aa_counts'] == 1,:]    
#         cols = ['#CHROM','POS','ID','REF','ALT'] + ['chr', 'nt_pos', 'nt_ref', 'nt_alt','gnomad_gc_homo_alt','gnomad_af','clin_sig','clinsig_level','review_star','evaluate_time','data_source','label'] +   ['SPLIT']     
#         duplicates = clinvar_gnomad_dbnsfp_srv.loc[clinvar_gnomad_dbnsfp_srv['srv_counts'] > 1,cols]
#         duplicates = duplicates.drop_duplicates()
#         duplicates.to_csv(self.db_path + 'clinvar/csv/clinvar_gnomad_dbnsfp_srv_duplicated.vcf',sep= '\t',index = False)
        print ('dbNSFP remove duplicated aa coordinates: ' + str(clinvar_gnomad_dbnsfp_srv.shape[0]))
        
        #get the most delterious score for scores have multiple values due to different transcripts
        clinvar_gnomad_dbnsfp_srv['SIFT_selected_score'] = clinvar_gnomad_dbnsfp_srv.apply(lambda x: get_value_byfun(x['SIFT_score'],'min'),axis = 1)
        clinvar_gnomad_dbnsfp_srv['Polyphen2_selected_HDIV_score'] = clinvar_gnomad_dbnsfp_srv.apply(lambda x: get_value_byfun(x['Polyphen2_HDIV_score'],'max'),axis = 1)
        clinvar_gnomad_dbnsfp_srv['Polyphen2_selected_HVAR_score'] = clinvar_gnomad_dbnsfp_srv.apply(lambda x: get_value_byfun(x['Polyphen2_HVAR_score'],'max'),axis = 1)
        clinvar_gnomad_dbnsfp_srv['MutationTaster_selected_score'] = clinvar_gnomad_dbnsfp_srv.apply(lambda x: get_value_byfun(x['MutationTaster_score'],'max'),axis = 1)
        clinvar_gnomad_dbnsfp_srv['FATHMM_selected_score'] = clinvar_gnomad_dbnsfp_srv.apply(lambda x: get_value_byfun(x['FATHMM_score'],'min'),axis = 1)
        clinvar_gnomad_dbnsfp_srv['PROVEAN_selected_score'] = clinvar_gnomad_dbnsfp_srv.apply(lambda x: get_value_byfun(x['PROVEAN_score'],'min'),axis = 1)
        clinvar_gnomad_dbnsfp_srv['VEST3_selected_score'] = clinvar_gnomad_dbnsfp_srv.apply(lambda x: get_value_byfun(x['VEST3_score'],'max'),axis = 1) 
          
        #************************************************************************************
        # mark polyphen training data
        #************************************************************************************
        self.polyphen_train = self.get_polyphen_train_data()
        clinvar_gnomad_dbnsfp_srv = pd.merge(clinvar_gnomad_dbnsfp_srv, self.polyphen_train, how='left')

        #************************************************************************************
        # replace "." with np.nan
        #************************************************************************************
        clinvar_gnomad_dbnsfp_srv = clinvar_gnomad_dbnsfp_srv.replace('.',np.nan)
        clinvar_gnomad_dbnsfp_srv = clinvar_gnomad_dbnsfp_srv.replace('-',np.nan)
        score_cols = [x for x in clinvar_gnomad_dbnsfp_srv.columns if '_score' in x]
        score_cols =  set(score_cols) - set(['SIFT_score','Polyphen2_HDIV_score','Polyphen2_HVAR_score','MutationTaster_score','FATHMM_score','PROVEAN_score','VEST3_score'])
        for score_col in score_cols:
            clinvar_gnomad_dbnsfp_srv[score_col] = clinvar_gnomad_dbnsfp_srv[score_col].astype(float)

        #************************************************************************************
        # save data
        #************************************************************************************                
        clinvar_gnomad_dbnsfp_srv.to_csv(self.db_path + 'clinvar/csv/clinvar_gnomad_dbnsfp_srv.csv',index = False)
                
        clinvar_gnomad_dbnsfp_srv_for_score = clinvar_gnomad_dbnsfp_srv[['p_vid', 'aa_pos', 'aa_ref', 'aa_alt']]
        clinvar_gnomad_dbnsfp_srv_for_score.drop_duplicates(inplace = True)        
        clinvar_gnomad_dbnsfp_srv_for_score.to_csv(self.db_path + 'clinvar/csv/clinvar_gnomad_dbnsfp_srv_for_score.txt', sep='\t', index=False, header=None)
        
        clinvar_gnomad_dbnsfp_srv_for_vcf = clinvar_gnomad_dbnsfp_srv[['#CHROM','POS','ID','REF','ALT']]
        clinvar_gnomad_dbnsfp_srv_for_vcf.to_csv(self.db_path + 'clinvar/csv/clinvar_gnomad_dbnsfp_srv.vcf', sep='\t', index=False)
        
        #for polyphen is ok to submit the for_score file but it is too big for provean
        indices = np.arange(0,clinvar_gnomad_dbnsfp_srv_for_score.shape[0], int(clinvar_gnomad_dbnsfp_srv_for_score.shape[0]/3))
        for i in range(len(indices)-1):
                clinvar_gnomad_dbnsfp_srv_for_score.loc[indices[i]:indices[i+1]-1,:].to_csv(self.db_path + 'clinvar/csv/clinvar_gnomad_dbnsfp_srv_for_score_' + str(i) + '.txt', sep='\t', index=False, header=None)
                if i == 2:
                    clinvar_gnomad_dbnsfp_srv_for_score.loc[indices[i]:clinvar_gnomad_dbnsfp_srv_for_score.shape[0]-1,['p_vid', 'aa_pos', 'aa_ref', 'aa_alt']].to_csv(self.db_path + 'clinvar/csv/clinvar_gnomad_dbnsfp_srv_for_score_' + str(i) + '.txt', sep='\t', index=False, header=None)

        #************************************************************************************
        # run variant processing
        #************************************************************************************     
        clinvar_plus_gnomad_final_df = self.variants_process('clinvar_gnomad_dbnsfp', clinvar_gnomad_dbnsfp_srv, self.uniprot_seqdict, self.flanking_k, nt_input = 1, gnomad_available = 1, gnomad_merge_id = 'uniprot_id')                  
        clinvar_plus_gnomad_final_df.to_csv(self.db_path + 'funregressor/funregressor_test_final.csv', index=False)  


        return (clinvar_plus_gnomad_final_df) 
     
        
    def variant_process_clinvar(self):
        clinvar_gnomad_dbnsfp_srv = pd.read_csv(self.db_path + 'clinvar/csv/clinvar_gnomad_dbnsfp_srv.csv')
        self.clinvar_gnomad_dbnsfp_processed_df = self.variants_process('clinvar_gnomad_dbnsfp', clinvar_gnomad_dbnsfp_srv, self.uniprot_seqdict, self.flanking_k, gnomad_merge_id='uniprot_id')                    
        self.clinvar_gnomad_dbnsfp_processed_df.to_csv(self.db_path + 'clinvar/csv/clinvar_plus_gnomad_final.csv', index=False)  
   
    def prepare_codonusage_data(self):
        codonusage_species_file = open(self.db_path + 'codon/codonusage_species.txt', 'w')
        codonusage_file = open(self.db_path + 'codon/codonusage.txt', 'w')
        
        str_codonusage_species = ''
        str_codonusage = ''
        
        file = self.db_path + 'codon/SPSUM_LABEL'
        with open(file) as infile:
            for line in infile: 
                if re.search(':', line):
                    str_codonusage_species += 'id|name|cds_num\n'
                else:
                    str_codonusage += line
        
        for file in glob.glob(self.db_path + 'codon/*.spsum'):
            with open(file) as infile:
                for line in infile: 
                    line = line.rstrip()  
                    if re.search(':', line):                      
                        lst_codonusage_species = line.split(':')
                        str_codonusage_species += '|'.join(lst_codonusage_species[0:3]) + '\n'
                    else:
                        str_codonusage += line + '\n'
        codonusage_species_file.write(str_codonusage_species)
        codonusage_species_file.close()
        codonusage_file.write(str_codonusage)
        codonusage_file.close()
    
    def prepare_asa_data(self):                
        ####***************************************************************************************************************************************************************
        # pdb to uniprot (be aware there are cases that PDB id is 'XEXX', the pandas will read it as float
        ####***************************************************************************************************************************************************************
        self.pdb_to_uniprot = pd.read_csv(self.db_path + 'pdb/csv/pdb_chain_uniprot.csv', dtype={"PDB": str})                 
        # get the structure data from clinvar data 
        asa_logFile = self.db_path + 'pdb/prepare_asa_data.log'
#         clinvar = pd.read_csv(self.db_path + 'clinvar/csv/clinvar_plus_gnomad_final.csv')
#         gene_lst = set(list(clinvar['p_vid']) + ['P35520','P63279','P63165','P62166','Q9H3S4','P0DP23'])           
#         gene_lst= list(self.gnomad_aa.loc[(self.gnomad_aa['gnomad_gc_homo_alt'] > 0) & (self.gnomad_aa['gnomad_af'] <0.0001), 'uniprot_id'].unique())
        gene_lst = ['P42898']        
        # print (str(gene_lst))      
        asa_url = 'http://www.ebi.ac.uk/pdbe/pisa/cgi-bin/interfaces.pisa?'
        for cur_gene in gene_lst:     
            if not os.path.isfile(self.db_path + 'pdb/org/' + cur_gene + '_asa.xml'):
                try:
                    xml_file = open(self.db_path + 'pdb/org/' + cur_gene + '_asa.xml', 'w')
                    asa_file = open(self.db_path + 'pdb/org/' + cur_gene + '_asa.txt', 'w')            
                    cur_gene_pdb = self.pdb_to_uniprot.loc[self.pdb_to_uniprot['SP_PRIMARY'] == cur_gene, :]
                    pdb_lst = ','.join(cur_gene_pdb['PDB'])
                    cur_asa_url = asa_url + pdb_lst
                    response = urllib.request.urlopen(cur_asa_url)
                    r = response.read().decode('utf-8')
                    xml_file.write(r)
                    tree = ET.fromstring(r)
                    for pdb_entry in tree.iter('pdb_entry'):
                        # print (pdb_entry[0].tag + ':' + pdb_entry[0].text)   #pdb_code
                        for interface in pdb_entry.iter("interface"):
                            # print (interface[0].tag + ':' + interface[0].text) #interface id 
                            for molecule in interface.iter("molecule"):
                                # print(molecule[1].tag +':' + molecule[1].text) #chain_id
                                for residue in molecule.iter("residue"):
                                    # print (residue[0].tag + ':' + residue[0].text + '|' + residue[1].tag + ':' + residue[1].text +'|' + residue[5].tag + ':' + residue[5].text)
                                    asa_file.write(pdb_entry[0].text + '\t' + interface[0].text + '\t' + molecule[1].text + '\t' + residue[0].text + '\t' + residue[1].text + '\t' + residue[2].text + '\t' + residue[5].text + '\n')
                    show_msg(asa_logFile, 1, cur_gene + ' is processed.\n')                                        
                except:
                    show_msg(asa_logFile, 1, cur_gene + ' process error: ' + traceback.format_exc() + '\n')
                asa_file.close() 
                xml_file.close()  
        count = 0
        for file in glob.glob(os.path.join(self.db_path + 'pdb/org/P42898_asa.txt')):
            if os.path.getsize(file) != 0 :
                try:
                    count += 1
                    cur_gene = os.path.basename(file)[0:6]
                    cur_pdb_to_uniprot = self.pdb_to_uniprot.loc[self.pdb_to_uniprot['SP_PRIMARY'] == cur_gene, ]
                    cur_pdb_to_uniprot['pdb_chain'] = cur_pdb_to_uniprot['PDB'] + '_' + cur_pdb_to_uniprot['CHAIN']
                    cur_asa_df = pd.read_csv(self.db_path + 'pdb/org/' + cur_gene + '_asa.txt', sep='\t', header=None, dtype={0: str})
                    cur_asa_df.columns = ['PDB', 'interface', 'CHAIN', 'ser_no', 'residue', 'aa_pos', 'asa']                        
                    cur_asa_df['PDB'] = cur_asa_df['PDB'].str.lower()
                    cur_asa_df['pdb_chain'] = cur_asa_df['PDB'] + '_' + cur_asa_df['CHAIN']        
                    cur_asa_df = cur_asa_df.loc[cur_asa_df['pdb_chain'].isin(cur_pdb_to_uniprot['pdb_chain']), :]
                    cur_asa_df_groupby = cur_asa_df.groupby(['residue', 'aa_pos'])
                    cur_asa_df = cur_asa_df_groupby.agg({'asa':['mean', 'std', 'count']}).reset_index().sort_values(['aa_pos'])
                    cur_asa_df.columns = ['residue', 'aa_pos', 'asa_mean', 'asa_std', 'asa_count']    
                    cur_asa_df['aa_ref'] = cur_asa_df['residue'].apply(lambda x: self.dict_aa3_upper.get(x, np.nan))
                    cur_asa_df = cur_asa_df.loc[cur_asa_df['aa_ref'].notnull(), ]
                    cur_asa_df['p_vid'] = cur_gene
                    cur_asa_df.to_csv(self.db_path + 'pdb/csv/asa_df.csv', index=False, header=False, mode='a')
                    show_msg(asa_logFile, 1, str(count) + ': ' + cur_gene + ' is saved to csv.\n')     
                except:
                    show_msg(asa_logFile, 1, str(count) + ': ' + cur_gene + ' process error: ' + traceback.format_exc() + '\n')
        print(count)  
                                       
    def prepare_gnomad_data(self):
        gnomad_snp = pd.read_csv(self.db_path + 'gnomad/gnomad_output_snp.txt', sep='\t', header=None)
        gnomad_snp.columns = ['chr', 'nt_pos', 'dbsnp_id', 'nt_ref', 'nt_alt', 'quality_score', 'filter', 'gnomad_ac', 'gnomad_an', 'gnomad_af', 'gnomad_esa_af', 'gnomad_gc', 'gnomad_gc_homo_ref', 'gnomad_gc_hetero', 'gnomad_gc_homo_alt', 'aa_ref', 'aa_pos', 'aa_alt', 'variant_type', 'gene_name', 'ensg_id', 'enst_id', 'cdna_variant_info', 'protein_variant_info', 'rs_id', 'ensp_id', 'uniprot_id', 'strand']
        gnomad_snp.loc[(gnomad_snp.gnomad_af == '.'), 'gnomad_af'] = 0        
        gnomad_snp.loc[(gnomad_snp.gnomad_esa_af == '.'), 'gnomad_esa_af'] = 0
        gnomad_snp.gnomad_ac = gnomad_snp.gnomad_ac.astype(int)
        gnomad_snp.gnomad_an = gnomad_snp.gnomad_an.astype(int)
        gnomad_snp.gnomad_af = gnomad_snp.gnomad_af.astype(float)
        gnomad_snp.gnomad_esa_af = gnomad_snp.gnomad_esa_af.astype(float)
        gnomad_snp_processed = gnomad_snp[['chr', 'nt_pos', 'dbsnp_id', 'nt_ref', 'nt_alt', 'quality_score', 'filter', 'gnomad_ac', 'gnomad_an', 'gnomad_af', 'gnomad_esa_af', 'gnomad_gc_homo_ref', 'gnomad_gc_hetero', 'gnomad_gc_homo_alt', 'aa_ref', 'aa_pos', 'aa_alt', 'gene_name', 'ensg_id', 'enst_id', 'ensp_id', 'uniprot_id', 'strand']]
        gnomad_snp_processed.to_csv(self.db_path + 'gnomad/gnomad_output_snp_processed.txt', index=False, sep='\t')

        # gnomad_snp_processed = pd.read_table(self.db_path + 'gnomad/gnomad_output_snp_processed.txt',sep = '\t')
        gnomad_snp_aa = gnomad_snp_processed.loc[gnomad_snp_processed['aa_ref'].notnull() & gnomad_snp_processed['aa_alt'].notnull() & (gnomad_snp_processed['aa_ref'] != gnomad_snp_processed['aa_alt']), :]
        gnomad_snp_aa.to_csv(self.db_path + 'gnomad/gnomad_output_snp_aa.txt', index=False, sep='\t')        
                
        # gnomad_snp_aa = pd.read_csv(self.db_path + 'gnomad/gnomad_output_snp_aa.txt',sep = '\t')
        gnomad_snp_aa_ensp_rs_groupby = gnomad_snp_aa.groupby(['ensp_id', 'dbsnp_id', 'aa_ref', 'aa_pos', 'aa_alt'])
        aggregations = {'gnomad_ac':'sum', 'gnomad_an':'mean', 'gnomad_af':'sum', 'gnomad_esa_af':'sum', 'gnomad_gc_homo_ref':'sum', 'gnomad_gc_hetero':'sum', 'gnomad_gc_homo_alt':'sum'}
        gnomad_snp_aa_ensp_rs = gnomad_snp_aa_ensp_rs_groupby.agg(aggregations).reset_index()
        gnomad_snp_aa_ensp_rs['rs_nt_count'] = gnomad_snp_aa_ensp_rs_groupby.size().reset_index(drop=True)
        gnomad_snp_aa_ensp_rs.to_csv(self.db_path + 'gnomad/gnomad_snp_aa_ensp_rs.txt', sep='\t', index=False) 
        
        gnomad_snp_aa_uniprot_rs_groupby = gnomad_snp_aa.groupby(['uniprot_id', 'dbsnp_id', 'aa_ref', 'aa_pos', 'aa_alt'])
        aggregations = {'gnomad_ac':'sum', 'gnomad_an':'mean', 'gnomad_af':'sum', 'gnomad_esa_af':'sum', 'gnomad_gc_homo_ref':'sum', 'gnomad_gc_hetero':'sum', 'gnomad_gc_homo_alt':'sum'}
        gnomad_snp_aa_uniprot_rs = gnomad_snp_aa_uniprot_rs_groupby.agg(aggregations).reset_index()
        gnomad_snp_aa_uniprot_rs['rs_nt_count'] = gnomad_snp_aa_uniprot_rs_groupby.size().reset_index(drop=True)
        gnomad_snp_aa_uniprot_rs.to_csv(self.db_path + 'gnomad/gnomad_snp_aa_uniprot_rs.txt', sep='\t', index=False) 
#         
        
        stime = time.time() 
        # gnomad_snp_aa_ensp_rs = pd.read_csv(self.db_path + 'gnomad/gnomad_snp_aa_ensp_rs.txt',sep = '\t')
        gnomad_snp_aa_ensp_groupby = gnomad_snp_aa_ensp_rs.groupby(['ensp_id', 'aa_ref', 'aa_pos', 'aa_alt'])        
        aggregations1 = {'gnomad_ac':'sum', 'gnomad_an':'mean', 'gnomad_af':'sum', 'gnomad_esa_af':'sum',
                        'gnomad_gc_homo_ref':'sum', 'gnomad_gc_hetero':'sum', 'gnomad_gc_homo_alt':'sum', 'rs_nt_count':'sum'}
        gnomad_snp_aa_ensp1 = gnomad_snp_aa_ensp_groupby.agg(aggregations1).reset_index()
        gnomad_snp_aa_ensp1.columns = ['ensp_id', 'aa_ref', 'aa_pos', 'aa_alt', 'gnomad_ac', 'gnomad_an', 'gnomad_af',
                                      'gnomad_esa_af', 'gnomad_gc_homo_ref', 'gnomad_gc_hetero', 'gnomad_gc_homo_alt', 'rs_nt_count']          
        aggregations2 = {'dbsnp_id': lambda x:','.join(x),
                         'rs_nt_count': lambda x:','.join(x.astype(str)),
                         'gnomad_gc_hetero': lambda x:','.join(x.astype(str)),
                         'gnomad_gc_homo_alt': lambda x:','.join(x.astype(str)),
                         'gnomad_gc_homo_ref': lambda x:','.join(x.astype(str))}
                                    
        gnomad_snp_aa_ensp2 = gnomad_snp_aa_ensp_groupby.agg(aggregations2).reset_index(drop=True)                            
        gnomad_snp_aa_ensp2.columns = ['rs_lst', 'rs_nt_count_lst', 'gnomad_gc_hetero_lst', 'gnomad_gc_homo_alt_lst', 'gnomad_gc_homo_ref_lst']
         
        gnomad_snp_aa_ensp = pd.concat([gnomad_snp_aa_ensp1, gnomad_snp_aa_ensp2], axis=1)
#         gnomad_snp_aa_ensp = gnomad_snp_aa_ensp1
        gnomad_snp_aa_ensp['rs_count'] = gnomad_snp_aa_ensp_groupby.size().reset_index(drop=True)         
        gnomad_snp_aa_ensp.to_csv(self.db_path + 'gnomad/gnomad_output_snp_aa_ensp.txt', index=False, sep='\t')
        etime = time.time() 
        print("Elapse time was %g seconds" % (etime - stime))
         
        stime = time.time() 
#         gnomad_snp_aa_uniprot_rs = pd.read_csv(self.db_path + 'gnomad/gnomad_snp_aa_uniprot_rs.txt',sep = '\t')
        gnomad_snp_aa_uniprot_groupby = gnomad_snp_aa_uniprot_rs.groupby(['uniprot_id', 'aa_ref', 'aa_pos', 'aa_alt'])        
        aggregations1 = {'gnomad_ac':'sum', 'gnomad_an':'mean', 'gnomad_af':'sum', 'gnomad_esa_af':'sum',
                        'gnomad_gc_homo_ref':'sum', 'gnomad_gc_hetero':'sum', 'gnomad_gc_homo_alt':'sum', 'rs_nt_count':'sum'}
        gnomad_snp_aa_uniprot1 = gnomad_snp_aa_uniprot_groupby.agg(aggregations1).reset_index()
        gnomad_snp_aa_uniprot1.columns = ['uniprot_id', 'aa_ref', 'aa_pos', 'aa_alt', 'gnomad_ac', 'gnomad_an', 'gnomad_af',
                                      'gnomad_esa_af', 'gnomad_gc_homo_ref', 'gnomad_gc_hetero', 'gnomad_gc_homo_alt', 'rs_nt_count']          
        aggregations2 = {'dbsnp_id': lambda x:','.join(x),
                         'rs_nt_count': lambda x:','.join(x.astype(str)),
                         'gnomad_gc_hetero': lambda x:','.join(x.astype(str)),
                         'gnomad_gc_homo_alt': lambda x:','.join(x.astype(str)),
                         'gnomad_gc_homo_ref': lambda x:','.join(x.astype(str))}
                                    
        gnomad_snp_aa_uniprot2 = gnomad_snp_aa_uniprot_groupby.agg(aggregations2).reset_index(drop=True)                            
        gnomad_snp_aa_uniprot2.columns = ['rs_lst', 'rs_nt_count_lst', 'gnomad_gc_hetero_lst', 'gnomad_gc_homo_alt_lst', 'gnomad_gc_homo_ref_lst']
          
        gnomad_snp_aa_uniprot = pd.concat([gnomad_snp_aa_uniprot1, gnomad_snp_aa_uniprot2], axis=1)
        # gnomad_snp_aa_uniprot = gnomad_snp_aa_uniprot1
        gnomad_snp_aa_uniprot['rs_count'] = gnomad_snp_aa_uniprot_groupby.size().reset_index(drop=True)         
        gnomad_snp_aa_uniprot.to_csv(self.db_path + 'gnomad/gnomad_output_snp_aa_uniprot.txt', index=False, sep='\t')
        etime = time.time() 
        print("Elapse time was %g seconds" % (etime - stime))
       
    def check_psipred_duplicated(self):
        psipred_path = self.db_path + 'psipred/'
        lst_ids = [] 
        for psipred_file in glob.glob(os.path.join(psipred_path, '*.ss2')):
            id = psipred_file[len(psipred_path):-4]
            if 'NP' in id:
                uniprot_id = self.refseq2uniprot_dict.get(id, None)
            else:
                uniprot_id = id
            
            if uniprot_id != None:
                lst_ids.append([id, uniprot_id])
        pass
        df_ids = pd.DataFrame(lst_ids, columns=['refseq_id', 'uniprot_id'])
        duplicated_ids = df_ids.loc[df_ids['uniprot_id'].duplicated(), 'uniprot_id']
        df_ids = df_ids.loc[df_ids['uniprot_id'].isin(duplicated_ids), :]
        df_ids = df_ids.sort_values('uniprot_id')
        print (df_ids)
    
    
    def prepare_eigen_data(self):
        eigenFile = self.db_path + 'eigen/org/Eigen_hg19_coding_annot_04092016.tab'
        eigenFile_missense = self.db_path + 'eigen/csv/eigen_missense.txt'
         
        with open(eigenFile) as infile:
            for line in infile:                                 
                lst_line = line.split('\t')
                if lst_line[16] == 'missense_variant':
                    with open(eigenFile_missense, "a") as f:
                        f.write(line)
        
    def prepare_pdbss_data(self):
        
        def phase_seq(x):                                
            x_processed_df = pd.DataFrame(np.vstack([[x['uniprot_id']] * (x['sp_end'] - x['sp_beg'] + 1), list(range(x['sp_beg'], x['sp_end'] + 1)), list(x['pdb_seq_sp']), list(x['pdb_ss_sp'])]).transpose())            
            x_processed_df.to_csv(self.db_path + 'pdbss/pdbss_processed_df.csv', mode='a', header=False, index=False)
            
#         pdbss_log_file = self.db_path + 'pdbss/pdbss_log.txt'
#         with open(self.db_path + 'pdbss/ss.txt','r') as f:
#             cur_pdb = '' 
#             cur_chain = ''
#             cur_info = ''    
#             cur_seq = ''
#             cur_ss = ''     
#             for line in f:
#                 line = line.rstrip("\n\r")
#                 if re.match('>',line):
#                     line_lst = line[1:].split(':')
#                     if line_lst[2] == 'sequence':
#                         if cur_pdb != '':
#                             with open(self.db_path + 'pdbss/ss_df.csv','a') as pdbss_df:
#                                 pdbss_df.write(cur_pdb + ',' + cur_chain +',' + cur_seq + ',' + cur_ss +'\n')
#                             cur_seq = ''
#                             cur_ss = ''
#                         cur_pdb = line_lst[0]
#                         cur_chain = line_lst[1]
#                         cur_info = line_lst[2]
#                     if line_lst[2] == 'secstr':
#                         cur_info = line_lst[2]
#                         if (cur_pdb != line_lst[0]) | (cur_chain != line_lst[1]):
#                             show_msg(pdbss_log_file, 1, line+'\n')
#                 else:
#                     if cur_info == 'sequence':
#                         cur_seq += line
#                     if cur_info == 'secstr':
#                         cur_ss += line        
        
#         pdbss_df = pd.read_csv(self.db_path + 'pdbss/ss_df.csv',header = None)
#         pdbss_df.columns = ['pdb','chain','pdb_seq','pdb_ss']
#         pdbss_df['pdb_seq_len'] = pdbss_df['pdb_seq'].str.len()
#         pdbss_df['pdb_ss_len'] = pdbss_df['pdb_ss'].str.len()
# 
#         pdb_to_uniprot = pd.read_csv(self.db_path + 'pdb/csv/pdb_chain_uniprot.csv',dtype={"PDB": str}) 
#         pdb_to_uniprot['PDB'] = pdb_to_uniprot['PDB'].str.upper()
#         pdb_to_uniprot.columns = ['pdb','chain','uniprot_id','res_beg','res_end','pdb_beg','pdb_end','sp_beg','sp_end']                        
#         pdbss_df = pd.merge(pdb_to_uniprot,pdbss_df,how = 'left')          
#         pdbss_df = pdbss_df[['uniprot_id','res_beg','res_end','pdb_beg','pdb_end','sp_beg','sp_end','pdb_seq','pdb_ss','pdb_seq_len','pdb_ss_len']]
#         pdbss_df.drop_duplicates(inplace = True)
#              
#         pdbss_df['uniprot_seq'] = pdbss_df['uniprot_id'].apply(lambda x: self.uniprot_seqdict.get(x,''))
#         pdbss_df = pdbss_df.loc[(pdbss_df['uniprot_seq'] != '') & (pdbss_df['pdb_seq'].notnull()),:]
#         
#         pdbss_df['pdb_seq_sp']= pdbss_df.apply(lambda x: x['pdb_seq'][x['res_beg']-1:x['res_end']],axis = 1)
#         pdbss_df['pdb_ss_sp']= pdbss_df.apply(lambda x: x['pdb_ss'][x['res_beg']-1:x['res_end']],axis = 1)
#         pdbss_df['pdb_seq_sp_len'] = pdbss_df['pdb_seq_sp'].str.len()
#         pdbss_df['uniprot_seq_sp']= pdbss_df.apply(lambda x: x['uniprot_seq'][x['sp_beg']-1:x['sp_end']],axis = 1)
#         pdbss_df['uniprot_seq_len'] = pdbss_df['uniprot_seq'].str.len()
#         pdbss_df['uniprot_seq_sp_len'] = pdbss_df['uniprot_seq_sp'].str.len()
#         pdbss_df['seq_match'] = pdbss_df.apply(lambda x: x['pdb_seq_sp'] == x['uniprot_seq_sp'],axis = 1)
#         
#         #Focus on the sequence matched ones for now
#         pdbss_seq_match_df = pdbss_df.loc[pdbss_df['seq_match'] == 1,['uniprot_id','sp_beg','sp_end','pdb_seq_sp','pdb_ss_sp']]
#         pdbss_seq_match_df.drop_duplicates(inplace = True)        
#         pdbss_seq_match_df.to_csv(self.db_path + 'pdbss/ss_seq_match_df.csv',index = False)
        
#         pdbss_seq_match_df = pd.read_csv(self.db_path + 'pdbss/ss_seq_match_df.csv')
#         pdbss_seq_match_df.apply(lambda x: phase_seq(x),axis = 1)
#         pdbss_seq_match_processed_df = pd.read_csv(self.db_path + 'pdbss/pdbss_processed_df.csv',header = None)
#         pdbss_seq_match_processed_df.columns = ['index','p_vid','aa_pos','aa_ref','aa_ss']
#         pdbss_seq_match_processed_df = pdbss_seq_match_processed_df[['p_vid','aa_pos','aa_ref','aa_ss']]

#         pdbss_seq_match_processed_unique_df= pdbss_seq_match_processed_df.groupby(['p_vid','aa_pos','aa_ref','aa_ss']).size().reset_index(name='count')
#         pdbss_seq_match_processed_unique_df.to_csv(self.db_path + 'pdbss/pdbss_processed_unique_df.csv',index = None)
#         
        pdbss_seq_match_processed_unique_df = pd.read_csv(self.db_path + 'pdbss/pdbss_processed_unique_df.csv')
        pdbss_seq_match_processed_unique_df.loc[pdbss_seq_match_processed_unique_df['aa_ss'] == ' ', 'aa_ss'] = 'C'
        pdbss_seq_match_processed_unique_max_df = pdbss_seq_match_processed_unique_df.groupby(['p_vid', 'aa_pos', 'aa_ref'])['count'].max().reset_index()
        pdbss_seq_match_processed_unique_max_df['max'] = 1
        
        pdbss_seq_match_processed_unique_max_df1 = pd.merge(pdbss_seq_match_processed_unique_df, pdbss_seq_match_processed_unique_max_df, how='left')
        pdbss_seq_match_processed_unique_max_df2 = pdbss_seq_match_processed_unique_max_df1.loc[pdbss_seq_match_processed_unique_max_df1['max'] == 1, ['p_vid', 'aa_pos', 'aa_ref', 'aa_ss', 'count']]
        
        pdbss_seq_match_processed_unique_max_df3 = pdbss_seq_match_processed_unique_max_df2.groupby(['p_vid', 'aa_pos', 'aa_ref']).size().reset_index(name='alt_ss_count')
        pdbss_seq_match_processed_unique_max_df4 = pd.merge(pdbss_seq_match_processed_unique_max_df2, pdbss_seq_match_processed_unique_max_df3, how='left')
        
        # not considering the conflict ones ( alt_ss_count > 1) for now, will consider if there are not enough evaluation data with secondary structure 
        pdbss_final_df = pdbss_seq_match_processed_unique_max_df4.loc[pdbss_seq_match_processed_unique_max_df4['alt_ss_count'] == 1, ['p_vid', 'aa_pos', 'aa_ref', 'aa_ss']]
        
        pdbss_final_df['aa_ss1'] = np.nan
        pdbss_final_df.loc[pdbss_final_df['aa_ss'].isin(['G', 'H', 'I']), 'aa_ss1'] = 'H'
        pdbss_final_df.loc[pdbss_final_df['aa_ss'].isin(['T', 'S']), 'aa_ss1'] = 'T'
        pdbss_final_df.loc[pdbss_final_df['aa_ss'].isin(['B', 'E']), 'aa_ss1'] = 'E'
        pdbss_final_df.loc[pdbss_final_df['aa_ss'] == 'C', 'aa_ss1'] = 'C'                        
        pdbss_final_df.to_csv(self.db_path + 'pdbss/pdbss_final.csv', index=False)

    def prepare_psipred_data(self):
        psipred_path = self.db_path + 'psipred/' 
        for psipred_file in glob.glob(os.path.join(psipred_path, '*.seq')):
            cur_psipred_df = pd.read_csv(psipred_file, header=None, sep='\t')
            cur_psipred_df.columns = ['aa_psipred', 'aa_pos', 'ss_end_pos']
            # convert to uniprot id 
            refseq_vid = psipred_file[len(psipred_path):-4]
            if 'NP' in refseq_vid:
                p_vid = self.refseq2uniprot_dict.get(refseq_vid, None)
                if p_vid != None:
                    cur_psipred_df['p_vid'] = p_vid                                       
                    cur_psipred_df.to_csv(self.db_path + 'psipred/psipred_region_df.csv', mode = 'a',index=None,header = None)


    def psipred_refseq2uniprot(self):
        psipred_path = self.db_path + 'psipred/' 
        for psipred_file in glob.glob(os.path.join(psipred_path, '*.seq')):
            cur_psipred_df = pd.read_csv(psipred_file, header=None, sep='\t')
            cur_psipred_df.columns = ['aa_psipred', 'aa_pos', 'ss_end_pos']
            # convert to uniprot id 
            refseq_vid = psipred_file[len(psipred_path):-4]
            if 'NP' in refseq_vid:
                p_vid = self.refseq2uniprot_dict.get(refseq_vid, None)
                if p_vid is not None:
                    cur_psipred_df.to_csv(self.db_path + 'psipred/dms_feature/' + p_vid + '_psipred.csv',index = False)


    


    def prepare_psipred_data_backup(self):
        psipred_path = self.db_path + 'psipred/' 
        for psipred_file in glob.glob(os.path.join(psipred_path, '*.ss2')):
            cur_psipred_df = pd.read_csv(psipred_file, skiprows=[0, 1], header=None, sep='\s+')
            cur_psipred_df = cur_psipred_df.loc[:, [0, 1, 2]]
            cur_psipred_df.columns = ['aa_pos', 'aa_ref', 'aa_psipred']
            
#             #create files for secondary structure regions
#             sc_seq = ''.join(list(cur_psipred_df['aa_psipred']))
#             psipred_seq_file =  open(psipred_file.replace('ss2','seq'),'w')
#             psipred_seq_str = ''
#             sc_start = ''
#             cur_start_pos = 1
#             for i in range(1,len(sc_seq)+1):
#                 cur_sc = sc_seq[i-1]
#                 if cur_sc != sc_start:
#                     pre_start_pos = cur_start_pos
#                     if i != 1:
#                         pre_end_pos = i-1
#                         psipred_seq_str += sc_start + '\t' + str(pre_start_pos) + '\t' + str(pre_end_pos) + '\n'                                        
#                     sc_start = cur_sc
#                     cur_start_pos = i
#                 if i == cur_psipred_df.shape[0]:
#                     psipred_seq_str += sc_start + '\t' + str(cur_start_pos) + '\t' + str(i) + '\n'
#             pass
#             psipred_seq_file.write(psipred_seq_str)      
#             psipred_seq_file.close()

            # convert to uniprot id 
            p_vid = psipred_file[len(psipred_path):-4]
            if 'NP' in p_vid:
                p_vid = self.refseq2uniprot_dict.get(p_vid, None)
            if p_vid != None:
                cur_psipred_df['p_vid'] = p_vid
                if 'psipred_df' not in locals():
                    psipred_df = cur_psipred_df
                else:
                    psipred_df = pd.concat([psipred_df, cur_psipred_df])
                                       
        psipred_df.to_csv(self.db_path + 'psipred/psipred_df.csv', index=None)
    
    def prepare_envision_data(self):
        
#         #first form a list of genes that we need from the dms data and clinvar_data        
#         clinvar_data = pd.read_csv(self.db_path + 'clinvar/csv/clinvar_plus_gnomad_final.csv')
#         clinvar_data_asa = clinvar_data.loc[clinvar_data['asa_mean'].notnull(),:]
#         lst_gene = list(clinvar_data_asa['p_vid'].unique()) + list(self.dict_dms_genes.keys())        
#         dict_gene = {x:1 for x in lst_gene}
#                        
#         envision_score_file = self.db_path + 'envision/org/envision_score_for_extrapolation.csv'
#         envision_org_file = self.db_path + 'envision/org/human_predicted_combined_20170925.csv'
#         envision_log_file = self.db_path + 'envision/org/envision_process_log'
#         f_new = open(envision_score_file,'w')
#         count = 0
#         with open(envision_org_file,'r') as f:
#             for line in f:
# #                 line = line.rstrip("\n\r")
#                 if (count % 10000 == 0):
#                     show_msg(envision_log_file,1,str(count) + ' records have been processed.\n')
#                 line_lst = line.split(',')                
#                 try: 
# #                     with open(envision_score_file,"a") as f_new:
# #                     f_new.write(line_lst[5] + ',' + line_lst[4] + ',' + line_lst[2] + ',' + line_lst[3] + ',' + line_lst[34] + '\n')
# #                     if line_lst[5] in lst_gene:
#                     if (dict_gene.get(line_lst[5]) == 1) :
#                         f_new.write (line)
#                     count += 1
#                 except:
#                     print ('Raise error:\n' + traceback.format_exc() + '\n')
#         f_new.close()
        
#         envision_data = pd.read_csv(self.db_path + 'envision/org/human_predicted_combined_20170925.csv',nrows = 100)
#         envision_columns = ['envi_' + x  for x in envision_data.columns]
#         envision_data_for_extrapolation = pd.read_csv(self.db_path + 'envision/org/envision_score_for_extrapolation.csv')
#         envision_data_for_extrapolation.columns = envision_columns
#         envision_data_for_extrapolation['p_vid'] = envision_data_for_extrapolation['envi_Uniprot']
#         envision_data_for_extrapolation['aa_pos'] = envision_data_for_extrapolation['envi_position']
#         envision_data_for_extrapolation['aa_ref'] = envision_data_for_extrapolation['envi_AA1']
#         envision_data_for_extrapolation['aa_alt'] = envision_data_for_extrapolation['envi_AA2']  
#         envision_data_for_extrapolation.to_csv(self.db_path + 'envision/csv/envision_score_for_extrapolation_processed.csv',index = False)

        envision_data_for_extrapolation = pd.read_csv(self.db_path + 'envision/csv/envision_score_for_extrapolation_processed.csv')
        print (envision_data_for_extrapolation.shape[0])
        envision_data_for_extrapolation.drop(['envi_X1'], axis=1, inplace=True)
        envision_data_for_extrapolation.drop_duplicates(inplace=True)
        print (envision_data_for_extrapolation.shape[0])
        envision_data_for_extrapolation.to_csv(self.db_path + 'envision/csv/envision_score_for_extrapolation_processed.csv', index=False)
        
#         envision_data_for_extrapolation['num'] = 1   
#         groupby_df = envision_data_for_extrapolation.groupby(['p_vid','aa_pos','aa_ref','aa_alt'])['num'].sum()
#         groupby_df = groupby_df.reset_index()
#         groupby_df.columns = ['p_vid','aa_pos','aa_ref','aa_alt','sum_num']
#         envision_data_for_extrapolation_final = pd.merge(envision_data_for_extrapolation,groupby_df,how = 'left')
#         envision_data_for_extrapolation_final.to_csv(self.db_path + 'envision/csv/envision_score_for_extrapolation_processed_final.csv',index = False)
#         envision_data_for_extrapolation_final.loc[envision_data_for_extrapolation_final['sum_num']>1,:].to_csv(self.db_path + 'envision/csv/envision_score_for_extrapolation_processed_final_duplicated.csv',index = False)
#         
#         envision_data_for_extrapolation_final_duplicated = pd.read_csv(self.db_path + 'envision/csv/envision_score_for_extrapolation_processed_final_duplicated.csv')                
#         print("OK")

    def prepare_evmutation_data(self):
#         first = 1
#         for evmutation_file in glob.glob(os.path.join(self.db_path + 'evmutation/effects/' , '*.csv')):
#             cur_evmutation_df = pd.read_csv(evmutation_file, sep=';')
#             cur_evmutation_df.columns = ['mutation', 'aa_pos', 'aa_ref', 'aa_alt', 'evm_epistatic_score', 'evm_independent_score', 'evm_frequency', 'evm_conservation']
#             
#             # convert to uniprot id 
#             filename_splits = evmutation_file.split('/')[-1].split('_')
#             cur_uniprot_id = filename_splits[0]
#             cur_pfam_id = filename_splits[1]
#             cur_evmutation_df['p_vid'] = cur_uniprot_id
#             cur_evmutation_df['evm_pfam_id'] = cur_pfam_id
# 
#             if first == 1:
#                 cur_evmutation_df.to_csv(self.db_path + 'evmutation/evmutation_df.csv', mode='w', index=None)
#                 first += 1
#             else:
#                 cur_evmutation_df.to_csv(self.db_path + 'evmutation/evmutation_df.csv', mode='a', index=None, header=False)
#                 first += 1
#                 print(first)
                
        evm_score = pd.read_csv(self.db_path + 'evmutation/csv/evmutation_df.csv')[['p_vid', 'aa_pos', 'aa_ref', 'aa_alt', 'evm_epistatic_score', 'evm_independent_score']]
        evm_groupby_df = evm_score.groupby(['p_vid', 'aa_pos', 'aa_ref', 'aa_alt'])['evm_epistatic_score','evm_independent_score'].mean()
        evm_groupby_df = evm_groupby_df.reset_index()
        evm_groupby_df.to_csv(self.db_path + 'evmutation/csv/evmutation_df_org.csv', index=None)
           
    def prepare_pfam_files(self):
        pfam = pd.read_csv(self.db_path + 'pfam/9606.tsv', header=None, skiprows=3, sep='\t')
        pfam.columns = ['p_vid', 'a_start', 'a_end', 'e_start', 'e_end', 'hmm_id', 'hmm_name', 'type', 'hmm_start', 'hmm_end', 'hmm_length', 'bit_score', 'e_value', 'clan']
        p_vids = pfam.p_vid.unique()        
        for p_vid in p_vids:
            pfam.loc[(pfam.p_vid == p_vid), :].to_csv(self.db_path + 'pfam/' + p_vid + '.pfam', sep='\t', index=None)
 
    def prepare_dms_data(self, fitness_only=False):
        self.dms_data_dict = {}
        for dms_gene in list(self.dict_dms_genes.keys()):
            dms_fitness_df = self.prepare_dms_normalization(dms_gene, self.dict_dms_genes[dms_gene])

            if fitness_only: 
                cur_dms_gene_df = dms_fitness_df.copy()
            else:
                dms_feature_df = self.prepare_dms_features(self.dict_dms_genes[dms_gene])
                cur_dms_gene_df = pd.merge(dms_feature_df, dms_fitness_df, how='left') 
                  
            cur_dms_gene_df.to_csv(self.db_path + 'dms/' + dms_gene + '_final.csv')
            self.dms_data_dict[dms_gene] = cur_dms_gene_df
            if 'dms_data_df' not in locals():
                dms_data_df = cur_dms_gene_df
            else:
                dms_data_df = pd.concat([dms_data_df, cur_dms_gene_df])   
        if fitness_only:       
            dms_data_df.to_csv(self.db_path + 'dms/dms_final_fitnessonly.csv')
        else:
            dms_data_df.to_csv(self.db_path + 'dms/dms_final.csv')
    
    def get_polyphen_sessionid(self, dms_gene_id):
        dms_gene_aa = self.uniprot_seqdict.get(dms_gene_id,'unknown')
        if dms_gene_aa == 'unknown':
            return("no_sequence")        
        dms_gene_matrix = np.full((21, len(dms_gene_aa)), np.nan)                                                           
        dms_feature_df = pd.DataFrame(dms_gene_matrix, columns=range(1, len(dms_gene_aa) + 1), index=self.lst_aa_21)
        dms_feature_df['aa_alt'] = dms_feature_df.index
        dms_feature_df = pd.melt(dms_feature_df, ['aa_alt'])
        dms_feature_df = dms_feature_df.rename(columns={'variable': 'aa_pos', 'value': 'aa_ref'})        
        dms_feature_df['aa_ref'] = dms_feature_df['aa_pos'].apply(lambda x: list(dms_gene_aa)[x - 1])
        dms_feature_df['p_vid'] = dms_gene_id
        dms_feature_df['uniprot_id'] = dms_gene_id        
        dms_feature_df[['p_vid', 'aa_pos', 'aa_ref', 'aa_alt']].to_csv(self.db_path + 'dms/for_score/' + dms_gene_id + '_for_score.txt', sep=' ', index=False, header=None)
        
#         polyphen_cmd = ['curl', '-F_ggi_project=PPHWeb2', '-F_ggi_origin=query', '-F', '_ggi_target_pipeline=1', '-F', 'MODELNAME=HumDiv', '-F', 'UCSCDB=hg19', '-F', 'SNPFUNC=m', '-F', 'NOTIFYME=joe.wu.ca@gmail.com', '-F_ggi_batch_file=@' + self.db_path + 'dms/for_score/' + dms_gene_id + '_for_score.txt', '-D' , '-', 'http://genetics.bwh.harvard.edu/cgi-bin/ggi/ggi2.cgi']
        polyphen_cmd = ['curl', '-F_ggi_project=PPHWeb2', '-F_ggi_origin=query', '-F', '_ggi_target_pipeline=1', '-F', 'MODELNAME=HumDiv', '-F', 'UCSCDB=hg19', '-F', 'SNPFUNC=m', '-F', 'NOTIFYME=joe.wu.ca@gmail.com', '-F_ggi_batch_file=@' + self.db_path + 'dms/for_score/' + dms_gene_id + '_for_score.txt', '-D' , '-', 'http://genetics.bwh.harvard.edu/cgi-bin/ggi/ggi2.cgi']
        x = subprocess.run(polyphen_cmd, stdout=subprocess.PIPE)
        # get the session_id        
        s_start = str(x).find('polyphenweb2') + 13
        s_end = s_start + 40        
        session_id = str(x)[s_start:s_end]        
        return(session_id)
        
    def get_polyphen_from_sessionid(self, dms_gene_id, session_id):
        session_url = 'http://genetics.bwh.harvard.edu/ggi/pph2/' + session_id + '/1/pph2-short.txt'
        response = urllib.request.urlopen(session_url)
        r = response.read().decode('utf-8')
#         f = open(self.db_path + "polyphen/org/" + dms_gene_id + "_pph2-full.txt", "w")
#         f.write(r)
#         f.close()
        return(r)
    
    def retrieve_psipred_by_uniprotid(self,dms_gene_id,dummy = 0):
        if dummy == 1:
            return(0)
        #assume you have the psipred output as Uniprot_ID.ss2 file
        
        psipred_file = self.db_path + 'psipred/' + dms_gene_id + '.ss2'        
        cur_psipred_df = pd.read_csv(psipred_file, skiprows=[0, 1], header=None, sep='\s+')
        cur_psipred_df = cur_psipred_df.loc[:, [0, 1, 2]]
        cur_psipred_df.columns = ['aa_pos', 'aa_ref', 'aa_psipred']
            
        #create files for secondary structure regions
        sc_seq = ''.join(list(cur_psipred_df['aa_psipred']))
        psipred_seq_file =  open(self.db_path + 'psipred/dms_feature/' + dms_gene_id + '_psipred.csv','w')
        psipred_seq_str = 'aa_psipred,aa_pos,ss_end_pos\n'
        sc_start = ''
        cur_start_pos = 1
        for i in range(1,len(sc_seq)+1):
            cur_sc = sc_seq[i-1]
            if cur_sc != sc_start:
                pre_start_pos = cur_start_pos
                if i != 1:
                    pre_end_pos = i-1
                    psipred_seq_str += sc_start + ',' + str(pre_start_pos) + ',' + str(pre_end_pos) + '\n'                                        
                sc_start = cur_sc
                cur_start_pos = i
            if i == cur_psipred_df.shape[0]:
                psipred_seq_str += sc_start + ',' + str(cur_start_pos) + ',' + str(i) + '\n'
        pass
        psipred_seq_file.write(psipred_seq_str)      
        psipred_seq_file.close()
        
         
        return(0)
    
    def retrieve_polyphen_by_uniprotid(self,dms_gene_id,dummy = 0):
        if dummy == 1:
            return(0)
        try:
            polyphen_cmd = ['curl', '-F_ggi_project=PPHWeb2', '-F_ggi_origin=query', '-F', '_ggi_target_pipeline=1', '-F', 'MODELNAME=HumDiv', '-F', 'UCSCDB=hg19', '-F', 'SNPFUNC=m', '-F', 'NOTIFYME=joe.wu.ca@gmail.com', '-F_ggi_batch_file=@' + self.db_path + 'dms/for_score/' + dms_gene_id + '_for_score.txt', '-D' , '-', 'http://genetics.bwh.harvard.edu/cgi-bin/ggi/ggi2.cgi']
            x = subprocess.run(polyphen_cmd, stdout=subprocess.PIPE)
            # get the session_id        
            s_start = str(x).find('polyphenweb2') + 13
            s_end = s_start + 40        
            session_id = str(x)[s_start:s_end]    
            session_url = 'http://genetics.bwh.harvard.edu/ggi/pph2/' + session_id + '/1/pph2-short.txt'
        
            while True:
                try:
                    time.sleep(10)
                    response = urllib.request.urlopen(session_url)
                except:
                    continue
                break
            r = response.read().decode('utf-8')
            r_1 = '\n'.join(r.split('\n')[:-6])
            polyphen = pd.read_csv(StringIO(r_1),sep = '\t',usecols = ['acc       ','   pos','aa1','aa2',' pph2_prob'],low_memory = False)
            polyphen.columns = ['p_vid','aa_pos','aa_ref','aa_alt','polyphen_score']
            polyphen['p_vid'] = polyphen['p_vid'].str.strip() 
            polyphen['aa_ref'] = polyphen['aa_ref'].str.strip()
            polyphen['aa_alt'] = polyphen['aa_alt'].str.strip()
            polyphen.to_csv(self.db_path + 'polyphen/dms_feature/' + dms_gene_id + '_polyphen.csv',header = False,index = False)
            retrun(1)
        except:
            return(0)
    
    
    def retrieve_asa_by_uniprotid_dummy(self,dms_gene_id):
        return(0)
    
    def retrieve_polyphen_by_uniprotid_dummy(self,dms_gene_id):
        return(0)
    
    def retrieve_psipred_by_uniprotid_dummy(self,dms_gene_id):
        return(0)
    
    def retrieve_asa_by_uniprotid(self,dms_gene_id,dummy = 0):
        if dummy == 1:
            return(0)
        try:
            asa_logFile = self.db_path + 'pdb/prepare_asa_data.log'
            asa_url = 'http://www.ebi.ac.uk/pdbe/pisa/cgi-bin/interfaces.pisa?'
            xml_file = open(self.db_path + 'pdb/org/' + dms_gene_id + '_asa.xml', 'w')
            asa_file = open(self.db_path + 'pdb/org/' + dms_gene_id + '_asa.txt', 'w')            
            cur_gene_pdb = self.pdb_to_uniprot.loc[self.pdb_to_uniprot['SP_PRIMARY'] == dms_gene_id, :]
            if cur_gene_pdb.shape[0] > 0 :   
    #             try:     
                    pdb_lst = ','.join(cur_gene_pdb['PDB'].unique())
                    cur_asa_url = asa_url + pdb_lst
                    response = urllib.request.urlopen(cur_asa_url)
                    r = response.read().decode('utf-8')
                    xml_file.write(r)
                    tree = ET.fromstring(r)
                    for pdb_entry in tree.iter('pdb_entry'):
                        # print (pdb_entry[0].tag + ':' + pdb_entry[0].text)   #pdb_code
                        for interface in pdb_entry.iter("interface"):
                            # print (interface[0].tag + ':' + interface[0].text) #interface id 
                            for molecule in interface.iter("molecule"):
                                # print(molecule[1].tag +':' + molecule[1].text) #chain_id
                                for residue in molecule.iter("residue"):
                                    # print (residue[0].tag + ':' + residue[0].text + '|' + residue[1].tag + ':' + residue[1].text +'|' + residue[5].tag + ':' + residue[5].text)
                                    asa_file.write(pdb_entry[0].text + '\t' + interface[0].text + '\t' + molecule[1].text + '\t' + residue[0].text + '\t' + residue[1].text + '\t' + residue[2].text + '\t' + residue[5].text + '\n')
                    show_msg(asa_logFile, 1, dms_gene_id + ' pisa is processed.\n')                                        
                    asa_file.close() 
                    xml_file.close()  
                 
                    cur_asa_df = pd.read_csv(self.db_path + 'pdb/org/' + dms_gene_id + '_asa.txt', sep='\t', header=None, dtype={0: str})
                    cur_asa_df.columns = ['PDB', 'interface', 'CHAIN', 'ser_no', 'residue', 'aa_pos', 'asa']                        
                    cur_asa_df_groupby = cur_asa_df.groupby(['residue', 'aa_pos'])
                    cur_asa_df = cur_asa_df_groupby.agg({'asa':['mean', 'std', 'count']}).reset_index().sort_values(['aa_pos'])
                    cur_asa_df.columns = ['residue', 'aa_pos', 'asa_mean', 'asa_std', 'asa_count']    
                    cur_asa_df['aa_ref'] = cur_asa_df['residue'].apply(lambda x: self.dict_aa3_upper.get(x, np.nan))
                    cur_asa_df = cur_asa_df.loc[cur_asa_df['aa_ref'].notnull(), ]
                    cur_asa_df['p_vid'] = dms_gene_id
                    cur_asa_df.drop(['residue'],axis = 1,inplace = True)
                    cur_asa_df[['aa_pos','aa_ref','asa_mean','asa_std','asa_count','p_vid']].to_csv(self.db_path + 'pdb/dms_feature/' + dms_gene_id + '_asa.csv', index=False)
                    show_msg(asa_logFile, 1, dms_gene_id + ' asa file is saved to csv.\n')
                    return(1)
    #             except:
    #                 show_msg(asa_logFile, 1, dms_gene_id + ' process error: ' + traceback.format_exc() + '\n')
    #                 return(0)
            else:
                return(0)    
        except:
            return(0) 
        
    def prepare_dms_input(self,dms_gene_id,syn_ratio = 0.6,stop_ratio = 0.6,missense_ratio = 0.6):
        dms_input = pd.read_csv(self.db_path + 'dms/for_score/' + dms_gene_id + '_for_score.txt',sep = ' ')
        dms_input.columns = ['p_vid','aa_pos','aa_ref','aa_alt']
        dms_input['quality_score'] = np.random.random(dms_input.shape[0])*2000
        dms_input['num_replicates'] = 2
        dms_input['fitness_input'] = np.nan
        dms_input['fitness_input_sd'] = np.random.random(dms_input.shape[0])*0.2
        
        #syn
        syn_index = dms_input.loc[dms_input['aa_ref'] == dms_input['aa_alt'],:].index
        syn_left_index = syn_index[np.random.randint(0,len(syn_index),int(len(syn_index)*syn_ratio))]
        dms_input.loc[syn_left_index,'fitness_input'] = np.random.random(len(syn_left_index))*0.1 + 0.9
        
        #stop
        stop_index = dms_input.loc[dms_input['aa_alt'] == '*',:].index
        stop_left_index = stop_index[np.random.randint(0,len(stop_index),int(len(stop_index)*stop_ratio))]
        dms_input.loc[stop_left_index,'fitness_input'] = np.random.random(len(stop_left_index))*0.1
        
        #missen
        missense_index = dms_input.loc[dms_input['aa_ref'] != dms_input['aa_alt'],:].index
        missense_left_index = missense_index[np.random.randint(0,len(missense_index),int(len(missense_index)*missense_ratio))]
        dms_input.loc[missense_left_index,'fitness_input'] = np.random.random(len(missense_left_index))*2
                
        dms_input = dms_input.loc[~dms_input['fitness_input'].isnull(),:][['aa_ref','aa_pos','aa_alt','quality_score','num_replicates','fitness_input','fitness_input_sd']]
        dms_input.to_csv(self.db_path + 'dms/dms_input/' + dms_gene_id + '_input.txt',sep = '\t',index = False)
        with open(self.db_path + 'dms/dms_input/' + dms_gene_id + '.fasta','w') as f:
            f.write('>' + dms_gene_id + '\n')
            f.write(self.uniprot_seqdict.get(dms_gene_id,'')) 
               
    def prepare_dms_features(self,dms_gene_id,k,dummy = 0):
        try:
            stime = time.time()
#             port = 3306
#             db = mysql.connector.connect(host="localhost", port=port, user="root", passwd="Vv8006201285", database="alphame")
            dms_gene_aa = self.uniprot_seqdict[dms_gene_id]
            aa_len = len(dms_gene_aa)
            dms_gene_matrix = np.full((21, len(dms_gene_aa)), np.nan)                                                           
            dms_feature_df = pd.DataFrame(dms_gene_matrix, columns=range(1, len(dms_gene_aa) + 1), index=self.lst_aa_21)
            dms_feature_df['aa_alt'] = dms_feature_df.index
            dms_feature_df = pd.melt(dms_feature_df, ['aa_alt'])
            dms_feature_df = dms_feature_df.rename(columns={'variable': 'aa_pos', 'value': 'aa_ref'})        
            dms_feature_df['aa_ref'] = dms_feature_df['aa_pos'].apply(lambda x: list(dms_gene_aa)[x - 1])
            dms_feature_df['p_vid'] = dms_gene_id
            # dms_feature_df['uniprot_id'] = dms_gene_id 
            dms_feature_df['aa_pos'] = dms_feature_df['aa_pos'].astype(int)
            dms_feature_df[['p_vid', 'aa_pos', 'aa_ref', 'aa_alt']].to_csv(self.db_path + 'dms/for_score/' + dms_gene_id + '_for_score.txt', sep=' ', index=False, header=None)
            dms_feature_df['aa_len'] = aa_len
            
            feature_log = open(self.db_path + 'dms/features_log/' + dms_gene_id + '_feature_log.txt','w') 
            
            ####*************************************************************************************************************************************************************
            # pdb_asa (can be retrieved on the fly if doesn't exist)
            ####*************************************************************************************************************************************************************
            pdb_asa_file = self.db_path + 'pdb/dms_feature/' + dms_gene_id + '_asa.csv'
            pdb_asa_ready = 0        
            if not os.path.isfile(pdb_asa_file):
                if self.retrieve_asa_by_uniprotid(dms_gene_id,dummy) == 1:
                    pdb_asa_ready = 1
            else:
                pdb_asa_ready = 1
            if pdb_asa_ready == 1:            
                pdb_asa = pd.read_csv(self.db_path + 'pdb/dms_feature/' + dms_gene_id + '_asa.csv')
                pdb_asa.columns = ['aa_pos','aa_ref','asa_mean','asa_std','asa_count','p_vid']
                dms_feature_df = pd.merge(dms_feature_df,pdb_asa,how = 'left')
                print ("merge with pdb_asa: " + str(dms_feature_df.shape[0]))
                feature_log.write("merge with pdb_asa: " + str(dms_feature_df.shape[0]) + '\n')
            else:
                dms_feature_df['asa_mean'] = np.nan
                dms_feature_df['asa_std'] = np.nan
                dms_feature_df['asa_count'] = np.nan            
                print (dms_gene_id +" pdb asa file doesn't exist.")
                feature_log.write(dms_gene_id +" pdb asa file doesn't exist.\n")                        
            ####*************************************************************************************************************************************************************
            #psipred (can be retrieved on the fly if doesn't exist)
            ####*************************************************************************************************************************************************************
            psipred_file = self.db_path + 'psipred/dms_feature/' + dms_gene_id + '_psipred.csv'      
            psipred_ready = 0
            if not os.path.isfile(psipred_file):
                if self.retrieve_psipred_by_uniprotid(dms_gene_id,dummy) == 1:
                    psipre_ready = 1
            else:
                psipred_ready = 1
            if psipred_ready == 1:
                psipred = pd.read_csv(self.db_path + 'psipred/dms_feature/' + dms_gene_id + '_psipred.csv')
                psipred.columns = ['aa_psipred','aa_pos','ss_end_pos']
                psipred['aa_alt'] = '*'
                dms_feature_df = pd.merge(dms_feature_df,psipred,how = 'left')
                print ("merge with psipred: " + str(dms_feature_df.shape[0]))
                feature_log.write("merge with psipred: " + str(dms_feature_df.shape[0]) + '\n')
            else:
                dms_feature_df['aa_psipred'] = np.nan
                dms_feature_df['ss_end_pos'] = np.nan            
                print (dms_gene_id +" psipred file doesn't exist.")
                feature_log.write(dms_gene_id +" psipred file doesn't exist.\n")
            ####*************************************************************************************************************************************************************
            #pfam 
            ####*************************************************************************************************************************************************************
            pfam_file = self.db_path + 'pfam/dms_feature/' + dms_gene_id + '_pfam.tsv'  
            pfam_ready = 0 
            if os.path.isfile(pfam_file):
                pfam_ready = 1
            if pfam_ready == 1:
                pfam = pd.read_csv(pfam_file,header = None,sep = '\t')
                pfam.columns = ['p_vid','a_start','a_end','e_start','e_end','hmm_id','hmm_name','type','hmm_start','hmm_end','hmm_length','bit_score','e_value','clan']
                pfam = pfam[['p_vid','a_start','a_end','hmm_id']]        
                dms_feature_df['in_domain'] = 0        
                for index, pfam_row in pfam.iterrows():
                    dms_feature_df.loc[(dms_feature_df['aa_pos'] >= pfam_row['a_start']) & (dms_feature_df['aa_pos']<= pfam_row['a_end']),'in_domain'] = 1        
                pfam.columns = ['p_vid','aa_pos','pfam_end_pos','hmm_id']
                pfam['aa_alt'] = '*'
                dms_feature_df = pd.merge(dms_feature_df,pfam,how = 'left')
                print ("merge with pfam: " + str(dms_feature_df.shape[0]))
                feature_log.write("merge with pfam: " + str(dms_feature_df.shape[0]) + '\n')
            else:
                dms_feature_df['hmm_id'] = np.nan
                dms_feature_df['pfam_end_pos'] = np.nan
                dms_feature_df['in_domain'] = np.nan 
                print (dms_gene_id +" pfam file doesn't exist.")
                feature_log.write(dms_gene_id +" pfam file doesn't exist.\n")
            
            ####*************************************************************************************************************************************************************
            #gnomad 
            ####*************************************************************************************************************************************************************
            gnomad_file = self.db_path + 'gnomad/dms_feature/' + dms_gene_id + '_gnomad.txt'
            gnomad_ready = 0           
            if os.path.isfile(gnomad_file):
                gnomad_ready = 1
            if gnomad_ready == 1:
                gnomad = pd.read_csv(gnomad_file,header = None,sep = '\t')
                gnomad.columns = ['p_vid','aa_ref','aa_pos','aa_alt','gnomad_ac','gnomad_an','gnomad_af']
                dms_feature_df = pd.merge(dms_feature_df,gnomad,how = 'left')
                print ("merge with gnomad: " + str(dms_feature_df.shape[0]))
                feature_log.write("merge with gnomad: " + str(dms_feature_df.shape[0]))
            else:
                dms_feature_df['gnomad_ac'] = np.nan
                dms_feature_df['gnomad_an'] = np.nan
                dms_feature_df['gnomad_af'] = np.nan
                print (dms_gene_id +" gnomad file doesn't exist.")
                feature_log.write(dms_gene_id +" gnomad file doesn't exist.\n")
                            
            ####*************************************************************************************************************************************************************    
            #polyphen (can be retrieved on the fly if doesn't exist)
            ####*************************************************************************************************************************************************************
            polyphen_file = self.db_path + 'polyphen/dms_feature/' + dms_gene_id + '_polyphen.csv'   
            polyphen_ready = 0
            if not (os.path.isfile(polyphen_file) & os.stat(polyphen_file).st_size != 0):
                if self.retrieve_polyphen_by_uniprotid(dms_gene_id,dummy) == 1:
                    polyphen_ready = 1
            else:
                polyphen_ready = 1    
            
            if polyphen_ready == 1:    
                polyphen = pd.read_csv(self.db_path + 'polyphen/dms_feature/' + dms_gene_id + '_polyphen.csv',header = None)
                polyphen.columns = ['p_vid','aa_pos','aa_ref','aa_alt','polyphen_score'] 
                dms_feature_df = pd.merge(dms_feature_df,polyphen,how = 'left')
                print ("merge with polyphen: " + str(dms_feature_df.shape[0]))
                feature_log.write("merge with polyphen: " + str(dms_feature_df.shape[0])  + '\n')       
            else: 
                dms_feature_df['polyphen_score'] = np.nan           
                print (dms_gene_id + "'s polyphen data has been retrieved on the fly.")
                feature_log.write(dms_gene_id + "'s polyphen data has been retrieved on the fly.\n")
    
            
            ####*************************************************************************************************************************************************************
            #determine which ensembl id to use 
            ####************************************************************************************************************************************************************        
            ensembl_ids = self.uniprot2ensembl_dict[dms_gene_id].split(";")
            ensembl_ok_ids = []
            sift_ensembl_id = None
            provean_ensembl_id = None
            sift_ready = 0
            provean_ready = 0
            if len(ensembl_ids) == 1:
                ensembl_ok_ids = ensembl_ids
            else:
                #compare seq to determine which ensembl id to use 
                for e_id in ensembl_ids:
                    if self.uniprot_seqdict.get(dms_gene_id,'unknown_uniprot_id') == self.ensembl_seqdict.get(e_id,'unknonw_ensembl_id'):
                        ensembl_ok_ids.append(e_id)
            print ("uniport -> ensembl : " + dms_gene_id + " -> " + str(ensembl_ok_ids))
            feature_log.write("uniport -> ensembl : " + dms_gene_id + " -> " + str(ensembl_ok_ids) + '\n')
                    
            for e_id in ensembl_ok_ids:
                sift_file = self.db_path + 'sift/dms_feature/' + e_id + '_sift.tsv'
                if os.path.isfile(sift_file):
                    sift_ready = 1
                    sift_ensembl_id = e_id 
                    break
            
            for e_id in ensembl_ok_ids:
                provean_file = self.db_path + 'provean/dms_feature/' + e_id + '_provean.tsv'
                if os.path.isfile(provean_file):
                    provean_ready = 1
                    provean_ensembl_id = e_id 
                    break
            
            
            ####*************************************************************************************************************************************************************
            #sift
            ####************************************************************************************************************************************************************
            if sift_ready == 1:
                sift = pd.read_csv(sift_file,header = None,sep = '\t')    
                sift.columns = ['ensp_id','aa_pos','A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','median','n_seq']      
                sift['p_vid'] = dms_gene_id
                sift.drop(['median','n_seq'],axis = 1,inplace = True)
                sift_melt = pd.melt(sift,id_vars=['p_vid','aa_pos'],value_vars = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'])
                sift_melt.columns = ['p_vid','aa_pos','aa_alt','sift_score']
                dms_feature_df = pd.merge(dms_feature_df,sift_melt,how = 'left')            
                print ("merge with sift: " + str(dms_feature_df.shape[0]))
                feature_log.write("merge with sift: " + str(dms_feature_df.shape[0]) + '\n')
            else:                 
          
                print (dms_gene_id + " sift ensembl file doesn't exist.")
                feature_log.write(dms_gene_id + " sift ensembl file doesn't exist.\n")
            
            ####*************************************************************************************************************************************************************
            #provean
            ####*************************************************************************************************************************************************************
            if provean_ready == 1:
                provean = pd.read_csv(provean_file,header = None,sep = '\t')    
                provean.columns = ['ensp_id','aa_pos','A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','Del']
                provean['p_vid'] = dms_gene_id
                provean_melt = pd.melt(provean,id_vars=['p_vid','aa_pos'],value_vars = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','Del'])
                provean_melt.columns = ['p_vid','aa_pos','aa_alt','provean_score']
                provean_melt.loc[provean_melt['aa_alt'] == 'Del','aa_alt'] = '*'
                dms_feature_df = pd.merge(dms_feature_df,provean_melt,how = 'left')            
                print ("merge with provean: " + str(dms_feature_df.shape[0]))
                feature_log.write("merge with provean: " + str(dms_feature_df.shape[0]) + '\n')        
            else:
                print (dms_gene_id + " provean ensembl file doesn't exist.")
                feature_log.write(dms_gene_id + " provean ensembl file doesn't exist.\n")
            
            ####*************************************************************************************************************************************************************
            #provean & sift manual file
            ####*************************************************************************************************************************************************************
            provean_sift_ready = 0
            provean_sift_file = self.db_path + 'provean/org/' + dms_gene_id + '_provean.tsv'
            if os.path.isfile(provean_sift_file):
                provean_sift_ready = 1
            if (provean_ready == 0) & (sift_ready == 0): 
                if provean_sift_ready ==1:
                    provean_sift_score = pd.read_csv(self.db_path + 'provean/org/' + dms_gene_id + '_provean.tsv', sep='\t')[['PROTEIN_ID', 'POSITION', 'RESIDUE_REF', 'RESIDUE_ALT', 'SCORE', 'SCORE.1']]
                    provean_sift_score.columns = ['p_vid', 'aa_pos', 'aa_ref', 'aa_alt', 'provean_score', 'sift_score']
                    provean_sift_score = provean_sift_score.drop_duplicates()
                    dms_feature_df = pd.merge(dms_feature_df,provean_sift_score,how = 'left')            
                    print ("merge with provean_sift: " + str(dms_feature_df.shape[0]))
                    feature_log.write("merge with provean_sift: " + str(dms_feature_df.shape[0]) + '\n')
                else:
                    dms_feature_df['provean_score'] = np.nan 
                    dms_feature_df['sift_score'] = np.nan  
                    print (dms_gene_id + " provean_sift uniprot file doesn't exist.")
                    feature_log.write(dms_gene_id + " provean_sift uniprot file doesn't exist.\n")
            
#             ####***************************************************************************************************************************************************************
#             #### aa_ref and aa_alt AA properties
#             ####***************************************************************************************************************************************************************           
#             aa_properties_features = self.aa_properties.columns        
#             aa_properties_ref_features = [x + '_ref' for x in aa_properties_features]
#             aa_properties_alt_features = [x + '_alt' for x in aa_properties_features]   
#             aa_properties_ref = self.aa_properties.copy()
#             aa_properties_ref.columns = aa_properties_ref_features
#             aa_properties_alt = self.aa_properties.copy()
#             aa_properties_alt.columns = aa_properties_alt_features                
#             dms_feature_df = pd.merge(dms_feature_df, aa_properties_ref, how='left')
#             dms_feature_df = pd.merge(dms_feature_df, aa_properties_alt, how='left')
#              
#             print ("merge with aa properties: " + str(dms_feature_df.shape[0]))
#             feature_log.write("merge with aa properties: " + str(dms_feature_df.shape[0]) + '\n')
#             ####***************************************************************************************************************************************************************
#             #### flanking kmer AA column and properties  
#             ####***************************************************************************************************************************************************************
#             for i in range(1, k + 1):    
#                 aa_left = 'aa_ref_' + str(i) + '_l'
#                 aa_right = 'aa_ref_' + str(i) + '_r'
#                 dms_feature_df[aa_left] = dms_feature_df[['p_vid', 'aa_pos']].apply(lambda x: seq_dict[x['p_vid']][max(0, (x['aa_pos'] - i - 1)):max(0, (x['aa_pos'] - i))], axis=1)
#                 dms_feature_df[aa_right] = dms_feature_df[['p_vid', 'aa_pos']].apply(lambda x: seq_dict[x['p_vid']][(x['aa_pos'] + i - 1):(x['aa_pos'] + i)], axis=1)
#                 aa_properties_ref_kmer_features = [x + '_ref_' + str(i) + '_l' for x in aa_properties_features]
#                 aa_properties_ref_kmer = self.aa_properties.copy()
#                 aa_properties_ref_kmer.columns = aa_properties_ref_kmer_features
#                 dms_feature_df = pd.merge(dms_feature_df, aa_properties_ref_kmer, how='left')
#                 aa_properties_ref_kmer_features = [x + '_ref_' + str(i) + '_r' for x in aa_properties_features]
#                 aa_properties_ref_kmer = self.aa_properties.copy()
#                 aa_properties_ref_kmer.columns = aa_properties_ref_kmer_features
#                 dms_feature_df = pd.merge(dms_feature_df, aa_properties_ref_kmer, how='left')
#             print ("merge with kmer properties: " + str(dms_feature_df.shape[0]))
#             feature_log.write("merge with aa properties: " + str(dms_feature_df.shape[0]) + '\n')
#             
#             ####***************************************************************************************************************************************************************
#             #### merge with the blosum properties
#             ####***************************************************************************************************************************************************************
#             df_blosums = self.df_blosums          
#             dms_feature_df = pd.merge(dms_feature_df, df_blosums, how='left')
#             print ("merge with blosums: " + str(dms_feature_df.shape[0]))
#             feature_log.write("merge with blosums: " + str(dms_feature_df.shape[0]) + '\n')
#             ####***************************************************************************************************************************************************************
#             #### merge with the funsum properties
#             ####***************************************************************************************************************************************************************
#             for funsum_key in self.dict_sums['funsum'].keys():
#                 dms_feature_df = pd.merge(dms_feature_df, self.dict_sums['funsum'][funsum_key], how='left')
#             print ("merge with funsums: " + str(dms_feature_df.shape[0]))
#             feature_log.write("merge with funsums: " + str(dms_feature_df.shape[0]) + '\n')   
#             ####*************************************************************************************************************************************************************
#             #### Encode name features
#             ####*************************************************************************************************************************************************************        
#             dms_feature_df['aa_ref_encode'] = dms_feature_df['aa_ref'].apply(lambda x: self.aa_encode_notnull(x))
#             dms_feature_df['aa_alt_encode'] = dms_feature_df['aa_alt'].apply(lambda x: self.aa_encode_notnull(x))
#     
#             for i in range(1, k + 1):    
#                 aa_left = 'aa_ref_' + str(i) + '_l'
#                 aa_right = 'aa_ref_' + str(i) + '_r'
#                 dms_feature_df[aa_left + '_encode'] = dms_feature_df[aa_left].apply(lambda x: self.aa_encode_notnull(x))
#                 dms_feature_df[aa_right + '_encode'] = dms_feature_df[aa_right].apply(lambda x:  self.aa_encode_notnull(x))
#             print ("encode name features: " + str(dms_feature_df.shape[0]))
#             feature_log.write("encode name features: " + str(dms_feature_df.shape[0]) + '\n')
            
            with open(self.db_path + 'dms/supported_uniprot_ids.txt','a') as f:
                f.write(dms_gene_id + '\t' + str(pdb_asa_ready) + '\t' + str(psipred_ready) + '\t' + str(pfam_ready) + '\t' + str(gnomad_ready) + '\t' + str(polyphen_ready) + '\t' + str(sift_ready) + '\t' + str(provean_ready) + '\t' + str(provean_sift_ready) + '\n')
                    
            dms_feature_df.to_csv(self.db_path + 'dms/features/' + dms_gene_id + '_features.csv')     
            etime = time.time()        
            print(dms_gene_id +" feature processing took %g seconds\n" % (etime - stime))
            return(1)
        except:
            with open(self.db_path + 'dms/feature_error.txt','a') as f:
                f.write(dms_gene_id +  ' raise error:\n' + traceback.format_exc() + '\n')
            return(0)       
             
    def prepare_dms_features_sql(self,dms_gene_id,k):
        stime = time.time()
        port = 3306
        db = mysql.connector.connect(host="localhost", port=port, user="root", passwd="Vv8006201285", database="alphame")
        dms_gene_aa = self.uniprot_seqdict[dms_gene_id]
        aa_len = len(dms_gene_aa)
        dms_gene_matrix = np.full((21, len(dms_gene_aa)), np.nan)                                                           
        dms_feature_df = pd.DataFrame(dms_gene_matrix, columns=range(1, len(dms_gene_aa) + 1), index=self.lst_aa_21)
        dms_feature_df['aa_alt'] = dms_feature_df.index
        dms_feature_df = pd.melt(dms_feature_df, ['aa_alt'])
        dms_feature_df = dms_feature_df.rename(columns={'variable': 'aa_pos', 'value': 'aa_ref'})        
        dms_feature_df['aa_ref'] = dms_feature_df['aa_pos'].apply(lambda x: list(dms_gene_aa)[x - 1])
        dms_feature_df['p_vid'] = dms_gene_id
        # dms_feature_df['uniprot_id'] = dms_gene_id 
        dms_feature_df['aa_pos'] = dms_feature_df['aa_pos'].astype(int)
        dms_feature_df[['p_vid', 'aa_pos', 'aa_ref', 'aa_alt']].to_csv(self.db_path + 'dms/features/' + dms_gene_id + '_for_score.txt', sep=' ', index=False, header=None)
        dms_feature_df['aa_len'] = aa_len
        ####*************************************************************************************************************************************************************
        # pdb_asa
        ####*************************************************************************************************************************************************************
        sql = "select * from pdb_asa where p_vid = " + "'"  + dms_gene_id + "';"
        pdb_asa = pd.read_sql(sql,db)
        dms_feature_df = pd.merge(dms_feature_df,pdb_asa,how = 'left')
        print ("merge with pdb_asa: " + str(dms_feature_df.shape[0]))
        
        ####*************************************************************************************************************************************************************
        #psipred
        ####*************************************************************************************************************************************************************
        sql = "select * from psipred_region where p_vid = " + "'"  + dms_gene_id + "';"
        psipred = pd.read_sql(sql,db)
        psipred['aa_alt'] = '*'
        dms_feature_df = pd.merge(dms_feature_df,psipred,how = 'left')
        print ("merge with psipred: " + str(dms_feature_df.shape[0]))
        
        ####*************************************************************************************************************************************************************
        #pfam
        ####*************************************************************************************************************************************************************
        sql = "select * from pfam where p_vid = " + "'"  + dms_gene_id + "';"
        pfam = pd.read_sql(sql,db)
        pfam = pfam[['p_vid','hmm_id', 'a_start', 'a_end']]
        dms_feature_df['in_domain'] = 0
        
        for index, pfam_row in pfam.iterrows():
            dms_feature_df.loc[(dms_feature_df['aa_pos'] >= pfam_row['a_start']) & (dms_feature_df['aa_pos']<= pfam_row['a_end']),'in_domain'] = 1        
        pfam.columns = ['p_vid','hmm_id', 'aa_pos', 'pfam_end_pos']
        pfam['aa_alt'] = '*'
        dms_feature_df = pd.merge(dms_feature_df,pfam,how = 'left')
        print ("merge with pfam: " + str(dms_feature_df.shape[0]))
        
        ####*************************************************************************************************************************************************************
        #gnomad
        ####*************************************************************************************************************************************************************
        sql = "select * from gnomad_aa where p_vid = " + "'"  + dms_gene_id + "';"
        gnomad = pd.read_sql(sql,db)
        dms_feature_df = pd.merge(dms_feature_df,gnomad,how = 'left')
        print ("merge with gnomad: " + str(dms_feature_df.shape[0]))
                
        ####*************************************************************************************************************************************************************    
        #polyphen
        ####*************************************************************************************************************************************************************
        sql = "select * from polyphen where p_vid = " + "'"  + dms_gene_id + "';"
        polyphen = pd.read_sql(sql,db)
        polyphen.rename(columns = { 'pph2_prob':'polyphen_score'},inplace = True)
        
        if polyphen.shape[0] == 0:
            polyphen_cmd = ['curl', '-F_ggi_project=PPHWeb2', '-F_ggi_origin=query', '-F', '_ggi_target_pipeline=1', '-F', 'MODELNAME=HumDiv', '-F', 'UCSCDB=hg19', '-F', 'SNPFUNC=m', '-F', 'NOTIFYME=joe.wu.ca@gmail.com', '-F_ggi_batch_file=@' + self.db_path + 'dms/for_score/' + dms_gene_id + '_for_score.txt', '-D' , '-', 'http://genetics.bwh.harvard.edu/cgi-bin/ggi/ggi2.cgi']
            x = subprocess.run(polyphen_cmd, stdout=subprocess.PIPE)
            # get the session_id        
            s_start = str(x).find('polyphenweb2') + 13
            s_end = s_start + 40        
            session_id = str(x)[s_start:s_end]    
            session_url = 'http://genetics.bwh.harvard.edu/ggi/pph2/' + session_id + '/1/pph2-short.txt'
        
            while True:
                try:
                    time.sleep(10)
                    response = urllib.request.urlopen(session_url)
                except:
                    continue
                break
            r = response.read().decode('utf-8')
            r_1 = '\n'.join(r.split('\n')[:-6])
            polyphen = pd.read_csv(StringIO(r_1),sep = '\t',usecols = ['acc       ','   pos','aa1','aa2',' pph2_prob'],low_memory = False)
            polyphen.columns = ['p_vid','aa_pos','aa_ref','aa_alt','polyphen_score']
            polyphen['p_vid'] = polyphen['p_vid'].str.strip() 
            polyphen['aa_ref'] = polyphen['aa_ref'].str.strip()
            polyphen['aa_alt'] = polyphen['aa_alt'].str.strip()
            polyphen.to_csv(self.db_path + 'polyphen/org/' + dms_gene_id + '_polyphen.csv',header = False,index = False)
        dms_feature_df = pd.merge(dms_feature_df,polyphen,how = 'left')
        print ("merge with polyphen: " + str(dms_feature_df.shape[0]))
        ####*************************************************************************************************************************************************************
        #sift
        ####*************************************************************************************************************************************************************
        ensembl_ids = self.uniprot2ensembl_dict[dms_gene_id].split(";")
        ensembl_ids = ["'" + x + "'" for x in ensembl_ids]
        ensembl_ids = ",".join(ensembl_ids)
        
        sql = "select * from sift where protein_id in (" + ensembl_ids +  ");"
        sift = pd.read_sql(sql,db)
        sift['p_vid'] = dms_gene_id
        sift.drop(['median'],axis = 1)
        sift_melt = pd.melt(sift,id_vars=['protein_id','p_vid','aa_pos'],value_vars = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'])
        sift_melt.columns = ['ensp_id','p_vid','aa_pos','aa_alt','sift_score']
        dms_feature_df = pd.merge(dms_feature_df,sift_melt,how = 'left')
        print ("merge with sift: " + str(dms_feature_df.shape[0]))
        
        ####*************************************************************************************************************************************************************
        #provean
        ####*************************************************************************************************************************************************************
        sql = "select * from provean where protein_id in (" + ensembl_ids +  ");"
        provean = pd.read_sql(sql,db)
        provean['p_vid'] = dms_gene_id
        provean_melt = pd.melt(provean,id_vars=['protein_id','p_vid','aa_pos'],value_vars = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','Del'])
        provean_melt.columns = ['ensp_id','p_vid','aa_pos','aa_alt','provean_score']
        provean_melt.loc[provean_melt['aa_alt'] == 'Del','aa_alt'] = '*'
        dms_feature_df = pd.merge(dms_feature_df,provean_melt,how = 'left') 
        print ("merge with provean: " + str(dms_feature_df.shape[0]))
        
        ####***************************************************************************************************************************************************************
        #### aa_ref and aa_alt AA properties
        ####***************************************************************************************************************************************************************           
        aa_properties_features = self.aa_properties.columns        
        aa_properties_ref_features = [x + '_ref' for x in aa_properties_features]
        aa_properties_alt_features = [x + '_alt' for x in aa_properties_features]   
        aa_properties_ref = self.aa_properties.copy()
        aa_properties_ref.columns = aa_properties_ref_features
        aa_properties_alt = self.aa_properties.copy()
        aa_properties_alt.columns = aa_properties_alt_features                
        dms_feature_df = pd.merge(dms_feature_df, aa_properties_ref, how='left')
        dms_feature_df = pd.merge(dms_feature_df, aa_properties_alt, how='left')
         
        print ("merge with aa properties: " + str(dms_feature_df.shape[0]))
 
        ####***************************************************************************************************************************************************************
        #### flanking kmer AA column and properties  
        ####***************************************************************************************************************************************************************
        for i in range(1, k + 1):    
            aa_left = 'aa_ref_' + str(i) + '_l'
            aa_right = 'aa_ref_' + str(i) + '_r'
            dms_feature_df[aa_left] = dms_feature_df[['p_vid', 'aa_pos']].apply(lambda x: seq_dict[x['p_vid']][max(0, (x['aa_pos'] - i - 1)):max(0, (x['aa_pos'] - i))], axis=1)
            dms_feature_df[aa_right] = dms_feature_df[['p_vid', 'aa_pos']].apply(lambda x: seq_dict[x['p_vid']][(x['aa_pos'] + i - 1):(x['aa_pos'] + i)], axis=1)
            aa_properties_ref_kmer_features = [x + '_ref_' + str(i) + '_l' for x in aa_properties_features]
            aa_properties_ref_kmer = self.aa_properties.copy()
            aa_properties_ref_kmer.columns = aa_properties_ref_kmer_features
            dms_feature_df = pd.merge(dms_feature_df, aa_properties_ref_kmer, how='left')
            aa_properties_ref_kmer_features = [x + '_ref_' + str(i) + '_r' for x in aa_properties_features]
            aa_properties_ref_kmer = self.aa_properties.copy()
            aa_properties_ref_kmer.columns = aa_properties_ref_kmer_features
            dms_feature_df = pd.merge(dms_feature_df, aa_properties_ref_kmer, how='left')
        print ("merge with kmer properties: " + str(dms_feature_df.shape[0]))
        
        ####***************************************************************************************************************************************************************
        #### merge with the blosum properties
        ####***************************************************************************************************************************************************************
        df_blosums = self.df_blosums          
        dms_feature_df = pd.merge(dms_feature_df, df_blosums, how='left')
        print ("merge with blosums: " + str(dms_feature_df.shape[0]))
        ####***************************************************************************************************************************************************************
        #### merge with the funsum properties
        ####***************************************************************************************************************************************************************
        for funsum_key in self.dict_sums['funsum'].keys():
            dms_feature_df = pd.merge(dms_feature_df, self.dict_sums['funsum'][funsum_key], how='left')
        print ("merge with funsums: " + str(dms_feature_df.shape[0]))
                
        ####*************************************************************************************************************************************************************
        #### Encode name features
        ####*************************************************************************************************************************************************************        
        dms_feature_df['aa_ref_encode'] = dms_feature_df['aa_ref'].apply(lambda x: self.aa_encode_notnull(x))
        dms_feature_df['aa_alt_encode'] = dms_feature_df['aa_alt'].apply(lambda x: self.aa_encode_notnull(x))

        for i in range(1, k + 1):    
            aa_left = 'aa_ref_' + str(i) + '_l'
            aa_right = 'aa_ref_' + str(i) + '_r'
            dms_feature_df[aa_left + '_encode'] = dms_feature_df[aa_left].apply(lambda x: self.aa_encode_notnull(x))
            dms_feature_df[aa_right + '_encode'] = dms_feature_df[aa_right].apply(lambda x:  self.aa_encode_notnull(x))
        
        dms_feature_df.to_csv(self.db_path + 'dms/features/' + dms_gene_id + '_features.csv')     
        etime = time.time()        
        print(dms_gene_id +" feature processing took %g seconds\n" % (etime - stime))
          
        return (dms_feature_df)

    def prepare_dms_features_backup(self, dms_gene_id):            
        dms_gene_aa = self.dms_seqdict[dms_gene_id]
        dms_gene_matrix = np.full((21, len(dms_gene_aa)), np.nan)                                                           
        dms_feature_df = pd.DataFrame(dms_gene_matrix, columns=range(1, len(dms_gene_aa) + 1), index=self.lst_aa_21)
        dms_feature_df['aa_alt'] = dms_feature_df.index
        dms_feature_df = pd.melt(dms_feature_df, ['aa_alt'])
        dms_feature_df = dms_feature_df.rename(columns={'variable': 'aa_pos', 'value': 'aa_ref'})        
        dms_feature_df['aa_ref'] = dms_feature_df['aa_pos'].apply(lambda x: list(dms_gene_aa)[x - 1])
        dms_feature_df['p_vid'] = dms_gene_id
        dms_feature_df['uniprot_id'] = dms_gene_id        
        dms_feature_df[['p_vid', 'aa_pos', 'aa_ref', 'aa_alt']].to_csv(self.db_path + 'dms/features/' + dms_gene_id + '_for_score.txt', sep=' ', index=False, header=None)                
        dms_feature_df = self.variants_process(dms_gene_id, dms_feature_df, self.dms_seqdict, self.flanking_k, 'uniprot_id')
        dms_feature_df.to_csv(self.db_path + 'dms/features/' + dms_gene_id + '_features.csv')
        return (dms_feature_df)
    
    def prepare_dms_normalization(self, dms_gene, dms_gene_id, proper_num_replicates=4, num_replicates=2): 
        dms_landscape_file = self.db_path + 'dms/' + dms_gene + '_FOLDCHANGE.txt'
        dms_fitness_df = pd.read_csv(dms_landscape_file, sep='\t')[['wt', 'mut', 'pos', 'foldchange', 'sdFoldchange', 'averageNonselect']]
        dms_fitness_df.columns = ['aa_ref', 'aa_alt', 'aa_pos', 'foldchange', 'foldchange_sd', 'quality_score']
         
        dms_fitness_df['num_replicates'] = num_replicates
        dms_fitness_df['pseudo_count'] = proper_num_replicates - dms_fitness_df['num_replicates']
        dms_fitness_df['foldchange_se'] = dms_fitness_df['foldchange_sd'] / np.sqrt(dms_fitness_df['num_replicates'])

        dms_fitness_df.loc[dms_fitness_df.aa_ref == dms_fitness_df.aa_alt, 'annotation'] = 'SYN'
        dms_fitness_df.loc[dms_fitness_df.aa_ref != dms_fitness_df.aa_alt, 'annotation'] = 'NONSYN'
        dms_fitness_df.loc[dms_fitness_df.aa_alt == '_', 'annotation'] = 'STOP'             
        dms_fitness_df['foldchange_log'] = np.log(dms_fitness_df['foldchange'])
        dms_fitness_df['foldchange_log_se'] = dms_fitness_df['foldchange_se'] / dms_fitness_df['foldchange']
         
        syn_foldchange = np.array(dms_fitness_df.loc[(dms_fitness_df['annotation'] == 'SYN') & dms_fitness_df['foldchange_log'].notnull(), 'foldchange_log'])
        fig = plt.figure(figsize=(16, 10))
        sns.distplot(syn_foldchange, color='red')
        fig.tight_layout()
        plt.savefig(self.db_path + 'dms/' + dms_gene + '_' + dms_gene_id + '_syn_distribution.png')
    
        syn_pdf = stats.gaussian_kde(syn_foldchange)
        syn_density = syn_pdf(syn_foldchange)
        syn_mode = syn_foldchange[syn_density == max(syn_density)][0]
          
        stop_foldchange = np.array(dms_fitness_df.loc[(dms_fitness_df['annotation'] == 'STOP') & dms_fitness_df['foldchange_log'].notnull(), 'foldchange_log'])
        fig = plt.figure(figsize=(16, 10))
        sns.distplot(stop_foldchange, color='blue')
        fig.tight_layout()
        plt.savefig(self.db_path + 'dms/' + dms_gene + '_' + dms_gene_id + '_stop_distribution.png')
        stop_pdf = stats.gaussian_kde(stop_foldchange)
        stop_density = stop_pdf(stop_foldchange)
        stop_mode = stop_foldchange[stop_density == max(stop_density)][0]
         
        dms_fitness_df['fitness'] = (dms_fitness_df['foldchange_log'] - stop_mode) / (syn_mode - stop_mode)
        dms_fitness_df['fitness_se'] = dms_fitness_df['foldchange_log_se'] / (syn_mode - stop_mode)
        dms_fitness_df['fitness_sd'] = dms_fitness_df['fitness_se'] * np.sqrt(dms_fitness_df['num_replicates'])
                
        dms_fitness_df['fitness_reverse'] = dms_fitness_df['fitness']              
        dms_fitness_df.loc[dms_fitness_df['fitness_reverse'] > 1 , 'fitness_reverse'] = 2 - dms_fitness_df.loc[dms_fitness_df['fitness_reverse'] > 1 , 'fitness_reverse'] 
        
        dms_fitness_df.to_csv(self.db_path + 'dms/' + dms_gene + '_' + dms_gene_id + '_normalized.csv')
        return(dms_fitness_df)
                
    def prepare_dms_singlegene_backup(self, dms_gene, dms_gene_id, dms_file, fitness_only, sample_size=2, quality_cutoff=100):            
        dms_gene_aa = self.dms_seqdict[dms_gene_id]
        dms_gene_matrix = np.full((21, len(dms_gene_aa)), np.nan)                                                           
        dms_gene_df = pd.DataFrame(dms_gene_matrix, columns=range(1, len(dms_gene_aa) + 1), index=self.lst_aa_21)
        dms_gene_df['aa_alt'] = dms_gene_df.index
        dms_gene_df = pd.melt(dms_gene_df, ['aa_alt'])
        dms_gene_df = dms_gene_df.rename(columns={'variable': 'aa_pos', 'value': 'aa_ref'})        
        dms_gene_df['aa_ref'] = dms_gene_df['aa_pos'].apply(lambda x: list(dms_gene_aa)[x - 1])
        
        # read fitness data and merge
        dms_gene_fitness = pd.read_csv(dms_file, sep='\t')[['wt', 'mut', 'pos', 'foldchange', 'sdFoldchange', 'averageNonselect', 'sdFitnessScore', 'fitnessScore']]
        dms_gene_fitness.columns = ['aa_ref', 'aa_alt', 'aa_pos', 'foldchange', 'foldchange_sd', 'quality_score', 'fitness_input_sd', 'fitness_input']

        dms_gene_fitness['foldchange_se'] = dms_gene_fitness['foldchange_sd'] / math.sqrt(sample_size)
        dms_gene_fitness['fitness_input_se'] = dms_gene_fitness['fitness_input_sd'] / math.sqrt(sample_size)        
        dms_gene_fitness['fitness_rse_input'] = abs(dms_gene_fitness['fitness_input_se'] / dms_gene_fitness['fitness_input'])
                
        # dms_gene_fitness.rename(columns={'pos': 'aa_pos', 'wt': 'aa_ref','mut': 'aa_alt'}, inplace=True)
        dms_gene_df = pd.merge(dms_gene_df, dms_gene_fitness, how='left')
        dms_gene_df.loc[dms_gene_df.aa_ref == dms_gene_df.aa_alt, 'annotation'] = 'SYN'
        dms_gene_df.loc[dms_gene_df.aa_ref != dms_gene_df.aa_alt, 'annotation'] = 'NONSYN'
        dms_gene_df.loc[dms_gene_df.aa_alt == '_', 'annotation'] = 'STOP'             
        # normalize the fitness (normalize with the mode of the synonymous, reverse using 1/x)
        dms_gene_df['foldchange_log'] = np.log(dms_gene_df['foldchange'])
        dms_gene_df['foldchange_log_se'] = dms_gene_df['foldchange_se'] / dms_gene_df['foldchange']
        syn_foldchange = np.array(dms_gene_df.loc[(dms_gene_df['annotation'] == 'SYN') & dms_gene_df['foldchange_log'].notnull(), 'foldchange_log'])
        syn_pdf = stats.gaussian_kde(syn_foldchange)
        syn_density = syn_pdf(syn_foldchange)
        syn_mode = syn_foldchange[syn_density == max(syn_density)][0]
         
        stop_foldchange = np.array(dms_gene_df.loc[(dms_gene_df['annotation'] == 'STOP') & dms_gene_df['foldchange_log'].notnull(), 'foldchange_log'])
        stop_pdf = stats.gaussian_kde(stop_foldchange)
        stop_density = stop_pdf(stop_foldchange)
        stop_mode = stop_foldchange[stop_density == max(stop_density)][0]
     
        # dms_gene_df['fitness'] = (dms_gene_df['foldchange_log'] - syn_mode) / (stop_mode - syn_mode)
        dms_gene_df['fitness'] = (dms_gene_df['foldchange_log'] - stop_mode) / (syn_mode - stop_mode)
        dms_gene_df['fitness_se'] = dms_gene_df['foldchange_log_se'] / (syn_mode - stop_mode)
        dms_gene_df['fitness_sd'] = dms_gene_df['fitness_se'] * math.sqrt(sample_size)
        dms_gene_df['fitness_rse'] = abs(dms_gene_df['fitness_se'] / dms_gene_df['fitness'])

        # reverse fitness
        dms_gene_df['fitness_reverse'] = dms_gene_df['fitness']
        # dms_gene_df.loc[dms_gene_df['fitness_reverse'] < 0 ,'fitness_reverse'] =  0 -dms_gene_df.loc[dms_gene_df['fitness_reverse'] < 0 ,'fitness_reverse']                  
        dms_gene_df.loc[dms_gene_df['fitness_reverse'] > 1 , 'fitness_reverse'] = 2 - dms_gene_df.loc[dms_gene_df['fitness_reverse'] > 1 , 'fitness_reverse'] 
        dms_gene_df['p_vid'] = dms_gene_id
        dms_gene_df['gene_name'] = dms_gene

        dms_gene_df = dms_gene_df.loc[dms_gene_df.annotation == 'NONSYN', :]  
         
        if fitness_only:
            dms_gene_df.to_csv(self.db_path + 'dms/' + dms_gene + '_data_fitnessonly.csv')
        else:
            dms_gene_df = self.variants_process(dms_gene, dms_gene_df, self.dms_seqdict, self.flanking_k, 'uniprot_id')
            dms_gene_df.to_csv(self.db_path + 'dms/' + dms_gene + '_data.csv')
        return (dms_gene_df)
        
    def plot_aa_matrix(self, matrix_df, value_column, value_ste_column, value_name, lst_aa=['I', 'L', 'V', 'M', 'F', 'W', 'H', 'K', 'T', 'A', 'G', 'P', 'Y', 'D', 'E', 'R', 'S', 'C', 'N', 'Q'] , aa_colors=['r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g'], vmin=np.nan, vmax=np.nan, vcenter=np.nan, annotation=False, show_ste=0, nonsyn_only=True):                
        lst_aa_reverse = list(reversed(np.array(lst_aa)))
        value_matrix = pd.DataFrame(np.zeros([len(lst_aa), len(lst_aa)]), columns=lst_aa, index=lst_aa_reverse)
        value_matrix[value_matrix == 0] = np.nan
        value_ste_matrix = pd.DataFrame(np.zeros([len(lst_aa), len(lst_aa)]), columns=lst_aa, index=lst_aa_reverse)
        value_ste_matrix[value_ste_matrix == 0] = np.nan        
        
        matrix_df = matrix_df.reset_index()
        
        if nonsyn_only:
            matrix_df.loc[matrix_df.aa_ref == matrix_df.aa_alt , value_column] = np.nan
            matrix_df.loc[matrix_df.aa_ref == matrix_df.aa_alt , value_ste_column] = np.nan
        
        for i in range(matrix_df.shape[0]):
            cur_aa_ref = matrix_df.loc[i, 'aa_ref']
            cur_aa_alt = matrix_df.loc[i, 'aa_alt']
            
            if (cur_aa_ref in lst_aa) & (cur_aa_alt in lst_aa):            
                value_matrix.loc[cur_aa_ref, cur_aa_alt] = matrix_df.loc[i, value_column]
                value_ste_matrix.loc[cur_aa_ref, cur_aa_alt] = matrix_df.loc[i, value_ste_column]
              
        if math.isnan(vmin) | math.isnan(vmax):    
            ax = sns.heatmap(value_matrix, annot=annotation, fmt='.2g', annot_kws={"size": 8}, cmap='Blues_r')  # previously RdBu_r
        else:
            ax = sns.heatmap(value_matrix, annot=annotation, fmt='.2g', annot_kws={"size": 8}, vmax=vmax, vmin=vmin, center=vcenter, cmap='Blues_r')
        ax.set_title(value_name, size=25)
        for xtick, color in zip(ax.get_xticklabels(), aa_colors):
            xtick.set_color(color)
        for ytick, color in zip(ax.get_yticklabels(), aa_colors):
            ytick.set_color(color)
        pass
        ax.set_xlabel('Alternative AA', size=20)
        ax.set_ylabel('Reference AA', size=20)
        
        if (show_ste == 1):  
            rows = value_ste_matrix.shape[0]      
            value_ste_matrix = np.matrix(value_ste_matrix)
            for x in range(value_ste_matrix.shape[0]):
                for y in range(value_ste_matrix.shape[1]):
                    cur_se = value_ste_matrix[rows - x - 1, y]
                    if not math.isnan(cur_se):
                        if cur_se < 1:                    
                            ax.plot([x + (1 - cur_se) / 2 + 0.05, x + (1 + cur_se) / 2 - 0.05], [value_ste_matrix.shape[0] - 1 - y + (1 - cur_se) / 2 + 0.05, value_ste_matrix.shape[0] - 1 - y + (1 + cur_se) / 2 - 0.05], color='purple', linestyle='-', linewidth=1)
                        else:
                            ax.plot(x + 0.5, value_ste_matrix.shape[0] - 1 - y + 0.5, color='purple', marker='.', markersize=10)
                    pass
        return (ax)
    
    def plot_aa_distribution(self, distribution_df, value_column, value_name, aa_ref, aa_alt):
        values = distribution_df.loc[(distribution_df.aa_ref == aa_ref) & (distribution_df.aa_alt == aa_alt), value_column]
        plt.clf()
        ax = plt.subplot()   
        plt.hist(values, color='green')
        ax.set_xlabel(value_name)
        ax.set_title(self.dict_aaname[aa_ref] + ' to ' + self.dict_aaname[aa_alt] + ' ' + value_name + ' Distribution', size=14)
    
    def plot_aasum(self, aasum_columnname, aasum_name, aasum_df, filter_columnname='', filter_value='', lst_aa=['I', 'L', 'V', 'M', 'F', 'W', 'H', 'K', 'T', 'A', 'G', 'P', 'Y', 'D', 'E', 'R', 'S', 'C', 'N', 'Q'] , colors_aa=['r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g'], vmin=np.nan, vmax=np.nan, annotation=True):
        if filter_columnname == '':
            aasum_df = aasum_df.loc[(aasum_df.aa_ref.isin(self.lst_aa)) & (aasum_df.aa_alt.isin(self.lst_aa)), :]
        else:
            aasum_df = aasum_df.loc[(aasum_df.aa_ref.isin(self.lst_aa)) & (aasum_df.aa_alt.isin(self.lst_aa)) & (aasum_df[filter_columnname] == filter_value), :]

        fig = plt.figure(figsize=(14, 10))
        plt.clf()            
        ax = plt.subplot(2, 2, 1)
        self.plot_aa_matrix(aasum_df, aasum_columnname, aasum_name, lst_aa, colors_aa, vmin, vmax, annotation)
        
        ax = plt.subplot(2, 2, 2)
        self.plot_aa_matrix(aasum_df, aasum_columnname + '_ste', aasum_name + ' STE', lst_aa, colors_aa, vmin, vmax, annotation)

        fig.tight_layout()
        plt.savefig(self.db_path + 'output/aasum_' + aasum_columnname + '.png')     
               
    def plot_aasum_backup(self, aasum_columnname, aasum_name, aasum_file, filter_columnname='', filter_value='', lst_aa=['I', 'L', 'V', 'M', 'F', 'W', 'H', 'K', 'T', 'A', 'G', 'P', 'Y', 'D', 'E', 'R', 'S', 'C', 'N', 'Q'] , colors_aa=['r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g'], vmin=np.nan, vmax=np.nan, annotation=True):
        aasum_df = pd.read_csv(self.db_path + aasum_file)
        if filter_columnname == '':
            aasum_df = aasum_df.loc[(aasum_df.aa_ref.isin(self.lst_aa)) & (aasum_df.aa_alt.isin(self.lst_aa)), :]
        else:
            aasum_df = aasum_df.loc[(aasum_df.aa_ref.isin(self.lst_aa)) & (aasum_df.aa_alt.isin(self.lst_aa)) & (aasum_df[filter_columnname] == filter_value), :]
            
        fig = plt.figure(figsize=(14, 10))
        plt.clf()            
        ax = plt.subplot(2, 2, 1)
        self.plot_aa_matrix(aasum_df, aasum_columnname, aasum_name, lst_aa, colors_aa, vmin, vmax, annotation)
        
        ax = plt.subplot(2, 2, 2)
        self.plot_aa_matrix(aasum_df, aasum_columnname + '_ste', aasum_name + ' STE', lst_aa, colors_aa, vmin, vmax, annotation)
        
        ax = plt.subplot(2, 2, 3)
        self.plot_aa_matrix(aasum_df, aasum_columnname + '_count', aasum_name + ' Count', lst_aa, colors_aa, vmin, vmax, annotation)
        
        ax = plt.subplot(2, 2, 4)
        ax.set_title(aasum_name + ' VS Standard Error', size=15)
        ax.scatter(-np.log10(aasum_df[aasum_columnname]), -np.log10(aasum_df[aasum_columnname + '_ste']))
        ax.set_xlabel(aasum_name + '(-log10)')
        ax.set_ylabel('Standard Error (-log10)')
        
        fig.tight_layout()
        plt.savefig(self.db_path + 'output/aasum_' + aasum_columnname + '.png')    
        
    def plot_psipred_funsum(self, funsum_file, funsum_columnname, funsum_name, funsum_psipred_file, funsum_psipred_columnname, funsum_psipred_name, lst_aa=['I', 'L', 'V', 'M', 'F', 'W', 'H', 'K', 'T', 'A', 'G', 'P', 'Y', 'D', 'E', 'R', 'S', 'C', 'N', 'Q'] , colors_aa=['r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g'], vmin=np.nan, vmax=np.nan, annotation=True):
        funsum_df = pd.read_csv(self.db_path + 'funsum/' + funsum_file)
        funsum_psipred_df = pd.read_csv(self.db_path + 'funsum/' + funsum_psipred_file)
        funsum_df_all = funsum_df.loc[(funsum_df.aa_ref.isin(self.lst_aa)) & (funsum_df.aa_alt.isin(self.lst_aa)), :]
        funsum_df_coil = funsum_psipred_df.loc[(funsum_psipred_df.aa_ref.isin(self.lst_aa)) & (funsum_psipred_df.aa_alt.isin(self.lst_aa)) & (funsum_psipred_df['aa_psipred'] == 'C'), :]
        funsum_df_helix = funsum_psipred_df.loc[(funsum_psipred_df.aa_ref.isin(self.lst_aa)) & (funsum_psipred_df.aa_alt.isin(self.lst_aa)) & (funsum_psipred_df['aa_psipred'] == 'H'), :]
        funsum_df_sheet = funsum_psipred_df.loc[(funsum_psipred_df.aa_ref.isin(self.lst_aa)) & (funsum_psipred_df.aa_alt.isin(self.lst_aa)) & (funsum_psipred_df['aa_psipred'] == 'E'), :]
                            
        fig = plt.figure(figsize=(14, 10))
        plt.clf()            
        ax = plt.subplot(2, 2, 1)
        self.plot_aa_matrix(funsum_df_all, funsum_columnname, funsum_name, lst_aa, colors_aa, vmin, vmax, annotation)
        
        ax = plt.subplot(2, 2, 2)
        self.plot_aa_matrix(funsum_df_coil, funsum_psipred_columnname, funsum_psipred_name + '(coil)', lst_aa, colors_aa, vmin, vmax, annotation)
        
        ax = plt.subplot(2, 2, 3)
        self.plot_aa_matrix(funsum_df_helix, funsum_psipred_columnname, funsum_psipred_name + '(helix)', lst_aa, colors_aa, vmin, vmax, annotation)
        
        ax = plt.subplot(2, 2, 4)
        self.plot_aa_matrix(funsum_df_sheet, funsum_psipred_columnname, funsum_psipred_name + '(sheet)', lst_aa, colors_aa, vmin, vmax, annotation)

        fig.tight_layout()
        plt.savefig(self.db_path + 'output/pispred_funsum_' + funsum_columnname + '.png')      
    
    def plot_aasums(self, lst_aa=['I', 'L', 'V', 'M', 'F', 'W', 'H', 'K', 'T', 'A', 'G', 'P', 'Y', 'D', 'E', 'R', 'S', 'C', 'N', 'Q'] , colors_aa=['r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g'], accessible_only=True, species='9606'):
        fig = plt.figure(figsize=(16, 10))
        fig.subplots_adjust(hspace=.5)
        plt.clf()
        plt.subplot(3, 3, 1)
        ax = self.plot_aa_matrix(self.dict_mutsums[species], 'accessibility', 'MUTSUM', lst_aa, colors_aa)
          
        for xtick, color in zip(ax.get_xticklabels(), colors_aa):
            xtick.set_color(color)    
        for ytick, color in zip(ax.get_yticklabels(), colors_aa):
            ytick.set_color(color)
        pass
        plt.subplot(3, 3, 2)
        mafsum = self.dict_mafsums['1']
        if accessible_only:
            mafsum = pd.merge(mafsum, self.dict_mutsums[species])
            mafsum.loc[mafsum.accessibility.isnull(), 'mafsum_fitness' ] = np.nan
        ax = self.plot_aa_matrix(mafsum, 'mafsum_fitness', 'MAFSUM', lst_aa, colors_aa)
        for xtick, color in zip(ax.get_xticklabels(), colors_aa):
            xtick.set_color(color)    
        for ytick, color in zip(ax.get_yticklabels(), colors_aa):
            ytick.set_color(color)  
        pass 
      
        plt.subplot(3, 3, 3)
        ax = self.plot_aa_matrix(self.dict_selsums[species], 'selsum_fitness', 'SELSUM', lst_aa, colors_aa)
        for xtick, color in zip(ax.get_xticklabels(), colors_aa):
            xtick.set_color(color)    
        for ytick, color in zip(ax.get_yticklabels(), colors_aa):
            ytick.set_color(color)  
        pass  

        plt.subplot(3, 3, 4)
        funsum = self.dict_funsums['fitness_median']
        if accessible_only:
            funsum = pd.merge(funsum, self.dict_mutsums[species])
            funsum.loc[funsum.accessibility.isnull(), 'funsum_fitness_median' ] = np.nan
        ax = self.plot_aa_matrix(funsum, 'funsum_fitness_median', 'FUNSUM', lst_aa, colors_aa)
        for xtick, color in zip(ax.get_xticklabels(), colors_aa):
            xtick.set_color(color)    
        for ytick, color in zip(ax.get_yticklabels(), colors_aa):
            ytick.set_color(color)
        pass
    
        plt.subplot(3, 3, 5)
        clinvarsum = self.clinvarsum_df
        if accessible_only:
            clinvarsum = pd.merge(clinvarsum, self.dict_mutsums[species])
            clinvarsum.loc[clinvarsum.accessibility.isnull(), 'clinvarsum_label_mean' ] = np.nan
        ax = self.plot_aa_matrix(clinvarsum, 'clinvarsum_label_mean', 'CLINVARSUM', lst_aa, colors_aa)
        for xtick, color in zip(ax.get_xticklabels(), colors_aa):
            xtick.set_color(color)    
        for ytick, color in zip(ax.get_yticklabels(), colors_aa):
            ytick.set_color(color)
        pass
    
        plt.subplot(3, 3, 6)
        humsavarsum = self.humsavarsum_df
        if accessible_only:
            humsavarsum = pd.merge(humsavarsum, self.dict_mutsums[species])
            humsavarsum.loc[humsavarsum.accessibility.isnull(), 'humsavarsum_label_mean' ] = np.nan
        ax = self.plot_aa_matrix(humsavarsum, 'humsavarsum_label_mean', 'HUMSAVARSUM', lst_aa, colors_aa)
        for xtick, color in zip(ax.get_xticklabels(), colors_aa):
            xtick.set_color(color)    
        for ytick, color in zip(ax.get_yticklabels(), colors_aa):
            ytick.set_color(color)
        pass

        plt.subplot(3, 3, 7)
        blosum100 = self.dict_blosums['blosum100']
        if accessible_only:
            blosum100 = pd.merge(blosum100, self.dict_mutsums[species])
            blosum100.loc[blosum100.accessibility.isnull(), 'blosum100' ] = np.nan
        
        ax = self.plot_aa_matrix(blosum100 , 'blosum100', 'BLOSUM 100', lst_aa, colors_aa)
        for xtick, color in zip(ax.get_xticklabels(), colors_aa):
            xtick.set_color(color)    
        for ytick, color in zip(ax.get_yticklabels(), colors_aa):
            ytick.set_color(color)
        pass
    
        plt.subplot(3, 3, 8)
        blosum62 = self.dict_blosums['blosum62']
        if accessible_only:
            blosum62 = pd.merge(blosum62, self.dict_mutsums[species])
            blosum62.loc[blosum62.accessibility.isnull(), 'blosum62' ] = np.nan
        ax = self.plot_aa_matrix(blosum62, 'blosum62', 'BLOSUM 62', lst_aa, colors_aa)
        for xtick, color in zip(ax.get_xticklabels(), colors_aa):
            xtick.set_color(color)    
        for ytick, color in zip(ax.get_yticklabels(), colors_aa):
            ytick.set_color(color)
        pass
    
        plt.subplot(3, 3, 9)
        blosum30 = self.dict_blosums['blosum30']
        if accessible_only:
            blosum30 = pd.merge(blosum30, self.dict_mutsums[species])
            blosum30.loc[blosum30.accessibility.isnull(), 'blosum30' ] = np.nan
        ax = self.plot_aa_matrix(blosum30, 'blosum30', 'BLOSUM 30', lst_aa, colors_aa)
        for xtick, color in zip(ax.get_xticklabels(), colors_aa):
            xtick.set_color(color)    
        for ytick, color in zip(ax.get_yticklabels(), colors_aa):
            ytick.set_color(color)
        pass

        fig.tight_layout()
        plt.savefig(self.db_path + 'output/different_sums.png')   
         
    def get_dms_seqdict(self):
        p_fa_dict = {}
        p_fa = open(self.db_path + 'dms/fasta/dms.fasta', 'r') 
        for line in p_fa:
           if line[0] == ">" :
               cur_key = line[1:].rstrip()
               p_fa_dict[cur_key] = ''
           else:
               p_fa_dict[cur_key] += line.strip() 
        return (p_fa_dict)    
        
    def get_uniprot_seqdict(self):
        p_fa_dict = {}
        uniprot_isoform_ids = []
        uniprot_reviewed_ids = []
        p_fa = open(self.db_path + 'uniprot/org/uniprot_sprot.fasta', 'r') 
        for line in p_fa:
            if line[0] == ">" :
                cur_key = line.split('|')[1]
                uniprot_reviewed_ids.append(cur_key)
                p_fa_dict[cur_key] = ''
            else:
                p_fa_dict[cur_key] += line.strip()
                
        p_fa_isoform = open(self.db_path + 'uniprot/org/uniprot_sprot_varsplic.fasta', 'r') 
        for line in p_fa_isoform:
            if line[0] == ">" :
                cur_key = line.split('|')[1]
                uniprot_isoform_ids.append(cur_key)
                p_fa_dict[cur_key] = ''
            else:
                p_fa_dict[cur_key] += line.strip()       
                
        np.save(self.db_path + 'uniprot/npy/uniprot_seq_dict.npy', p_fa_dict)
        np.save(self.db_path + 'uniprot/npy/uniprot_reviewed_ids.npy', uniprot_reviewed_ids)
        np.save(self.db_path + 'uniprot/npy/uniprot_isoform_ids.npy', uniprot_isoform_ids)
        return (p_fa_dict, uniprot_reviewed_ids, uniprot_isoform_ids)
    
    def get_uniprot_info(self):
        # manually dowonload from Uniprot with filter [reviewed:yes AND organism:"Homo sapiens (Human) [9606]"], and with primary gene name
        uniprot_df = pd.read_csv(self.db_path + 'uniprot/org/uniprot_human_all.tab', sep='\t')
        uniprot_df.columns = ['uniprot_ac', 'gene_name', 'gene_syn', 'length', 'seq', 'reviewed']
        
        uniprot_df['gene_name'] = uniprot_df['gene_name'].replace(np.nan, '')
        uniprot_df['gene_syn'] = uniprot_df['gene_syn'].replace(np.nan, '')
        uniprot_df['gene_name_syn'] = uniprot_df['gene_name'] + ';' + uniprot_df['gene_syn']
        uniprot_df['gene_name_syn'] = uniprot_df['gene_name_syn'].apply(lambda x: ','.join(re.compile(";|\s").split(str(x))))
        # return a sequence dictionary 
        dict_uniprot_ac2gname_gsyn = uniprot_df[['uniprot_ac', 'gene_name_syn']].set_index('uniprot_ac').T.to_dict('records')[0]
        
        # return a sequence dictionary 
        dict_uniprot_ac2seq = uniprot_df[['uniprot_ac', 'seq']].set_index('uniprot_ac').T.to_dict('records')[0]
        
        # handel the reviewed records first
        uniprot_reviewed_df = uniprot_df.loc[uniprot_df.reviewed == 'reviewed', :]

        # return a gene_name to uniprot_ac dictionary
        dict_uniprot_gname2ac = uniprot_reviewed_df[['gene_name', 'uniprot_ac']].set_index('gene_name').T.to_dict('records')[0]
        dict_uniprot_gname2ac_copy = dict_uniprot_gname2ac.copy()
        for key in dict_uniprot_gname2ac_copy.keys():
            key = str(key)
            sub_keys = key.split(sep=";")
            if len(sub_keys) > 1 :
                for sub_key in sub_keys:
                    dict_uniprot_gname2ac[sub_key.lstrip()] = dict_uniprot_gname2ac[key]
            sub_keys = key.split(sep=" ")
            if len(sub_keys) > 1 :
                for sub_key in sub_keys:
                    dict_uniprot_gname2ac[sub_key.lstrip()] = dict_uniprot_gname2ac[key]                    
        
        dict_uniprot_gsyn2ac = uniprot_reviewed_df[['gene_syn', 'uniprot_ac']].set_index('gene_syn').T.to_dict('records')[0]
        dict_uniprot_gsyn2ac_copy = dict_uniprot_gsyn2ac.copy()
        for key in dict_uniprot_gsyn2ac_copy.keys():
            key = str(key)
            sub_keys = key.split(sep=";")
            if len(sub_keys) > 1 :
                for sub_key in sub_keys:
                    dict_uniprot_gsyn2ac[sub_key.lstrip()] = dict_uniprot_gsyn2ac[key]
            sub_keys = key.split(sep=" ")
            if len(sub_keys) > 1 :
                for sub_key in sub_keys:
                    dict_uniprot_gsyn2ac[sub_key.lstrip()] = dict_uniprot_gsyn2ac[key]       
                    
        dict_uniprot_gene2ac_reviewed = dict_uniprot_gsyn2ac.copy()
        dict_uniprot_gene2ac_reviewed.update(dict_uniprot_gname2ac.items())      

        # handel the unreviewed records first
        uniprot_unreviewed_df = uniprot_df.loc[uniprot_df.reviewed == 'unreviewed', :]
        
        # return a gene_name to uniprot_ac dictionary
        dict_uniprot_gname2ac = uniprot_unreviewed_df[['gene_name', 'uniprot_ac']].set_index('gene_name').T.to_dict('records')[0]
        dict_uniprot_gname2ac_copy = dict_uniprot_gname2ac.copy()
        for key in dict_uniprot_gname2ac_copy.keys():
            key = str(key)
            sub_keys = key.split(sep=";")
            if len(sub_keys) > 1 :
                for sub_key in sub_keys:
                    dict_uniprot_gname2ac[sub_key.lstrip()] = dict_uniprot_gname2ac[key]
            sub_keys = key.split(sep=" ")
            if len(sub_keys) > 1 :
                for sub_key in sub_keys:
                    dict_uniprot_gname2ac[sub_key.lstrip()] = dict_uniprot_gname2ac[key]                    
        
        dict_uniprot_gsyn2ac = uniprot_unreviewed_df[['gene_syn', 'uniprot_ac']].set_index('gene_syn').T.to_dict('records')[0]
        dict_uniprot_gsyn2ac_copy = dict_uniprot_gsyn2ac.copy()
        for key in dict_uniprot_gsyn2ac_copy.keys():
            key = str(key)
            sub_keys = key.split(sep=";")
            if len(sub_keys) > 1 :
                for sub_key in sub_keys:
                    dict_uniprot_gsyn2ac[sub_key.lstrip()] = dict_uniprot_gsyn2ac[key]
            sub_keys = key.split(sep=" ")
            if len(sub_keys) > 1 :
                for sub_key in sub_keys:
                    dict_uniprot_gsyn2ac[sub_key.lstrip()] = dict_uniprot_gsyn2ac[key]       
                    
        dict_uniprot_gene2ac_unreviewed = dict_uniprot_gsyn2ac.copy()
        dict_uniprot_gene2ac_unreviewed.update(dict_uniprot_gname2ac.items())      
        
        dict_uniprot_gene2ac = dict_uniprot_gene2ac_unreviewed.copy()
        dict_uniprot_gene2ac.update(dict_uniprot_gene2ac_reviewed)
        
        uniprot_df.to_csv(self.db_path + 'uniprot/csv/uniprot_df.csv', index=False)
        np.save(self.db_path + 'uniprot/npy/dict_uniprot_ac2seq.npy', dict_uniprot_ac2seq)
        np.save(self.db_path + 'uniprot/npy/dict_uniprot_gene2ac.npy', dict_uniprot_gene2ac)
        np.save(self.db_path + 'uniprot/npy/dict_uniprot_ac2gname_gsyn.npy', dict_uniprot_ac2gname_gsyn)                    
        return [uniprot_df, dict_uniprot_ac2seq, dict_uniprot_gene2ac, dict_uniprot_ac2gname_gsyn]
        
    def generate_uniprot_individual_fasta(self):
        for uniprot_id in self.uniprot_seqdict:
            outfile = open(self.db_path + 'uniprot/fasta/' + uniprot_id + '.fasta', 'w')
            outfile.write(self.uniprot_seqdict[uniprot_id]) 
            outfile.close() 
    
    def generate_refseq_individual_fasta(self):
        for refseq_id in self.refseq_seqdict:
            outfile = open(self.db_path + 'refseq/fasta/' + refseq_id + '.fasta', 'w')
            outfile.write(self.refseq_seqdict[refseq_id]) 
            outfile.close() 
    
    def get_refseq_seqdict(self):
        p_fa_dict = {}
        for i in range(1, 500):
            cur_file = self.db_path + 'refseq/org/human.' + str(i) + '.protein.faa'
            if os.path.isfile(cur_file):
                p_fa = open(cur_file, 'r') 
                for line in p_fa:
                    if line[0] == ">" :
                        cur_key = line.split(' ')[0][1:]
                        p_fa_dict[cur_key] = ''
                    else:
                        p_fa_dict[cur_key] += line.strip() 
        np.save(self.db_path + 'refseq/npy/refseq_seq_dict.npy', p_fa_dict) 
        return (p_fa_dict)  

    def get_ensembl_seqdict(self):
        p_fa_dict = {}
        for i in range(1, 50):
            cur_file = self.db_path + 'ensembl/org/Homo_sapiens.GRCh37.66.pep.all.fa'
            if os.path.isfile(cur_file):
                p_fa = open(cur_file, 'r') 
                for line in p_fa:
                    if line[0] == ">" :
                        cur_key = line.split(' ')[0][1:]
                        p_fa_dict[cur_key] = ''
                    else:
                        p_fa_dict[cur_key] += line.strip() 
        np.save(self.db_path + 'ensembl/npy/ensembl_seq_dict.npy', p_fa_dict) 
        return (p_fa_dict)  

    def get_id_mapping(self):
        refseq2uniprot_dict = {}  
        ensp2uniprot_dict = {}      

        def fill_dict(alt_ids, uniprot_id):
            alt_ids = str(alt_ids)
            lst_alt_ids = alt_ids.split(';')
            for alt_id in lst_alt_ids:
                if (alt_id != 'NaN'):
                    alt_id = alt_id.strip()
                    refseq2uniprot_dict[alt_id] = uniprot_id
        
        id_maps = pd.read_table(self.db_path + 'uniprot/org/HUMAN_9606_idmapping_selected.tab', sep='\t', header=None) 
        id_maps.columns = ['UniProtKB-AC', 'UniProtKB-ID', 'GeneID', 'RefSeq', 'GI', 'PDB', 'GO', 'UniRef100', 'UniRef90', 'UniRef50', 'UniParc', 'PIR', 'NCBI-taxon', 'MIM', 'UniGene', 'PubMed', 'EMBL', 'EMBL-CDS', 'Ensembl', 'Ensembl_TRS', 'Ensembl_PRO', 'Additional PubMed']
        id_maps = id_maps.loc[id_maps['UniProtKB-AC'].isin(list(self.uniprot_reviewed_ids)), :]       
        id_maps[['UniProtKB-AC', 'RefSeq']].apply(lambda x: fill_dict(x['RefSeq'], x['UniProtKB-AC']), axis=1)
        id_maps[['UniProtKB-AC', 'Ensembl']].apply(lambda x: fill_dict(x['Ensembl'], x['UniProtKB-AC']), axis=1)
        return (refseq2uniprot_dict)
    
    def get_uniprot_id_mapping(self,target_id):
        target2uniprot_dict = {}
        uniprot2target_dict = {}       
        def fill_target2uniprot_dict(alt_ids, uniprot_id):
            alt_ids = str(alt_ids)
            lst_alt_ids = alt_ids.split(';')
            for alt_id in lst_alt_ids:
                if (alt_id != 'NaN'):
                    alt_id = alt_id.strip()
                    target2uniprot_dict[alt_id] = uniprot_id
                    
        def fill_uniprot2target_dict(alt_ids, uniprot_id):
            uniprot2target_dict[uniprot_id] = alt_ids
        
        id_maps = pd.read_table(self.db_path + 'uniprot/org/HUMAN_9606_idmapping_selected.tab', sep='\t', header=None) 
        id_maps.columns = ['UniProtKB-AC', 'UniProtKB-ID', 'GeneID', 'RefSeq', 'GI', 'PDB', 'GO', 'UniRef100', 'UniRef90', 'UniRef50', 'UniParc', 'PIR', 'NCBI-taxon', 'MIM', 'UniGene', 'PubMed', 'EMBL', 'EMBL-CDS', 'Ensembl', 'Ensembl_TRS', 'Ensembl_PRO', 'Additional PubMed']
#         id_maps = id_maps.loc[id_maps['UniProtKB-AC'].isin(list(self.uniprot_reviewed_ids)), :]       
        id_maps[['UniProtKB-AC', target_id]].apply(lambda x: fill_target2uniprot_dict(x[target_id], x['UniProtKB-AC']), axis=1)
        id_maps[['UniProtKB-AC', target_id]].apply(lambda x: fill_uniprot2target_dict(x[target_id], x['UniProtKB-AC']), axis=1)
        return ([target2uniprot_dict,uniprot2target_dict])
    
    
    def get_ucsc_idmapping(self):
#         http://hgdownload.soe.ucsc.edu/goldenPath/proteinDB/proteins140122/database/hgnc.txt.gz
#         http://hgdownload.soe.ucsc.edu/goldenPath/proteinDB/proteins140122/database/tableDescriptions.txt.gz
#         http://hgdownload.soe.ucsc.edu/goldenPath/hg19/UCSCGenes/uniProtToUcscGenes.txt (didn't use this, check it out if necessary)
        target2uniprot_dict = {}
        def fill_target2uniprot_dict(alt_ids, uniprot_id):
            alt_ids = str(alt_ids)
            lst_alt_ids = alt_ids.split(';')
            for alt_id in lst_alt_ids:
                if (alt_id != 'NaN'):
                    alt_id = alt_id.strip()
                    target2uniprot_dict[alt_id] = uniprot_id
        ucsc_map = pd.read_csv(self.db_path + 'ucsc/org/hgnc.txt',sep = '\t',encoding = "latin1" ,header = None)
        ucsc_map.columns = ['hgncId','symbol','name','status','locusType','locusGroup','prvSymbols','prvNames','synonyms','nameSyns','chrom','dateApprv','dateMod','dateSymChange','dateNmChange','accession','enzymeIds','entrezId','ensId','mgdId','miscDbs','miscIds','pubMed','refSeqIds','geneFamilyNm',
                            'geneFamilyDesc','recType','primaryId','secondaryId','ccdsId','vegaId','locusDbs','gdbMapped','entrezMapped','omimMapped','refSeqMapped','uniportMapped','ensMapped','ucscMapped','mgiMapped','rgdMapped']
        ucsc2uniprot_map = ucsc_map.loc[ucsc_map['uniportMapped'].notnull() & ucsc_map['ucscMapped'].notnull(),['uniportMapped','ucscMapped']]
        
        ucsc2uniprot_map.apply(lambda x: fill_target2uniprot_dict(x['ucscMapped'], x['uniportMapped']), axis=1)
        
        np.save(self.db_path + 'ucsc/npy/ucsc2uniprot_dict.npy',target2uniprot_dict)
        return (target2uniprot_dict)
        
                   
    def get_refseq_idmapping(self):

        def process_idmap(refseq_vid, isoform, uniprot_id):
            refseq_vids = refseq_vid.split(',')
            refseq_vids_dict = {x :uniprot_id for x in refseq_vids}
            refseq2uniprot_dict.update(refseq_vids_dict)

            isoforms = str(isoform).split(',')
            for x in isoforms:
                if '->' in x:
                    x_s = x.split(' -> ')
                    if x_s[1] in self.uniprot_isoform_ids:                    
                        refseq2uniprot_isoform_dict[x_s[0]] = x_s[1]

        pass    

        # create refseq to uniprot map
        refseq2uniprot_dict = {}
        refseq2uniprot_isoform_dict = {}
        refseq2uniprot = pd.read_table(self.db_path + 'refseq/org/refseq_vid2uniprot.txt', sep='\t')  # generate from uniprot id mapping tool on the website (from refseq_vid -> uniprot_id)
        refseq2uniprot.columns = ['refseq_pvid', 'isoform', 'uniprot_id', 'uniprot_name', 'review', 'p_name', 'g_name', 'organism', 'length']
        refseq2uniprot = refseq2uniprot.loc[refseq2uniprot['review'] == 'reviewed', :]
        refseq2uniprot.apply(lambda x: process_idmap(x['refseq_pvid'], x['isoform'], x['uniprot_id']), axis=1)
        refseq2uniprot_dict.update(refseq2uniprot_isoform_dict)
        
        # Refseq ids
        LRG = pd.read_table(self.db_path + 'refseq/org/LRG_RefSeqGene', sep='\t')
        LRG.columns = ['tax_id', 'gene_id', 'symbol', 'refseq_gvid', 'lrg_id', 'refseq_tvid', 't_alt', 'refseq_pvid', 'p_alt', 'category']
        LRG = LRG.loc[LRG['refseq_pvid'].notnull(), :]
        LRG['refseq_gid'] = LRG['refseq_gvid'].apply(lambda x: x.split('.')[0])
        LRG['refseq_tid'] = LRG['refseq_tvid'].apply(lambda x: x.split('.')[0])
        LRG['refseq_pid'] = LRG['refseq_pvid'].apply(lambda x: x.split('.')[0])
        LRG['refseq_pid_version'] = LRG['refseq_pvid'].apply(lambda x: x.split('.')[1])
        LRG['refseq_pid_version'] = LRG['refseq_pid_version'].astype(int)
                  

        # it is OK that same transcripts  map to different protein verison id, but we want to have the latest protein id version 
        refseq_vids = LRG[['refseq_tvid', 'refseq_pid', 'refseq_pid_version']]
        refseq_vids = refseq_vids.groupby(['refseq_tvid', 'refseq_pid'])['refseq_pid_version'].max().reset_index()
          
        refseq_vids['refseq_pvid'] = refseq_vids.apply(lambda x: x['refseq_pid'] + '.' + str(x['refseq_pid_version']) , axis=1)
        refseq_vids = refseq_vids[['refseq_tvid', 'refseq_pvid', 'refseq_pid']]
        refseq_vids = refseq_vids.drop_duplicates()

        # map to uniprot_id
        refseq_vids['uniprot_id'] = refseq_vids['refseq_pvid'].apply(lambda x: refseq2uniprot_dict.get(x, np.nan))
        refseq_vids['isoform'] = 0
        refseq_vids.loc[refseq_vids['uniprot_id'].isin (self.uniprot_isoform_ids), 'isoform'] = 1
        
        #check if refseq_pvid has sequences 
        refseq_vids['seq_available'] = 0        
        refseq_vids.loc[refseq_vids['refseq_pvid'].isin(self.refseq_seqdict.keys()), 'seq_available'] = 1
        
        # check if refseq_pvid to uniprot_id mapping has the same sequence
        refseq_vids['seq_match'] = refseq_vids.apply(lambda x: self.refseq_seqdict.get(x['refseq_pvid'],np.nan) == self.uniprot_seqdict.get(x['uniprot_id'],np.nan), axis=1)
#         refseq_vids_notmatch = refseq_vids.loc[refseq_vids['seq_match'] == 0, 'refseq_pvid']
#         refseq_vids_notmatch_dict = {x: None for x in refseq_vids_notmatch}
#         refseq2uniprot_dict.update(refseq_vids_notmatch_dict)
        
        print ("Number of human refseq NM -> NP records form LRG_RefSeqGene: " + str(refseq_vids.shape[0]))
        refseq_vids.to_csv(self.db_path + 'refseq/csv/refseq_vids.csv', index=None)    
        np.save(self.db_path + 'refseq/npy/refseq2uniprot_isoform_dict.npy', refseq2uniprot_isoform_dict)
        np.save(self.db_path + 'refseq/npy/refseq2uniprot_dict.npy', refseq2uniprot_dict)
        return ([refseq_vids, refseq2uniprot_isoform_dict, refseq2uniprot_dict]) 
    
    def check_new(self, ftp_obj, ftp_path, ftp_file):    
        return (False)
    
    def get_gnomad_af(self):
        stime = time.time()
        gnomadLogFile = self.db_path + 'gnomad/gnomad_process_log'
        gnomadFile = self.db_path + 'gnomad/gnomad.exomes.r2.0.2.sites.vcf'
        gnomFile_output = self.db_path + 'gnomad/gnomad_output_snp.txt'
        gnomFile_output_af = self.db_path + 'gnomad/gnomad_output_snp_af.txt'
        vcfheaderFile_output = self.db_path + 'gnomad/vcfheader_output_snp.txt'
        count = 0
        newline = ''
        newline_af = ''
        vcfheader = ''
         
        with open(gnomadFile) as infile:
            for line in infile:                                 
                if not re.match('#', line):
                    count += 1  
                    if (count % 10000 == 0):
                        show_msg(gnomadLogFile, 1, str(count) + ' records have been processed.\n')
                        with open(gnomFile_output, "a") as f:
                            f.write(newline)
                            newline = ''
                        with open(gnomFile_output_af, "a") as f:
                            f.write(newline_af)
                            newline_af = ''                                                        
                    line_list = line.split('\t')             
                    info = line_list[-1]
                    line_list.pop(-1)
                    if re.match('AC', info): 
                        # print(info)
                        try:                
                            info_dict = {x.split("=")[0]:x.split("=")[1] for x in info.split(';')}
                        except:
                            info_dict = {x.split("=")[0]:x.split("=")[1] if "=" in x else x + '=' for x in info.split(';')}

                        try:
                            line_list.append(info_dict.get('AC', ''))  # add AC
                            line_list.append(info_dict.get('AN', ''))  # add AN
                            line_list.append(info_dict.get('AF', ''))  # add AF
                            line_list.append(info_dict.get('AF_EAS', ''))  # add AF_ESA
                            line_list.append(info_dict.get('GC', ''))  # add GC
     
                            vep_list = info_dict.get('CSQ', '').split(',')                            
                            vep_allele_dict = {}    
                            for vep in vep_list:
                                vep_sub_list = vep.split('|')                      
                                if (vep_sub_list[26] == 'YES') & (vep_sub_list[1] == 'missense_variant') :  # 'CANONICAL'      
                                    if vep_sub_list[0] not in vep_allele_dict.keys():  
                                        vep_allele_dict[vep_sub_list[0]] = []                                                          
                                    vep_allele_dict[vep_sub_list[0]].append([vep_sub_list[15].split('/')[0], vep_sub_list[14], vep_sub_list[15].split('/')[1], vep_sub_list[1], vep_sub_list[3], vep_sub_list[4], vep_sub_list[6], vep_sub_list[10], vep_sub_list[11], vep_sub_list[17], vep_sub_list[30], vep_sub_list[31], vep_sub_list[20]])
#                                     vep_allele_dict[vep_sub_list[0]] = [vep_sub_list[15].split('/')[0],vep_sub_list[14],vep_sub_list[15].split('/')[1],vep_sub_list[1],vep_sub_list[3],vep_sub_list[4],vep_sub_list[6],vep_sub_list[10],vep_sub_list[11],vep_sub_list[17],vep_sub_list[30],vep_sub_list[31],vep_sub_list[20]]
                                if (vep_sub_list[26] == 'YES') & (vep_sub_list[1] == 'synonymous_variant') :  # 'CANONICAL'
                                    if vep_sub_list[0] not in vep_allele_dict.keys():  
                                        vep_allele_dict[vep_sub_list[0]] = []   
                                    vep_allele_dict[vep_sub_list[0]].append([vep_sub_list[15], vep_sub_list[14], vep_sub_list[15], vep_sub_list[1], vep_sub_list[3], vep_sub_list[4], vep_sub_list[6], vep_sub_list[10], vep_sub_list[11], vep_sub_list[17], vep_sub_list[30], vep_sub_list[31], vep_sub_list[20]])                                    
#                                     vep_allele_dict[vep_sub_list[0]] = [vep_sub_list[15],vep_sub_list[14],vep_sub_list[15],vep_sub_list[1],vep_sub_list[3],vep_sub_list[4],vep_sub_list[6],vep_sub_list[10],vep_sub_list[11],vep_sub_list[17],vep_sub_list[30],vep_sub_list[31],vep_sub_list[20]]                                                                                                                                                                           
                            # multiple allele for same position and ref
                            allele_list = line_list[4].split(',')                            
                            if len(allele_list) > 1:
                                ac_list = line_list[7].split(',')
                                an_list = line_list[8].split(',')
                                af_list = line_list[9].split(',')
                                af_esa_list = line_list[10].split(',')
                                if line_list[11] != '':
                                    gc_list_1 = line_list[11].split(',')
                                    gc_list = []
                                    for i in range(int(len(gc_list_1) / 3)):
                                        gc_list.append(gc_list_1[3 * i:3 * i + 3])
                                else:
                                    gc_list = [['-1', '-1', '-1']] * len(allele_list)                             
                                
                                for i in range(len(allele_list)):
                                    newline_af += '\t'.join (line_list[0:4] + [allele_list[i]] + line_list[5:7] + [ac_list[i]] + [line_list[8]] + [af_list[i]] + [af_esa_list[i]] + [str(gc_list[i])] + [gc_list[i][0]] + [gc_list[i][1]] + [gc_list[i][2]]) + '\n'
                                    if allele_list[i] in list(vep_allele_dict.keys()):   
                                        for j in range(len(vep_allele_dict[allele_list[i]])):                                                
                                            newline += '\t'.join (line_list[0:4] + [allele_list[i]] + line_list[5:7] + [ac_list[i]] + [line_list[8]] + [af_list[i]] + [af_esa_list[i]] + [str(gc_list[i])] + [gc_list[i][0]] + [gc_list[i][1]] + [gc_list[i][2]] + vep_allele_dict[allele_list[i]][j]) + '\n'
                                    else:
                                        newline += '\t'.join (line_list[0:4] + [allele_list[i]] + line_list[5:7] + [ac_list[i]] + [line_list[8]] + [af_list[i]] + [af_esa_list[i]] + [str(gc_list[i])] + [gc_list[i][0]] + [gc_list[i][1]] + [gc_list[i][2]] + ['N/A'] * 13) + '\n'
                            else:                                
                                if line_list[11] != '':
                                    gc_list = line_list[11].split(',')
                                else:
                                    gc_list = ['-1', '-1', '-1']
                                
                                newline_af += '\t'.join (line_list[0:11] + [str(gc_list)] + [gc_list[0]] + [gc_list[1]] + [gc_list[2]]) + '\n'                                
                                if allele_list[0] in list(vep_allele_dict.keys()):    
                                    for j in range(len(vep_allele_dict[allele_list[0]])):                                                
                                        newline += '\t'.join (line_list[0:11] + [str(gc_list)] + [gc_list[0]] + [gc_list[1]] + [gc_list[2]] + vep_allele_dict[allele_list[0]][j]) + '\n'
                                else:
                                    newline += '\t'.join (line_list[0:11] + [str(gc_list)] + [gc_list[0]] + [gc_list[1]] + [gc_list[2]] + ['N/A'] * 13) + '\n'
                        except:
                            show_msg(gnomadLogFile, 1, 'Records ' + str(count) + ' raise error:\n' + traceback.format_exc() + '\n' + str(line_list))
                            newline += '\t'.join (line_list + ['N/A'] * 16) + '\n'
                            newline_af += '\t'.join (line_list + ['N/A'] * 16) + '\n'
                else:
                    vcfheader += line + '\n'                
        with open(gnomFile_output, "a") as f:
            f.write(newline)
        with open(gnomFile_output_af, "a") as f:
            f.write(newline_af)
        with open(vcfheaderFile_output, "a") as f:
            f.write(vcfheader)  
        etime = time.time() 
        print("Elapse time was %g seconds" % (etime - stime))
        
    def get_gnomad_af_beta(self):
        gnomadLogFile = self.db_path + 'gnomad/gnomad_process_log'
        gnomadFile = self.db_path + 'gnomad/gnomad.exomes.r2.0.2.sites.vcf'
        gnomFile_output = self.db_path + 'gnomad/gnomad_output_snp.txt'
        gnomFile_output1 = self.db_path + 'gnomad/gnomad_output_snp_1.txt'
        vcfheaderFile_output = self.db_path + 'gnomad/vcfheader_output_snp.txt'
        count = 0
        newline = ''
        vcfheader = ''
        
        # rs782326802
#         f = open(gnomFile_output1,'w')
#         stime = time.time()
#         count = 0
#         with open(gnomFile_output) as infile:
#             for line in infile:                                 
#                 count +=1
#                 f.write(line)
#                 if (count > 14999999):
#                     if re.search('rs782326802',line): 
#                         f.close()
#                         break
                                
#         etime = time.time()
#         print("Elapse time was %g seconds" % (etime - stime))
         
        with open(gnomadFile) as infile:
            for line in infile:                                 
                if not re.match('#', line):
                    count += 1  
                    if (count % 10000 == 0):
                        print(str(count) + '\n') 
                    if count >= 15000000:   
                        if (count % 10000 == 0):
                            show_msg(gnomadLogFile, 1, str(count) + ' records have been processed.\n')
                            with open(gnomFile_output1, "a") as f:
                                f.write(newline)
                        line_list = line.split('\t')             
                        info = line_list[-1]
                        line_list.pop(-1)
                        if re.match('AC', info): 
                            # print(info)
                            try:                
                                info_dict = {x.split("=")[0]:x.split("=")[1] for x in info.split(';')}
                            except:
                                info_dict = {x.split("=")[0]:x.split("=")[1] if "=" in x else x + '=' for x in info.split(';')}
    
                            try:
                                line_list.append(info_dict.get('AC', ''))  # add AC
                                line_list.append(info_dict.get('AN', ''))  # add AN
                                line_list.append(info_dict.get('AF', ''))  # add AF
                                line_list.append(info_dict.get('AF_EAS', ''))  # add AF_ESA
                                line_list.append(info_dict.get('GC', ''))  # add GC
         
                                vep_list = info_dict.get('CSQ', '').split(',')                            
                                vep_allele_dict = {}    
                                for vep in vep_list:
                                    vep_sub_list = vep.split('|')                          
                                    if (vep_sub_list[26] == 'YES') & (vep_sub_list[1] == 'missense_variant') :  # 'CANONICAL'                                                               
                                        vep_allele_dict[vep_sub_list[0]] = [vep_sub_list[15].split('/')[0], vep_sub_list[14], vep_sub_list[15].split('/')[1], vep_sub_list[1], vep_sub_list[3], vep_sub_list[4], vep_sub_list[6], vep_sub_list[10], vep_sub_list[11], vep_sub_list[17], vep_sub_list[30], vep_sub_list[31]]
                                    if (vep_sub_list[26] == 'YES') & (vep_sub_list[1] == 'synonymous_variant') :  # 'CANONICAL' 
                                        vep_allele_dict[vep_sub_list[0]] = [vep_sub_list[15], vep_sub_list[14], vep_sub_list[15], vep_sub_list[1], vep_sub_list[3], vep_sub_list[4], vep_sub_list[6], vep_sub_list[10], vep_sub_list[11], vep_sub_list[17], vep_sub_list[30], vep_sub_list[31]]                                                                                                                                                                           
                                # multiple allele for same position and ref
                                allele_list = line_list[4].split(',')
                                if len(allele_list) > 1:
                                    ac_list = line_list[7].split(',')
                                    an_list = line_list[8].split(',')
                                    af_list = line_list[9].split(',')
                                    af_esa_list = line_list[10].split(',')
                                    if line_list[11] != '':
                                        gc_list_1 = line_list[11].split(',')
                                        gc_list = []
                                        for i in range(int(len(gc_list_1) / 3)):
                                            gc_list.append(','.join(gc_list_1[3 * i:3 * i + 3]))
                                    else:
                                        gc_list = [''] * len(allele_list)                             
                                    
                                    for i in range(len(allele_list)):
                                        if allele_list[i] in list(vep_allele_dict.keys()):                                                   
                                            newline += '\t'.join (line_list[0:4] + [allele_list[i]] + line_list[5:7] + [ac_list[i]] + [line_list[8]] + [af_list[i]] + [af_esa_list[i]] + [gc_list[i]] + vep_allele_dict[allele_list[i]]) + '\n'
                                        else:
                                            newline += '\t'.join (line_list[0:4] + [allele_list[i]] + line_list[5:7] + [ac_list[i]] + [line_list[8]] + [af_list[i]] + [af_esa_list[i]] + [gc_list[i]] + ['N/A'] * 12) + '\n'
                                else:
                                    if allele_list[0] in list(vep_allele_dict.keys()):                                                   
                                        newline += '\t'.join (line_list + vep_allele_dict[allele_list[0]]) + '\n'
                                    else:
                                        newline += '\t'.join (line_list + ['N/A'] * 12) + '\n'
                            except:
                                show_msg(gnomadLogFile, 1, 'Records ' + str(count) + ' raise error:\n' + traceback.format_exc() + '\n' + str(line_list))
                                newline += '\t'.join (line_list + ['N/A'] * 12) + '\n'
                else:
                    vcfheader += line + '\n'
        with open(gnomFile_output1, "a") as f:
            f.write(newline)
        
#         with open(vcfheaderFile_output,"a") as f:
#             f.write(vcfheader)       

    def find_gnomad_record(self, rs_id):
        count = 0
        gnomFile = self.db_path + 'gnomad/gnomad.exomes.r2.0.2.sites.vcf'
        with open(gnomFile) as infile:
            for line in infile:  
                count += 1
                if (count % 10000 == 0):
                    print (str(count) + ' records have been processed.\n')                             
                if re.search(rs_id, line):
                    print(line)
                    break
       
    def get_gnomad_snp(self):
        gnomLogFile = self.db_path + 'gnomad/gnomad_process_log'
        gnomFile = self.db_path + 'gnomad/gnomad.exomes.r2.0.2.sites.vcf'
        count = 0
        newline = ''
        vcfheader = ''
        with open(gnomFile) as infile:
            for line in infile:                                
                if not re.match('#', line):
                    count += 1
                    if (count % 10000 == 0):
                        show_msg(gnomLogFile, 1, str(count) + ' records have been processed.')
                    line_list = line.split('\t')             
                    info = line_list[-1]
                    line_list.pop(-1)
                    if re.match('AC', info):                 
                        info_list = info.split(';')
                        line_list.append(info_list[0].split("=")[1])  # add AC
                        line_list.append(info_list[1].split("=")[1])  # add AF
                        line_list.append(info_list[2].split("=")[1])  # add AN  
                        vep_list = info_list[-1][4:].split(',')                            
                        vep_allele_dict = {}    
                        for vep in vep_list:
                            vep_sub_list = vep.split('|')                          
                            if (vep_sub_list[26] == 'YES') & ('missense_variant' in vep_sub_list[1]) :  # 'CANONICAL'                                                               
                                vep_allele_dict[vep_sub_list[0]] = [vep_sub_list[15].split('/')[0], vep_sub_list[14], vep_sub_list[15].split('/')[1], vep_sub_list[1], vep_sub_list[3], vep_sub_list[4], vep_sub_list[6], vep_sub_list[10], vep_sub_list[11], vep_sub_list[17], vep_sub_list[30], vep_sub_list[31]]
                            if (vep_sub_list[26] == 'YES') & ('synonymous_variant' in vep_sub_list[1]) :  # 'CANONICAL' 
                                vep_allele_dict[vep_sub_list[0]] = [vep_sub_list[15], vep_sub_list[14], vep_sub_list[15], vep_sub_list[1], vep_sub_list[3], vep_sub_list[4], vep_sub_list[6], vep_sub_list[10], vep_sub_list[11], vep_sub_list[17], vep_sub_list[30], vep_sub_list[31]]                                                                                                                                                                           
                        # multiple allele for same position and ref
                        allele_list = line_list[4].split(',')
                        ac_list = line_list[7].split(',')
                        af_list = line_list[8].split(',')
                        for i in range(len(allele_list)):
                            if allele_list[i] in list(vep_allele_dict.keys()):                                                   
                                newline += '\t'.join (line_list[0:4] + [allele_list[i]] + line_list[5:7] + [ac_list[i]] + [af_list[i]] + [line_list[9]] + vep_allele_dict[allele_list[i]]) + '\n'
                else:
                    print(line)
                    vcfheader += line + '\n'
                                  
        gnomFile_output = open(self.db_path + 'gnomad/gnomad_output_snp.txt', 'w')
        gnomFile_output.write(newline) 
        gnomFile_output.close() 
        
        vcfheader_output = open(self.db_path + 'gnomad/vcfheader_output_snp.txt', 'w')
        vcfheader_output.write(vcfheader) 
        vcfheader_output.close()         
                                 
    def exac_process(self):
        exac_File = self.db_path + 'exac/ExAC.r1.sites.vep.vcf'
        count = 0
        newline = ''
        with open(exac_File) as infile:
            for line in infile:
        #         if count == 1000:
        #             break
                if not re.match('#', line):
                    count += 1
                    line_list = line.split('\t')
                    info = line_list[-1]
                    line_list.pop(-1)
                    if re.match('AC', info):                 
                        info_list = info.split(';')
                        line_list.append(info_list[0].split("=")[1])  # add AC
                          
                        if re.match('AF=', info_list[11]):
                            line_list.append(info_list[11].split("=")[1])  # add AF
                        if re.match('AF=', info_list[12]):
                            line_list.append(info_list[12].split("=")[1])  # add AF                    
                        if re.match('AN=', info_list[12]):
                            line_list.append(info_list[12].split("=")[1])  # add AN  
                        if re.match('AN=', info_list[13]):
                            line_list.append(info_list[13].split("=")[1])  # add AN  
                                                                     
                        # multiple allele for same position and ref
                        if re.search(',', line_list[4]):
                            allele_list = line_list[4].split(',')
                            ac_list = line_list[7].split(',')
                            af_list = line_list[8].split(',')
                            for i in range(len(allele_list)):
                                if float(af_list[i]) > 1:
                                    print (line)
                                else:
                                    newline += '\t'.join (line_list[0:4] + [allele_list[i]] + line_list[5:7] + [ac_list[i]] + [af_list[i]] + [line_list[9]]) + '\n'
                        else:                    
                            newline += '\t'.join(line_list) + '\n'  
                                  
        exac_File_output = open(self.db_paht + 'exac/exac_output.txt', 'w')
        exac_File_output.write(newline) 
        exac_File_output.close() 
        
    def exac_notcga_process(self): 
        exac_notcga_File = self.db_path + 'exac/ExAC_nonTCGA.r1.sites.vep.vcf'
        count = 0
        newline = ''
        with open(exac_notcga_File) as infile:
            for line in infile:
                if not re.match('#', line):
                    count += 1
                    line_list = line.split('\t')
                    info = line_list[-1]
                    line_list.pop(-1)
                    if re.match('AC', info):                 
                        info_list = info.split(';')
                        line_list.append(info_list[0].split("=")[1])  # add AC
                           
                        if re.match('AF=', info_list[11]):
                            line_list.append(info_list[11].split("=")[1])  # add AF
                        if re.match('AF=', info_list[12]):
                            line_list.append(info_list[12].split("=")[1])  # add AF                    
                        if re.match('AN=', info_list[12]):
                            line_list.append(info_list[12].split("=")[1])  # add AN  
                        if re.match('AN=', info_list[13]):
                            line_list.append(info_list[13].split("=")[1])  # add AN  
                              
                        # multiple allele for same position and ref
                        if re.search(',', line_list[4]):
                            allele_list = line_list[4].split(',')
                            ac_list = line_list[7].split(',')
                            af_list = line_list[8].split(',')
                            for i in range(len(allele_list)):
                                if float(af_list[i]) > 1:
                                    print (line)
                                else:
                                    newline += '\t'.join (line_list[0:4] + [allele_list[i]] + line_list[5:7] + [ac_list[i]] + [af_list[i]] + [line_list[9]]) + '\n'
                        else:                    
                            newline += '\t'.join(line_list) + '\n'  
                                   
        exac_notcga_File_output = open(self.db_path + 'exac/exac_notcga_output.txt', 'w')
        exac_notcga_File_output.write(newline) 
        exac_notcga_File_output.close()                
    
    def get_clinvar_raw(self):
        # check if update needed 
        if self.check_new(self.ncbi_ftp_obj, self.clinvar_ftp_path, self.clinvar_ftp_file):
            self.ftp_download(self.ncbi_ftp_obj, self.clinvar_ftp_path, self.clinvar_ftp_file, self.self.clinvar_ftp_file)
            inF = gzip.GzipFile(self.db_path + self.clinvar_ftp_file, 'rb')
            s = inF.read()
            inF.close()
            outF = open(self.db_path + self.clinvar_raw_file, 'wb')
            outF.write(s)
            outF.close()     
        x = pd.read_table(self.db_path + self.clinvar_raw_file, sep='\t', dtype={"#chr": str})
        return(x) 

    def update_ftp_files(self):
        ####***************************************************************************************************************************************************************
        # NCBI FTP
        ####***************************************************************************************************************************************************************
        self.ncbi_ftp = 'ftp.ncbi.nlm.nih.gov' 
        self.ncbi_ftp_obj = self.obj_ftp(self.ncbi_ftp)
        # 1. LRG_RefSeqGene (ftp://ftp.ncbi.nih.gov/refseq/H_sapiens/RefSeqGene/LRG_RefSeqGene) for transcript_id, protein_id mapping 
        # 2. human.#.protein.faa.gz (ftp://ftp.ncbi.nih.gov/refseq/H_sapiens/mRNA_Prot/) for refseq protein sequences
        # 3. variant_summary.txt.gz (ftp://ftp.ncbi.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz) for clinvar raw data
        self.ftp_download(self.ncbi_ftp_obj, '/refseq/H_sapiens/RefSeqGene/', 'LRG_RefSeqGene', self.db_path + 'refseq/org/LRG_RefSeqGene')
        
        for i in range(1, 500):
            cur_file = 'human.' + str(i) + '.protein.faa.gz'
            cur_file_unzipped = 'human.' + str(i) + '.protein.faa'
            return_info = self.ftp_download(self.ncbi_ftp_obj, '/refseq/H_sapiens/mRNA_Prot/', cur_file, self.db_path + 'refseq/org/' + cur_file)
            if return_info == '0':
                self.gzip_decompress(self.db_path + 'refseq/org/' + cur_file, self.db_path + 'refseq/org/' + cur_file_unzipped) 
                
            if (return_info != "0") & (return_info != "1"):
                break
                
         
        return_info = self.ftp_download(self.ncbi_ftp_obj, '/pub/clinvar/tab_delimited/', 'variant_summary.txt.gz', self.db_path + 'clinvar/org/variant_summary.txt.gz')
        if return_info == '0':
            self.gzip_decompress(self.db_path + 'clinvar/org/variant_summary.txt.gz', self.db_path + 'clinvar/org/variant_summary.txt') 
         
        ####***************************************************************************************************************************************************************    
        # UniProt FTP
        ####***************************************************************************************************************************************************************
        # 1. uniprot_human_all.tab (Uniprot filter [reviewed:yes AND organism:"Homo sapiens (Human) [9606]"]) for gene name,alias
        # 2. uniprot_sprot.fasta.gz (ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz)
        # 3. uniprot_sprot_varsplic.gz (ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot_varsplic.fasta.gz
        # 4. ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/idmapping/by_organism/HUMAN_9606_idmapping_selected.tab.gz
        # 5.  humsavar.txt (http://www.uniprot.org/docs/humsavar.txt) for humsavar raw data

        ####***************************************************************************************************************************************************************    
        # Ensembl FTP
        ####***************************************************************************************************************************************************************
#         self.ensembl_ftp_obj = self.obj_ftp('ftp.ensembl.org')
#         # 1. stable_id_lookup.txt.tab (ftp://ftp.ensembl.org/pub/release-91/mysql/ensembl_stable_ids_91/stable_id_lookup.txt.gz)
#         return_info = self.ftp_download(self.ensembl_ftp_obj, '/pub/release-91/mysql/ensembl_stable_ids_91/', 'stable_id_lookup.txt.gz', self.db_path + 'ensembl/stable_id_lookup.txt.gz')
#         if return_info == '0':
#                 self.gzip_decompress(self.db_path + 'ensembl/stable_id_lookup.txt.gz', self.db_path + 'ensembl/stable_id_lookup.txt') 
        ####***************************************************************************************************************************************************************
        # Refseq to Uniprot mapping
        ####***************************************************************************************************************************************************************
#         LRG_RefSeqGene = pd.read_table(self.db_path + 'refseq/org/LRG_RefSeqGene',sep = '\t')
#         LRG_RefSeqGene.columns = ['tax_id','gene_id','symbol','g_vid','lrg_id','t_vid','t_alt','refseq_vid','p_alt','category']
#         refseq_vids = LRG_RefSeqGene.loc[LRG_RefSeqGene['refseq_vid'].notnull(),'refseq_vid']
#         refseq_vids_query = ' '.join(refseq_vids.unique())
#         url = 'http://www.uniprot.org/uploadlists/'
#         params = {
#         'from':'P_REFSEQ_AC',
#         'to':'ACC',
#         'format':'tab',
#         'query': refseq_vids_query
#         }
#          
#         data = urllib.parse.urlencode(params).encode('utf-8')
#         request = urllib.request.Request(url)
#         contact = "joe.wu.ca@gmail.com" # Please set your email address here to help us debug in case of problems.
#         request.add_header('User-Agent', 'Python %s' % contact)
#         response = urllib.request.urlopen(request,data = data)
#         r = response.read().decode('utf-8')
#         f = open(self.db_path + "refseq/org/refseq_vid2uniprot.txt", "w")
#         f.write(r)
#         f.close()
        
        ####***************************************************************************************************************************************************************
        # Polyphen
         ####***************************************************************************************************************************************************************
        # 1. humdiv-2011_12.deleterious.pph.input (ftp://genetics.bwh.harvard.edu/pph2/training/training-2.2.2.tar.gz)
        # 2. humdiv-2011_12.neutral.pph.input (ftp://genetics.bwh.harvard.edu/pph2/training/training-2.2.2.tar.gz)
        # 3. humvar-2011_12.deleterious.pph.input (ftp://genetics.bwh.harvard.edu/pph2/training/training-2.2.2.tar.gz)
        # 4. humvar-2011_12.neutral.pph.input (ftp://genetics.bwh.harvard.edu/pph2/training/training-2.2.2.tar.gz)

        ####***************************************************************************************************************************************************************
        # gnomad
        ####***************************************************************************************************************************************************************
        # 1. gnomad.exomes.r2.0.2.sites.vcf #(https://storage.googleapis.com/gnomad-public/release/2.0.2/vcf/exomes/gnomad.exomes.r2.0.2.sites.vcf.bgz)
        
        ####***************************************************************************************************************************************************************
        # pdb to uniprot
        ####***************************************************************************************************************************************************************
        # 1. pdb_chain_uniprot.csv (ftp://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/csv/pdb_chain_uniprot.csv.gz)
        ####***************************************************************************************************************************************************************
        # evmutation
        ####***************************************************************************************************************************************************************
        # 1. https://marks.hms.harvard.edu/evmutation/data/effects.tar.gz
#         evmutation_url = "https://marks.hms.harvard.edu/evmutation/data/effects.tar.gz"
#         response = urllib.request.urlopen(evmutation_url)
#         r = response.read().decode('utf-8')
#         f = open(self.db_path + "evmutation/effects.tar.gz", "w")
#         f.write(r)
#         f.close() 
#         self.gzip_decompress(self.db_path + "evmutation/effects.tar.gz", self.db_path + "evmutation/effects.tar.gz")   

        return(0)
        
    def get_uniprot_mapping_api(self, ids, id_from, id_to, tofile):
        query_ids = ' '.join(ids)
        url = 'http://www.uniprot.org/uploadlists/'
        params = {
        'from':id_from,
        'to':id_to,
        'format':'tab',
        'query': query_ids
        }          
        data = urllib.parse.urlencode(params).encode('utf-8')
        request = urllib.request.Request(url)
        contact = "joe.wu.ca@gmail.com"  # Please set your email address here to help us debug in case of problems.
        request.add_header('User-Agent', 'Python %s' % contact)
        response = urllib.request.urlopen(request, data=data)
        r = response.read().decode('utf-8')
        f = open(tofile, "w")
        f.write(r)
        f.close()
    
    def gzip_decompress (self, zipped_file_name, unzipped_file_name):
        inF = gzip.GzipFile(zipped_file_name, 'rb')
        s = inF.read()
        inF.close()
        outF = open(unzipped_file_name, 'wb')
        outF.write(s)
        outF.close()   
        
    def dms_foldchange_calculation(self, x, nonselect_cutoff, std_num):        
        x['mean_nonselect'] = np.mean([x['nonselect1'], x['nonselect2']])
        x['std_nonselect'] = np.std([x['nonselect1'], x['nonselect2']])
        x['mean_select'] = np.mean([x['select1'], x['select2']])
        x['std_select'] = np.std([x['select1'], x['select2']])
        x['mean_controlNS'] = np.mean([x['controlNS1'], x['controlNS2']])
        x['std_controlNS'] = np.std([x['controlNS1'], x['controlNS2']])
        x['mean_controlS'] = np.mean([x['controlS1'], x['controlS2']])
        x['std_controlS'] = np.std([x['controlS1'], x['controlS2']])

        nonselect_sub_mean = np.nan
        nonselect_sub_std = np.nan
        if std_num * x['std_controlNS'] + x['mean_controlNS'] < x['mean_nonselect'] - std_num * x['std_nonselect']:
            nonselect_sub_mean = x['mean_nonselect'] - x['mean_controlNS']
            nonselect_sub_std = np.sqrt(x['std_controlNS'] ** 2 + x['std_nonselect'] ** 2)
                    
        select_sub_mean = np.nan
        select_sub_std = np.nan                 
        if nonselect_sub_mean >= nonselect_cutoff:
            if std_num * x['std_select'] + x['mean_select'] >= x['mean_controlS']:
                select_sub_mean = np.abs(x['mean_select'] - x['mean_controlS'])
                select_sub_std = np.sqrt(x['std_controlS'] ** 2 + x['std_select'] ** 2)
        else:
            if std_num * x['std_controlS'] + x['mean_controlS'] < x['mean_select'] - std_num * x['std_select']:
                select_sub_mean = x['mean_select'] - x['mean_controlS']
                select_sub_std = np.sqrt(x['std_controlS'] ** 2 + x['std_select'] ** 2)
        
        foldchange_mean = np.nan
        foldchange_std = np.nan       
        if (not math.isnan(nonselect_sub_mean)) & (not math.isnan(select_sub_mean)):
            foldchange_mean = select_sub_mean / nonselect_sub_mean 
            foldchange_std = foldchange_mean * np.sqrt((select_sub_std / select_sub_mean) ** 2 + (nonselect_sub_std / nonselect_sub_mean) ** 2)
            
        # return (nonselect_sub_mean)
        
        x = pd.Series([nonselect_sub_mean, nonselect_sub_std, select_sub_mean, select_sub_std, foldchange_mean, foldchange_std])
        x.index = ['nonselect_sub_mean', 'nonselect_sub_std', 'select_sub_mean', 'select_sub_std', 'foldchange_mean', 'foldchange_std']
        
        return (x)

    def preprocess_dms(self, gene, nonselect_cutoff, std_num):
        dms_raw = pd.read_csv(self.db_path + 'dms/' + gene + '_raw.txt', sep='\t')        
        dms_raw = dms_raw.groupby(['wt_aa', 'mut_aa', 'pos'])['nonselect1', 'nonselect2', 'select1', 'select2', 'controlNS1', 'controlNS2', 'controlS1', 'controlS2'].sum().reset_index()
#         dms_raw['mean_nonselect'] = dms_raw.apply(lambda x : np.mean([x['nonselect1'],x['nonselect2']]),axis = 1)
#         dms_raw['std_nonselect'] = dms_raw.apply(lambda x : np.std([x['nonselect1'],x['nonselect2']]),axis = 1)
#         dms_raw['mean_select'] = dms_raw.apply(lambda x : np.mean([x['select1'],x['select2']]),axis = 1)
#         dms_raw['std_select'] = dms_raw.apply(lambda x : np.std([x['select1'],x['select2']]),axis = 1)
#         dms_raw['mean_controlNS'] = dms_raw.apply(lambda x : np.mean([x['controlNS1'],x['controlNS2']]),axis = 1)
#         dms_raw['std_controlNS'] = dms_raw.apply(lambda x : np.std([x['controlNS1'],x['controlNS2']]),axis = 1)
#         dms_raw['mean_controlS'] = dms_raw.apply(lambda x : np.mean([x['controlS1'],x['controlS2']]),axis = 1)
#         dms_raw['std_controlS'] = dms_raw.apply(lambda x : np.std([x['controlS1'],x['controlS2']]),axis = 1)
        
        x = dms_raw.loc[1:10, :].apply(lambda x : self.dms_foldchange_calculation(x, nonselect_cutoff, std_num), axis=1)
        z = pd.concat([dms_raw.loc[1:10, :], x], axis=1)
        z.to_csv(self.db_path + 'dms/' + gene + '_foldchange1.csv')
    
    def get_aa_ref_humsavar(self, x):
        x = x[33:47].strip()
        if 'p.' not in x:
            return np.nan
        else:
            y = x[2:5]
            if y in self.dict_aa3.keys():
                return self.dict_aa3[y]
            else:
                return '?'            
    
    def get_aa_pos_humsavar(self, x):
        x = x[33:47].strip()        
        if 'p.' not in x:
            return -1
        else:
            y = x[5:-3]
            if y.isdigit():
                return y
            else:
                return -1    
            
    def get_aa_alt_humsavar(self, x):
        x = x[33:47].strip()
        if 'p.' not in x:
            return np.nan
        else:
            if '=' in x:
                return '*'
            else:  
                y = x[-3:]
                if y in self.dict_aa3.keys():
                    return self.dict_aa3[y]
                else:
                    return '?' 
                 
    def get_aa_ref_clinvar(self, x):
        if '(p.' not in x:
            return np.nan
        else:
            y = x.split(':')[1].split('(')[1][2:5]
            if y in self.dict_aa3.keys():
                return self.dict_aa3[y]
            else:
                return '?'            
    
    def get_aa_pos_clinvar(self, x):
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
            
    def get_aa_alt_clinvar(self, x):
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
    
    def get_aa_properties(self):
        aa_properties = pd.read_table(self.db_path + 'other/aa.txt', sep='\t')
        aa_properties.drop_duplicates(inplace=True)
        aa_properties.drop(['aa_name'], axis=1, inplace=True)
        # aa_properties_features = ['aa','mw','pka','pkb','pi', 'cyclic','charged','charge','hydropathy_index','hydrophobic','polar','ionizable','aromatic','aliphatic','hbond','sulfur','pbr','avbr','vadw','asa']
        aa_properties_features = ['aa', 'mw', 'pka', 'pkb', 'pi', 'hi', 'pbr', 'avbr', 'vadw', 'asa', 'pbr_10', 'avbr_100', 'vadw_100', 'asa_100', 'cyclic', 'charge', 'positive', 'negative', 'hydrophobic', 'polar', 'ionizable', 'aromatic', 'aliphatic', 'hbond', 'sulfur', 'essential', 'size']
        aa_properties.columns = aa_properties_features    
        return (aa_properties)
    
    def get_blosums(self):
        df_blosums = None
        dict_blosums = {} 
        blosums = ['blosum30', 'blosum35', 'blosum40', 'blosum45', 'blosum50', 'blosum55', 'blosum60', 'blosum62', 'blosum65', 'blosum70', 'blosum75', 'blosum80', 'blosum85', 'blosum90', 'blosum95', 'blosum100']
        for blosum in blosums:
            blosum_raw = pd.read_table(self.db_path + "blosum/new_" + blosum + ".sij", sep='\t')
            col_names = blosum_raw.columns.get_values()
            b = blosum_raw.replace(' ', 0).astype('float') 
            bv = b.values + b.transpose().values
            bv[np.diag_indices_from(bv)] = bv[np.diag_indices_from(bv)] / 2
            blosum_new = pd.DataFrame(bv, columns=col_names)
            blosum_new['aa_ref'] = col_names
            blosum_new = pd.melt(blosum_new, id_vars=['aa_ref'])
            blosum_new.columns = ['aa_ref', 'aa_alt', blosum]
#             blosum_new[blosum] = 0-blosum_new[blosum]
            if df_blosums is None:
                df_blosums = blosum_new.copy()
            else:
                df_blosums = df_blosums.join(blosum_new[blosum]) 
            dict_blosums[blosum] = blosum_new   
        return [df_blosums, dict_blosums]       
    
    def get_funsums(self):
#         self.funsum_genes = ['P35520','P63279','P63165','P62166','Q9H3S4','P0DP23']
#         self.funsum_scores = ['fitness','fitness_reverse']
#         self.funsum_centralities = ['mean','median']
#         self.funsum_properties = ['aa_psipred','in_domain']
#         self.funsum_dmsfiles = self.db_path +'dms/all_dms_maps'
#         self.funsum_value_score_names = ['fitness']
#         self.funsum_weightedby_columns = ['']
#         self.funsum_weightedby_columns_inverse = [0]
        
        dict_funsums = {}
        aasum_prefix = 'funsum_'
        for value_score in self.funsum_scores: 
            for centrality in self.funsum_centralities: 
                for i in range(len(self.funsum_weightedby_columns)):
                    weighted_by = self.funsum_weightedby_columns[i]
                    try:                        
                        if weighted_by == '' :      
                            cur_name = aasum_prefix + value_score + '_' + centrality
                        else:
                            cur_name = aasum_prefix + value_score + '_' + centrality + '_' + weighted_by
                        
                        dict_funsums[cur_name] = pd.read_csv(self.db_path + 'funsum/' + cur_name + '.csv')                       
                        for property in self.funsum_properties:                                
                            if weighted_by == '' :      
                                cur_property_name = aasum_prefix + value_score + '_' + property.replace(',', '_') + '_' + centrality
                            else:
                                cur_property_name = aasum_prefix + value_score + '_' + property.replace(',', '_') + '_' + centrality + '_' + weighted_by
                                           
                            dict_funsums[cur_property_name] = pd.read_csv(self.db_path + 'funsum/' + cur_property_name + '.csv')
                    except:
                        print ("Error:", sys.exc_info()[0])
        return (dict_funsums)
    
    def get_mafsums_blosum(self):
        self.dict_mafsums = {}
        for maf_cutoff in [1]:
            self.dict_mafsums[str(maf_cutoff)] = pd.read_csv(self.db_path + 'mafsum/mafsum_melt_' + str(maf_cutoff) + '.csv')               
                    
    def get_mafsums(self):
        self.dict_mafsums = {}
        for maf_score in self.mafsum_scores:  
            for centrality in self.mafsum_centralities: 
                try:
                    self.dict_mafsums[maf_score + '_' + centrality] = pd.read_csv(self.db_path + 'mafsum/mafsum_' + maf_score + '_' + centrality + '.csv')
                    for property in self.mafsum_properties:                
                        self.dict_mafsums[maf_score + '_' + property + '_' + centrality] = pd.read_csv(self.db_path + 'mafsum/mafsum_' + maf_score + '_' + property + '_' + centrality + '.csv', index_col=0)
                except:               
                    print ("Error:", sys.exc_info()[0])

    def get_sum_corrletions(self, sum1, value1, sum2, value2):         
        allsums = pd.merge(sum1, sum2, how='left')
        allsum_ind = ~allsums.isnull().any(axis=1)
        allsums = allsums.loc[allsum_ind, :]
        return (stats.spearmanr(allsums[value1], allsums[value2])[0])               

    def obj_ftp(self, ftp_site):   
        cur_ftp = FTP(ftp_site)
        cur_ftp.login()
        return(cur_ftp)
    
    def ftp_download(self, ftp, ftp_path, ftp_file, local_file):
        try:
            if os.path.isfile(local_file):
                statinfo = os.stat(local_file)
                local_size = statinfo.st_size
                ftp.cwd(ftp_path)
                remote_size = ftp.size(ftp_path + ftp_file)
                if local_size != remote_size :
                    ftp.retrbinary('RETR ' + ftp_file, open(local_file, 'wb').write)
                    print (local_file + ' is updated.')
                    return('0')
                else:
                    print (local_file + ' is up to date.')
                    return('1')
            else: 
                ftp.cwd(ftp_path)
                ftp.retrbinary('RETR ' + ftp_file, open(local_file, 'wb').write)
                print (local_file + ' is downloaded.')
                return('0')
        except Exception as e:
            os.remove(local_file)
            print (str(e))
            return(str(e))
            
    def vcf_process(self, vcf_file, vcf_type):
        stime = time.time()  
        if vcf_type == 'exome':
            vcf_df = pd.read_csv(vcf_file, sep='\t')
            vcf_df.columns = ['chr', 'nt_pos', 'nt_ref', 'nt_alt', 'genotype']
            # merge with gnomad
            vcf_df1 = pd.merge(vcf_df, self.gnomad_nt, how='left')
        return (vcf_df1)
    
    
     
    def variants_process(self, gene_id, variants_df, seq_dict, k, nt_input, gnomad_available, gnomad_merge_id):
        total_stime = time.time()  
        print ("variants processing for : [" + str(gene_id) + "]")
        variants_df = variants_df.loc[variants_df['p_vid'].isin(seq_dict.keys()), :]
        variants_df['aa_len'] = variants_df['p_vid'].apply(lambda x: len(seq_dict[x]))
        variants_df['aa_pos'] = variants_df['aa_pos'].astype(int)
#         variants_df['chr'] = variants_df['chr'].astype(str)        
        varaints_df = variants_df.drop_duplicates()
                  
        ####***************************************************************************************************************************************************************
        #### merge with the asa (accessible solvent area)
        ####***************************************************************************************************************************************************************  
        stime = time.time() 
        self.pdb_asa = pd.read_csv(self.db_path + 'pdb/csv/asa_df.csv', header=None)
        self.pdb_asa.columns = ['aa_pos', 'aa_ref', 'asa_mean', 'asa_std', 'asa_count', 'p_vid']                    
        variants_df = pd.merge(variants_df, self.pdb_asa, how='left') 
        etime = time.time() 
        print ("merge with pdb asa: " + str(variants_df.shape[0]) + " took %g seconds\n" % (etime - stime))
    
        ####***************************************************************************************************************************************************************
        #### merge with the pdb secondary structure
        ####***************************************************************************************************************************************************************  
        stime = time.time() 
        self.pdbss = pd.read_csv(self.db_path + 'pdbss/pdbss_final.csv')
        variants_df = pd.merge(variants_df, self.pdbss, how='left') 
        # encode      
        pdbss1_dict = {'E':1, 'H':2, 'C':3, 'T':4}
        pdbss_dict = {'H':1, 'G':2, 'I':3, 'B':4, 'E':5, 'T':6, 'S':7, 'C':8}
        variants_df['aa_ss_encode'] = variants_df['aa_ss'].apply(lambda x:pdbss_dict.get(x, np.nan))
        variants_df['aa_ss1_encode'] = variants_df['aa_ss1'].apply(lambda x:pdbss1_dict.get(x, np.nan)) 
        etime = time.time()        
        print ("merge with pdbss: " + str(variants_df.shape[0]) + " took %g seconds\n" % (etime - stime))
          
        ####***************************************************************************************************************************************************************
        #### merge with the psipred secondary structure
        ####***************************************************************************************************************************************************************
        stime = time.time() 
        self.psipred = pd.read_csv(self.db_path + 'psipred/psipred_df.csv')
        self.psipred = self.psipred.drop_duplicates()
        variants_df = pd.merge(variants_df, self.psipred, how='left') 
        # encode      
        psipred_dict = {'E':1, 'H':2, 'C':3}
        variants_df['aa_psipred_encode'] = variants_df['aa_psipred'].apply(lambda x:psipred_dict.get(x, np.nan))
        etime = time.time()        
        print ("merge with psipred: " + str(variants_df.shape[0]) + " took %g seconds\n" % (etime - stime))
            
        ####***************************************************************************************************************************************************************
        #### merge with the pfam domain information
        ####****************************************************************************************************************************************
        stime = time.time() 
        self.pfam = pd.read_csv(self.db_path + 'pfam/9606.tsv', header=None, skiprows=3, sep='\t')
        self.pfam.columns = ['p_vid', 'a_start', 'a_end', 'e_start', 'e_end', 'hmm_id', 'hmm_name', 'type', 'hmm_start', 'hmm_end', 'hmm_length', 'bit_score', 'e_value', 'clan']
         
        p_vids = variants_df['p_vid'].unique()
        cur_pfam = self.pfam.loc[(self.pfam['p_vid'].isin(p_vids)), :]
        variants_df['pfam_id'] = np.nan
        count = 0
        for i in cur_pfam.index:
            cur_hmmid = cur_pfam.loc[i, "hmm_id"]
            cur_pvid = cur_pfam.loc[i, 'p_vid'] 
            cur_aa_start = cur_pfam.loc[i, 'a_start']
            cur_aa_end = cur_pfam.loc[i, 'a_end']
            variants_df.loc[(variants_df['p_vid'] == cur_pvid) & (variants_df.aa_pos >= cur_aa_start) & (variants_df.aa_pos <= cur_aa_end), 'pfam_id'] = cur_hmmid
            count += 1
           
        variants_df['in_domain'] = np.nan
        variants_df.loc[variants_df['pfam_id'].notnull(), 'in_domain'] = 1
        variants_df.loc[variants_df['pfam_id'].isnull(), 'in_domain'] = 0        
        etime = time.time()
        print ("merge with pfam: " + str(variants_df.shape[0]) + " took %g seconds\n" % (etime - stime)) 
            
        ####***************************************************************************************************************************************************************
        #### merge with the allel frequency properties
        ####***************************************************************************************************************************************************************
        if gnomad_available == 0:
            stime = time.time() 
            if nt_input == 1:    
                gnomad_nt = pd.read_csv(self.db_path+'gnomad/gnomad_output_snp_nt.txt',sep = '\t',dtype={"chr": str})
                variants_df = pd.merge(variants_df, self.gnomad_nt, how='left')
            else: 
                gnomad_aa = pd.read_csv(self.db_path+'gnomad/gnomad_output_snp_aa.txt',sep = '\t')
                gnomad_aa.rename(columns={gnomad_merge_id: 'p_vid'}, inplace=True)
                variants_df = pd.merge(variants_df, gnomad_aa, how='left') 
            etime = time.time()         
            print ("merge with gnomad: " + str(variants_df.shape[0]) + " took %g seconds\n" % (etime - stime))     
                
            
        ####***************************************************************************************************************************************************************   
        # Polyphen (http://genetics.bwh.harvard.edu/pph2/bgi.shtml)
        ####***************************************************************************************************************************************************************
        stime = time.time()
        polyphen_new_score = pd.read_csv(self.db_path + 'polyphen/org/' + gene_id + '_pph2-full.txt', sep='\t')
        polyphen_new_score.columns = np.char.strip(polyphen_new_score.columns.get_values().astype(str))
        polyphen_new_score = polyphen_new_score[['#o_acc', 'o_pos', 'o_aa1', 'o_aa2', 'pph2_prob']]
        polyphen_new_score.columns = ['p_vid', 'aa_pos', 'aa_ref', 'aa_alt', 'polyphen_new_score']
        polyphen_new_score['p_vid'] = polyphen_new_score['p_vid'].str.strip()
        polyphen_new_score['aa_ref'] = polyphen_new_score['aa_ref'].str.strip()
        polyphen_new_score['aa_alt'] = polyphen_new_score['aa_alt'].str.strip()
          
        polyphen_new_score['polyphen_new_score'] = polyphen_new_score['polyphen_new_score'].astype(str)
        polyphen_new_score['polyphen_new_score'] = polyphen_new_score['polyphen_new_score'].str.strip()
        polyphen_new_score = polyphen_new_score.loc[polyphen_new_score['polyphen_new_score'] != '?', :]
        polyphen_new_score['polyphen_new_score'] = polyphen_new_score['polyphen_new_score'].astype(float)
        polyphen_new_score = polyphen_new_score.loc[polyphen_new_score['aa_pos'].notnull(), :]
        polyphen_new_score['aa_pos'] = polyphen_new_score['aa_pos'].astype(int)
    
        polyphen_new_score.drop_duplicates(inplace = True)
        variants_df = pd.merge(variants_df, polyphen_new_score, how='left')
        etime = time.time()  
        print ("merge with polyphen: " + str(variants_df.shape[0]) + " took %g seconds\n" % (etime - stime))
        
        ####***************************************************************************************************************************************************************   
        # SIFT and PROVEAN (http://provean.jcvi.org/protein_batch_submit.php?species=human)
        ####***************************************************************************************************************************************************************
        stime = time.time()
        sift_new_score = pd.read_csv(self.db_path + 'provean/org/' + gene_id + '_provean.tsv', sep='\t')[['PROTEIN_ID', 'POSITION', 'RESIDUE_REF', 'RESIDUE_ALT', 'SCORE', 'SCORE.1']]
        sift_new_score.columns = ['p_vid', 'aa_pos', 'aa_ref', 'aa_alt', 'provean_new_score', 'sift_new_score']
        sift_new_score = sift_new_score.drop_duplicates()
        variants_df = pd.merge(variants_df, sift_new_score, how='left')
        etime = time.time()
        print ("merge with sift and provean: " + str(variants_df.shape[0]) + " took %g seconds\n" % (etime - stime))
        
        ####***************************************************************************************************************************************************************
        #### EV-mutation score (https://marks.hms.harvard.edu/evmutation/human_proteins.html)
        ####***************************************************************************************************************************************************************                
        stime = time.time()
        self.evm_score = pd.read_csv(self.db_path + 'evmutation/csv/evmutation_df_org.csv')
 
        # ['mutation','aa_pos','aa_ref','aa_alt','evm_epistatic_score','evm_independent_score','evm_frequency','evm_conservation']
        variants_df = pd.merge(variants_df, self.evm_score, how='left')
        etime = time.time()
        print ("merge with evmutation: " + str(variants_df.shape[0]) + " took %g seconds\n" % (etime - stime))
         
        ####***************************************************************************************************************************************************************
        #### envision
        ####***************************************************************************************************************************************************************
        stime = time.time()
        self.envision_score = pd.read_csv(self.db_path + 'envision/csv/envision_score_for_extrapolation_processed.csv')
        variants_df = pd.merge(variants_df, self.envision_score, how='left')
        etime = time.time()
        print ("merge with envision: " + str(variants_df.shape[0]) + " took %g seconds\n" % (etime - stime))
        
        ###***************************************************************************************************************************************************************
        ### primateAI
        ###***************************************************************************************************************************************************************
        if nt_input == 1:
            stime = time.time()
            self.primateai_score = pd.read_csv(self.db_path + 'primateai/PrimateAI_scores_v0.2.tsv', skiprows = 10, sep = '\t')
            self.primateai_score.columns = ['chr_org','nt_pos','nt_ref','nt_alt','aa_ref','aa_alt','strand','codon','ucsc_id','exac_af','primateai_score']
            self.primateai_score['chr'] = self.primateai_score['chr_org'].apply(lambda x:  x[3:])
            self.primateai_score['chr'] =   self.primateai_score['chr'].astype(str)
            variants_df = pd.merge(variants_df,self.primateai_score[['chr','nt_pos','nt_ref','nt_alt','aa_ref','aa_alt','primateai_score']],how = 'left')
            etime = time.time()
            print ("merge with primateai: " + str(variants_df.shape[0]) + " took %g seconds\n" % (etime - stime)) 
            
        ####***************************************************************************************************************************************************************
        #### aa_ref and aa_alt AA properties
        ####***************************************************************************************************************************************************************           
        aa_properties_features = self.aa_properties.columns                
        aa_properties_ref_features = [x + '_ref' for x in aa_properties_features]
        aa_properties_alt_features = [x + '_alt' for x in aa_properties_features]   
        aa_properties_ref = self.aa_properties.copy()
        aa_properties_ref.columns = aa_properties_ref_features
        aa_properties_alt = self.aa_properties.copy()
        aa_properties_alt.columns = aa_properties_alt_features                
        variants_df = pd.merge(variants_df, aa_properties_ref, how='left')
        variants_df = pd.merge(variants_df, aa_properties_alt, how='left')
         
 
        for x in aa_properties_features:
            if x != 'aa':
                variants_df[x+'_delta'] = variants_df[x+'_ref'] - variants_df[x+'_alt']        
          
        print ("merge with aa properties: " + str(variants_df.shape[0]))
  
        ####***************************************************************************************************************************************************************
        #### flanking kmer AA column and properties  
        ####***************************************************************************************************************************************************************
        for i in range(1, k + 1):    
            aa_left = 'aa_ref_' + str(i) + '_l'
            aa_right = 'aa_ref_' + str(i) + '_r'
            variants_df[aa_left] = variants_df[['p_vid', 'aa_pos']].apply(lambda x: seq_dict[x['p_vid']][max(0, (x['aa_pos'] - i - 1)):max(0, (x['aa_pos'] - i))], axis=1)
            variants_df[aa_right] = variants_df[['p_vid', 'aa_pos']].apply(lambda x: seq_dict[x['p_vid']][(x['aa_pos'] + i - 1):(x['aa_pos'] + i)], axis=1)
            aa_properties_ref_kmer_features = [x + '_ref_' + str(i) + '_l' for x in aa_properties_features]
            aa_properties_ref_kmer = self.aa_properties.copy()
            aa_properties_ref_kmer.columns = aa_properties_ref_kmer_features
            variants_df = pd.merge(variants_df, aa_properties_ref_kmer, how='left')
            aa_properties_ref_kmer_features = [x + '_ref_' + str(i) + '_r' for x in aa_properties_features]
            aa_properties_ref_kmer = self.aa_properties.copy()
            aa_properties_ref_kmer.columns = aa_properties_ref_kmer_features
            variants_df = pd.merge(variants_df, aa_properties_ref_kmer, how='left')
        print ("merge with kmer properties: " + str(variants_df.shape[0]))
        
        ####***************************************************************************************************************************************************************
        #### merge with the blosum properties
        ####***************************************************************************************************************************************************************
        df_blosums = self.df_blosums          
        variants_df = pd.merge(variants_df, df_blosums, how='left')
        print ("merge with blosums: " + str(variants_df.shape[0]))
        
        ####***************************************************************************************************************************************************************
        #### merge with the funsum properties
        ####***************************************************************************************************************************************************************
        for funsum_key in self.dict_sums['funsum'].keys():
            variants_df = pd.merge(variants_df, self.dict_sums['funsum'][funsum_key], how='left')
            print ("merge with funsums - " + funsum_key + " :" + str(variants_df.shape[0]))
  
        ####***************************************************************************************************************************************************************
        ## Accessibility  
        ####***************************************************************************************************************************************************************
        variants_df = pd.merge(variants_df,self.accsum_df,how = 'left')
        print ("merge with accsums: " + str(variants_df.shape[0]))
 
        ####*************************************************************************************************************************************************************
        #### Encode name features (one hot encode)
        ####*************************************************************************************************************************************************************        
        variants_df['aa_ref_encode'] = variants_df['aa_ref'].apply(lambda x: self.aa_encode_notnull(x))
        variants_df['aa_alt_encode'] = variants_df['aa_alt'].apply(lambda x: self.aa_encode_notnull(x))
        
        for aa in self.lst_aa_20:
            variants_df['aa_ref_' + aa] = variants_df['aa_ref'].apply(lambda x: int(x == aa))

        for i in range(1, k + 1):    
            aa_left = 'aa_ref_' + str(i) + '_l'
            aa_right = 'aa_ref_' + str(i) + '_r'
            variants_df[aa_left + '_encode'] = variants_df[aa_left].apply(lambda x: self.aa_encode_notnull(x))
            variants_df[aa_right + '_encode'] = variants_df[aa_right].apply(lambda x:  self.aa_encode_notnull(x))
            
            for aa in self.lst_aa_20:
                variants_df['aa_ref_' + str(i) + '_l_' + aa] = variants_df['aa_ref_' + str(i) + '_l'].apply(lambda x: int(x == aa))
                variants_df['aa_ref_' + str(i) + '_r_' + aa] = variants_df['aa_ref_' + str(i) + '_r'].apply(lambda x: int(x == aa))
            
              
        total_etime = time.time()
        print("Variants processing took %g seconds\n" % (total_etime - total_stime)) 
        
         
        ####~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #### old and backup codes
        ####~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#         variants_df.to_csv(self.db_path + 'dms/clinvar_gnomad_process_after_primateai.csv',index = False)

#         print(str(variants_df.loc[variants_df['primateai_score'].notnull(),:].shape))        
#         x = self.primateai_score.groupby(['chr','nt_pos','nt_ref','nt_alt','aa_ref','aa_alt']).size().reset_index()
#         y = x.loc[x[0]>1,:]
#         
#         z = variants_df.groupby(['chr','nt_pos','nt_ref','nt_alt','aa_ref','aa_alt']).size().reset_index()
#         z.loc[z[0]>1]
        
        # MCAP (http://bejerano.stanford.edu/MCAP/) , not batch mode, pre_computered score downloaded
#         mcap_score = pd.read_csv(self.db_path +'mcap/mcap_v1_0.txt',sep = '\t')
#         # mcap_score[['PROTEIN_ID','POSITION','RESIDUE_REF','RESIDUE_ALT','SCORE','SCORE.1']]
#         mcap_score.columns = ['chr','nt_pos','nt_ref','nt_alt','mcap_score']
#         #mcap_score = mcap_score.drop_duplicates()
#         variants_df = pd.merge(variants_df,mcap_score,how = 'left')
#         variants_df['mcap_score'] = 0-variants_df['mcap_score']
        
#         varaint_df = pd.read_csv(self.db_path + 'dms/clinvar_gnomad_process_after_pfam.csv')
#         

     
        ####***************************************************************************************************************************************************************          
        # use aa_pos to check if aa_ref is consistent with the sequence     
        ####***************************************************************************************************************************************************************
#         variants_df['aa_ref_chk'] = variants_df.apply(lambda x: seq_dict[x.p_vid][(x.aa_pos-1):(x.aa_pos)],axis = 1)
#         #stop position
#         variants_df.loc[variants_df['aa_len'] == variants_df['aa_pos'] - 1,'aa_ref_chk'] = '_'
#         #remove the aa_ref record that is not consistent with the seqeunce
#         variants_df = variants_df.loc[variants_df['aa_ref_chk'] == variants_df['aa_ref'],:]

        
        
        return (variants_df)
              
    def aa_encode_notnull(self, x):
        if x not in self.dict_aaencode.keys():
            return -1
        else:
            return self.dict_aaencode[x]
      
    def switch_aa(self, x):
        if x.gnomad_af > 0.5:            
            aa_ref = x.aa_ref
            aa_alt = x.aa_alt
            x.gnomad_af = 1 - x.gnomad_af
            x.aa_ref = aa_alt
            x.aa_alt = aa_ref
        return(x)   
        
