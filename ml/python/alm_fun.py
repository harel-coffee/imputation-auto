#!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3
import numpy as np
import pandas as pd
import smtplib
from email.message import EmailMessage
import csv
import os
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import matplotlib.pyplot as plt 
import matplotlib.path as mpath
import matplotlib.patches as patches  
from matplotlib.lines import Line2D  
from matplotlib.gridspec import GridSpec
import matplotlib.collections as collections

import operator
import itertools
import time
import math
import random
import codecs
import pydotplus 
import copy
import pickle

# sklearn
import tensorflow as tf
from sklearn import linear_model as lm
from sklearn import svm
from sklearn import feature_selection as fs
from sklearn import model_selection as ms
from sklearn import ensemble as es
from sklearn import tree
from sklearn import pipeline
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics.ranking import roc_auc_score
 
from scipy import stats 
from functools import partial
from datetime import datetime
from numpy import inf
from cgi import log
from decimal import *
from collections import Counter
sns.set(rc={'axes.facecolor':'#C0C0C0'}) 




def quality_evaluation(name, path, extra_train_file, use_extra_train_data, train_file, test_file, dependent_variable, feature_engineer, ml_type, estimators, cv_split_method, cv_split_folds, verbose, onehot_features, initial_features, percent_min_feature, quality_feature, quality_feature_direction, quality_feature_cutoffs):
    cutoff_objects = []    
    # generate all quality cutoff permutation for current quality_feature_cutoffs
    for quality_feature_cutoff in itertools.product(*quality_feature_cutoffs):
        ex = alm_ml(name, path, extra_train_file, use_extra_train_data, train_file, test_file, dependent_variable, feature_engineer, ml_type, estimators, cv_split_method, cv_split_folds, verbose, onehot_features, initial_features, percent_min_feature, quality_feature, quality_feature_direction, quality_feature_cutoff)
        ex.data_filter() 
        ex.feature_evaluation()
        cutoff_objects.append(ex) 
    return  cutoff_objects
                  
def show_msg(infile, verbose, msg):
    infile.write(msg + '\n')
    if (verbose == 1): print (msg)     
        
def show_start_msg(infile,verbose,class_name,fun_name):
    msg = 'Class: [' + class_name + '] Fun: [' + fun_name + '] .... start @' + str(datetime.now())
    show_msg(infile,verbose,msg)
    return(time.time()) 
        
def show_end_msg(infile,verbose,class_name,fun_name,stime):
    msg = 'Class: [' + class_name + '] Fun: [' + fun_name + '] .... done @' + str(datetime.now()) + ' time spent: %g seconds' % (time.time() - stime)
    show_msg(infile,verbose,msg)

def dependency_matrix(x, f, symmetric=1):
    n = x.shape(1)
    names = x.columns.get_values()
    dependency_matrix = pd.DataFrame(np.zeros((n, n)), columns=names)   
    score_interaction_forward.index = names
     
    for i in range(n):
        for j in range(i, n):
            dependency_matrix.loc[names[i], names[j]] = f(x[names[i]], x[names[j]]) 
            if i == j:
                break
            if symmetric == 1:
                dependency_matrix.loc[names[j], names[i]] = dependency_matrix.loc[names[i], names[j]]
            else:
                dependency_matrix.loc[names[j], names[i]] = f(x[names[j]], x[names[i]]) 
    return dependency_matrix            
                 
# pearson correlation coefficient

 
def pcc_cal(x, y, if_abs=False):  
    x = np.array(x)
    y = np.array(y)
    
    x_nullidx = list(np.where(np.isnan(x))[0])
    y_nullidx = list(np.where(np.isnan(y))[0])
    
    nullidx = set(x_nullidx + y_nullidx) 
    idx = list(set(range(len(x))) - nullidx)
    
    if len(idx) == 0 :
        return(np.nan)
    
    x = x[idx]
    y = y[idx]
      
    r = stats.pearsonr(x, y)
    pcc = r[0]
     
    if math.isnan(pcc):
        pcc = 0
    if if_abs:
        return (np.abs(pcc))
    else:        
        return pcc 

 
def spc_cal(x, y, if_abs=False):  
    x = np.array(x)
    y = np.array(y)
    
    x_nullidx = list(np.where(np.isnan(x))[0])
    y_nullidx = list(np.where(np.isnan(y))[0])
    
    nullidx = set(x_nullidx + y_nullidx) 
    idx = list(set(range(len(x))) - nullidx)
    
    if len(idx) == 0 :
        return(np.nan)
    
    x = x[idx]
    y = y[idx]
      
    r = stats.spearmanr(x, y)
    spc = r[0]
     
    if math.isnan(spc):
        spc = 0

    if if_abs:
        return (np.abs(spc))
    else:        
        return spc 

 
def linear_rmse_cal(y, y_predicted):
    y_predicted = y_predicted.reshape(y_predicted.shape[0], 1)
    es = lm.ElasticNet(alpha=1.0, l1_ratio=0.5)
    es.fit(y_predicted, y)
    y1 = es.predict(y_predicted)
    return(np.sqrt(np.sum((y - y1) ** 2) / len(y)))    
    # pearson correlation coefficient - pvalue

 
def pcc_pvalue(x, y):
    r = stats.pearsonr(x, y)
    return r[1]   

 
def rmse_cal(y, y_predicted):
    # if y_predicted contains nan
    y = np.array(y)
    y_predicted = np.array(y_predicted)
    notnull_idx = ~np.isnan(y_predicted)
    y_predicted = y_predicted[notnull_idx]
    y = y[notnull_idx]
    
    return(np.sqrt(np.sum((y - y_predicted) ** 2) / len(y)))    

 
def mse_cal(y, y_predicted):
    return(np.sum((y - y_predicted) ** 2) / len(y))    

       
def auprc_cal(y, y_predicted):
    prc = round(metrics.average_precision_score(y, y_predicted), 4)
    precision, recall, thresholds = metrics.precision_recall_curve(y, y_predicted)
    return [prc, precision, recall, thresholds]     

 
def auroc_cal(y, y_predicted):
    roc = round(metrics.roc_auc_score(y, y_predicted), 4)
    fpr, tpr, thresholds = metrics.roc_curve(y, y_predicted)
    return [roc, fpr, tpr, thresholds]  

 
def roc_prc_cal(y, y_predicted):
    metric = {}
    reverse = 0
    roc = round(metrics.roc_auc_score(y, y_predicted), 4)
    if roc < 0.5:
        reverse = 1
        y_predicted = 0 - y_predicted
        roc = round(metrics.roc_auc_score(y, y_predicted), 4)
     
    prc = round(metrics.average_precision_score(y, y_predicted), 4)
    precision, recall, thresholds = metrics.precision_recall_curve(y, y_predicted)
    fpr, tpr, thresholds = metrics.roc_curve(y, y_predicted)
    metric['roc'] = roc
    metric['prc'] = prc
    metric['fpr'] = fpr
    metric['tpr'] = tpr
    metric['precision'] = precision
    metric['recall'] = recall
    metric['reverse'] = reverse    
    return metric

 
def get_classification_metrics(metrics_name, round_digits, y, y_predicted):
    if metrics_name == 'auroc':
        result = round(metrics.roc_auc_score(y, y_predicted), round_digits)
     
    if metrics_name == 'auprc':
        result = round(metrics.average_precision_score(y, y_predicted), round_digits)
     
    if metrics_name == 'neg_log_loss':
        result = round(metrics.log_loss(y, y_predicted), round_digits)
    return result

 
def classification_metrics(y, y_predicted, cutoff=np.nan, test_precision=0.9, test_recall=0.9,find_best_cutoff=0):    
    y_classes = np.unique(y)
    # if y_predicted contains nan
    y = np.array(np.squeeze(y))
    y_predicted = np.array(np.squeeze(y_predicted))
#     notnull_idx = ~np.isnan(y_predicted)
#     y_predicted = y_predicted[notnull_idx]
#     y = y[notnull_idx]
        
    #if y_predicted contains nan, then do random prediction on the Nan     
    for i in range(len(y_predicted)):
        if np.isnan(float(y_predicted[i])) == True :
#             y_predicted[i] = random.choice(y_classes)            
#             y_predicted[i] = np.random.uniform(np.nanmin(y_predicted),np.nanmax(y_predicted),1)[0]
            y_predicted[i] = np.nanmax(y_predicted)
        
    # check if multi-classification
    multiclass_metrics_dict = {}

    if len(y_classes) > 2:
        for y_class in y_classes:
            y_new = y.copy()
            y_new[y == y_class] = 1
            y_new[y != y_class] = 0
            multiclass_metrics_dict[y_class] = (classification_metrics_sub(y_new, y_predicted, cutoff, test_precision, test_recall))
     
        for y_class in y_classes:
            cur_y_predicted = multiclass_metrics_dict[y_class][0]
            cur_y_predicted[cur_y_predicted == 1] = y_class
            cur_y_predicted = pd.DataFrame(cur_y_predicted, columns=[y_class])
            if 'combined_best_y_predicted_df' not in locals():
                combined_best_y_predicted_df = cur_y_predicted
            else:
                combined_best_y_predicted_df = pd.concat([combined_best_y_predicted_df, cur_y_predicted], axis=1)
                        
            cur_y_metrics = multiclass_metrics_dict[y_class][1]
            cur_y_metrics = pd.DataFrame(list(cur_y_metrics.items()), columns=['key', y_class])
            cur_y_metrics.set_index('key', drop=True, inplace=True)
            if 'combined_metrics_dict_df' not in locals():
                combined_metrics_dict_df = cur_y_metrics
            else:
                combined_metrics_dict_df = pd.concat([combined_metrics_dict_df, cur_y_metrics], axis=1)
                 
            combined_best_y_predicted = combined_best_y_predicted_df.mean(axis=1)
            combined_metrics_dict = pd.DataFrame(combined_metrics_dict_df.mean(axis=1)).T.to_dict('records')[0]
    else:        
        [combined_best_y_predicted, combined_metrics_dict] = classification_metrics_sub(y, y_predicted, cutoff, test_precision, test_recall,find_best_cutoff)
             
    return [combined_best_y_predicted, combined_metrics_dict, multiclass_metrics_dict]

 
def classification_metrics_sub(y, y_predicted, cutoff=np.nan, test_precision=0.9, test_recall=0.9, find_best_cutoff=0):      
    reverse = 0
    roc = round(metrics.roc_auc_score(y, y_predicted), 4)
    if roc < 0.5:
        reverse = 1
        y_predicted = 0 - y_predicted
#         y = 1-y
    
    if find_best_cutoff == 1: 
        y_p_max = max(y_predicted)
        y_p_min = min(y_predicted)
         
        y_p_cutoffs = np.arange(y_p_min, y_p_max, (y_p_max - y_p_min) / 100)
           
        best_mcc = -1
        best_cutoff = -inf       
        if math.isnan(cutoff): 
            for cutoff in y_p_cutoffs:
                cur_y_predicted = np.zeros(len(y_predicted))
                cur_y_predicted[y_predicted > cutoff] = 1
                if cutoff != best_cutoff:
                    cur_mcc = metrics.matthews_corrcoef(y, cur_y_predicted)
                    if cur_mcc >= best_mcc:
                        best_mcc = cur_mcc
                        best_cutoff = cutoff
                        best_y_predicted = cur_y_predicted
        else:
            best_cutoff = cutoff 
            best_y_predicted = np.zeros(len(y_predicted))
            best_y_predicted[y_predicted > cutoff] = 1
    else:
        best_mcc = np.nan
        best_cutoff = np.nan
        best_y_predicted = None
      
    metrics_dict = {}    
    y_p = np.zeros(len(y))
    y_p[y_predicted > best_cutoff] = 1     
    tp = ((y == 1) & (y_p == 1)).sum() 
    fp = ((y == 0) & (y_p == 1)).sum()
    tn = ((y == 0) & (y_p == 0)).sum()
    fn = ((y == 1) & (y_p == 0)).sum()    
    metrics_dict = {}
    metrics_dict['tp'] = tp
    metrics_dict['fp'] = fp
    metrics_dict['tn'] = tn
    metrics_dict['fn'] = fn    
    # metrics_dict['confusion'] = [tp,fp,tn,fn]
    metrics_dict['precision'] = float(round(Decimal(tp / (tp + fp)), 4))
    metrics_dict['recall'] = float(round(Decimal(tp / (tp + fn)), 4))
    metrics_dict['specificity'] = float(round(Decimal(tn / (tn + fp)), 4))
    metrics_dict['accuracy'] = float(round(Decimal((tp + fp) / (tn + fp + tp + fn)), 4))
    metrics_dict['mcc'] = float(round(Decimal((tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))), 4))
     
    metrics_dict['prior'] = get_prior(y)
    metrics_dict['auroc'] = float(round(Decimal(metrics.roc_auc_score(y, y_predicted)), 4))
    metrics_dict['auprc'] = float(round(metrics.average_precision_score(y, y_predicted), 4))
    metrics_dict['auprc-prior'] = float(round(metrics.average_precision_score(y, y_predicted) - metrics_dict['prior'], 4)) 
     
    precisions, recalls, prc_thresholds = metrics.precision_recall_curve(y, y_predicted)
    fprs, tprs, roc_thresholds = metrics.roc_curve(y, y_predicted)
    
    precisions[-1] = precisions[-2] #  The precsison dosen't neccsarily starts from 1
     
    metrics_dict['precisions'] = precisions
    metrics_dict['recalls'] = recalls
    metrics_dict['prc_thresholds'] = prc_thresholds
     
    metrics_dict['fprs'] = fprs
    metrics_dict['tprs'] = tprs
    metrics_dict['roc_thresholds'] = roc_thresholds
     
    precision_diff = precisions - test_precision
    if max(precision_diff) > 0:
        min_diff = min(abs(precision_diff))
        idx_precision = list(abs(precision_diff)).index(min_diff)
        metrics_dict['recall_fixed_precision'] = round(recalls[idx_precision],4)
    else:
        metrics_dict['recall_fixed_precision'] = 0
     
    recall_diff = abs(recalls - test_recall)
    min_diff = min(recall_diff)
    idx_recall = list(recall_diff).index(min_diff)
    metrics_dict['precision_fixed_recall'] = round(precisions[idx_recall],4)
 
    best_cutoff = float(round(Decimal(best_cutoff), 4))      
    best_mcc = float(round(Decimal(best_mcc), 4))
    metrics_dict['best_cutoff'] = best_cutoff
    metrics_dict['best_mcc'] = best_mcc
    metrics_dict['reverse'] = reverse
    metrics_dict['size'] = len(y)
    return([best_y_predicted, metrics_dict])


def plot_prc(y, y_predicted, plot_name, fig_w, fig_h, cutoff=np.nan, test_precision=0.9, test_recall=0.9, title_name='AUPRC'):
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax_auprc = plt.subplot()   
    ax_auprc.margins(1,1)
    color_lst = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000']
    
    if y_predicted.shape[1] > 1:
        if y_predicted.shape[1] > 8:
#             lst_colors = list(matplotlib.colors.CSS4_COLORS.keys())
            lst_colors = color_lst
        else:
            lst_colors = list(matplotlib.colors.BASE_COLORS.keys())
        
        for i in range(y_predicted.shape[1]):
            metrics_dict = classification_metrics(y, y_predicted[y_predicted.columns[i]], cutoff, test_precision, test_recall)[1]
#             metrics_dict['precisions'][-1] = metrics_dict['precisions'][-2]                          
#             ax_auprc.plot(metrics_dict['recalls'] + i*0.005 ,metrics_dict['precisions'] + i*0.005,marker='o',markersize=5,color = lst_colors[i],label = y_predicted.columns[i] + '(' + str(metrics_dict['auprc']) + ')')
            ax_auprc.plot(metrics_dict['recalls'], metrics_dict['precisions'], marker='o', markersize=2, color=lst_colors[i], label = y_predicted.columns[i] + '(' + str(metrics_dict['auprc']) + ',' + str(metrics_dict['recall_fixed_precision']) +')')
        # Now add the legend with some customizations.
        legend = ax_auprc.legend(loc='upper left',bbox_to_anchor=(1, 1),shadow=True)        
        # The frame is matplotlib.patches.Rectangle instance surrounding the legend.
        frame = legend.get_frame()
        frame.set_facecolor('0.90')
        
        # Set the fontsize
        for label in legend.get_texts():
            label.set_fontsize('large')
        
        for label in legend.get_lines():
            label.set_linewidth(1.5)  # the legend line width    
    else:        
        metrics_dict = classification_metrics(y, y_predicted, cutoff, test_precision, test_recall)[1]    
        fig = plt.figure() 
        ax_auprc = plt.subplot()   
#         metrics_dict['precisions'][-1] = metrics_dict['precisions'][-2]          
        ax_auprc.plot(metrics_dict['recalls'], metrics_dict['precisions'], marker='o', markersize=2, color='black', label='AUPRC: (' + str(metrics_dict['auprc']) + ')')
        # Now add the legend with some customizations.
        legend = ax_auprc.legend(loc='upper right', shadow=True)
        frame = legend.get_frame()
        frame.set_facecolor('0.90')
        for label in legend.get_texts():
            label.set_fontsize('large')
        
        for label in legend.get_lines():
            label.set_linewidth(1.5)  # the legend line width   
    pass
    ax_auprc.set_title(title_name, size=15)
    ax_auprc.set_xlabel('Recall')
    ax_auprc.set_ylabel('Precision') 
    ax_auprc.set_xlim(0, 1)
    ax_auprc.set_ylim(0, 1.05)
    fig.tight_layout()
    plt.savefig(plot_name)   


def plot_roc(y, y_predicted, plot_name, fig_w, fig_h, cutoff=np.nan, test_precision=0.9, test_recall=0.9, title_name='AUROC'):
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax_auroc = plt.subplot()   
    
    if y_predicted.shape[1] > 1:
        if y_predicted.shape[1] > 8:
            lst_colors = list(matplotlib.colors.CSS4_COLORS.keys())
        else:
            lst_colors = list(matplotlib.colors.BASE_COLORS.keys())
        
        for i in range(y_predicted.shape[1]):
            metrics_dict = classification_metrics(y, y_predicted[y_predicted.columns[i]], cutoff, test_precision, test_recall)[1]  
            ax_auroc.plot(metrics_dict['fprs'], metrics_dict['tprs'], marker='o', markersize=2, color=lst_colors[i], label=y_predicted.columns[i] + '(' + str(metrics_dict['auroc']) + ')')
            
        # Now add the legend with some customizations.
        legend = ax_auroc.legend(loc='bottom left', shadow=True)
        
        # The frame is matplotlib.patches.Rectangle instance surrounding the legend.
        frame = legend.get_frame()
        frame.set_facecolor('0.90')
        
        # Set the fontsize
        for label in legend.get_texts():
            label.set_fontsize('large')
        
        for label in legend.get_lines():
            label.set_linewidth(1.5)  # the legend line width    
    else:
        
        metrics_dict = classification_metrics(y, y_predicted, cutoff, test_precision, test_recall)    
        fig = plt.figure() 
        ax_auroc = plt.subplot()            
        ax_auroc.plot(metrics_dict['fprs'], metrics_dict['tprs'], marker='o', markersize=2, color='black', label='AUROC:(' + str(metrics_dict['auroc']) + ')')
    pass
    ax_auroc.set_title(title_name, size=15)
    ax_auroc.set_xlabel('False Positive Rate')
    ax_auroc.set_ylabel('True Positive Rate') 
    ax_auroc.set_xlim(0, 1)
    ax_auroc.set_ylim(0, 1)
    # fig.tight_layout()
    plt.savefig(plot_name)   


def plot_prc_backup(metrics_dict):
    fig = plt.figure() 
    ax_auprc = plt.subplot()            
    ax_auprc.scatter(metrics_dict['recalls'], metrics_dict['precisions'], s=10)    
     
    diff = abs(metrics_dict['prc_thresholds'] - metrics_dict['best_cutoff'])
    diff_min = min(diff)
    cutoff_ind = list(diff).index(diff_min)
     
    cutoff_recall = round(metrics_dict['recalls'][cutoff_ind], 3)
    cutoff_precision = round(metrics_dict['precisions'][cutoff_ind], 3)
    best_cutoff = round(metrics_dict['best_cutoff'], 3)
     
    ax_auprc.scatter(cutoff_recall, cutoff_precision, color='red')  
    ax_auprc.annotate('[x: ' + str(cutoff_recall) + ' y: ' + str(cutoff_precision) + ' cutoff: ' + str(best_cutoff) + ']',
                xy=(cutoff_recall, cutoff_precision), xytext=(cutoff_recall - 0.1, cutoff_recall - 0.1), arrowprops=dict(facecolor='black', shrink=0.05))
       
    if metrics_dict['reverse'] == 1:      
        ax_auprc.set_title(' AUPRC: ' + str(metrics_dict['auprc']) + ' [reverse]', size=15)
    else:
        ax_auprc.set_title(' AUPRC: ' + str(metrics_dict['auprc']), size=15)
    pass
    ax_auprc.set_xlabel('Recall')
    ax_auprc.set_ylabel('Precision') 
    ax_auprc.set_xlim(0, 1)
    ax_auprc.set_ylim(0, 1)
    fig.tight_layout()
    plt.savefig(plot_name)   

     
def plot_roc_backup(metrics_dict):
    fig = plt.figure()    
    ax_auroc = plt.subplot()            
    ax_auroc.scatter(metrics_dict['fprs'], metrics_dict['tprs'], s=10)   
     
    diff = abs(metrics_dict['roc_thresholds'] - metrics_dict['best_cutoff'])
    diff_min = min(diff)
    cutoff_ind = list(diff).index(diff_min)
     
    cutoff_tpr = round(metrics_dict['tprs'][cutoff_ind], 3)
    cutoff_fpr = round(metrics_dict['fprs'][cutoff_ind], 3)
    best_cutoff = round(metrics_dict['best_cutoff'], 3)
     
    ax_auroc.scatter(cutoff_fpr, cutoff_tpr, color='red')  
    ax_auroc.annotate('[x: ' + str(cutoff_fpr) + ' y: ' + str(cutoff_tpr) + ' cutoff: ' + str(best_cutoff) + ']',
                xy=(cutoff_fpr, cutoff_tpr), xytext=(cutoff_fpr + 0.1, cutoff_tpr - 0.1),)
       
    if metrics_dict['reverse'] == 1:      
        ax_auroc.set_title(' AUROC: ' + str(metrics_dict['auroc']) + ' [reverse]', size=15)
    else:
        ax_auroc.set_title(' AUROC: ' + str(metrics_dict['auroc']), size=15)
    pass
    ax_auroc.set_xlabel('False Positive Rate')
    ax_auroc.set_ylabel('Ture Positive Rate') 
    ax_auroc.set_xlim(0, 1)
    ax_auroc.set_ylim(0, 1)         

             
def get_prior(y):
    return (y == 1).sum() / len(y)

     
def label_vector(y):
    y_unique = np.unique(y)
    y_vector = np.zeros([len(y), len(y_unique)])
    for i in range(len(y_unique)):
        y_vector[:, i] = (y == y_unique[i]).astype(int)
    return y_vector

 
def down_sampling(x):   
    x_p = x.loc[x.label == 1, :]
    x_n = x.loc[x.label == 0, :]
    if x_p.shape[0] > x_n.shape[0]:
        x_p = x_p.loc[np.random.permutation(x_p.index)[range(x_n.shape[0])], :]
    else:
        x_n = x_n.loc[np.random.permutation(x_n.index)[range(x_p.shape[0])], :]        
    x = pd.concat([x_p, x_n])
    return(x)

 
def sentence_count(text_string):
    return len(text_string.split('. ')) + 1

     
def word_count(text_string):
    return len(text_string.split()) + 1

 
def word_frequency(text_string, words, reg, normalization=0):
    text_frequency = dict(Counter(re.findall(reg, text_string)))
    if len(words) == 0:
        init_frequency = text_frequency
    else:
        init_frequency = {k: text_frequency.get(k, 0) for k in words}   
 
    if normalization == 0:
        frequency_normalized = init_frequency
     
    if normalization == 1:            
        n = sentence_count(text_string)
        frequency_normalized = dict((k, v / n) for k, v in init_frequency.items())
     
    if normalization == 2:            
        n = word_count(text_string)
        frequency_normalized = dict((k, v / n) for k, v in init_frequency.items())
     
    return frequency_normalized

 
def double_word_frequency(text_string, words):
    n_words = len(words)
    s_wf = np.empty((0, len(words)), int)
    sentences = text_string.split('. ')
    for s in sentences:                
        s_wf = np.vstack((s_wf, list(word_frequency(s, words, 0, 0).values())))
    pass
    x = np.matmul(np.transpose(s_wf), s_wf)
    idx_tri = np.triu_indices(n_words)    
    return list(x[idx_tri])

 
def multiclass_to_vectors(x):
    x_classes = x.unique()
    x_classes = sorted(x_classes)
    x_vectors = np.zeros([len(x), len(x_classes)])
     
    for i in range(len(x_classes)):
        x_vectors[x == x_classes[i], i] = 1
    return x_vectors

 
def bin_data(x, n_bins):
    bins = np.linspace(x.min(), x.max(), num=n_bins)    
    return np.digitize(x, bins) 

 
def plot_barplot(data, fig_w, fig_h, title, title_size, x_label, y_label, label_size, tick_size, ylim_min, ylim_max, plot_name):
    fig = plt.figure(figsize=(fig_w, fig_h))
    # data_plot = data.stack().reset_index(inplace = True)
    
    data = data.loc[~data.isnull().any(axis=1), :]
    data_plot = data.copy()
    data_plot['x'] = data_plot.index
    # data_plot = data.reset_index(inplace = True)
    data_plot.columns = ['y', 'ci', 'x']
    list_ci = list(data_plot['ci'])
    ax = sns.barplot(x='x', y='y', data=data_plot)
    ax.set_xlabel(x_label, size=label_size)
    ax.set_ylabel(y_label, size=label_size)
    ax.tick_params(labelsize=tick_size)
    ax.set_ylim(ylim_min, ylim_max)
    ttl = ax.set_title(title, size=title_size)
    ttl.set_position([.5, 1.05])
    i = 0
    for p in ax.patches:
        height = p.get_height()
        # plot error bar
        line_x = [p.get_x() + p.get_width() / 2, p.get_x() + p.get_width() / 2]
        line_y = [height - list_ci[i], height + list_ci[i]]
        ax.plot(line_x, line_y, 'k-', color='black' , linewidth=1) 
        # plot caps
        line_x = [p.get_x() + p.get_width() / 2 - p.get_width() / 20 , p.get_x() + p.get_width() / 2 + p.get_width() / 20]
        line_y = [height - list_ci[i], height - list_ci[i]]
        ax.plot(line_x, line_y, 'k-', color='black' , linewidth=1) 
        line_x = [p.get_x() + p.get_width() / 2 - p.get_width() / 20 , p.get_x() + p.get_width() / 2 + p.get_width() / 20]
        line_y = [height + list_ci[i], height + list_ci[i]]
        ax.plot(line_x, line_y, 'k-', color='black', linewidth=1) 
        # plot value   
        # ax.text(p.get_x()+p.get_width()/2,height + list_ci[i] + 0.01,str(np.round(height,4)) + '±' + str(list_ci[i]),ha="center",size = tick_size) 
        ax.text(p.get_x() + p.get_width() / 2, height + list_ci[i] + 0.01, str(np.round(height, 4)), ha="center", size=tick_size)
        i += 1    
    pass
    fig.tight_layout()
    plt.savefig(plot_name)   

    
def plot_scatter(fig_w, fig_h, x, y, x_test_name, y_test_name, hue, hue_name, title_extra, marker_size, plot_name):    
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = plt.subplot()    
    if hue is None:
        ax.scatter(x, y, cmap='Blues', s=marker_size)
    else:
        ax.scatter(x, y, c=hue, cmap='Blues', s=marker_size)
 
    ax.set_title(x_test_name + ' VS ' + y_test_name + ' [pcc:' + str(round(pcc_cal(x, y), 3)) + 
                     '][spc:' + str(round(spc_cal(x, y), 3)) + '][color: ' + hue_name + '] ' + title_extra, size=15)
    ax.set_ylabel(y_test_name, size=10)
    ax.set_xlabel(x_test_name, size=10) 
    ax.tick_params(size=10)
    ax.legend()
    fig.tight_layout()
    plt.savefig(plot_name)  
    
    # class alm_tf:
#     def __init__(self,loss_name = 'cross_entropy',hidden_units = [],activation_fn = tf.nn.relu,n_classes = 9,batch_gd = 1,batch_size = 10,num_epochs=10,learning_rate = 0.1,use_saved_weight = 0):
#         self.loss_name = loss_name
#         self.hidden_units = hidden_units
#         self.n_classes = n_classes
#         self.batch_gd = batch_gd
#         self.batch_size = batch_size
#         self.num_epochs = num_epochs
#         self.activation_fn = activation_fn
#         self.leraning_rate = learning_rate
#         self.use_saved_weight = use_saved_weight
#         self.sess = tf.Session()
#           
#     def weight_variable(self,shape):
#         initial = tf.truncated_normal(shape, stddev=0.1)
#         return tf.Variable(initial)
#   
#     def bias_variable(self,shape):
#         initial = tf.constant(0.1, shape=shape)
#         return tf.Variable(initial)        
#     
#     def fit(self,x,y):        
#         #take 20% as validation set
#         # x is input , y_ is output 
#         self.x_input = np.array(x).astype('float32')
#         self.y_input = np.array(y).astype('float32')
#         self.y_input = label_vector(self.y_input)
#           
#         validation_pivot = int(self.x_input.shape[0]*0.2)
#         rnd_idx = np.random.permutation(self.x_input.shape[0])
#         validation_idx = rnd_idx[range(validation_pivot)]
#         train_idx = rnd_idx[range(validation_pivot,self.x_input.shape[0])]
#           
#         self.x_train = self.x_input[train_idx,:]
#         self.x_validation = self.x_input[validation_idx,:]
#         self.y_train = self.y_input[train_idx,:]
#         self.y_validation = self.y_input[validation_idx,:]
#   
#         self.x = tf.placeholder(tf.float32, shape=[None, self.x_train.shape[1]])
#         self.y_ = tf.placeholder(tf.float32, shape=[None, self.n_classes])
#   
#         #add hidden layers
#         self.n_hlayer = len(self.hidden_units)
#         self.W = list(np.zeros(self.n_hlayer))
#         self.b = list(np.zeros(self.n_hlayer))
#         self.h = list(np.zeros(self.n_hlayer))
#         self.h_active =list(np.zeros(self.n_hlayer))
#         self.W_s = list(np.zeros(self.n_hlayer))
#         self.b_s = list(np.zeros(self.n_hlayer))
#         self.h_s = list(np.zeros(self.n_hlayer))
#         self.h_active_s = list(np.zeros(self.n_hlayer))
#           
#         for i in range(self.n_hlayer):
#             if i == 0:
#                 self.W[i] =  tf.Variable(tf.zeros([self.x_train.shape[1],self.hidden_units[i]]))
#                 self.b[i] =  tf.Variable(tf.zeros([self.hidden_units[i]]))
#                 #self.W[i] =  self.weight_variable([self.x_train.shape[1],self.hidden_units[i]])
#                 #self.b[i] =  self.bias_variable([self.hidden_units[i]])
#                 self.h[i] =  tf.matmul(self.x,self.W[i]) + self.b[i]
#                 self.h_active[i] = self.activation_fn(self.h[i])
#                 self.W_s[i] = np.zeros([self.num_epochs,self.x_train.shape[1],self.hidden_units[i]])
#             else:
#                 self.W[i] =  tf.Variable(tf.zeros([self.hidden_units[i-1],self.hidden_units[i]]))
#                 self.b[i] =  tf.Variable(tf.zeros([self.hidden_units[i]]))
# #                 self.W[i] =  self.weight_variable([self.hidden_units[i-1],self.hidden_units[i]])
# #                 self.b[i] =  self.bias_variable([self.hidden_units[i]])
#                 self.h[i] =  tf.matmul(self.h_acitve[i-1],self.W[i]) + self.b[i]
#                 #self.h[i] =  tf.matmul(self.h_acitve[i-1],self.W[i])
#                 self.h_active[i] = self.activation_fn(self.h[i])   
#                 self.W_s[i] = np.zeros([self.num_epochs,self.hidden_units[i],self.hidden_units[i]])
#                   
#             self.b_s[i] = np.zeros([self.num_epochs,self.hidden_units[i]])
#             self.h_s[i] = np.zeros([self.num_epochs,self.x_train.shape[0],self.hidden_units[i]])
#             self.h_active_s[i] = np.zeros([self.num_epochs,self.x_train.shape[0],self.hidden_units[i]])
#      
#         if self.n_hlayer == 0: # no hidden units
#             #self.W_output = tf.Variable(tf.zeros([self.x_train.shape[1],self.n_classes]))
#             #self.b_output = tf.Variable(tf.zeros([self.n_classes]))
#             if self.use_saved_weight == 0:
#                 self.W_output =  self.weight_variable([self.x_train.shape[1],self.n_classes])
#                 self.b_output =  self.bias_variable([self.n_classes])
#             else:
#                 self.W_output = tf.Variable(np.array(pd.read_csv('W_output.csv')).astype('float32'))   
#                 self.b_output = tf.Variable(np.squeeze(np.array(pd.read_csv('b_output.csv'))).astype('float32'))   
#             #matrix multiplication
#             self.y = tf.matmul(self.x,self.W_output) + self.b_output
#             self.y_prob = tf.nn.softmax(self.y)
#               
#             self.weight_output = np.zeros([self.num_epochs,self.x_train.shape[1],self.n_classes])
#             self.bias_output = np.zeros([self.num_epochs,self.n_classes])
#         else:
#             #self.W_output = tf.Variable(tf.zeros([self.hidden_units[self.n_hlayer-1],self.n_classes]))
#             #self.b_output = tf.Variable(tf.zeros([self.n_classes]))  
#             if self.use_saved_weight == 0: 
#                 self.W_output =  self.weight_variable([self.hidden_units[self.n_hlayer-1],self.n_classes])
#                 self.b_output =  self.bias_variable([self.n_classes])
#             else:
#                 self.W_output = tf.Variable(np.array(pd.read_csv('W_output.csv')).astype('float32'))     
#                 self.b_output = tf.Variable(np.squeeze(np.array(pd.read_csv('b_output.csv'))).astype('float32'))   
#   
#             #matrix multiplication
#             self.y = tf.matmul(self.h_active[self.n_hlayer-1],self.W_output) + self.b_output
#             self.y_prob = tf.nn.softmax(self.y)
#               
#             self.weight_output = np.zeros([self.num_epochs,self.hidden_units[self.n_hlayer-1],self.n_classes])
#             self.bias_output = np.zeros([self.num_epochs,self.n_classes])     
#   
#           
#         #loss function
#         if self.loss_name == 'cross_entropy':
#             self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y))
#   
#         #loss function optimization
#         self.train_step = tf.train.GradientDescentOptimizer(self.leraning_rate).minimize(self.loss)
#           
#         #evaluation metrics    
#         self.correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.y_,1))
#         self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
#           
#         #run training
#         self.sess.run(tf.global_variables_initializer())
#           
#         #save initiated parameters
#         pd.DataFrame(self.W_output.eval(session = self.sess)).to_csv('W_output.csv',index = False)
#         pd.DataFrame(self.b_output.eval(session = self.sess)).to_csv('b_output.csv',index = False)
#           
#           
#         if self.batch_gd == 1 :
#             self.batch_size = self.x_train.shape[0]
#           
#         self.batch_idx = np.arange(0,self.x_train.shape[0],self.batch_size)
#         if self.batch_idx[-1] < self.x_train.shape[0]:
#             self.batch_idx = np.append(self.batch_idx,self.x_train.shape[0])
#                           
#         self.loss_value = np.zeros(self.num_epochs)    
#         self.train_accuracy = np.zeros(self.num_epochs)    
#         self.validation_accuracy = np.zeros(self.num_epochs)   
#   
#         self.train_y_predicted_prob = np.zeros([self.num_epochs,self.x_train.shape[0],self.n_classes])
#         self.train_label_predicted = np.zeros([self.num_epochs,self.x_train.shape[0]])
#           
#         self.validation_y_predicted_prob = np.zeros([self.num_epochs,self.x_validation.shape[0],self.n_classes])
#         self.validation_label_predicted = np.zeros([self.num_epochs,self.x_validation.shape[0]])
#           
#         self.train_auroc = np.zeros(self.num_epochs)
#         self.train_auprc = np.zeros(self.num_epochs)
#         self.validation_auroc = np.zeros(self.num_epochs)
#         self.validation_auprc = np.zeros(self.num_epochs)
#           
#         self.train_lable_truth = self.y_train[:,1]
#         self.validation_lable_truth = self.y_validation[:,1]
#           
#         self.train_prior = (self.train_lable_truth == 1).sum()/self.train_lable_truth.shape[0]
#         self.validation_prior = (self.validation_lable_truth == 1).sum()/self.validation_lable_truth.shape[0]
#            
#           
#         for i in range(self.num_epochs):
#             for j in range(len(self.batch_idx)-1):
#                 self.x_train_batch = self.x_train[range(self.batch_idx[j],self.batch_idx[j+1]),:]
#                 self.y_train_batch = self.y_train[range(self.batch_idx[j],self.batch_idx[j+1]),:]
#                 self.loss_value[i] = self.loss.eval(session = self.sess,feed_dict={self.x: self.x_train, self.y_: self.y_train}) 
# #                 self.y.eval(session = self.sess,feed_dict={self.x: self.x_train, self.y_: self.y_train})
#   
# #                 self.weight_output[i,:,:] =  self.W_output.eval(session = self.sess)
# #                 self.bias_output[i,:] =  self.b_output.eval(session = self.sess)
# #                 
# #                 for k in range(self.n_hlayer):
# #                     self.W_s[k][i,:,:] = self.W[k].eval(session = self.sess)
# #                     self.b_s[k][i,:] = self.b[k].eval(session = self.sess)
# #                     self.h_s[k][i,:,:] = self.h[k].eval(session = self.sess, feed_dict={self.x: self.x_train_batch, self.y_: self.y_train_batch})
# #                     self.h_active_s[k][i,:,:] = self.h_active[k].eval(session = self.sess, feed_dict={self.x: self.x_train_batch, self.y_: self.y_train_batch})
# # #                 
#                 self.train_accuracy[i] = self.accuracy.eval(session = self.sess,feed_dict={self.x: self.x_train, self.y_: self.y_train})
#                 #self.train_y_predicted_prob[i,:,:] = self.y_prob.eval(session = self.sess,feed_dict={self.x: self.x_train, self.y_: self.y_train})  
#                 #self.train_label_predicted[i,:] = self.train_y_predicted_prob[i,:,1]  
#                 #self.train_auroc[i] = auroc_cal(self.train_lable_truth,self.train_y_predicted_prob[i,:,1])
#                 #self.train_auprc[i] = auprc_cal(self.train_lable_truth,self.train_y_predicted_prob[i,:,1])
#                 train_label_predicted = self.y_prob.eval(session = self.sess,feed_dict={self.x: self.x_train, self.y_: self.y_train})[:,1]
#                   
#                 metric = roc_prc_cal(self.train_lable_truth,train_label_predicted)
#                 self.train_auroc[i] = metric['roc']
#                 self.train_auprc[i] = metric['prc']
#                   
#                 self.validation_accuracy[i] = self.accuracy.eval(session = self.sess,feed_dict={self.x: self.x_validation, self.y_: self.y_validation})
#                 #self.validation_y_predicted_prob[i,:,:] = self.y_prob.eval(session = self.sess,feed_dict={self.x: self.x_validation, self.y_: self.y_validation})  
#                 #self.validation_label_predicted[i,:] = self.validation_y_predicted_prob[i,:,1]                  
#                 #self.validation_auroc[i] = auroc_cal(self.validation_lable_truth,self.validation_y_predicted_prob[i,:,1])
#                 #self.validation_auprc[i] = auprc_cal(self.validation_lable_truth,self.validation_y_predicted_prob[i,:,1])
#                   
#                 validation_label_predicted = self.y_prob.eval(session = self.sess,feed_dict={self.x: self.x_validation, self.y_: self.y_validation})[:,1]
#                   
#                 metric = roc_prc_cal(self.validation_lable_truth,validation_label_predicted)
#                 self.validation_auroc[i] = metric['roc']
#                 self.validation_auprc[i] = metric['prc']
#                   
#                 # spot check the weight update for first feature and first hidden units  
# #                 g00 = 0              
# #                 for n in range(self.x_train_batch.shape[0]):
# #                     x0 = self.x_train_batch[n,0]
# #                     a0 = self.h_active_s[0][i,n,0]
# #                     w00 = self.weight_output[i,0,0]
# #                     w01 = self.weight_output[i,0,1]
# #                     p0 = self.train_y_predicted_prob[i,n,0]
# #                     p1 = self.train_y_predicted_prob[i,n,1]
# #                     y0 = self.y_train[n,0]
# #                     y1 = self.y_train[n,1]
# #                     g00 = g00 + (p0-y0)*w00*a0*(1-a0)*x0 + (p1-y1)*w01*a0*(1-a0)*x0
# #                 g00 = g00*self.leraning_rate/self.x_train_batch.shape[0]
#   
#                 print ('[epochs:' + str(i)  + '] loss:' + str(self.loss_value[i]) + ', train_auroc:' + str(self.train_auroc[i]) + ', validation_auroc:' + str(self.validation_auroc[i]))
#                 self.train_step.run(session = self.sess, feed_dict={self.x: self.x_train_batch, self.y_: self.y_train_batch})
#           
#         fit_result = pd.DataFrame(np.transpose(np.vstack([self.loss_value,self.train_auroc,self.validation_auroc,self.train_auprc,self.validation_auprc,self.train_accuracy,self.validation_accuracy])))
#         fit_result.columns = ['loss','train_auroc','validation_auroc','train_auprc','validation_auprc','train_accuracy','validation_accuracy']
#         fit_result.to_csv('fit_result_'+ str(datetime.now()) + '.csv',index = False)
#         plt.figure()
#         p1 = plt.subplot2grid((2,2),(0,0))
#         p1.plot(range(self.num_epochs),self.loss_value)
#         p1.set_title('Loss function VS epochs')
#         p1.set_xlabel('epochs')
#         p1.set_ylabel('loss function')
#         p2 = plt.subplot2grid((2,2),(0,1))
#         p2.plot(range(self.num_epochs),self.train_accuracy,label = 'train accuracy')
#         p2.plot(range(self.num_epochs),self.validation_accuracy, label = 'validation_accuracy')
#         p2.set_title('training and validation performance VS epochs')
#         p2.set_xlabel('epochs')
#         p2.set_ylabel('performance')
#         p2.legend(loc = "upper right")
#         p3 = plt.subplot2grid((2,2),(1,0))
#         p3.plot(range(self.num_epochs),self.train_auroc,label = 'train AUROC')
#         p3.plot(range(self.num_epochs),self.validation_auroc, label = 'validation AUROC')
#         p3.set_title('AUROC VS epochs')
#         p3.set_xlabel('epochs')
#         p3.set_ylabel('AUROC')
#         p3.legend(loc = "upper right")
#         p4 = plt.subplot2grid((2,2),(1,1))
#         p4.plot(range(self.num_epochs),self.train_auprc,label = 'train AUPRC')
#         p4.plot(range(self.num_epochs),self.validation_auprc, label = 'validation AUPRC')
#         p4.set_title('AUPRC VS epochs' + '-Train prior:' + str(self.train_prior) + ' -Validation prior:' + str(self.validation_prior))
#         p4.set_xlabel('epochs')
#         p4.set_ylabel('AUPRC')
#         p4.legend(loc = "upper right")
#   
#     def predict_proba(self,x):           
#         self.x_validation = np.array(x).astype('float32')        
#         self.proba = tf.nn.softmax(self.y)
#         self.prediction_proba = self.proba.eval(session = self.sess, feed_dict={self.x:self.x_validation})
#         return self.prediction_proba

 
def plot_color_gradients(vmax, vmin, vcenter, max_color, min_color, center_color, max_step, min_step, fig_width, fig_height, orientation, legend_name, fig_savepath):
    [lst_max_colors, lst_min_colors] = create_color_gradients(vmax, vmin, vcenter, max_color, min_color, center_color, max_step, min_step)                           
    fig = plt.figure(figsize=(fig_width, fig_height)) 
    ax = plt.subplot()
    if lst_min_colors is not None:
        lst_colors_new = lst_max_colors + lst_min_colors[1:]
    else:
        lst_colors_new = lst_max_colors
    n_tracks = 1
    if orientation == 'H':
        x = [0] + [0] * (n_tracks + 1) + list(range(0, len(lst_colors_new) + 1))
        y = [-1] + list(range(0, n_tracks + 1)) + [0] * (len(lst_colors_new) + 1)
        
    if orientation == 'V':
        y = [0] + [0] * (n_tracks + 1) + list(range(0, len(lst_colors_new) + 1))
        x = [-1] + list(range(0, n_tracks + 1)) + [0] * (len(lst_colors_new) + 1)

    ax.plot(x, y, alpha=0)

    # add rectangles
    for i in range(len(lst_colors_new)):
        xy_color = lst_colors_new[len(lst_colors_new) - i - 1] 
        if orientation == 'V':
            rect = patches.Rectangle((0, i), 1, 1, linewidth=1, edgecolor=xy_color, facecolor=xy_color, fill=True, clip_on=False)
        if orientation == 'H':
            rect = patches.Rectangle((i, 0), 1, 1, linewidth=1, edgecolor=xy_color, facecolor=xy_color, fill=True, clip_on=False)
            
        ax.add_patch(rect)
    pass

    # add lines
    min_pos = 0.5
    max_pos = len(lst_colors_new) - 0.5
    center_pos = (len(lst_colors_new) - 1) / 2 + 0.5
    tick_offset = -0.1
    lines = []    
    if orientation == 'H':
        lines.append(((min_pos, tick_offset), (max_pos, tick_offset)))
        lines.append(((min_pos, tick_offset), (min_pos, tick_offset * 2)))
        if vmin != vcenter :
            lines.append(((center_pos, tick_offset), (center_pos, tick_offset * 2)))
        lines.append(((max_pos, tick_offset), (max_pos, tick_offset * 2)))
    if orientation == 'V':
        lines.append((tick_offset, min_pos), (tick_offset, max_pos))
        lines.append((tick_offset, min_pos), (tick_offset * 2, min_pos))
        if vmin != vcenter :
            lines.append((tick_offset, center_pos), (tick_offset * 2, center_pos))
        lines.append((tick_offset, max_pos), (tick_offset * 2, max_pos))        
    lc = collections.LineCollection(lines, linewidth=2, color='black', clip_on=False)
    ax.add_collection(lc) 
    
    if orientation == 'H':
        ax.text(min_pos, tick_offset * 4, vmin, rotation=360, fontsize=10, ha='center', weight='bold')
        ax.text(max_pos, tick_offset * 4, vmax, rotation=360, fontsize=10, ha='center', weight='bold')
        if vmin != vcenter :
            ax.text(center_pos, tick_offset * 4, vcenter, rotation=360, fontsize=10, ha='center', weight='bold')
        ax.text(center_pos, tick_offset * 6, legend_name, rotation=360, fontsize=12, ha='center', weight='bold')
    if orientation == 'V':
        ax.text(tick_offset * 4, min_pos, vmin, rotation=360, fontsize=10, va='center', weight='bold')
        ax.text(tick_offset * 4, max_pos, vmax, rotation=360, fontsize=10, va='center', weight='bold')
        if vmin != vcenter :
            ax.text(tick_offset * 4, vcenter, center_pos, rotation=360, fontsize=10, va='center', weight='bold')
        ax.text(tick_offset * 6, center_pos, legend_name, rotation=360, fontsize=12, va='center', weight='bold')
    
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.patch.set_facecolor('white')
    ax.patch.set_facecolor('white')
    plt.savefig(fig_savepath, format='png', dpi=300, transparent=True)


def create_color_gradients(vmax, vmin, vcenter, max_color, min_color, center_color, max_step, min_step):
    if vmax > vcenter:
        lst_max_colors = color_gradient(center_color, max_color, max_step)
    else:
        lst_max_colors = None
    if vmin < vcenter:
        lst_min_colors = color_gradient(min_color, center_color, min_step)
    else:
        lst_min_colors = None
    return [lst_max_colors, lst_min_colors]

    
def get_colorcode(value, vmax, vmin, vcenter, max_step, min_step, lst_max_colors, lst_min_colors):
    if value > vmax: value = vmax  
    if value < vmin: value = vmin        
    if np.isnan(value): 
        colorcode = '#C0C0C0'
    else:
        if value >= vcenter:            
            colorcode = lst_max_colors[(len(lst_max_colors) - 1) - int(round((value - vcenter) * max_step / (vmax - vcenter)))]
        else:
            colorcode = lst_min_colors[(len(lst_min_colors) - 1) - int(round((value - vmin) * min_step / (vcenter - vmin)))]
    colorcode = colorcode.replace('x', '0')
    return colorcode


def color_arrayMultiply(array, c):
    return [element * c for element in array]


def color_arraySum(a, b):
    return list(map(sum, zip(a, b)))


def color_intermediate(a, b, ratio):
    aComponent = color_arrayMultiply(a, ratio)
    bComponent = color_arrayMultiply(b, 1 - ratio)
    decimal_color = color_arraySum(aComponent, bComponent)
    hex_color = '#' + str(hex(int(decimal_color[0])))[-2:] + str(hex(int(decimal_color[1])))[-2:] + str(hex(int(decimal_color[2])))[-2:]
    hex_color = hex_color.replace('x', '0')
    return hex_color


def color_gradient(a, b, steps):
    lst_gradient_colors = []
    start_color = [ int(a[1:3], 16), int(a[3:5], 16), int(a[5:7], 16)]
    end_color = [ int(b[1:3], 16), int(b[3:5], 16), int(b[5:7], 16)]
    steps = [n / float(steps) for n in range(steps)]
    for step in steps:
        hex_color = color_intermediate(start_color, end_color, step)
        lst_gradient_colors.append(hex_color)
    lst_gradient_colors = lst_gradient_colors + [a]
    return lst_gradient_colors

def send_email(server_address,server_port,login_user,login_password,from_address,to_address, subject, msg_content):
    server = smtplib.SMTP('smtp-relay.gmail.com', 587)
    server.starttls()
    server.login(login_user, login_password)  
    msg = EmailMessage()       

    msg.set_content(msg_content)
    msg['Subject'] = subject
    msg['From'] = from_address
    msg['To'] = to_address
    server.sendmail(from_address, to_address, msg.as_string())
    server.quit()
           
def error_propagation_fun(value1, value1_err, inFun):
    if inFun == 'log':
        value = np.log(value1)
        value_err = np.abs(value1_err / value1)
        
    if inFun == 'log10':
        value = np.log(value1)
        value_err = np.abs(value1_err / (value1 * log(10)))

    return pd.Series([value, value_err])   

def error_propagation_operation(value1, value1_err, value2, value2_err, inOperation):        
    if inOperation == 'addition':
        value = value1 + value2
        value_err = np.sqrt(value1_err ** 2 + value2_err ** 2)            
    if inOperation == 'subtraction':
        value = value1 - value2
        value_err = np.sqrt(value1_err ** 2 + value2_err ** 2)                 
    if inOperation == 'division':
        value = value1 / value2
        value_err = np.abs(value) * np.sqrt((value1_err / value1) ** 2 + (value2_err / value2) ** 2)   
    if inOperation == 'multiplication':
        value = value1 / value2
        value_err = np.abs(value) * np.sqrt((value1_err / value1) ** 2 + (value2_err / value2) ** 2)   
    
    return pd.Series([value, value_err])
