#!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3
import sys
import numpy as np
import pandas as pd
import random
import codecs
import os
import re
import datetime
import time
import warnings
import pickle
import glob
from scipy import stats
from sklearn import preprocessing
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt 
python_path = '/usr/local/projects/ml/python/'
project_path = '/usr/local/projects/imputation/gwt/www/'
humandb_path = '/usr/local/database/humandb/'
sys.path.append(python_path)

import imputation_web 


#***************************************************************************************************************************************************************
# debug for the web application 
#***************************************************************************************************************************************************************
dict_arg = pickle.load(open(project_path + "output/P63279[Mon_Feb_18_21_21_06_44]_1.pickle", "rb"))
JSON_Return = imputation_web.run_imputation(1,dict_arg) 
print (JSON_Return)   

