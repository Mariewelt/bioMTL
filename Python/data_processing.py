import csv
import numpy as np

def read_data(design_matrix_file='features_2.csv', activity_file='all_endpoints_with_missing_values_012615.csv'):
    # reading CSV file
    reader = csv.reader(open(design_matrix_file, 'r'), delimiter=',')
    data_full = np.array(list(reader))
    reader = csv.reader(open(activity_file, 'r'), delimiter=',')
    activity_full = np.array(list(reader))
    
    # feature names
    feature_names = data_full[0, 1:]

    # names of the proteins
    protein_names = data_full[1:, 0]
    protein_names1 = activity_full[1:, 0]
    print 'Protein names equality check:', np.array_equal(protein_names1, protein_names)
    
    # names of receptors
    receptor_names = activity_full[0, 1:]

    # Object-Feature matrix (proteins description)
    X = data_full[1:, 1:].astype('double')

    # Activity matrix
    Y = activity_full[1:, 1:].astype('int16')
    
    return X, Y, feature_names, receptor_names, protein_names

def remove_constant_features(X, feature_names, eps=1e-2):
    # Removing constant features
    ind = np.var(X, axis = 0) > eps
    X = X[:, ind]
    feature_names = feature_names[ind]
    return X, feature_names

def select_tasks(X, Y, receptor_ind, missing_values='remove', eps=1e-2):
    X_k = np.copy(X)
    Y_k = Y[:, receptor_ind]
    
    if missing_values == 'remove':
        if Y_k.ndim == 1:
            ind_full = Y_k != 999
        else:
            ind_full = np.all(Y_k != 999, axis=1)
        Y_k = Y_k[ind_full]
        X_k = X_k[ind_full]
        
        feat_ind = np.var(X_k, axis = 0) > eps
        X_k = X_k[:, feat_ind]
        
    return X_k, Y_k   

import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.cross_validation import KFold, cross_val_score, train_test_split
from sklearn.linear_model import Lasso
from sklearn.svm import SVC
import scipy as sp
from scipy.optimize import minimize
from time import time
import warnings
warnings.filterwarnings('ignore')