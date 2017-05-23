#!/usr/bin/python
#http://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
import csv
import sys

from sklearn import manifold, datasets
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
import scrape as sc
from sklearn.naive_bayes import BernoulliNB

import argparse

# general parameters

parser = argparse.ArgumentParser(description='Perform SVM machine learning.')

sc.model_options(parser)
sc.all_options(parser)

# SVC parameters

parser.add_argument('--C',type=float,default=1.0,help='Penalty parameter C of the error term.')

parser.add_argument('--kernel',default='rbf',help='Specifies the kernel type to be used in the algorithm. It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable. If none is given, ‘rbf’ will be used. If a callable is given it is used to pre-compute the kernel matrix from data matrices; that matrix should be an array of shape (n_samples, n_samples).')

parser.add_argument('--degree',type=int,default=3,help='Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.')

parser.add_argument('--gamma',default='auto',help='Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’. If gamma is ‘auto’ then 1/n_features will be used instead.')

parser.add_argument('--coef0',type=float,default=0.0,help='Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.')
                    
parser.add_argument('--probability',type=bool,default=False,help='Whether to enable probability estimates. This must be enabled prior to calling fit, and will slow down that method.')

parser.add_argument('--shrinking',type=bool, default=True, help='Whether to use the shrinking heuristic.')

parser.add_argument('--tol',type=float,default=1e-3,help='Tolerance for stopping criterion.')

parser.add_argument('--cache_size',default=100.0,type=float,help='Specify the size of the kernel cache (in MB).')

parser.add_argument('--class_weight',help='Set the parameter C of class i to class_weight[i]*C for SVC. If not given, all classes are supposed to have weight one. The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y))')

parser.add_argument('--verbose',type=bool,default=False,help='Enable verbose output. Note that this setting takes advantage of a per-process runtime setting in libsvm that, if enabled, may not work properly in a multithreaded context.')

parser.add_argument('--max_iter',type=int,default=-1,help='Hard limit on iterations within solver, or -1 for no limit.')

parser.add_argument('--decision_function_shape',choices=['ovo','ovr'],help='Whether to return a one-vs-rest (‘ovr’) decision function of shape (n_samples, n_classes) as all other classifiers, or the original one-vs-one (‘ovo’) decision function of libsvm which has shape (n_samples, n_classes * (n_classes - 1) / 2). The default of None will currently behave as ‘ovo’ for backward compatibility and raise a deprecation warning, but will change ‘ovr’ in 0.19. New in version 0.17: decision_function_shape=’ovr’ is recommended. Changed in version 0.17: Deprecated decision_function_shape=’ovo’ and None.')

parser.add_argument('--random_state',type=int,help='The seed of the pseudo random number generator to use when shuffling the data for probability estimation.')


args = sc.parse_args(parser)

if not args.gamma == 'auto':    # gamma can be auto or float
    setattr(args, "gamma", float(args.gamma))

#svc = sc.load_model(args.model_file)

svc = BernoulliNB()

   
sc.save_model(svc, args.model_output_file)


    
    


