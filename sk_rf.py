#!/usr/bin/python
#http://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
import csv
import sys
from sklearn.ensemble import RandomForestClassifier

from sklearn import manifold, datasets
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
import scrape as sc

import argparse

# general parameters

parser = argparse.ArgumentParser(description='Create Random Forest Classifier.')

sc.model_options(parser)
sc.all_options(parser)

# RF parameters

parser.add_argument('--n_estimators', type=int,default=10,help='The number of trees in the forest.')
parser.add_argument('--criterion',default='gini', help='The function to measure the quality of a split. Supported criteria are gini for the Gini impurity and entropy for the information gain.')
parser.add_argument('--max_features',default='auto', help='The number of features to consider when looking for the best split: If int, then consider max_features features at each split. '
                    'If float, then max_features is a percentage and int(max_features * n_features) features are considered at each split. If auto, then max_features=sqrt(n_features). If sqrt, then max_features=sqrt(n_features) (same as auto). If log2, then max_features=log2(n_features). If None, then max_features=n_features. Note: the search for a split does not stop until at least one valid partition of the node samples is found, even if it requires to effectively inspect more than max_features features.')
parser.add_argument('--max_depth',type=int,help='The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.')
parser.add_argument('--min_samples_split',type=float,default=2,help='The minimum number of samples required to split an internal node: If int, then consider min_samples_split as the minimum number. If float, then min_samples_split is a percentage and ceil(min_samples_split * n_samples) are the minimum number of samples for each split. Changed in version 0.18: Added float values for percentages.')
parser.add_argument('--min_samples_leaf',type=float,default=1,help='The minimum number of samples required to be at a leaf node: If int, then consider min_samples_leaf as the minimum number. If float, then min_samples_leaf is a percentage and ceil(min_samples_leaf * n_samples) are the minimum number of samples for each node. Changed in version 0.18: Added float values for percentages.')

parser.add_argument('--min_weight_fraction_leaf',type=float,default=0.,help='The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided.')

parser.add_argument('--max_leaf_nodes',type=int,help='Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.')

parser.add_argument('--min_impurity_split',type=float,default=1e-7,help='Threshold for early stopping in tree growth. A node will split if its impurity is above the threshold, otherwise it is a leaf. New in version 0.18.')

parser.add_argument('--bootstrap',type=bool,default=True,help='Whether bootstrap samples are used when building trees.')

parser.add_argument('--oob_score', type=bool,default=False, help='Whether to use out-of-bag samples to estimate the generalization accuracy.')

sc.add_arguments(parser,["n_jobs","random_state","verbose"])

parser.add_argument('--warm_start',type=bool,default=False,help='When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just fit a whole new forest.')

parser.add_argument('--class_weight',help='list of dicts, balanced, balanced_subsample or None, optional (default=None) Weights associated with classes in the form {class_label: weight}. If not given, all classes are supposed' 
        ' to have weight one. For multi-output problems, a list of dicts can be provided in the same order as the columns of y. The balanced mode uses the values of y to automatically adjust weights inversely '
        'proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y)) The balanced_subsample mode is the same as balanced except that weights are computed based on the bootstrap sample for '
        'every tree grown. For multi-output, the weights of each column of y will be multiplied. Note that these weights will be multiplied with sample_weight (passed through the fit method) if sample_weight is specified.')


args = sc.parse_args(parser)

rf = RandomForestClassifier(n_estimators=args.n_estimators, criterion=args.criterion, max_depth=args.max_depth, min_samples_split=args.min_samples_split, min_samples_leaf=args.min_samples_leaf, min_weight_fraction_leaf=args.min_weight_fraction_leaf, max_features=args.max_features, max_leaf_nodes=args.max_leaf_nodes, min_impurity_split=args.min_impurity_split, bootstrap=args.bootstrap, oob_score=args.oob_score, n_jobs=args.n_jobs, random_state=args.random_state, verbose=args.verbose, warm_start=args.warm_start, class_weight=args.class_weight)
   
sc.save_model(rf, args.model_output_file)


    
    


