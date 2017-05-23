#!/usr/bin/python
# -*- coding: ascii -*-
import numpy as np
import csv
import sys
from sklearn.datasets import make_blobs

from sklearn import manifold, datasets


import argparse


parser = argparse.ArgumentParser(description='Generate isotropic Gaussian blobs for clustering.')


parser.add_argument('--centers',default='3',help=' or array of shape [n_centers, n_features], optional (default=3) The number of centers to generate, or the fixed center locations.')

parser.add_argument('--cluster_std',help=' or sequence of floats, optional (default=1.0) The standard deviation of the clusters.')

parser.add_argument('--center_box',default='(-10.0,10.0)',help=' : pair of floats (min, max), optional (default=(-10.0, 10.0)) The bounding box for each cluster center when centers are generated at random.')

parser.add_argument('--n_samples',default=100,type=int,help='')

parser.add_argument('--n_features',type=int,default=20, help='The total number of features. These comprise n_informative informative features, n_redundant redundant features, n_repeated duplicated features and n_features-n_informative-n_redundant- n_repeated useless features drawn at random.')

parser.add_argument('--shuffle',type=bool,default=True,help='Shuffle the samples and the features.')

parser.add_argument('--random_state',type=int,help=' RandomState instance or None, optional (default=None) If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random.')

import scrape as sc

sc.output_options(parser)
sc.all_options(parser)


from scrape import write_dict
from scrape import load_file


args = sc.parse_args(parser)


#results = np.loadtxt(open("test_1_centroids.csv","rb"),delimiter=",",skiprows=1)

#print (results)

# bcml mode feature_class version
#  0    1        2           3
# mode { baseline, update }


np.random.seed(0)

X, y = make_blobs(n_samples=args.n_samples, n_features=args.n_features, centers=sc.safe_eval(args.centers), cluster_std=sc.safe_eval(args.cluster_std), center_box=sc.safe_eval(args.center_box), shuffle=args.shuffle, random_state=args.random_state)

datasets.dump_svmlight_file(X,y,args.output_file,zero_based=args.zero_based,query_id=args.query_id,multilabel=args.multilabel,comment=args.comment)

write_dict({'feature_file':args.output_file})
