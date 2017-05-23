#!/usr/bin/python
# -*- coding: ascii -*-
import numpy as np
import csv
import sys
from sklearn.datasets import make_circles

from sklearn import manifold, datasets


import argparse

parser = argparse.ArgumentParser(description='Generate classification data sets')

parser.add_argument('--n_samples',default=100,type=int,help='')

parser.add_argument('--n_features',type=int,default=20, help='The total number of features. These comprise n_informative informative features, n_redundant redundant features, n_repeated duplicated features and n_features-n_informative-n_redundant- n_repeated useless features drawn at random.')

parser.add_argument('--n_informative',type=int,default=2, help='The number of informative features. Each class is composed of a number of gaussian clusters each located around the vertices of a hypercube in a subspace of dimension n_informative. For each cluster, informative features are drawn independently from N(0, 1) and then randomly linearly combined within each cluster in order to add covariance. The clusters are then placed on the vertices of the hypercube.')
parser.add_argument('--n_redundant', type=int,default=2, help='The number of redundant features. These features are generated as random linear combinations of the informative features.')

parser.add_argument('--n_repeated',type=int,default=0, help='The number of duplicated features, drawn randomly from the informative and the redundant features.')

parser.add_argument('--n_classes',type=int,default=2, help='The number of classes (or labels) of the classification problem.')

parser.add_argument('--n_clusters_per_class',type=int,default=2,help='The number of clusters per class.')

parser.add_argument('--weights',help='The proportions of samples assigned to each class. If None, then classes are balanced. Note that if len(weights) == n_classes - 1, then the last class weight is automatically inferred. More than n_samples samples may be returned if the sum of weights exceeds 1.')

parser.add_argument('--flip_y',type=float,default=0.01,help='The fraction of samples whose class are randomly exchanged.')

parser.add_argument('--class_sep',type=float,default=1.0,help='The factor multiplying the hypercube dimension.')

parser.add_argument('--hypercube',type=bool,default=True,help='If True, the clusters are put on the vertices of a hypercube. If False, the clusters are put on the vertices of a random polytope.')

parser.add_argument('--shift',type=float,default=0.0,help='Shift features by the specified value. If None, then features are shifted by a random value drawn in [-class_sep, class_sep].')

parser.add_argument('--scale',type=float,help='shape [n_features] or None, optional (default=1.0) Multiply features by the specified value. If None, then features are scaled by a random value drawn in [1, 100]. Note that scaling happens after shifting.')

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


from os.path import basename

def write_dict(data):
    for key in data:
        print('{"%s":"%s"}'%(key,data[key]))


np.random.seed(0)

X, y = datasets.make_classification(args.n_samples, n_features=args.n_features, n_informative=args.n_informative, n_redundant=args.n_redundant, n_repeated=args.n_repeated,
                          n_classes=args.n_classes, n_clusters_per_class=args.n_clusters_per_class, weights=args.weights, flip_y=args.flip_y, class_sep=args.class_sep, hypercube=args.hypercube, shift=args.shift,
                          scale=args.scale, shuffle=args.shuffle, random_state=args.random_state)

datasets.dump_svmlight_file(X,y,args.output_file,zero_based=args.zero_based,query_id=args.query_id,multilabel=args.multilabel,comment=args.comment)

write_dict({'feature_file':args.output_file})
