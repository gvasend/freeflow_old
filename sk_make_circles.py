#!/usr/bin/python
#http://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html
import numpy as np
import csv
import sys
from sklearn.datasets import make_circles

from sklearn import manifold, datasets


import argparse

parser = argparse.ArgumentParser(description='Generate test data sets')

parser.add_argument('--type',default='',help='')

parser.add_argument('--n_samples',default=100,type=int,help='')

parser.add_argument('--factor',type=float,default=0.8,help='')

parser.add_argument('--noise',help='',type=float)

parser.add_argument('--shuffle',type=bool,default=True,help='')

import scrape as sc

sc.all_options(parser)
sc.output_options(parser)

from scrape import write_dict

args = sc.parse_args(parser)


#results = np.loadtxt(open("test_1_centroids.csv","rb"),delimiter=",",skiprows=1)

#print (results)

# bcml mode feature_class version
#  0    1        2           3
# mode { baseline, update }


np.random.seed(0)

X, y = make_circles(n_samples=args.n_samples, factor=args.factor, noise=args.noise, shuffle=args.shuffle)

datasets.dump_svmlight_file(X,y,args.output_file,zero_based=args.zero_based,query_id=args.query_id,multilabel=args.multilabel,comment=args.comment)

write_dict({'feature_file':args.output_file})
