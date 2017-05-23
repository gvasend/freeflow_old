#!/usr/bin/python
#http://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html
import numpy as np
import csv
import sys
from sklearn.datasets import make_s_curve

from sklearn import manifold, datasets


import argparse

parser = argparse.ArgumentParser(description='Generate S curve datasets')

parser.add_argument('--n_samples',default=100,type=int,help='')

parser.add_argument('--noise',default=0.0,type=float,help='The standard deviation of the gaussian noise.')


parser.add_argument('--random_state',type=int,help=' RandomState instance or None, optional (default=None) If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random.')


import scrape as sc

sc.all_options(parser)
sc.output_options(parser)

from scrape import write_dict

args = sc.parse_args(parser)


np.random.seed(0)

X, y = make_s_curve(n_samples=args.n_samples, noise=args.noise, random_state=args.random_state)

datasets.dump_svmlight_file(X,y,args.output_file,zero_based=args.zero_based,query_id=args.query_id,multilabel=args.multilabel,comment=args.comment)

write_dict({'feature_file':args.output_file})
