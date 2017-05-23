#!/usr/bin/python
#http://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
import csv
import sys
import logging

from sklearn import manifold, datasets


import argparse

parser = argparse.ArgumentParser(description='Perform PCA analysis on feature data.')

parser.add_argument('--n_components',type=int,default=2,help='number of components to use in PCA analysis')

parser.add_argument('--copy',type=bool,default=True,help='If False, data passed to fit are overwritten and running fit(X), transform(X) will not yield the expected results, use fit_transform(X) instead.')

parser.add_argument('--whiten',type=bool,default=False,help='')

parser.add_argument('--svd_solver',default='auto',choices=['auto','full','arpack','randomized'],help='')


parser.add_argument('--tol',type=float,default=0.0,help='Convergence tolerance for arpack. If 0, optimal value will be chosen by arpack.')

parser.add_argument('--iterated_power',default='auto',help='')

parser.add_argument('--random_state',help='A pseudo random number generator used for the initialization of the residuals when eigen_solver == ‘arpack’.')

import scrape as sc

sc.model_options(parser)
sc.all_options(parser)
sc.output_options(parser)

from scrape import write_dict

args = sc.parse_args(parser)







from sklearn.decomposition import PCA
pca = PCA(n_components=2, whiten=args.whiten, copy=args.copy, svd_solver=args.svd_solver, tol=args.tol, iterated_power=args.iterated_power, random_state=args.random_state)

   
sc.save_model(pca, args.model_output_file)




