#!/usr/bin/python
#http://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
import csv
import sys

from sklearn import manifold, datasets


import argparse

parser = argparse.ArgumentParser(description='Perform Kmeans analysis on feature data.')

parser.add_argument('--n_clusters',type=int,default=8, help='The number of clusters to form as well as the number of centroids to generate.')

parser.add_argument('--max_iter',type=int,default=300,help='Maximum number of iterations of the k-means algorithm for a single run.')

parser.add_argument('--n_init',type=int,default=10,help='Number of time the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia.')

parser.add_argument('--init',choices=['k-means++','random','ndarray'],default='k-means++',help='Method for initialization, defaults to k-means++: k-means++ : selects initial cluster centers for k-mean clustering in a smart way to speed up convergence. See section Notes in k_init for more details. random: choose k observations (rows) at random from data for the initial centroids. If an ndarray is passed, it should be of shape (n_clusters, n_features) and gives the initial centers.')

parser.add_argument('--algorithm',choices=['auto', 'full', 'elkan'], default='auto', help='K-means algorithm to use. The classical EM-style algorithm is “full”. The “elkan” variation is more efficient by using the triangle inequality, but currently doesn’t support sparse data. “auto” chooses “elkan” for dense data and “full” for sparse data.')

parser.add_argument('--precompute_distances', choices=['auto','True','False'], default='auto', help='Precompute distances (faster but takes more memory). ‘auto’ : do not precompute distances if n_samples * n_clusters > 12 million. This corresponds to about 100MB overhead per job using double precision. True : always precompute distances False : never precompute distances')

parser.add_argument('--tol',type=float,default=1e-4,help='Relative tolerance with regards to inertia to declare convergence')

parser.add_argument('--n_jobs',default=1,type=int, help='The number of jobs to use for the computation. This works by computing each of the n_init runs in parallel. If -1 all CPUs are used. If 1 is given, no parallel computing code is used at all, which is useful for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one are used.')
                    
parser.add_argument('--random_state',type=int,help='The generator used to initialize the centers. If an integer is given, it fixes the seed. Defaults to the global numpy random number generator.')

parser.add_argument('--verbose',type=int,default=0,help='Verbosity mode.')

parser.add_argument('--copy_x',type=bool,default=True,help='When pre-computing distances it is more numerically accurate to center the data first. If copy_x is True, then the original data is not modified. If False, the original data is modified, and put back before the function returns, but small numerical differences may be introduced by subtracting and then adding the data mean.')


import scrape as sc
from scrape import write_dict
from scrape import load_file

sc.all_options(parser)
sc.model_options(parser)

args = sc.parse_args(parser)


   
kmeans = KMeans(n_clusters=args.n_clusters, init=args.init, n_init=args.n_init, max_iter=args.max_iter, tol=args.tol, precompute_distances=args.precompute_distances, verbose=args.verbose, random_state=args.random_state, 
            copy_x=args.copy_x, n_jobs=args.n_jobs, algorithm=args.algorithm)

 
sc.save_model(kmeans, args.model_output_file)





    
    


