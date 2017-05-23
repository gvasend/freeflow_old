#!/usr/bin/python
#http://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
import csv
import sys

from sklearn import manifold, datasets


import argparse

parser = argparse.ArgumentParser(description='Create a cluster model using Mean Shift model.')

parser.add_argument('--components',default=2,help='number of components to use in PCA analysis')



parser.add_argument('--bandwidth',type=float,help='Bandwidth used in the RBF kernel. If not given, the bandwidth is estimated using sklearn.cluster.estimate_bandwidth')
parser.add_argument('--seeds',help='shape=[n_samples, n_features], Seeds used to initialize kernels. If not set, the seeds are calculated by clustering.get_bin_seeds with bandwidth as the grid size and default values for other parameters.')
parser.add_argument('--bin_seeding',type=bool,help='If true, initial kernel locations are not locations of all points, but rather the location of the discretized version of points, where points are binned onto a grid whose coarseness corresponds to the bandwidth. Setting this option to True will speed up the algorithm because fewer seeds will be initialized. default value: False Ignored if seeds argument is not None.')
parser.add_argument('--min_bin_freq',type=int,help='To speed up the algorithm, accept only those bins with at least min_bin_freq points as seeds. If not defined, set to 1.')
parser.add_argument('--cluster_all',type=bool,default=True,help='If true, then all points are clustered, even those orphans that are not within any kernel. Orphans are assigned to the nearest kernel. If false, then orphans are given cluster label -1.')
parser.add_argument('--n_jobs',default=1,type=int,help='The number of jobs to use for the computation. This works by computing each of the n_init runs in parallel. If -1 all CPUs are used. If 1 is given, no parallel computing code is used at all, which is useful for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one are used.')


import scrape as sc

sc.model_options(parser)
sc.output_options(parser)

import scrape as sc
from scrape import write_dict
from scrape import load_file

sc.all_options(parser)

args = sc.parse_args(parser)



#results = np.loadtxt(open("test_1_centroids.csv","rb"),delimiter=",",skiprows=1)

#print (results)

# bcml mode feature_class version
#  0    1        2           3
# mode { baseline, update }

	  
ms = MeanShift(bandwidth=args.bandwidth, seeds=args.seeds, bin_seeding=args.bin_seeding, min_bin_freq=args.min_bin_freq, cluster_all=args.cluster_all, n_jobs=args.n_jobs)

 
sc.save_model(ms, args.model_output_file)

