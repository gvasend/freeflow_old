#!/usr/bin/python
#http://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html
import numpy as np
from sklearn import manifold, datasets

import csv
import sys
import logging

from sklearn import manifold, datasets


import argparse

parser = argparse.ArgumentParser(description='Perform Isomap analysis on feature data.')

parser.add_argument('--n_neighbors',default=5,type=int,help='number of neighbors to consider for each point.')

parser.add_argument('--n_components',default=2,type=int,help='number of coordinates for the manifold')

parser.add_argument('--eigen_solver',default='auto',choices=['auto','arpack','dense'], help='auto: Attempt to choose the most efficient solver for the given problem. arpack : Use Arnoldi decomposition to find the eigenvalues and eigenvectors. dense : Use a direct solver (i.e. LAPACK) for the eigenvalue decomposition.')

parser.add_argument('--tol',default=0.0,type=float,help='Convergence tolerance passed to arpack or lobpcg. not used if eigen_solver == dense.')

parser.add_argument('--max_iter',type=int,help='Maximum number of iterations for the arpack solver. not used if eigen_solver == dense.')

parser.add_argument('--path_method',default='auto',choices=['auto','FW','D'],help='Method to use in finding shortest path. auto : attempt to choose the best algorithm automatically. FW : Floyd-Warshall algorithm. D : Dijkstra’s algorithm.')

parser.add_argument('--neighbors_algorithm',default='auto',choices=['auto','brute','kd_tree','ball_tree'],help='Algorithm to use for nearest neighbors search, passed to neighbors.NearestNeighbors instance.')

parser.add_argument('--n_jobs',type=int,default=1,help='The number of parallel jobs to run. If -1, then the number of jobs is set to the number of CPU cores.')

import scrape as sc

sc.model_options(parser)
sc.all_options(parser)
sc.output_options(parser)

from scrape import write_dict

args = sc.parse_args(parser)

from sklearn.manifold import Isomap

model = Isomap(n_neighbors=args.n_neighbors, n_components=args.n_components, eigen_solver=args.eigen_solver, tol=args.tol, max_iter=args.max_iter, path_method=args.path_method, neighbors_algorithm=args.neighbors_algorithm, n_jobs=args.n_jobs)

# model = Isomap(n_neighbors=args.n_neighbors, n_components=args.n_components)
   
sc.save_model(model, args.model_output_file)




