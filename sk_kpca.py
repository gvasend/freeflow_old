"""
==========
Kernel PCA
==========

This example shows that Kernel PCA is able to find a projection of the data
that makes data linearly separable.
"""
print(__doc__)

# Authors: Mathieu Blondel
#          Andreas Mueller
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import make_circles
from sklearn import datasets



import argparse

parser = argparse.ArgumentParser(description='Perform Kernel PCA analysis.')

parser.add_argument('--n_components',type=int,default=2,help='number of components to use in PCA analysis')

parser.add_argument('--kernel',default='rbf',choices=['linear','poly','rbf','sigmoid','cosine','precomputed'],help='number of components to use in PCA analysis')

parser.add_argument('--degree',type=int,default=3,help='Degree for poly kernels. Ignored by other kernels.')

parser.add_argument('--gamma',default=1.0,type=float,help='Kernel coefficient for rbf and poly kernels. Ignored by other kernels.')

parser.add_argument('--coef0',type=float,default=1.0,help='Independent term in poly and sigmoid kernels. Ignored by other kernels.')

parser.add_argument('--kernel_params',help='Parameters (keyword arguments) and values for kernel passed as callable object. Ignored by other kernels.')

parser.add_argument('--alpha',type=int,default=1,help='Hyperparameter of the ridge regression that learns the inverse transform (when fit_inverse_transform=True).')

parser.add_argument('--fit_inverse_transform',type=bool,default=False,help='Learn the inverse transform for non-precomputed kernels. (i.e. learn to find the pre-image of a point)')

parser.add_argument('--eigen_solver',default='auto',choices=['auto','dense','arpack'],help='Select eigensolver to use. If n_components is much less than the number of training samples, arpack may be more efficient than the dense eigensolver')

parser.add_argument('--tol',type=float,default=0.0,help='Convergence tolerance for arpack. If 0, optimal value will be chosen by arpack.')

parser.add_argument('--max_iter',help='Maximum number of iterations for arpack. If None, optimal value will be chosen by arpack.')

parser.add_argument('--remove_zero_eig',help='If True, then all components with zero eigenvalues are removed, so that the number of components in the output may be < n_components (and sometimes even zero due to numerical instability). When n_components is None, this parameter is ignored and components with zero eigenvalues are removed regardless.')
parser.add_argument('--random_state',help='A pseudo random number generator used for the initialization of the residuals when eigen_solver == ‘arpack’.')
parser.add_argument('--n_jobs',type=int,default=1,help='The number of parallel jobs to run. If -1, then the number of jobs is set to the number of CPU cores.')
parser.add_argument('--copy_X',help='If True, input X is copied and stored by the model in the X_fit_ attribute. If no further changes will be done to X, setting copy_X=False saves memory by storing a reference.')



parser.add_argument('--plot',default='no',choices=['no','save','show'],help='plot options: no, save, show')

parser.add_argument('--plot_title',default='PCA Transform',help='Title for the plot')

import scrape as sc
sc.model_options(parser)
sc.all_options(parser)
sc.output_options(parser)

from scrape import write_dict

args = sc.parse_args(parser)



#
# Compute PCA
#

kpca = KernelPCA(kernel=args.kernel, fit_inverse_transform=args.fit_inverse_transform, gamma=args.gamma, n_components=args.n_components,
                 degree=args.degree, coef0=args.coef0, kernel_params=args.kernel_params, alpha=args.alpha, eigen_solver=args.eigen_solver,
                 tol=args.tol,max_iter=args.max_iter, random_state=args.random_state, n_jobs=args.n_jobs, copy_X=args.copy_X)

   
sc.save_model(kpca, args.model_output_file)







