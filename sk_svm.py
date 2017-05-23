#!/usr/bin/python
#http://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
import csv
import sys

from sklearn import manifold, datasets
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
import scrape as sc

h = 0.02

import argparse

# general parameters

parser = argparse.ArgumentParser(description='Perform SVM machine learning.')

sc.model_options(parser)

parser.add_argument('--feature_file',required=True,help='file containing feature data')

parser.add_argument('--plot',default='no',choices=['no','save','show'],help='plot options: no, save, show')

parser.add_argument('--plot_title',default='KMeans Centers',help='Title for the plot')

# SVC parameters

parser.add_argument('--C',type=float,default=1.0,help='Penalty parameter C of the error term.')

parser.add_argument('--kernel',default='rbf',help='Specifies the kernel type to be used in the algorithm. It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable. If none is given, ‘rbf’ will be used. If a callable is given it is used to pre-compute the kernel matrix from data matrices; that matrix should be an array of shape (n_samples, n_samples).')

parser.add_argument('--degree',type=int,default=3,help='Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.')

parser.add_argument('--gamma',default='auto',help='Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’. If gamma is ‘auto’ then 1/n_features will be used instead.')

parser.add_argument('--coef0',type=float,default=0.0,help='Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.')
                    
parser.add_argument('--probability',type=bool,default=False,help='Whether to enable probability estimates. This must be enabled prior to calling fit, and will slow down that method.')

parser.add_argument('--shrinking',type=bool, default=True, help='Whether to use the shrinking heuristic.')

parser.add_argument('--tol',type=float,default=1e-3,help='Tolerance for stopping criterion.')

parser.add_argument('--cache_size',default=100.0,type=float,help='Specify the size of the kernel cache (in MB).')

parser.add_argument('--class_weight',help='Set the parameter C of class i to class_weight[i]*C for SVC. If not given, all classes are supposed to have weight one. The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y))')

parser.add_argument('--verbose',type=bool,default=False,help='Enable verbose output. Note that this setting takes advantage of a per-process runtime setting in libsvm that, if enabled, may not work properly in a multithreaded context.')

parser.add_argument('--max_iter',type=int,default=-1,help='Hard limit on iterations within solver, or -1 for no limit.')

parser.add_argument('--decision_function_shape',choices=['ovo','ovr'],help='Whether to return a one-vs-rest (‘ovr’) decision function of shape (n_samples, n_classes) as all other classifiers, or the original one-vs-one (‘ovo’) decision function of libsvm which has shape (n_samples, n_classes * (n_classes - 1) / 2). The default of None will currently behave as ‘ovo’ for backward compatibility and raise a deprecation warning, but will change ‘ovr’ in 0.19. New in version 0.17: decision_function_shape=’ovr’ is recommended. Changed in version 0.17: Deprecated decision_function_shape=’ovo’ and None.')

parser.add_argument('--random_state',type=int,help='The seed of the pseudo random number generator to use when shuffling the data for probability estimation.')



from scrape import write_dict
from scrape import load_file

args = sc.parse_args(parser)



feature_file = args.feature_file

from os.path import basename

def write_dict(data):
    for key in data:
        print('{"%s":"%s"}'%(key,data[key]))

write_dict({'pca_file':'pca_plot.png'})


import datetime
print(datetime.datetime.now())

from numpy import genfromtxt
print ("feature_file:",feature_file)

if '.csv' in feature_file:
    X = genfromtxt(feature_file, delimiter=',')
elif '.libsvm' in feature_file:
    X, y = datasets.load_svmlight_files([feature_file])
    X = X.toarray()
#    y = y.toarray()

svc = sc.load_model(args.model_file)

if svc == None:
    svc = svm.SVC(C=args.C, kernel=args.kernel, degree=args.degree, gamma=args.gamma, coef0=args.coef0, shrinking=args.shrinking, probability=args.probability, tol=args.tol, cache_size=args.cache_size, 
              class_weight=args.class_weight, verbose=args.verbose, max_iter=args.max_iter, decision_function_shape=args.decision_function_shape, random_state=args.random_state)
    
svc.fit(X,y)

figure = plt.figure(figsize=(27, 9))
i = 1
# iterate over datasets
#for ds_cnt, ds in []:
if not args.plot == 'no':
    ds_cnt = 0
    # preprocess dataset, split into training and test part
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(1, 2, i)
    if ds_cnt == 0:
        ax.set_title("Input data")
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
    # and testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
#    for name, clf in zip(['svc'], [svc]):
    if not args.plot == 'no':
        name = 'svc'
        clf = svc
        ax = plt.subplot(1, 2, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot also the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
        i += 1

#plt.tight_layout()

#plt.figure()

#    plt.show() g
if args.plot == 'save':
    fname = 'svc_plot.png'
    plt.savefig(fname)
    write_dict({'svc_plot_file':fname})
elif args.plot == 'show':
    plt.show()
    
sc.save_model(svc, args.model_output_file)


    
    


