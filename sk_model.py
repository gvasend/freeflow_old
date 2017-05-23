#!/usr/bin/python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
import csv
import sys
import uuid

from sklearn import manifold, datasets
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
import scrape as sc
from sklearn.decomposition import PCA

h = 0.02

import argparse

# general parameters

parser = argparse.ArgumentParser(description='Perform a function using an classifier pickle.')

sc.model_options(parser)
sc.all_options(parser)

parser.add_argument('--pca')
parser.add_argument('--feature_file',default='*feature_file',help='file containing feature data')

parser.add_argument('--plot',default='no',choices=['no','save','show'],help='plot options: no, save, show')

parser.add_argument('--plot_title',default='Model Plot',help='Title for the plot')

# SVC parameters

sc.output_options(parser)

from scrape import write_dict
from scrape import load_file

args = sc.parse_args(parser)

from numpy import genfromtxt
feature_file = args.feature_file
# print ("feature_file:",feature_file)

y = None
if '.csv' in feature_file:
    X = genfromtxt(feature_file, delimiter=',')
    y = [1 for y1 in range(len(X))]
elif '.libsvm' in feature_file:
    X, y = datasets.load_svmlight_files([feature_file])
    X = X.toarray()
#    y = y.toarray()

pca = PCA()
pca.fit(X)
# print("matrix shape ",len(X),len(X[0]))
sc.write_dict({'rows':len(X),'features':len(X[0])})

svc = sc.load_model(args.model_file)


labels_created = False
data_changed = False
model_change = False
if args.action == 'fit':
    svc.fit(X,y)
    model_change=True
elif args.action == 'fit_transform':
    X = svc.fit_transform(X,y)
    model_change = True
    data_changed = True
elif args.action == 'predict':
    y = svc.predict(X)
    labels_created = True
elif args.action == 'transform':
    X = svc.transform(X,y)
    data_changed = True
elif args.action == 'fit_predict':
    y = svc.fit_predict(X)
    model_change = True
    labels_created = True
elif args.action == 'score':
    svc.score(X,y)

write_dict(svc.__dict__)

    
figure = plt.figure(figsize=(27, 9))
i = 1
# iterate over datasets
#for ds_cnt, ds in []:
if not args.plot == 'no':
    ds_cnt = 0
    # preprocess dataset, split into training and test part
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.3)

    X_test = np.nan_to_num(X_test)
    y_test = np.nan_to_num(y_test)

    X_test = (X_test - np.mean(X_test, axis=0)) / np.std(X_test, axis=0)



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
    X_train2d = pca.transform(X_train)
    X_test2d = pca.transform(X_test)
    ax.scatter(X_train2d[:, 0], X_train2d[:, 1], c=y_train, cmap=cm_bright)
    # and testing points
    ax.scatter(X_test2d[:, 0], X_test2d[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
#    for name, clf in zip(['svc'], [svc]):
    if not args.plot == 'no' and 0==1:
        name = 'svc'
        clf = svc
        ax = plt.subplot(1, 2, i)
        clf.fit(X_train, y_train)
        
        score_op = getattr(clf, "score", None)
        try:
            score = clf.score(X_test, y_test)
        except:
            score = 1.0

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        plot_z = False
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            plot_z = True
        else:
            if hasattr(clf, "predict_proba"):
                Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
                plot_z = True

        # Put the result into a color plot
        if plot_z == True:
           try:
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
           except:
            print("reshape failed")

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
            ax.set_title(type(clf))
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
        i += 1

#plt.tight_layout()

#plt.figure()

fname = 'model_plot.png'

if args.plot == 'save':
    plt.savefig(fname)
    write_dict({'plot_file':fname})
elif args.plot == 'show':
    plt.savefig(fname)
    write_dict({'plot_file':fname})
    plt.show()

if model_change == True:    
    sc.save_model(svc, args.model_output_file)

if data_changed == True or labels_created == True:
    if args.output_file == None:
        setattr(args, 'output_file', "sk_ff_"+str(uuid.uuid1()))+'.libsvm'
    datasets.dump_svmlight_file(X,y,args.output_file,zero_based=args.zero_based,query_id=args.query_id,multilabel=args.multilabel,comment=args.comment)
    sc.write_dict({'feature_file':args.output_file})
    sc.write_dict({'out_rows':len(X),'out_features':len(X[0])})
    
    


