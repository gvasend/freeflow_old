#!/usr/bin/python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
import csv
from sklearn.dummy import DummyClassifier
import sys
import uuid

from sklearn import manifold, datasets
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D

Axes3D

import scrape as sc

h = 0.02

import argparse

# general parameters

parser = argparse.ArgumentParser(description='Perform SVM machine learning.')

sc.all_options(parser)

parser.add_argument('--feature_file',default='*feature_file',help='file containing feature data')

parser.add_argument('--plot',default='show',choices=['no','save','show'],help='plot options: no, save, show')

parser.add_argument('--plot_title',default='Model Plot',help='Title for the plot')

parser.add_argument('--plot_file',help='Plot output file')

parser.add_argument('--model_file',help='load existing model from file., default is DummyClassifier')

parser.add_argument('--type',default='input')

parser.add_argument('--projection',choices=['3d','polar'])

parser.add_argument('--feature_file2')

from scrape import write_dict
from scrape import load_file

args = sc.parse_args(parser)

plots = args.type.split(",")


from os.path import basename


import datetime
print(datetime.datetime.now())

from numpy import genfromtxt
feature_file = args.feature_file
print ("feature_file:",feature_file)

if '.csv' in feature_file:
    X = genfromtxt(feature_file, delimiter=',')
elif '.libsvm' in feature_file:
    X, y = datasets.load_svmlight_files([feature_file])
    X = X.toarray()
    
feature_file2 = args.feature_file2
print ("feature_file2:",feature_file2)

if not feature_file2 == None:
    if '.csv' in feature_file2:
        X2 = genfromtxt(feature_file, delimiter=',')
    elif '.libsvm' in feature_file2:
        X2, y2 = datasets.load_svmlight_files([feature_file2])
        X2 = X2.toarray()
    #    y = y.toarray()

print("matrix shape ",len(X),len(X[0]))

if not args.model_file == None:
    svc = sc.load_model(args.model_file)
else:
    svc = DummyClassifier()
  
clf = svc

figure = plt.figure(figsize=(27, 9))
i = 1
# iterate over datasets
#for ds_cnt, ds in []:
from sklearn.decomposition import PCA

if len(X[0]) > 3:
    X = PCA().fit_transform(X)

if not args.plot == 'no' and 'basic' in plots:
    sc.write_dict({'plot_type':'basic'})
    plt.title(args.plot_title)
    ax = figure.add_subplot(111)
    if feature_file2 == None:
        yd = y
    else:
        y_not_noise = [abs(item[0]-item[1]) for item in zip(y,y2) if item[0] > -1 and item[0] > -1]
        yd_not_noise = [item for item in y_not_noise if item == 0]
        score1 = float(len(yd_not_noise))/float(len(y_not_noise))
        yd = [abs(item[0]-item[1]) for item in zip(y,y2)]
        yd1 = [item for item in yd if item == 0]
        scr = float(len(yd1))/float(len(yd))
        sc.write_dict({'score_noise':scr,'score':score1})
        ax.text(0.95, 0.1, ('score %.2f' % (score1)).lstrip('0'),
                    size=15, horizontalalignment='right', transform=ax.transAxes)
    ax.scatter(X[:, 0], X[:, 1], marker='o', c=yd)

#    print(yd)

if not args.plot == 'no' and 'input' in plots:
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
    ax = figure.add_subplot(121, projection=args.projection)
    # Plot the training points
    if args.projection == None:
#        ax = plt.subplot(1, 2, i)
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
    # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
    if args.projection == '3d':
#        ax = plt.subplot(251, projection='3d')
        ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:,2], c=y_train, cmap=cm_bright)
    # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=y_test, cmap=cm_bright, alpha=0.6)
    if ds_cnt == 0:
        ax.set_title("Input data")
    score_op = getattr(clf, "score", None)
    if not score_op == None:
        score = clf.score(X_test, y_test)
    else:
        score = 1.0
    sc.write_dict({'score':score})
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
#    for name, clf in zip(['svc'], [svc]):
    try:
        if not args.plot == 'no' and 'classifier' in plots:
            name = type(svc)
            ax = plt.subplot(1, 2, i)
            clf.fit(X_train, y_train)
            


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
    except:
        print("plot exception")
        pass

#plt.tight_layout()

#plt.figure()

#    plt.show() g
if args.plot == 'save':
    if args.plot_file == None:
        id = str(uuid.uuid1())
        setattr(args, 'plot_file', "sk_plot_"+id+'.png')

    fname = args.plot_file
    plt.savefig(fname)
    write_dict({'svc_plot_file':fname})
elif args.plot == 'show':
    plt.show()

   


