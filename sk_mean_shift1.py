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

parser.add_argument('--mode',default='baseline',help='baseline or update')

parser.add_argument('--components',default=2,help='number of components to use in PCA analysis')

parser.add_argument('--feature_file',required=True,help='file containing feature data')

import scrape as sc

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


n_components = 2


mode = args.mode

feature_file = args.feature_file

from os.path import basename

feature_class = basename(feature_file)
feature_extraction = feature_class
#labels_file = feature_class+"_labels.csv"
#known_labels_file = feature_class+"_known_labels.csv"
pickle_file  = feature_class+"_model.pkl"
legends_file =  feature_class+"_legend.csv"


write_dict({'pickle_file':pickle_file,'legend_file':legends_file})



 
X, y = load_file(feature_file)
    
X1 = X.copy()

print("feature file load complete")

import matplotlib.pyplot as plt
from itertools import cycle

			  
ms = MeanShift()
ms.fit(X1)
labels = ms.labels_
n_clusters_ = len(np.unique(labels))

#kmeans = KMeans(n_clusters=n_clusters_)
#kmeans.fit(X)

centroids = ms.cluster_centers_
labels = ms.labels_


from sklearn.externals import joblib
joblib.dump(ms, pickle_file) 

cluster_centers = ms.cluster_centers_

write_dict({'cluster_centers':cluster_centers,'n_clusters':len(ms.cluster_centers_)})

#****************************************************
# correlate generic labels to known labels

if 0==1:
    with open(known_labels_file, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        known_labels = [row for row in spamreader]
    import statistics

    label_key = []
    for n in range(n_clusters_):
        this_label = []
        for m in range(len(labels)):
            if labels[m] == n:
                this_label.append(known_labels[m][1])
        label_key.append(statistics.mode(this_label))

    with open(legends_file, "w") as f:
        for m in range(len(label_key)):
            print(m,",",label_key[m],file=f)
else:
    label_key = []
		
#*****************************************************************

# set baseline



if not args.output_file == None:
    datasets.dump_svmlight_file(X,labels,args.output_file,zero_based=args.zero_based,query_id=args.query_id,multilabel=args.multilabel,comment=args.comment)
    write_dict({'output_file':args.output_file})


