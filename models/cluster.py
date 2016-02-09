#!/usr/bin/python


# Clustering

import sys

import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs



# Gives the usage of this program
def usage():
  print "Usage: python run.py cluster [data_file]"

# Executes a model for clustering data. Treats the first feature as the dependent
# feature.
def execute(args):
  if len(args) < 1:
    usage()
    sys.exit()
  names, y, x = parse(args[0])
  train_x = x[:3 * len(x) / 4]
  train_y = y[:3 * len(y) / 4]
  test_x = x[3 * len(x) / 4:]
  test_y = y[3 * len(y) / 4:]

  # The following bandwidth can be automatically detected using
  bandwidth = estimate_bandwidth(train_x, quantile=0.2, n_samples=len(train_x))

  ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
  ms.fit_predict(train_x, train_y)
  cluster_centers = ms.cluster_centers_

  labels_unique = np.unique(labels)
  n_clusters_ = len(labels_unique)

  print("number of estimated clusters : %d" % n_clusters_)

  predict_y = ms.predict(test_x)
  print labels_unique

# Parses the given file into a matrix of data. The depenedent variable is assumed
# to be at the beginning
def parse(filename):
  raw = [[feature for feature in line.strip().split(',')] for line in open(filename, 'r')]
  names = raw[0]
  raw = raw[1:]
  np.random.shuffle(raw)
  dependent = [sample[0] for sample in raw]
  independent = [sample[1:] for sample in raw]
  independent = [[float(sample_point) for sample_point in sample] for sample in independent]
  return names, dependent, np.asarray(independent)
