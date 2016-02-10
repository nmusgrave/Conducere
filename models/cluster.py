#!/usr/bin/python


# Clustering

import sys

import numpy as np
from sklearn.cluster import KMeans



# Gives the usage of this program
def usage():
  print "Usage: python run.py cluster [data_file] [features to use...]"

# Executes a model for clustering data. Treats the first feature as the dependent
# feature.
#
# For arguments, takes the data file, and an optional list of features to use. If
# no list is given, will use all features. Outputs a map of each cluster, in the
# form of majority playlist and the percentage of the cluster belonging to that
# playlist.
def execute(args):
  if len(args) < 1:
    usage()
    sys.exit()
  names, y, x = parse(args[0])
  indices = args[1:]
  if len(indices) > 0:
    x = [[sample[i] for i in indices] for sample in x]

  kmeans = KMeans(n_clusters=3, random_state=0)
  y_pred = kmeans.fit_predict(x)

  counts = get_cluster_counts(y, y_pred)
  totals = [0] * len(counts)
  for i, mapping in counts.iteritems():
    totals[i] = sum(mapping.values())
  finals = get_final_mapping(counts, totals)
  if len(finals) < len(counts):
    print "WARNING: Insufficient data to distinguish all playlists"
  print finals


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

# Given the actual y-values 'y' and the predicted values 'y_pred', returns the counts
# for each playlist in each cluster.
#
# Returns a map, containing an id for each cluster. Each id maps to another map,
# containing playlist labels as keys. Each label maps to the count of that label
# in the cluster.
def get_cluster_counts(y, y_pred):
  unique = np.unique(y_pred)
  labels = np.unique(y)
  counts = {un : {label : 0 for label in labels} for un in unique}
  for i in range(len(y)):
    counts[y_pred[i]][y[i]] += 1
  return counts

# Returns the final mapping, which is a decided playlist and a percentage of that cluster
# made up of that playlist
def get_final_mapping(counts, totals):
  return {max(mapping, key = lambda k : mapping[k]) : max(mapping.values()) / float(totals[key]) for key, mapping in counts.iteritems()}
