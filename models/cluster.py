#!/usr/bin/python

# CSE 481I - Sound Capstone wi16
# Conducere (TM)

# Clustering

import sys

import numpy as np
from sklearn.cluster import KMeans
import math

# The number of clusters should be this * number of y values
CLUSTER_FACTOR = 4

# Gives the usage of this program
def usage():
  print "Usage: python run.py cluster [iterations] [data_file] [features to use...]"

# Executes a model for clustering data. Treats the first feature as the dependent
# feature.
#
# For arguments, takes the data file, and an optional list of features to use. If
# no list is given, will use all features. Outputs a map of each cluster, in the
# form of majority playlist and the percentage of the cluster belonging to that
# playlist.
def execute(args):
  np.random.seed(42)
  if len(args) < 1:
    usage()
    sys.exit()
  names, y, x = parse(args[0])
  indices = [int(i) for i in args[1:]]
  relevant_names = names
  if len(indices) > 0:
    x = [[sample[i] for i in indices] for sample in x]
    relevant_names = [names[i] for i in indices]
  print "Clustering on", str(relevant_names) + "..."

  labels = np.unique(y)
  kmeans = KMeans(n_clusters= CLUSTER_FACTOR * len(labels), random_state=0)
  y_pred = kmeans.fit_predict(x)

  counts = get_cluster_counts(y, y_pred)
  totals = [0] * len(counts)
  print counts
  for i, mapping in counts.iteritems():
    totals[i] = sum(mapping.values())
  finals = get_final_mapping(counts, totals)
  if len(finals) < len(labels):
    print "WARNING: Not all clusters unique!"
  print "FINAL CLUSTERS", finals
  print
  print "ACCURACY", accuracy(finals, labels)
  return accuracy(finals, labels)


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
  clusters = [(max(mapping, key = lambda k : mapping[k]), mapping) for key, mapping in counts.iteritems()]
  combined = {}
  for cluster in clusters:
    old = combined.get(cluster[0], {})
    new = {}
    for key in cluster[1]:
      new[key] = cluster[1][key] + old.get(key, 0)
    combined[cluster[0]] = new
  return combined

def probability(final):
  return {name : value[name] / float(sum([v for k, v in value.iteritems()])) for name, value in final.iteritems()}

def accuracy(final, labels):
  # tuple for (right, wrong)
  accuracy = {label : [0, 0] for label in labels}
  for name, data in final.iteritems():
    for dataName, dataValue in data.iteritems():
      if name == dataName:
        accuracy[dataName][0] += dataValue
      else:
        accuracy[dataName][1] += dataValue
  final_accuracy = {name : value[0] / float(sum(value)) for name, value in accuracy.iteritems()}
  return final_accuracy
