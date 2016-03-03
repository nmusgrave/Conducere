#!/usr/bin/python

# CSE 481I - Sound Capstone wi16
# Conducere (TM)

# Clustering

import sys
import importlib

import numpy as np
import math

# The number of clusters should be this * number of y values
CLUSTER_FACTOR = 3

# Gives the usage of this program
def usage():
  print "Usage: python run.py kMeansCluster [iterations] [data_file] [features to use...]"

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
  library = importlib.import_module("models")
  kmeans_problem_file = getattr(library, "kMeansCluster")
  kmeans = getattr(kmeans_problem_file, "execute")(args)
  affinity_problem_file = getattr(library, "affinityPropagationCluster")
  affinity = getattr(affinity_problem_file, "execute")(args)

  merge = merge_accuracies([(kmeans, .7), (affinity[0], .3)])
  print
  print
  print "ACCURACY", merge
  return merge

def merge_accuracies(accuracies):
  merged = {}
  for accuracy in accuracies:
    for name, val in accuracy[0].iteritems():
      if name not in merged:
        merged[name] = 0
      merged[name] = merged[name] + val * accuracy[1]
  return merged
