#!/usr/bin/python

import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs

# Clustering

import sys

# Gives the usage of this program
def usage():
  print "Usage: python run.py cluster [data_file]"

# Executes a model for clustering data. Treats the first feature as the dependent
# feature.
def execute(args):
  if len(args) < 1:
    usage()
    sys.exit()
  y, x = parse(args[0])

# Parses the given file into a matrix of data. The depenedent variable is assumed
# to be at the beginning
def parse(filename):
  raw = [[feature for feature in line.strip().split(',')] for line in open(filename, 'r')]
  dependent = [sample[0] for sample in raw]
  independent = [sample[1:] for sample in raw]
  return dependent, independent
