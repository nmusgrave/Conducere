#!/usr/bin/python


# Clustering

import sys

import numpy as np
from sklearn.linear_model import ARDRegression



# Gives the usage of this program
def usage():
  print "Usage: python run.py cluster [data_file]"

# Executes a model for clustering data. Treats the first feature as the dependent
# feature.
def execute(args):
  if len(args) < 1:
    usage()
    sys.exit()

  # Parse and partition
  names, y, x = parse(args[0])
  train_x = x[:3 * len(x) / 4]
  train_y = y[:3 * len(y) / 4]
  test_x = x[3 * len(x) / 4:]
  test_y = y[3 * len(y) / 4:]

  

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
  return names, dependent, np.asarray(independent)irint "iteration:", t

