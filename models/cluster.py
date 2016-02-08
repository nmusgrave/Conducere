#!/usr/bin/python

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
