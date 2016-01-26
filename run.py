#!/usr/bin/python

import sys
import importlib

# prints out how to use this script
def usage():
  print "Usage: python run.py [model_name] [parameters..]"


# note that sys.argv[0] is always the name of the script run
if len(sys.argv) < 2:
  usage()
  sys.exit()

# dynamically imports the model from models, and calls with arguments sys.argv[2..]
i = importlib.import_module("models")
try:
  problem_file = getattr(i, str(sys.argv[1]))
  problem = getattr(problem_file, "execute")
  problem(sys.argv[2:])
except AttributeError:
  print "Error: no module/method found for", sys.argv[1]
  sys.exit()
