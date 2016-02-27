#!/usr/bin/python

# CSE 481I - Sound Capstone wi16
# Conducere (TM)

import sys
import importlib

# prints out how to use this script
def usage():
  print "Usage: python run.py [model_name] [num_iterations] [parameters..]"


# note that sys.argv[0] is always the name of the script run
if len(sys.argv) < 3:
  usage()
  sys.exit()

# dynamically imports the model from models, and calls with arguments sys.argv[2..]
library = importlib.import_module("models")
try:
  problem_file = getattr(library, str(sys.argv[1]))
  problem = getattr(problem_file, "execute")
  for i in range(int(sys.argv[2])):
    problem(sys.argv[3:])
except AttributeError as a:
  print "Error: no module/method found for", sys.argv[1]
  print a
  sys.exit()
