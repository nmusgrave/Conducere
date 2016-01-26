#!/usr/bin/python

# An example on how to write an ML model in accordance with run.py

import sys

# Gives the usage of this program
def usage():
  print "Usage: python run.py example arg1 arg2 arg3"

# This is the important method required for every ML module in this
# architecture. It takes in an array of arguments, which will contain
# 0 or more elements. execute() could impose further requirements, as
# is shown with this example.
def execute(args):
  if len(args) < 3:
    usage()
    sys.exit()
  print "Hi! Here are your first three arguments:", args[0], ",", args[1], ",", args[2]
