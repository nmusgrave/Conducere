#!/usr/bin/python

# An example on how to write an ML model in accordance with run.py

import sys
import importlib

# This is the important method required for every ML module in this
# architecture. It takes in an array of arguments, which will contain
# 0 or more elements. execute() could impose further requirements, as
# is shown with this example.
def execute(args):
  library = importlib.import_module("models")
  problem_file = getattr(library, "cluster")
  problem = getattr(problem_file, "execute")
  possibles = lists([0, 1, 2, 3, 4, 5, 6, 7, 8])
  features = get_features(args[0])
  results = {}
  for i in range(len(possibles)):
    possible = possibles[i]
    possible.insert(0, args[0])
    results[i] = problem(possible)
  maxSumIndex = max(results, key = lambda k : sum([v for name, v in results[k].iteritems()]))
  print
  print
  print "BY AVERAGE"
  print "\tFEATURE NUMBERS", possibles[maxSumIndex][1:]
  print "\tFEATURES LEARNED ON", [features[i] for i in possibles[maxSumIndex][1:]]
  print "\tMAX", results[maxSumIndex]

# Returns all possible subsets of numbers
def lists(numbers):
  return lists_helper(numbers, [], 0)

def lists_helper(numbers, current, index):
  if index >= len(numbers):
    return [list(current)]
  elements = lists_helper(numbers, current, index + 1)
  current.append(numbers[index])
  elements += lists_helper(numbers, current, index + 1)
  current.pop()
  return elements

def get_features(filename):
  raw = [feature for feature in open(filename, 'r').readline().strip().split(',')]
  return raw[1:]
