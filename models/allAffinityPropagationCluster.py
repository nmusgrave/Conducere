#!/usr/bin/python

# An example on how to write an ML model in accordance with run.py

import sys
import importlib
from math import sqrt

# This is the important method required for every ML module in this
# architecture. It takes in an array of arguments, which will contain
# 0 or more elements. execute() could impose further requirements, as
# is shown with this example.
def execute(args):
  library = importlib.import_module("models")
  problem_file = getattr(library, "affinityPropagationCluster")
  problem = getattr(problem_file, "execute")
  possibles = lists([0, 1, 2, 3, 4, 5, 6, 7, 8])
  features = get_features(args[0])
  results = {}
  clusters = {}
  for i in range(len(possibles)):
    possible = possibles[i]
    possible.insert(0, args[0])
    p = problem(possible)
    results[i] = p[0]
    clusters[i] = p[1]
  m = best_mean(results)
  dev = lowest_std_dev(results)
  both = best_of_both_worlds(results, clusters, 50)
  print
  print
  print "BEST SUBSET BY MEAN"
  print possibles[m]
  print results[m]
  print "MEAN", mean(results[m]), "STD DEV", std_dev(results[m])
  print
  print "LOWEST STANDARD DEVIATION"
  print possibles[dev]
  print results[dev]
  print "MEAN", mean(results[dev]), "STD DEV", std_dev(results[dev])
  print
  print "BEST OF BOTH WORLDS"
  print possibles[both]
  print results[both]
  print "MEAN", mean(results[both]), "STD DEV", std_dev(results[both])

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

def best_mean(results):
  means = {key: mean(result) for key, result in results.iteritems()}
  return max(means, key=lambda x: means[x])

def mean(result):
  return sum([v for k, v in result.iteritems()]) / len(result)

def lowest_std_dev(results):
  devs = {key : std_dev(result) for key, result in results.iteritems()}
  return min(devs, key=lambda x: devs[x])

def std_dev(data):
  m = mean(data)
  return sqrt(sum([(datum - m) ** 2 for _, datum in data.iteritems()]) / len(data))

def best_of_both_worlds(results, clusters, cluster_limit):
  boths = {key: mean(result) - std_dev(result) for key, result in results.iteritems() if clusters[key] <= cluster_limit}
  return max(boths, key=lambda x: boths[x])
