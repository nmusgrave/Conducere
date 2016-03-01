import numpy as np
from sklearn import metrics

###############################################################################
# Setting up
# Construct feature vectors from each song's echonest analysis.
# The dependent variable is assumed to be at the beginning
def parse(filename):
  raw = [[feature for feature in line.strip().split(',')] for line in open(filename, 'r')]
  names = raw[0][1:]

  raw = raw[1:]
  np.random.shuffle(raw)
  dependent = [sample[0] for sample in raw]
  independent = [sample[1:] for sample in raw]
  independent = [[abs(float(sample_point)) for sample_point in sample] for sample in independent]
  return names, dependent, np.asarray(independent)

# Loudness and tempo are not on the [0-1] range, so must be modified
# Scale loudness from [-100 - 100], and tempo from [0 - 500], to [0-1]
def clean(features, x):
  # columns to adjust
  loudness = features.index('loudness')
  tempo = features.index('tempo')
  for row in x:
    row[loudness] = (row[loudness] + 100)/200
    row[tempo] = row[tempo] / 500
  return x

###############################################################################
# Evaluation
def evaluate(classifier, logistic_classifier, x_test, y_test):
  results = metrics.classification_report(y_test, classifier.predict(x_test))
  print("Logistic regression using RBM features:\n%s\n" % (results))
  print("Logistic regression using raw song features:\n%s\n" % (
    metrics.classification_report(
        y_test,
        logistic_classifier.predict(x_test))))
  return results


###############################################################################
# Feature Selection
# all       all features
# target    desired features
# x         data set
def selectFeatures(all, target, x):
  a = all
  arr = np.array(x)
  remove = set(a) - set(target)
  for f in remove:
    # determine index for feature
    i = a.index(f)
    del a[i]
    arr = np.delete(arr, i, 1)
  return arr

# Generator to build all subsets of a set
def powerset(s):
  if len(s) <= 1:
    yield s
    yield []
  else:
    for item in powerset(s[1:]):
      yield [s[0]]+item
      yield item
