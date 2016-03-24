# CSE 481I - Sound Capstone wi16
# Conducere (TM)

# Random Forest Classification

import sys
import numpy as np
import util
import copy

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.cross_validation import train_test_split


###############################################################################
# Parameters for training Random Forest Classifier
#
TRAIN_PARTITION = 0.7       # train/test partition ratio
FOREST_SIZE = 1000           # number of trees in forest
CRITERION = 'entropy'          # criterion for splits (default='gini')
MAX_FEATURES = 'auto'       # max number of features to consider during split
VERBOSE = False             # verbose setting during forest construction

ATTRIBUTES = ['danceability','energy','liveness','loudness','speechiness','tempo','valence','instrumentalness','acousticness']


###############################################################################
# Setting up
# Construct feature vectors from each song's echonest analysis.
# The dependent variable is assumed to be at the beginning
def parse(filename):
  raw = [[feature for feature in line.strip().split(',')] for line in open(filename, 'r')]
  names = raw[0]
  raw = raw[1:]
  np.random.shuffle(raw)
  dependent = [sample[0] for sample in raw]
  independent = [sample[1:] for sample in raw]
  independent = [[float(sample_point) for sample_point in sample] for sample in independent]
  return names, dependent, np.asarray(independent)

def average_score_k_trials(model, x, y, k):
  """
  Takes the average accuracy score over k trials with the given
  model, x (independent data), and y (dependent data).
  """
  avg = 0
  for i in range(k):
    avg += model.score(x, y)
  return avg / float(k)

###############################################################################
# Evaluation
#
def evaluate(model, x_test, y_test):
  """
  Prints a summary of results for multiclass classification using a RFC.
  Also prints feature importances for the RFC.
  """
  avg_over_10 = average_score_k_trials(model, x_test, y_test, 10)
  
  print "================================================================================"
  print "Summary of Results:"
  print
  print "Forest Size = %d trees" % FOREST_SIZE
  print "Random Guess Mean Accuracy: %f" % (1.0 / model.n_classes_)
  print "Avg Accuracy (10 trials): ", avg_over_10
  print
  print "================================================================================"
  print("Results using Random Forest Classification:\n\n%s\n" % (
    metrics.classification_report(
        y_test,
        model.predict(x_test))))
 
  print
  print "================================================================================"
  print "Feature importance for Random Forest Classifier:\n"
  for i in range(len(model.feature_importances_)):
    print "%20s:\t\t%f" % (ATTRIBUTES[i], model.feature_importances_[i])
  print
  print "================================================================================"
  print "Done with evaluation"


def evaluate_forest_size(x_train, y_train, x_test, y_test):
  """
  Collects analysis data on forest size vs. average accuracy.
  """
  for i in range(1, FOREST_SIZE+2, 20):
    # Set up Random Forest Classifier
    model = RandomForestClassifier(
              n_estimators=i, 
              criterion=CRITERION, 
              max_features=MAX_FEATURES,
              verbose=VERBOSE,
        )

    model.fit(x_train, y_train)
    avg_score = average_score_k_trials(model, x_test, y_test, 10)
    print str(i) + "," + str(avg_score) 


###############################################################################
# Invocation
#

# Gives the usage of this program
def usage():
  print "Usage: python run.py random_forest [num iterations] [data_file]"

# Executes a model for clustering data. Treats the first feature as the dependent
# feature.
def execute(args):
  if len(args) < 1:
    usage()
    sys.exit()

  # Parse data
  #   names == feature labels
  #   x     == features that correspond to shuffled names
  #   y     == shuffled names
  names, y, x = parse(args[0])
  x = util.clean(names, x)

  # Runs a multi-class classification using Random Forest.
  # The number of possible class predictions = number of users.

  print "Running full multi-class classification:"
  print "Number of users: %d" % (len(set(y)))
  print

  x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                  test_size=TRAIN_PARTITION,
                                                  random_state=0)

  # Set up Random Forest Classifier
  model = RandomForestClassifier(
            n_estimators=FOREST_SIZE, 
            criterion=CRITERION, 
            max_features=MAX_FEATURES,
            verbose=VERBOSE,
      )

  model.fit(x_train, y_train)

  # Evaluation
  evaluate(model, x_test, y_test)


  