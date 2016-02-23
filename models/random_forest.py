# CSE 481I - Sound Capstone wi16
# Conducere (TM)

# Random Forest Classification

import sys
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.cross_validation import train_test_split

###############################################################################
# Parameters for training Random Forest Classifier
#
TRAIN_PARTITION = 0.7       # train/test partition ratio
FOREST_SIZE = 500           # number of trees in forest
CRITERION = 'gini'          # criterion for splits (default='gini')
MAX_FEATURES = 'auto'       # max number of features to consider during split
VERBOSE = False             # verbose setting during forest construction


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


###############################################################################
# Evaluation
#
def evaluate(model, x_test, y_test):
  print("Results using Random Forest Classification:\n\n%s\n" % (
    metrics.classification_report(
        y_test,
        model.predict(x_test))))


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
  evaluate(model, x_test, y_test)
