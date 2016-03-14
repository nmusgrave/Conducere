# CSE 481I - Sound Capstone wi16
# Conducere (TM)

# Random Forest Binary Classification

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


def get_binary_user_combinations(users):
  """
  For the given list of users, finds every unique binary combination
  of those users and returns a list of tuples corresponding to these
  unique combinations.
  """
  results = []
  sorted_uniq = sorted(set(users))
  for i in range(len(sorted_uniq)):
    for j in range(i, len(sorted_uniq)):
      if i is not j:
        results.append((sorted_uniq[i], sorted_uniq[j]))
  return results


def prune_data(x, y, users):
  """
  Given a collection of users, prunes independent (x) data and dependent (y) 
  data to only contain data pertaining to the users present in the collection.

  In our case, a user corresponds to a y-value (label). 
  """
  res_x, res_y = [], []
  for i in range(len(x)):
    if y[i] in users:
      res_x.append(x[i])
      res_y.append(y[i])
  return res_x, res_y


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
def evaluate(scores, num_classes):
  """ 
  Evaluates the average accuracies for each user from the provided dictionary
  of scores and prints the results.

    scores is of the form {user_1 : {user_2 : score}}
  
  """
  print 
  print "================================================================================"
  print "Calculating average accuracies for each user based on combination"
  print "accuracy results."
  print
  print "Random Guess Mean Accuracy: %f" % (1.0 / num_classes)
  print

  count = 0
  print "%20s\t\t%s" % ("User", "Avg. Accuracy")
  print
  for u in scores:
    u_avg = sum(scores[u].values()) / len(scores[u].values())
    print "%20s:\t\t%f" % (u, u_avg)
    count += 1

  print
  print "================================================================================"
  print "Done with evaluation"



###############################################################################
# Invocation
#

# Gives the usage of this program
def usage():
  print "Usage: python run.py random_forest_binary [num iterations] [data_file]"

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

  # Runs RFC every combination of pairs of users for a binary classification.
  # The number of possible class predictions = 2.

  num_users = len(set(y))
  num_combos = np.math.factorial(num_users) / (2 * np.math.factorial(num_users - 2))

  print "Testing 2-way combinations of users for binary classification:"
  print "Number of users: %d" % (num_users)
  print "Number of combinations: %d" % (num_combos)
  print
  print "================================================================================"
  print "Evaluating Combinations of Users:"
  print 

  combos = get_binary_user_combinations(y)
  COMBO_SCORES = {}
  for c in combos:
    x_pruned, y_pruned = prune_data(x, y, c)

    x_train, x_test, y_train, y_test = train_test_split(x_pruned, y_pruned,
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

    # Updating combination scores
    if c[0] not in COMBO_SCORES:
      COMBO_SCORES[c[0]] = {}
    if c[1] not in COMBO_SCORES:
      COMBO_SCORES[c[1]] = {}

    score = average_score_k_trials(model, x_test, y_test, 5)
    COMBO_SCORES[c[0]][c[1]] = score
    COMBO_SCORES[c[1]][c[0]] = score

    print "\tEvaluating users: %35s       %f" % (c, score)
  
  # Evaluate final results
  evaluate(COMBO_SCORES, 2)

