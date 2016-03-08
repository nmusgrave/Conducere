# CSE 481I - Sound Capstone wi16
# Conducere (TM)

# Artifical Neural Network
# Unsupervised feature learning performed by Restricted Boltzmann Machine
# Classification by Logistic Regression

# Input data should either be binary, or real-valued between 0 and 1 signifying
# the probability that the visible unit would turn on or off

import sys
from collections import defaultdict, Counter

from sklearn import linear_model, metrics
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler

from util import parse, clean, selectFeatures, evaluate, powerset
import copy

###############################################################################
# Parameters for training
# On data_2_19_16.txt:
# The best parameters are {'logistic__C': 100, 'rbm__n_iter': 1,
# 'rbm__learning_rate': 0.001, 'rbm__n_components': 300} with a score of 0.30

# On data_3_1_16.txt:
# The best parameters are {'logistic__C': 100, 'rbm__n_iter': 21,
# 'rbm__learning_rate': 0.0046415888336127772, 'rbm__n_components': 300} with a score of 0.22

# Logistic regression features
L_REGULARIZATION=100
# Neural network features
# More components tend to give better prediction performance, but larger
# fitting time
N_LEARNING_RATE = 0.001
N_ITER=31
N_COMPONENTS=300


###############################################################################
# Invocation

# Gives the usage of this program
# Iterations should be 1, since best number of iterations for the model are found via grid search
# Use powerset, to examine a powerset of all features specified
# Can
def usage():
  print "Usage: python run.py ann [int iterations] [bool use powerset] [data_file] [features]"
  print "(required) Iterations should be 1, since best number of iterations for the model are found via grid search"
  print "(required) Enable the powerset, to run the model on all possible subsets of the features"
  print "(optional) List no features, to run with all features"

# Command line arguments: data file
def execute(args):
  print 'Starting the artificial neural network'
  if len(args) < 2:
    usage()
    sys.exit()

  ###############################################################################
  # Data

  # names     feature labels
  # y         shuffled names
  # x         features that correspond to shuffled names
  names, y, x = parse(args[1])
  x = clean(names, x)
  usePowerset = args[0]

  # Build features to include in test
  features = args[2:]
  if len(features) == 0:
    features = names
  # print 'Selected features:', features

  # Build all subsets of features, if requested
  if usePowerset.lower() == 'true':
    combos = powerset(features)
  else:
    combos = [features]

  # map from feature set, to map of correct counts for each person
  feature_performance = {}
  highest_correct = 0
  best_combo = {}
  for c in combos:
    if len(c) == 0:
      continue
    print 'Attempting feature set:', c
    x_selected = selectFeatures(copy.copy(names), c, x)

    # Split into testing and traiing data
    x_train, x_test, y_train, y_test = train_test_split(x_selected, y,
                                                      test_size=0.2,
                                                      random_state=0)

    ###############################################################################
    # Models

    logistic = linear_model.LogisticRegression(C=L_REGULARIZATION)
    rbm = BernoulliRBM(random_state=0, verbose=True, learning_rate=N_LEARNING_RATE, n_iter=N_ITER, n_components=N_COMPONENTS)

    # Note: attempted StandardScaler, MinMaxScaler, MaxAbsScaler, without strong results
    # Not needed, since data is scaled to the [0-1] range by clean()
    classifier = Pipeline(steps=[('rbm', rbm),('logistic', logistic)])

    # ###############################################################################
    # Training
    print 'Training the classifier...'
    # Training RBM-Logistic Pipeline
    classifier.fit(x_train,y_train)
    correct = 0
    label_counts = defaultdict(int)
    for i in range(len(x_test)):
      test = x_test[i]
      if len(test) == 1:
        test = test.reshape(-1, 1)
      else:
        test = [test]
      predicted = classifier.predict(test)

      if predicted == y_test[i]:
        correct += 1
        label_counts[predicted[0]] += 1

    if correct >= highest_correct:
      highest_correct = correct
      best_combo = c
    feature_performance[str(c)] = {'predictions':label_counts,'expected':Counter(y_test)}

    ###############################################################################
    # Evaluation
    # evaluate(classifier, x_test, y_test)

  summary = feature_performance[str(best_combo)]
  print 'Accuracy:\t\t\t', highest_correct, 'correct gives', (highest_correct * 1.0/len(y_test)), 'compared to guessing', (1.0/len(summary['expected']))
  print 'Best feature set:\t\t', best_combo
  print 'Identified %d out of %d labels'%(len(summary['predictions']),len(summary['expected']))
  for p in summary['predictions']:
    pred = summary['predictions'][p]
    tot = summary['expected'][p]
    print '\t %s \t\t %d\t of %d \t (%f)'%(p, pred, tot, pred * 1.0/tot)
