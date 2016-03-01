# CSE 481I - Sound Capstone wi16
# Conducere (TM)

# Artifical Neural Network
# Unsupervised feature learning performed by Restricted Boltzmann Machine
# Classification by Logistic Regression

# Input data should either be binary, or real-valued between 0 and 1 signifying
# the probability that the visible unit would turn on or off

from sklearn import linear_model
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler

from util import parse, clean, selectFeatures, evaluate, powerset
import copy

###############################################################################
# Parameters for training
# The best parameters are {'logistic__C': 100, 'rbm__n_iter': 1,
# 'rbm__learning_rate': 0.001, 'rbm__n_components': 300} with a score of 0.30

# Logistic regression features
L_COMPONENTS=100
# Neural network features
# More components tend to give better prediction performance, but larger
# fitting time
N_LEARNING_RATE = 0.001
N_ITER=10
N_COMPONENTS=300

###############################################################################
# Invocation

# Gives the usage of this program
# Iterations should be 1, since best number of iterations for the model are found via grid search
def usage():
  print "Usage: python run.py ann [iterations] [data_file] [features] (note: iterations should be 1)"

# Command line arguments: data file
def execute(args):
  print 'Starting the artificial neural network'
  if len(args) < 1:
    usage()
    sys.exit()

  ###############################################################################
  # Data

  # names     feature labels
  # y         shuffled names
  # x         features that correspond to shuffled names
  names, y, x = parse(args[0])
  x = clean(names, x)

  # Build features to include in test
  features = args[1:]
  if len(features) == 0:
    features = names
  print 'Selected features:', features
  combos = powerset(features)
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

    logistic = linear_model.LogisticRegression(C=L_COMPONENTS)
    rbm = BernoulliRBM(random_state=0, verbose=True, learning_rate=N_LEARNING_RATE, n_iter=N_ITER, n_components=N_COMPONENTS)

    # Note: attempted StandardScaler, MinMaxScaler, MaxAbsScaler, without strong results
    # Not needed, since data is scaled to the [0-1] range by clean()
    classifier = Pipeline(steps=[('rbm', rbm),('logistic', logistic)])

    # ###############################################################################
    # Training
    print 'Training the classifier...'
    # Training RBM-Logistic Pipeline
    classifier.fit(x_train, y_train)

    # Training Logistic regression
    logistic_classifier = linear_model.LogisticRegression(C=100.0)
    logistic_classifier.fit(x_train, y_train)

    ###############################################################################
    # Evaluation
    evaluate(classifier, logistic_classifier, x_test, y_test)
