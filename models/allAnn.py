# CSE 481I - Sound Capstone wi16
# Conducere (TM)

# Artifical Neural Network
# Unsupervised feature learning performed by Restricted Boltzmann Machine
# Classification by Logistic Regression

# Input data should either be binary, or real-valued between 0 and 1 signifying
# the probability that the visible unit would turn on or off

import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report

from util import parse, clean, selectFeatures

###############################################################################
# Parameters for training
# More components tend to give better prediction performance, but larger
# fitting time
rbm_learning = np.logspace(-3, 0, 10)
rbm_iter = range(1, 51, 10)
rbm_components = [10, 50, 100, 300]
log_C = [10, 40, 60, 100]
param_grid = dict(rbm__learning_rate=rbm_learning, rbm__n_iter=rbm_iter,
    rbm__n_components=rbm_components, logistic__C=log_C)

###############################################################################
# Invocation

# Gives the usage of this program
# Iterations should be 1, since best number of iterations for the model are found via grid search
def usage():
  print "Usage: python run.py ann [iterations] [data_file] [features] (note: iterations should be 1)"

# Command line arguments: data file
def execute(args):
  print 'Starting the gridsearch on the artificial neural network'
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
  x = selectFeatures(names, features, x)

  # Split into testing and traiing data
  x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.2,
                                                    random_state=0)

  ###############################################################################
  # Models

  logistic = LogisticRegression()
  rbm = BernoulliRBM(random_state=0, verbose=True)
  classifier = Pipeline(steps=[('rbm', rbm),('logistic', logistic)])

  # ###############################################################################
  # Training
  print 'Training the classifier...'
  # Training RBM-Logistic Pipeline
  # classifier.fit(x_train, y_train)

  # Training Logistic regression
  # logistic_classifier = LogisticRegression(C=100.0)
  # logistic_classifier.fit(x_train, y_train)
  # evaluate(classifier, logistic_classifier, x_test, y_test)

  ###############################################################################
  # Evaluation

  scores = ['precision', 'recall']
  for score in scores:
      print("# Tuning hyper-parameters for %s" % score)
      print()

      clf = GridSearchCV(classifier, param_grid=param_grid, cv=3, scoring='%s_weighted' % score)
      clf.fit(x_train, y_train)

      print("The best parameters are %s with a score of %0.2f"
       % (clf.best_params_, clf.best_score_))
      print()
      print("Grid scores on development set:")
      print()
      for params, mean_score, scores in clf.grid_scores_:
          print("%0.3f (+/-%0.03f) for %r"
                % (mean_score, scores.std() * 2, params))
      print()

      print("Detailed classification report:")
      print()
      print("The model is trained on the full development set.")
      print("The scores are computed on the full evaluation set.")
      print()
      y_true, y_pred = y_test, clf.predict(x_test)
      print(classification_report(y_true, y_pred))
      print()

