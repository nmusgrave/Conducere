# CSE 481I - Sound Capstone wi16
# Conducere (TM)

# Artifical Neural Network
# Unsupervised feature learning performed by Restricted Boltzmann Machine
# Classification by Logistic Regression


import numpy as np

from scipy.ndimage import convolve
from sklearn import linear_model, datasets, metrics
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline

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

# 
def construct_classifier():
  # Models we will use
  logistic = linear_model.LogisticRegression()
  rbm = BernoulliRBM(random_state=0, verbose=True)

  # Hyper-parameters. These were set by cross-validation,
  # using a GridSearchCV. Here we are not performing cross-validation to
  # save time.
  rbm.learning_rate = 0.06
  rbm.n_iter = 20
  # More components tend to give better prediction performance, but larger
  # fitting time
  rbm.n_components = 100
  logistic.C = 6000.0

  classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])
  return classifier


###############################################################################
# Training
# 
def train(classifier, x_train, y_train):
  # Training RBM-Logistic Pipeline
  classifier.fit(x_train, y_train)

  # Training Logistic regression
  logistic_classifier = linear_model.LogisticRegression(C=100.0)
  logistic_classifier.fit(x_train, y_train)
  return logistic_classifier


###############################################################################
# Evaluation
# 
def evaluate(classifier, logistic_classifier, x_test, y_test):
  print()
  print("Logistic regression using RBM features:\n%s\n" % (
    metrics.classification_report(
        y_test,
        classifier.predict(x_test))))

  print("Logistic regression using raw pixel features:\n%s\n" % (
    metrics.classification_report(
        y_test,
        logistic_classifier.predict(x_test))))


###############################################################################
# Invocation

# Gives the usage of this program
def usage():
  print "Usage: python run.py ann [iterations] [data_file]"

# Command line arguments: data file
def execute(args):
  print args[0]
  if len(args) < 1:
    usage()
    sys.exit()
  # names == feature labels
  # x     == features that correspond to shuffled names
  # y     == shuffled names
  names, y, x = parse(args[0])

  x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.2,
                                                    random_state=0)

  classifier = construct_classifier()
  logisitic_classifier = train(classifier, x_train, y_train)
  evaluate(classifier, logisitic_classifier, x_test, y_test)
  