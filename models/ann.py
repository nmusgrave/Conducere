# CSE 481I - Sound Capstone wi16
# Conducere (TM)

# Artifical Neural Network
# Unsupervised feature learning performed by Restricted Boltzmann Machine
# Classification by Logistic Regression

# Input data should either be binary, or real-valued between 0 and 1 signifying
# the probability that the visible unit would turn on or off


import numpy as np

from sklearn import linear_model, metrics
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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
  independent = [[abs(float(sample_point)) for sample_point in sample] for sample in independent]
  return names, dependent, np.asarray(independent)


###############################################################################
# Evaluation
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
# Feature Selection
# all       all features
# target    desired features
# x         data set
def selectFeatures(all, target, x):
  arr = np.array(x)
  remove = set(all) - set(target)
  for f in remove:
    # determine index for feature
    i = all.index(f)
    del all[i]
    arr = np.delete(arr, i, 1)
  return arr
  

###############################################################################
# Invocation

# Gives the usage of this program
def usage():
  print "Usage: python run.py ann [iterations] [data_file] [features]"

# Command line arguments: data file
def execute(args):
  if len(args) < 1:
    usage()
    sys.exit()

  # names     feature labels
  # y         shuffled names
  # x         features that correspond to shuffled names
  names, y, x = parse(args[0])
  # remove user_id column
  names = names[1:]

  # Build features to include in test
  features = args[1:]
  if len(features) == 0:
    features = names

  print 'Testing on features:', features

  x = selectFeatures(names, features, x)

  # Split into testing and traiing data
  x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.2,
                                                    random_state=0)

  ###############################################################################
  # Models
  # TODO: explore best params using cross-validation (GridSearchCV)
  # More components tend to give better prediction performance, but larger
  # fitting time
  logistic = linear_model.LogisticRegression(C=6000.0)
  rbm = BernoulliRBM(random_state=0, verbose=True, learning_rate=0.006, n_iter=50, n_components=300)

  # note: sometimes undefinedmetricwarning appears for bad data
  # this shows up occasionally regardless of preprocessing step
  # others: ('min/max scaler', MinMaxScaler()), ('maxabs scaler', MaxAbsScaler())
  classifier = Pipeline(steps=[
    ('standard scaler', StandardScaler()),        # best results so far
    ('rbm', rbm),
    ('logistic', logistic)
  ])

  # ###############################################################################
  # Training

  # Training RBM-Logistic Pipeline
  classifier.fit(x_train, y_train)

  # Training Logistic regression
  logistic_classifier = linear_model.LogisticRegression(C=100.0)
  logistic_classifier.fit(x_train, y_train)

  ###############################################################################
  # Evaluation

  evaluate(classifier, logistic_classifier, x_test, y_test)

