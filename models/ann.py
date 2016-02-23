# CSE 481I - Sound Capstone wi16
# Conducere (TM)

# Artifical Neural Network
# Unsupervised feature learning performed by Restricted Boltzmann Machine
# Classification by Logistic Regression


import numpy as np

from scipy.ndimage import convolve
from sklearn import linear_model, datasets, metrics, preprocessing
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
  independent = [[abs(float(sample_point)) for sample_point in sample] for sample in independent]
  return names, dependent, np.asarray(independent)

# Binarize the data
def binarize(x):
  #X = [x[0], x[1], x[2]] # for testing below process
  # print '----------'
  arr = np.array(x)
  arr = arr.astype(float)
  col = arr[:,0]
  x = col
  complete_bin = np.empty([len(arr), len(arr[0])])
  for i in range(len(arr[0])):
    x = [arr[:,i]]
    # print '----------'
    # == process 1: normalize first ==
    # scale data so falls between -1 and 1
    # normalizer = preprocessing.Normalizer().fit(x)
    # nor = normalizer.transform(x)
    nor = preprocessing.normalize(x, norm='l2', axis=1, copy=True)
    nor = map(lambda x: x - 0.5, nor)
    # print 'nor1', nor
    sca1 = preprocessing.scale(nor, axis=0, with_mean=False, with_std=True, copy=True)
    # print 'sca1', sca1
    binarizer = preprocessing.Binarizer().fit(sca1)
    bin = binarizer.transform(sca1)
    # print 'bin1', bin
    complete_bin[:,i] = bin
  return complete_bin

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
  # y     == shuffled names
  # x     == features that correspond to shuffled names
  names, y, x = parse(args[0])
  # print y
  # print 'x---', x
  # x = binarize(x)
  # print 'bx--', x

  # create fake data for testing
  # naomi = [10] * len(names)
  # naomi = [1, 2, 3, 9, 9, 9, 9, 9, 9, 9]
  # megan = [5] * len(names)
  # connor = [1] * len(names)
  # y = ['naomi', 'megan', 'connor']
  # x = [naomi, megan, connor]
  # print len(x), len(y), len(names)
  # print y
  # print x
  # x = binarize(x)
  # print x

  # print len(x), len(y), len(names)

  x = np.asarray(x, 'float32')
  # X, Y = nudge_dataset(X, digits.target)
  x = (x - np.min(x, 0)) / (np.max(x, 0) + 0.0001)  # 0-1 scaling
  print 'x==', x[1]


  x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.2,
                                                    random_state=0)

  # classifier = construct_classifier()
  # logisitic_classifier = train(classifier, x_train, y_train)
  # evaluate(classifier, logisitic_classifier, x_test, y_test)
  

  # Models we will use
  logistic = linear_model.LogisticRegression()
  rbm = BernoulliRBM(random_state=0, verbose=True)

  classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])

  ###############################################################################
  # Training

  # Hyper-parameters. These were set by cross-validation,
  # using a GridSearchCV. Here we are not performing cross-validation to
  # save time.
  rbm.learning_rate = 0.06
  rbm.n_iter = 20
  # More components tend to give better prediction performance, but larger
  # fitting time
  rbm.n_components = 100
  logistic.C = 6000.0

  # Training RBM-Logistic Pipeline
  classifier.fit(x_train, y_train)

  # Training Logistic regression
  logistic_classifier = linear_model.LogisticRegression(C=100.0)
  logistic_classifier.fit(x_train, y_train)

  ###############################################################################
  # Evaluation

  print()
  print("Logistic regression using RBM features:\n%s\n" % (
      metrics.classification_report(
          y_test,
          classifier.predict(x_test))))

  print("Logistic regression using raw pixel features:\n%s\n" % (
      metrics.classification_report(
          y_test,
          logistic_classifier.predict(x_test))))