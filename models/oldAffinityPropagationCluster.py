import numpy as np

from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

def usage():
  print "Usage: python run.py kMeansCluster [iterations] [data_file] [features to use...]"

def execute(args):
  ##############################################################################
  if len(args) < 1:
    usage()
    sys.exit()

  names, labels_true, X = parse(args[0])
  indices = [int(i) for i in args[1:]]
  relevant_names = names[1:]
  if len(indices) > 0:
    X = np.asarray([[sample[i] for i in indices] for sample in X])
    relevant_names = [relevant_names[i] for i in indices]
  print "Clustering on", str(relevant_names) + "..."

  
  ##############################################################################
  # Compute Affinity Propagation
  af = AffinityPropagation(preference=-50)
  # cluster_centers_indices = af.cluster_centers_indices_
  # labels = af.labels_
  # 
  # n_clusters_ = len(cluster_centers_indices)

  y_pred = af.fit_predict(X)
  if y_pred is None or len(y_pred) is 0 or type(y_pred[0]) is np.ndarray:
    return 0
  counts = get_cluster_counts(labels_true, y_pred)
  print counts
  
  # print('Estimated number of clusters: %d' % n_clusters_)
  # print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
  # print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
  # print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
  # print("Adjusted Rand Index: %0.3f"
  #     % metrics.adjusted_rand_score(labels_true, labels))
  # print("Adjusted Mutual Information: %0.3f"
  #     % metrics.adjusted_mutual_info_score(labels_true, labels))
  # print("Silhouette Coefficient: %0.3f"
  #     % metrics.silhouette_score(X, labels, metric='sqeuclidean'))
  # return metrics.silhouette_score(X, labels, metric='sqeuclidean')
  
# Parses the given file into a matrix of data. The depenedent variable is assumed
# to be at the beginning
def parse(filename):
  raw = [[feature for feature in line.strip().split(',')] for line in open(filename, 'r')]
  names = raw[0]
  raw = raw[1:]
  np.random.shuffle(raw)
  dependent = [sample[0] for sample in raw]
  independent = [sample[1:] for sample in raw]
  independent = [[float(sample_point) for sample_point in sample] for sample in independent]
  return names, dependent, np.asarray(independent)

# Given the actual y-values 'y' and the predicted values 'y_pred', returns the counts
# for each playlist in each cluster.
#
# Returns a map, containing an id for each cluster. Each id maps to another map,
# containing playlist labels as keys. Each label maps to the count of that label
# in the cluster.
def get_cluster_counts(y, y_pred):
  unique = np.unique(y_pred)
  labels = np.unique(y)
  counts = {un : {label : 0 for label in labels} for un in unique}
  for i in range(len(y)):
    counts[y_pred[i]][y[i]] += 1
  return counts

# Returns the final mapping, which is a decided playlist and a percentage of that cluster
# made up of that playlist
def get_final_mapping(counts, totals):
  clusters = [(max(mapping, key = lambda k : mapping[k]), mapping) for key, mapping in counts.iteritems()]
  combined = {}
  for cluster in clusters:
    old = combined.get(cluster[0], {})
    new = {}
    for key in cluster[1]:
      new[key] = cluster[1][key] + old.get(key, 0)
    combined[cluster[0]] = new
  return combined

def probability(final):
  return {name : value[name] / float(sum([v for k, v in value.iteritems()])) for name, value in final.iteritems()}

def accuracy(final, labels):
  # tuple for (right, wrong)
  accuracy = {label : [0, 0] for label in labels}
  for name, data in final.iteritems():
    for dataName, dataValue in data.iteritems():
      if name == dataName:
        accuracy[dataName][0] += dataValue
      else:
        accuracy[dataName][1] += dataValue
  final_accuracy = {name : value[0] / float(sum(value)) for name, value in accuracy.iteritems()}
  return final_accuracy
