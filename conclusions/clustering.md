# Clustering

Clustering is an unsupervised classification model that attempts to separate
n-dimensional data into a number of clusters. We explored two different kinds
of clustering, k-means and affinity propagation. In both models, the metric is
based on accuracy of the clusters. First, each cluster is labeled the majority
element from that cluster. For example, if cluster 1 is {A: 35, B: 20, C:
45}, then cluster 1 is labeled C. Second, we iterate over all labeled
clusters, marking any they got correct and any they got wrong. We keep the
total right and total wrong for each label. In this example, after seeing
cluster 1, we woudl have {A: [0 right, 35 wrong], B: [0 right, 20 wrong], C:
[45 right, 0 wrong]}. We then use these final counts to determine accuracy for
each label. In each model, we try every possible subset of independent
features, and pick the approximate best, based on mean and standard deviation
of accuracies.

For reference, our final data had 11 labels, meaning a random guess would give
an average mean of 0.0909, or 9.09%.

K-means clustering will look for exactly k clusters, and will shift the mean of
the k clusters to place the data as well as it can. We set k = C * len(labels),
and tried a number of values for C. The approximate best results were as
follows, where the first line is what features were clustered on and the second
is mean and standard deviation:

C = 1: 
  ['danceability', 'liveness', 'loudness', 'tempo', 'valence', 'acousticness']
  mean: 0.151    standard deviation: 0.142

C = 2:
  ['danceability', 'loudness', 'tempo', 'valence', 'acousticness']
  mean: 0.175   standard deviation: 0.115

C = 3:
  ['danceability', 'tempo', 'valence', 'acousticness']
  mean: 0.200   standard deviation: 0.096

C = 4:
  ['energy', 'liveness', 'valence']
  mean: 0.221   standard deviation: 0.133

Any more values of C likely risks overfitting


Affinity Propagation looks at all of the data, and in an iterative process
looks for the best mean centers of the data, attempting to cluster all of the
data in some center or another. It doesn't limit the number of clusters, and
our data can be very spread at parts. As a result, it was difficult to limit
overfitting. We split the data to try to reduce overfitting, fitting to a
training data set, a subset of the entire data set. We attempted to use a
damping factor, which will dampen your normalized data by adding a certain
amount to each of it, bringing it farther from the origin of the means but
reducing the difference between the distances of individual data points from
the means. Here are four runs from different damping constants. The first line
is data clustered on, the second is the number of clusters (a strong indication
of overfitting), and the third is mean and standard deviation:

damping = 0.5:
  ['danceability']
  number of clusters: 1002
  mean: 0.911   standard deviation: 0.050

damping = 0.55:
  ['danceability']
  number of clusters: 984
  mean: 0.898   standard deviation: 0.048

damping = 0.6:
  ['valence']
  number of clusters: 857
  mean: 0.371   standard deviation: 0.112

damping = 0.65:
  ['valence']
  number of clusters: 701
  mean: 0.332   standard deviation: 0.125


Perhaps the most interesting result came from combining these two models. If
k-means is too coarse, and affinity propagation overfits, it seems like a good
idea to to combine the two models in some way. The following is what resulted
from letting affinity propagation overfit with a damping of 0.52, and combining
the two models by making the accuracy for each label 0.7 * kMeansPrediction +
0.3 * affinityPropagationPrediction:

  ['acousticness']
  mean: 0.400   standard deviation: 0.140

Clustering is a plain and simple classification aglorithm. For our project,
this means that uncertainty in the model can be as interesting as certainty. If
you look closer at the data, some labels were clustered very diffinitively,
consistenly getting an accuracy around 0.6 or 0.7. Others not so much. Some
playlists never rose above 0.1. But uncertainty could indicate a similarity in
music taste between two labels that got clustered together consistently. It
could indicate that some music tastes are more explorative and spread out,
having many data points in many different clusters. It seems that going
forward, it would be very interesting to see a measurement of the spread of
music tastes, analyzing numbers between labels as well as the individual data
points. I also think that data would be better split by "taste" rather than by
individuals, to give a more accurate model for music listening tendancies. This
data would seem to corroborate the obvious, that people have a number of
chaning music tastes, and that taste is often shared between many groups of
people. I think it would be more interesting in the future to look at a model
where, say, person A has a music tastes {0, 1, 3} and person B has music tastes
{1, 2, 4}. This could then be used to suggest to Person A that they listen to
selections from music taste 1, from Person B or otherwise, but not necessarily
suggest 2 or 4 from Person B's selection.
