# Neural Network

## Motivations for Model

The goal is to determine if it is possible to identify a user's listening habits. If it is possible to do so, a neural network will be able to extract a feature set that describes the listening habits, and be able to correctly identify songs that align with individuals' tastes. At a high level, the model uses a combination of a neural network and classifier. The model is provided song data, presented as a vector of features, and an associated user profile. 

### Artificial Neural Network

In the first step, a Bernoulli Restricted Bolztmann Machine builds a feature space describing the training set of song vectors. It is an unsupervised model, so builds the feature space only from the song data, without any provided labels. During training, the RBM aims to maximize the product of probabilities assigned to the features it learns. The RBM constructs a bipartite graph of neurons to describe features. The parts are divided into hidden and visible nodes. Since this is a restricted Boltzmann Machine, there are no connections between hidden nodes. Therefore, node activations are mutually independent of each other. This is desirable in this case, since the given feature set is already dependent (for example, a song with high danceability is more likely to be highly energetic).

### Logistic Regression

Given the features learned by the neural network, a logistic regression function performs classification. It assigns labels to the song data. In general, logistic regression is useful for computing the probability of class membership. It maps a data point in a n-dimensional feature space to a value, which in turn maps to a label. Due to this behavior, logistic regression performs well on transforming a complex input set to a subset of labels. In particular, logistic regression favors and disfavors outcomes using log odds, rather than probabilities. Probalities treat positive and negative outcomes fairly, but we prefer to discount negative outcomes and reward positive.

### Steps

On the training subset of song vectors:

+ Step 1: train RBM to build up representation of feature space
+ Step 2: train a logistic classifier using the representation

On the testing subset of song vectors:

+ Predict the user associated with the song, and examine the accuracy

## Parameter tuning
Explore all possible combinations of features in a space

rbm_learning = np.logspace(-3, 0, 10)

+ Learning rate for weight updates
+ 10 values in the range 10^-3 to 1
+ Smaller values for slower learning

rbm_iter = range(1, 51, 10)

+ Train for 1, 11, 21, …, 51 iterations
+ At risk for over-fitting with more iterations

rbm_components = [10, 50, 100, 300]

+ Number of binary hidden components

log_C = [10, 40, 60, 100]

+ Smaller values gives stronger regularization
+ Regularization helps prevent overfitting (esp. Necessary since the data set for each user is small)

## Feature performance
Trained on power set of all features, using best parameters found above

## Results

+ Performed v poorly on small sample sets
+ Performed best on larger sets
+ Performed even better on sets with v similar genres (Mallika’s playlists) 
Data: vector of song features, and an associated user profile. Forms collections of playlists describing user listening habits. Each user has a varying distribution of song varieties. Some have songs tightly grouped within a genre (reference graph- Mallika), and others have a broader distribution of genres (reference graph- hunter?)


## References

[Scikit-learn on digit classification, using rbm & logistic regression](http://www.pyimagesearch.com/2014/06/23/applying-deep-learning-rbm-mnist-using-python/)

[Rbm info](http://deeplearning.net/tutorial/rbm.html)
[Logistic regression info](http://courses.washington.edu/css490/2012.Winter/lecture_slides/05b_logistic_regression.pdf)
[Scikit-learn rbm API](http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.BernoulliRBM.html)
[Sckikit-learn rbm](http://scikit-learn.org/stable/modules/neural_networks.html#rbm)
[Sckit-learn logistic regression API](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
[Sckit-learn logistic regression](http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
[Regularization](https://cs.brown.edu/courses/archive/2006-2007/cs195-5/lectures/lecture13.pdf)


# Trial 1

## input

100 samples from each person: sana, connor, mlhopp, naomimusgrave, jungleprince, 1246241522, hunter5474

## parameters

L_COMPONENTS = 100
N_LEARNING_RATE = 0.0046415888336127772
N_ITER = 21
N_COMPONENTS = 300
With score of XXX

## features

danceability energy liveness loudness speechiness acousticness

Best feature set: ['danceability', 'energy', 'liveness', 'loudness', 'speechiness', 'acousticness']
Identified 6 out of 7 labels
    naomimusgrave     8     of 17    (0.470588)
    hunter5474     2     of 21    (0.095238)
    jungleprince   13    of 22    (0.590909)
    sana     10    of 18    (0.555556)
    mlhopp   3     of 18    (0.166667)
    connor   4     of 22    (0.181818)

## Results

Avg Accuracy = 27.8

Accuracy:   33 correct gives 0.235714285714 compared to guessing 0.142857142857
Best feature set: ['danceability', 'energy', 'liveness', 'loudness', 'speechiness', 'acousticness']
Identified 5 out of 7 labels
    jungleprince   11    of 18    (0.611111)
    mlhopp   5     of 17    (0.294118)
    sana     6     of 19    (0.315789)
    hunter5474     8     of 17    (0.470588)
    connor   3     of 21    (0.142857)

Accuracy:   32 correct gives 0.228571428571 compared to guessing 0.142857142857
Best feature set: ['danceability', 'energy', 'liveness', 'loudness', 'speechiness', 'acousticness']
Identified 5 out of 7 labels
    jungleprince   14    of 20    (0.700000)
    mlhopp   3     of 15    (0.200000)
    sana     4     of 20    (0.200000)
    hunter5474     7     of 20    (0.350000)
    connor   4     of 17    (0.235294)

Accuracy:   29 correct gives 0.207142857143 compared to guessing 0.142857142857
Best feature set: ['danceability', 'energy', 'liveness', 'loudness', 'speechiness', 'acousticness']
Identified 4 out of 7 labels
    jungleprince   13    of 17    (0.764706)
    connor   5     of 17    (0.294118)
    naomimusgrave     1     of 17    (0.058824)
    sana     10    of 17    (0.588235)

Accuracy:   28 correct gives 0.2 compared to guessing 0.142857142857
Best feature set: ['danceability', 'energy', 'liveness', 'loudness', 'speechiness', 'acousticness']
Identified 5 out of 7 labels
    jungleprince   11    of 20    (0.550000)
    mlhopp   1     of 21    (0.047619)
    connor   3     of 18    (0.166667)
    hunter5474     3     of 19    (0.157895)
    sana     10    of 16    (0.625000)

Accuracy:   17 correct gives 0.121428571429 compared to guessing 0.142857142857
Best feature set: ['danceability', 'energy', 'liveness', 'loudness', 'speechiness', 'acousticness']
Identified 5 out of 7 labels
    jungleprince   3     of 25    (0.120000)
    mlhopp   5     of 15    (0.333333)
    naomimusgrave     5     of 12    (0.416667)
    hunter5474     1     of 19    (0.052632)
    connor   3     of 18    (0.166667)

Accuracy:   28 correct gives 0.2 compared to guessing 0.142857142857
Best feature set: ['danceability', 'energy', 'liveness', 'loudness', 'speechiness', 'acousticness']
Identified 5 out of 7 labels
    jungleprince   8     of 23    (0.347826)
    mlhopp   6     of 16    (0.375000)
    connor   3     of 21    (0.142857)
    hunter5474     7     of 17    (0.411765)
    sana     4     of 19    (0.210526)

Accuracy:   25 correct gives 0.178571428571 compared to guessing 0.142857142857
Best feature set: ['danceability', 'energy', 'liveness', 'loudness', 'speechiness', 'acousticness']
Identified 3 out of 7 labels
    jungleprince   14    of 18    (0.777778)
    mlhopp   4     of 15    (0.266667)
    sana     7     of 16    (0.437500)

Accuracy:   31 correct gives 0.221428571429 compared to guessing 0.142857142857
Best feature set: ['danceability', 'energy', 'liveness', 'loudness', 'speechiness', 'acousticness']
Identified 4 out of 7 labels
    jungleprince   15    of 17    (0.882353)
    mlhopp   3     of 15    (0.200000)
    hunter5474     3     of 20    (0.150000)
    sana     10    of 16    (0.625000)

Accuracy:   27 correct gives 0.192857142857 compared to guessing 0.142857142857
Best feature set: ['danceability', 'energy', 'liveness', 'loudness', 'speechiness', 'acousticness']
Identified 5 out of 7 labels
    jungleprince   8     of 23    (0.347826)
    sana     1     of 26    (0.038462)
    1246241522     4     of 14    (0.285714)
    hunter5474     10    of 16    (0.625000)
    connor   4     of 20    (0.200000)

Accuracy:   28 correct gives 0.2 compared to guessing 0.142857142857
Best feature set: ['danceability', 'energy', 'liveness', 'loudness', 'speechiness', 'acousticness']
Identified 5 out of 7 labels
    jungleprince   13    of 21    (0.619048)
    mlhopp   5     of 15    (0.333333)
    1246241522     1     of 15    (0.066667)
    hunter5474     7     of 19    (0.368421)
    connor   2     of 22    (0.090909)


# Trial 2

## input

100 samples from each person: sana, connor, mlhopp, naomimusgrave, jungleprince, 1246241522, hunter5474

## parameters

L_COMPONENTS = 100
N_LEARNING_RATE = 0.001
N_ITER = 1
N_COMPONENTS = 300
With score of 0.26

## features

liveness valence instrumentalness acousticness

## Results

Avg Accuracy = 25.9

Accuracy: 26 correct gives 0.185714285714 compared to guessing 0.142857142857
Best feature set: ['liveness', 'valence', 'instrumentalness', 'acousticness']
Identified 4 out of 7 labels
   jungleprince    8   of 19   (0.421053)
   1246241522    3   of 19   (0.157895)
   hunter5474    11    of 14   (0.785714)
   connor    4   of 13   (0.307692)

Accuracy: 23 correct gives 0.164285714286 compared to guessing 0.142857142857
Best feature set: ['liveness', 'valence', 'instrumentalness', 'acousticness']
Identified 5 out of 7 labels
   jungleprince    7   of 22   (0.318182)
   mlhopp    2   of 21   (0.095238)
   connor    1   of 27   (0.037037)
   1246241522    10    of 11   (0.909091)
   naomimusgrave   3   of 21   (0.142857)

Accuracy: 29 correct gives 0.207142857143 compared to guessing 0.142857142857
Best feature set: ['liveness', 'valence', 'instrumentalness', 'acousticness']
Identified 5 out of 7 labels
   jungleprince    11    of 23   (0.478261)
   sana    2   of 18   (0.111111)
   1246241522    2   of 23   (0.086957)
   hunter5474    7   of 14   (0.500000)
   connor    7   of 18   (0.388889)

Accuracy: 29 correct gives 0.207142857143 compared to guessing 0.142857142857
Best feature set: ['liveness', 'valence', 'instrumentalness', 'acousticness']
Identified 5 out of 7 labels
   jungleprince    10    of 23   (0.434783)
   mlhopp    7   of 16   (0.437500)
   1246241522    6   of 19   (0.315789)
   hunter5474    5   of 17   (0.294118)
   naomimusgrave   1   of 19   (0.052632)

Accuracy: 16 correct gives 0.114285714286 compared to guessing 0.142857142857
Best feature set: ['liveness', 'valence', 'instrumentalness', 'acousticness']
Identified 5 out of 7 labels
   jungleprince    7   of 15   (0.466667)
   sana    2   of 21   (0.095238)
   naomimusgrave   4   of 12   (0.333333)
   1246241522    2   of 24   (0.083333)
   connor    1   of 29   (0.034483)

Accuracy: 29 correct gives 0.207142857143 compared to guessing 0.142857142857
Best feature set: ['liveness', 'valence', 'instrumentalness', 'acousticness']
Identified 5 out of 7 labels
   jungleprince    7   of 20   (0.350000)
   mlhopp    2   of 14   (0.142857)
   connor    6   of 19   (0.315789)
   naomimusgrave   10    of 17   (0.588235)
   sana    4   of 17   (0.235294)

Accuracy: 27 correct gives 0.192857142857 compared to guessing 0.142857142857
Best feature set: ['liveness', 'valence', 'instrumentalness', 'acousticness']
Identified 5 out of 7 labels
   jungleprince    9   of 24   (0.375000)
   mlhopp    1   of 16   (0.062500)
   1246241522    7   of 16   (0.437500)
   hunter5474    4   of 17   (0.235294)
   connor    6   of 20   (0.300000)

Accuracy: 31 correct gives 0.221428571429 compared to guessing 0.142857142857
Best feature set: ['liveness', 'valence', 'instrumentalness', 'acousticness']
Identified 6 out of 7 labels
   naomimusgrave   6   of 18   (0.333333)
   jungleprince    13    of 23   (0.565217)
   connor    2   of 22   (0.090909)
   mlhopp    2   of 17   (0.117647)
   1246241522    6   of 20   (0.300000)
   sana    2   of 17   (0.117647)

Accuracy: 26 correct gives 0.185714285714 compared to guessing 0.142857142857
Best feature set: ['liveness', 'valence', 'instrumentalness', 'acousticness']
Identified 6 out of 7 labels
   naomimusgrave   2   of 21   (0.095238)
   hunter5474    3   of 17   (0.176471)
   jungleprince    10    of 20   (0.500000)
   mlhopp    2   of 16   (0.125000)
   1246241522    5   of 19   (0.263158)
   connor    4   of 22   (0.181818)

Accuracy: 23 correct gives 0.164285714286 compared to guessing 0.142857142857
Best feature set: ['liveness', 'valence', 'instrumentalness', 'acousticness']
Identified 4 out of 7 labels
   jungleprince    11    of 20   (0.550000)
   1246241522    1   of 25   (0.040000)
   hunter5474    8   of 16   (0.500000)
   connor    3   of 17   (0.176471)
