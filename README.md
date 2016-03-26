# Conducere: An Application of Music Learning
By [Connor Moore](https://www.linkedin.com/in/connor-moore-190a30a3), [Naomi Musgrave](), [Megan Hopp](https://www.linkedin.com/in/hoppm), and [Svetlana Grabar](https://www.linkedin.com/in/svetlana-grabar-71aa6a83). Created for Prof. Bruce Hemingway's Winter 2016 Sound Capstone course at the University of Washington, Computer Science Department.

## Contents
+ [Summary](#summary)
+ [Design](#design)
  + [Overview](#overview)
  + [Machine Learning Models](#machine-learning-models)
    + [Random Forest Classifier](#random-forest-classifier)
    + [Clustering](#clustering)
    + [Artificial Neural Network with Logistic Regression](#artificial-neural-network-with-logistic-regression)
+ [Results](#results)
+ [How To Use](#how-to-use)
  + [Example](#example)
  + [Learning Module](#learning-module)
  + [Data Collection](#data-collection)
+ [Data Sources](#data-sources)
+ [Related Research](#related-research)

## Summary

Conducere is a project that delves into the analysis of musical features. Many projects attempt to reccommend songs to users based on the listening habits of their friends. Other projects attempt to categorize individual tracks by determining the song's mood or musical genre. While musical analysis of attributes, such as chord progression, timbre, pitch, and tempo, has been substantially researched, this realm is a relatively untouched field when it comes to predicting music preference of individuals based on these attributes. 

This field of research spawns a question: Can we determine individual music preferences from these musical attributes in prior listening trends, excluding the social aspect? Conducere is an effort to investigate this question through analyzing music and applying various machine learning techniques to build different models of a user's listening habits. Along the way, we hope to determine what learning models yield the best results.

## Design

### Overview
There are two major steps in the processing of our data. The first is the [EchoNest API](https://github.com/echonest/pyechonest), to collect data describing an individual’s music libraries. Provided a user's Spotify ID, we extract songs from one of their public playlists, and query the EchoNest API for the features of each song. The Echonest API extracts the following features for each track

* danceability      [0 - 1]
* energy            [0 - 1]
* liveness          [0 - 1]
* loudness          [-100 - 100] decibels
* speechiness       [0 - 1]
* tempo             [0 - 500] BPM
* valence           [0 - 1]
* instrumentalness  [0 - 1]
* acousticness      [0 - 1]

The [Scikit-learn](http://scikit-learn.org/) machine learning libraries for Python provided algorithms for modeling our dataset. By using a reliable machine learning library, we easily repurposed aspects of our architecture for experimenting with various machine learning models.

We defined a data processing module to interacts with the Echo Nest API and the machine learning libraries. This component collects and sanitizes the data, splits it into testing and training sets, furnishes it to a machine learning model, and analyzes the model's accuracy. If the model performs well, it will correctly identify the user associated with each song in the testing data set.

### Machine Learning Models

#### Random Forest Classifier

A [random forest classifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) is assembled from a collection of decision trees. A single tree has low bias (error due to incorrect assumptions) but high variance (very sensitive to noise in the sample set). By using a forest of decision trees, rather than a single tree, the model is less sensitive to noise in the sample set. Each decision tree is constructed using a subset of the training data set. Samples from the training set are chosen using bootstrap aggregation, or bagging. With this technique, bags of samples are chosen uniformly and with replacement. This helps reduce the effects of high variance and overfitting that decision trees are susceptible to. 

The decision trees learn rules to classify the data set. Each interior node describes a threshold on a feature of the data set. Each leaf node is a possible label the item may be assigned. When constructing the tree, the number of nodes is increased as the data set is recursively partitioned by repeated tests of feature values. The splitting halts when a subset has all the same value, or splitting yields no knew labels. When a data point is given to a decision tree, it flows from the root to a leaf node, following branches based upon the values of its attributes. The random forest classifies a data point by collecting a vote by the trees in the forest, weighted by their probability estimates.

When using the random forest for multiclass classification, we varied the forest size from 1 to 1000 trees. When using the full set of features, the accuracy ranged from 15.8% to 25.7%. When using selecting the best-performing features, the accuracies fell in the 30-35% range. 

We also attempted treating the random forest as a binary classifier. Splitting the data by user, we constructed every possible user pairing and ran the model. By random guessing, the model would have 50% accuracy. The best accuracy was 93.7%, with some others falling closer to 50%. The average was 71.1% accuracy. User pairs with classification accuracies closer to 50% reflect that those two users have very similar music tastes. By grouping the pairings by user, and averaging the model's accuracies for all the pairings for that user, we calculated the distinctiveness of each user's listening habits. These measures ranged from 0.65 to 0.867.

Overall, the random forest classifier performed much better than random guessing. By using the binary classifier, we found which users had the most similar tastes. From the perspective of music reccomendation, it is useful to understand what users have similar tastes, in order to reccomend appropriate songs to that group, rather than on a user-by-user basis.

#### Clustering


#### Artificial Neural Network with Logistic Regression


## Results


## How To Use

### Example

To get started right away, you can run a pre-existing model, an artificial neural network. Below runs the model on a powerset of all the Echonest features, on the tracks described in the provided data file:

```
python run.py ann 1 true data/data_3_8_16.txt
```

### Learning Module

To make a new learning module, you can put a new python file in the `models` folder.
The only requirement on a module is that it has an execute method, which should have
the following signature:

```
def execute(args)
```

Other than that, run.py does not require anything else of your module. To make sure
that it is added to the `models` import, add `import <my_model>` into `models/__init.py__`.

To run it from the command line, make sure you're in the root of the project, and type:
```
python run.py <my_model> [arguments..]
```
And that's it! You should now be able to run your model with any number of arguments.

### Data Collection

To collect a Spotify user's playlist, and analyze the tracks with Echonest, set the following environment variables: 

* SPOTIPY_CLIENT_ID
* SPOTIPY_CLIENT_SECRET
* SPOTIPY_REDIRECT_URI
* ECHO_NEST_API_KEY

And execute
```
python data/collect.py <username> <playlist name> ...
```
To then parse the collected echonest data, execute
```
python parse.py
```
which prints comma-separated data to standard out (redirect to filepath, if desired)

## Data Sources

| user name       | user id       | playlist name      | playlist id            |
|-----------------|---------------|--------------------|------------------------|
| Naomi Musgrave  | naomimusgrave | Conducere          | 5PncMLe2hgXNShCMjTczcJ |
| Megan Hopp      | mlhopp        | conducere          | 7g45qlGsYfZSxIAioYBD8N |
| Connor Moore    | 1260365679    | Capstone           | 1FRHfvYqQBnZfWwZ0aXHFB |
| Svetlana Grabar | svetlanag     | Calm               | 30ICfBesEb5uvrhfyfI6DU |
| Svetlana Grabar | svetlanag     | Upbeat             | 2QdiJzfIFh2UZnItHeB3DS |
| Svetlana Grabar | svetlanag     | Happy              | 5uhT2QnPCcwT4oq0KILy76 |
| Vincent Chan    | 1257662670    | quiet yearning     | 1nmlQhiuMGBxOGtH8fz3D2 |
| Mallika Potter  | 1246241522    | vaguely indie      | 3xuiTGv241bH8BER0U9ANo |
| Mallika Potter  | 1246241522    | feminist pop       | 1VnkZa21CrQBG9EGA4Lpxl |
| Becca Saunders  | 1257552049    |                    |                        |
| Punya Jain      | 1215184557    |                    |                        |
| Sana Nagar      | sana          | Emerald City Vibes | 5S7yyHr7zCU3liLN9ina7x |
| Scott Shively   | scottshively  | HC                 | 475nZAiQKPmGsrpInmWcUv |
| Corne Strootman | corne         | CrD'A              | 4x7OoNFxGu4RkCwvQkgMys |

## Related Research

[A Method for Comparative Analysis of Folk Music Based on Musical Feature Extraction and Neural Networks](http://users.jyu.fi/~ptee/publications/2_2001.pdf): A study presenting a simple data-mining tool for databases that uses a symbolic representation of melodic information. Their methods include musical feature extraction and then applying various tools to classify a large music data set. Uses a self-organizing map.

[Features for Audio and Music Classification](https://jscholarship.library.jhu.edu/bitstream/handle/1774.2/22/paper.pdf?sequence=1): Evaluates four audio feature sets in their ability to classify five general audio classes and seven popular music genres. Uses Gaussian-based quadratic discriminant analysis

[Music Genre Classification Using Machine Learning Techniques](https://www.cs.swarthmore.edu/~meeden/cs81/s12/papers/AdrienDannySamPaper.pdf): Uses a neural network with a growing neural gas to improve accuracy. The neural network is trained to label the data. The growing neural gas is used to build a map of song features. Data sets are songs from LastFM. Uses EchoNest to label songs with metadata.

[Automatic music emotion classification using artificial neural network based on vocal and instrumental sound timbres](https://www.researchgate.net/publication/276432106_Automatic_music_emotion_classification_using_artificial_neural_network_based_on_vocal_and_instrumental_sound_timbres): By analyzing the timbre of vocal and instrumental components, can identify emotions in music. Two main modes of emotion identification are pattern matching (use memory to find strongest set of parameters that match) or signal modeling (translate audio into features). The research uses a binary feedback neural network. Data sets are songs with timbre features (spectral rolloff, zero-cross, spectral centroid) extracted.

[Classification of Musical Genre: A ML Approach](http://art.uniroma2.it/research/musicIR/BasSeraStel_ISMIR04.pdf): This work examines various musical features and categorization approaches. Multiple binary classifiers are more accurate than a single multi-class classifier. Simple musical features can provide very reasonable classification results.



