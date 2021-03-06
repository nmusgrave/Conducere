# Random Forest 

A random forest classifier is an ensemble learning method that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. During training, individual decision trees within the forest are assigned weights according to their individual accuracies. This allows better decision trees to be weighted more highly, and also prevents really poor (overfitting) decision trees from holding too much power in the prediction process. Predictions are determined by a random forest through introducing a set of input data to the trained forest, and the predicted class is a weighted vote of all the trees in the forest. Each weight corresponds to the probability estimates for each tree from training. For example, if a forest had the following 3 trees:

    [T1 (avg. accuracy / weight = 0.4), 
     T2 (avg. accuracy / weight = 0.9),
     T3 (avg. accuracy / weight = 0.1)]

 After introducing a data point for binary class prediction [A=(1) or B=(-1)], the results are as follows:

    [T1= A (0.4), T2= B (0.9), T3= A (0.1)] 
    = (1)(0.4) + (-1)(0.9) + (1)(0.1) 
    = -0.4 

 Since the result was less than 0, the forest would classify this data as B.

 We used random forests for multi-class classification, but the underlying principle of weighted classification votes is the same. Typically, the more trees in your forest, the better. We tried different sizes of the forest, along with many different parameters for fine-tuning both the classifier and the features we used in predictions. Below are some  average accuracies for various sizes of the forest, predicting over the full set of music features, with default settings. (For reference, this data had 11 labels, meaning a random guess would yield an average mean of 0.0909, or 9.09%):

    Forest Size = 	1 
    Accuracy:  		0.158027812895

    Forest Size = 	10 
    Accuracy:  		0.226295828066

    Forest Size = 	50 
    Accuracy:  		0.236637168142

    Forest Size = 	250 
    Accuracy:  		0.242730720607

    Forest Size = 	1000 
    Accuracy:  		0.256637168142

The results above were produced with the random forest classifier setting the weights for each decision tree. The weights of the attributes for the 1000-tree forest are shown below:

        danceability:		0.111842
              energy:		0.101156
            liveness:		0.087901
            loudness:		0.120813
         speechiness:		0.108170
               tempo:		0.101866
             valence:		0.112708
    instrumentalness:		0.138057
        acousticness:		0.117487

As evidenced, for the most part, the features were ranked similarly, but a few were seen to be more important by the classifier. We also worked on feature selection, in order to only use the most relevant features. This yielded slightly better results of ~30%-35% mean accuracy. 

However, the most interesting results came in an experiment of splitting the input data by user and using the random forest as a binary classifier. We ran our classifier on a pruned train/test data, based on user pairs, for every user matching in our full data set, so the random forest was only evaluating tracks/features of at most 2 users. We evaluated each run separately. For reference, with this new approach, a random guess would yield a mean accuracy of 50%.

    Min Accuracy Reported for a Matching: 0.533898
    Max Accuracy Reported for a Matching: 0.936508
    Avg Accuracy Reported for a Matching: 0.710830

We can see that at it's best it does extremely well in predicting the correct user for a set of track features. The average case is also a much bigger improvement over randomly guessing. Even the lower accuracy results have valuable conclusions to be made. The closer to 50% an accuracy produced by the classifier for a pair of users is, the closer those users' music tastes are. On the flip side, as we get farther from 50%, it indicates a larger distinction between the music preferences of the two users. 

With these preliminary results, we wanted to see which user's music preference was the most distinct (i.e. which user, in all of his/her matchings, yielded the highest average accuracy) To evaluate this, we compiled the accuracies for each matching, grouped by user and took the average. Below, the results show that user 1257662670 had the most distinct music taste.

	User ID 		Avg. Accuracy

	svetlanag		0.7479269
	naomimusgrave	0.6904512
	hunter5474		0.6553261
	jungleprince	0.724423
	corne			0.6956241
	connor			0.6756102
	mlhopp			0.6613407
	scott			0.6730724
	1246241522		0.7353428
	sana 			0.6930337
	1257662670 		0.8669809


Random forests are powerful for classification, particularly with their resistance to overfitting. While it is still possible to overfit a random forest, the inherent weighting property and increasing the size of the forest make it difficult. In our experimentation, the RFC consistently performed better than the baseline of random guessing. Going forward, it would be interesting to expand the user-matching technique, forming 3-way comparisons and seeing what results are produced from this. It would also be an interesting experiment to try to pinpoint what makes a user's music taste 'distinct' and launch a more comprehensive investigation in which feature combinations correspond to this attribute.

