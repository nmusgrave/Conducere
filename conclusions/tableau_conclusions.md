# Overview

Using Tableau to look at the data proved to be a good way to identify patterns in the data to supplement the models. From looking at the graphs of feature average, median with quartiles, standard deviation, and range, it was possible to see how features varied in different playlists. Danceability, instrumentalness, loudness, and speechiness seemed to be the most interesting.

## Danceability

Danceability’s average was fairly consistent between different playlists, ranging from 0.46 to 0.59, but the range was quite varied. As an example, one playlist had a min of .15 and a max of .98, while another had a min of .22 and a max of .67. It is very possible that liking songs with high danceability could be an indicator for music taste based on this, the playlists are fairly large. In fact, this is exactly what some of the models found- danceability was a good feature to determine music taste.

## Instrumentalness

Instrumentalness was an odd feature as a lot of songs had no intrumentalness. If the data was not corrupted in any way, then this is a great indicator as playlists seemed to either have a lot of instrumental music, or almost none. In the playlists we looked at, there was no middle ground. 

## Loudness

Loudness is another interesting feature as the averages were the same for most playlists, but the mins and maxes were varied. For one playlist, loudness ranged from -0.9 to -16, while for another, loudness ranged from -2.5 to 28.6. 

## Speechiness

Speechiness had a very varied average and range. The average speechiness ranged from 0.045 to 0.082. Some playlists had a big range of speechiness, while some had a smaller one. One of the playlists had speechiness range from 0 to .8 while another ranged from 0.02 to 0.13. 

# Reflection

There were a few features where it wasn’t conclusive if they are important or not. Energy and Acousticness stand out as playlists were somewhat varied in their averages, but not as much as the other features. The rest of the features showed the playlists having very similar averages and ranges, suggesting that it was mostly random. 

Tableau was useful as it makes it easy to visualize and understand the data more. It is only a supplement to models, however, as models can actually calculate how easy it is to learn on the chosen features. The features that stood out to the different models, however, were also found using Tableau, which is a good indicator that we can somewhat separate or guess music tastes.

# Preliminary Patterns Observed in Playlists
1.  Acousticness
  1.  Average acousticness ranged from 0.16 to 0.65, which is quite a big range. One playlist in particular stood out but the rest of the data was also fairly varied on this. The range, however, was pretty similar. Every playlist had at least one song with low acousticness and one with very high acousticness.
2.  Danceability
  1.  Average ranged from 0.46 to 0.59, but the range (max – min) was actually quite varied
  2.  Connor’s playlist had a min of .15 and a max of .98, another playlist had a min of .22 and a max of .67. 
3.  Energy
  1.  One playlist had very low average energy but besides that, average energy and the range of energy was very similar. Most of the playlists had at least one song with energy 0.95-1.0
  2.  this is probably not a great feature to look at.
4.  Instrumentalness
  1.  One of the more interesting features to look at as the average is very varied.
  2.  Every playlist has a song with almost no instrumentalness. Might be worthwhile to check out if this is some sort of bug in collecting data, or if every playlist had some songs that just don’t have any instruments. 
  3.  However 
5.  Liveness
  1.  Probably not interesting, has almost no variation
6.  Loudness
  1.  Loudness is interesting as 6/7 playlists have a very similar average but the min/maxes are different
  2.  For 6/7 playlists, the difference between the min and max loudness is between -13 and -15, which is a fairly small range, but  the mins and maxes differ a lot. For example, one playlist ranges from -5.5 to -20 loudness, while another ranges from -2 to -13, while another ranges from -0.9 to -16. It is possible that people then have a strong preference to how loud their songs are
7.  Speechiness
  1.  Min for speechiness was 0 for all playlists. Data could be skewed or messed up
  2.  Very large range for max and large-ish range for average speechiness.
8.  Tempo
  1.  Very boring data, everything is similar. Not much of a range and similar averages
9.  Valence
  1.  Except for one playlist, the averages and ranges were very similar. Probably doesn’t make a big difference
10. Conclusion/more analysis
  1.  Danceability, Instrumentalness, Loudness, Speechiness were the most interesting that seemed to change between playlists
  2.  Not sure about Acousticness- it seems like one playlist is very different than the rest. That playlist, number 125766.. etc is the outlier in many of the features. Should try to run data without it, or add more data so it doesn’t skew results.
  3.  Liveness, Energy, Tempo, Valence probably don’t make a difference at all
  4.  Might want to double check speechiness and instrumentalness as some data is at 0. The median instrumentality of 4 of the playlists is just about 0. Only one of them is significantly high on that. Looking at the median and the 25 percentile data, speechiness isn’t too bad, but instrumentality has some very extreme results.
