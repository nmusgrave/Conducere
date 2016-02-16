# CSE 481I - Sound Capstone wi16
# Conducere (TM)

# Utility data/functions for data collection and analysis


# List of relevant attributes
# TODO: Add new attributes for analysis to this set
#
#   attribute        value space
#   -------------    ------------
#   danceability     (0.0 - 1.0)
#   energy           (0.0 - 1.0)
#   liveness         (0.0 - 1.0)
#   loudness         (dB)
#   speechiness      (0.0 - 1.0)
#   tempo            (BPM)
#   valence          (0.0 - 1.0)
#   instrumentalness (0.0 - 1.0)
#   acousticness     (0.0 - 1.0)
#
ATTRIBUTES = [ \
    'danceability', \
    'energy', \
    'liveness', \
    'loudness', \
    'speechiness', \
    'tempo', \
    'valence', \
    'instrumentalness', \
    'acousticness']

# Checks if the given analyzed track contains all 
# featured attributes
def contains_all_attributes(track):
  for attr in ATTRIBUTES:
    if not hasattr(track, attr):
      return False
  return True

