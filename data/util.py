# CSE 481I: Sound Capstone
# Conducere (TM)
#
# This file holds utility data/functions for 
# data collection and analysis


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
#
ATTRIBUTES = { \
    'danceability', \
    'energy', \
    'liveness', \
    'loudness', \
    'speechiness', \
    'tempo'}

# Checks if the given analyzed track contains all 
# featured attributes
def contains_all_attributes(track):
  for attr in ATTRIBUTES:
    if not hasattr(track, attr):
      return False
  return True

