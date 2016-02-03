# CSE 481I: Sound Capstone
# Conducere (TM)
#
# This file holds utility data/functions for 
# data collection and analysis


# List of relevant attributes
# TODO: Add new attributes for analysis to this set
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

