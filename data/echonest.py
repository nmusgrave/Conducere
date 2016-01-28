# CSE 481I - Sound Capstone
# Conducere (TM)

# This file contains logic for data collection through the Echonest
# API. 

import time
import os
from pyechonest import config, artist, song, track

# ==========================================
# TODO: Make sure the following variables is set
#       appropriately before trying to run.
#
#    ECHO_NEST_API_KEY='YOUR API KEY'
#
# ==========================================

def get_playlist_track_analysis(playlist_tracks):
  """
  Description:
    Analyzes the audio features of a playlist's tracks through
    Echonest, given a list of Spotify track IDs.

  Return:
    A list of track features for songs in the specified playlist
  """
  set_api_key()
  analysis = []
  for i in range(len(playlist_tracks)):
    analysis.append(track.track_from_id(playlist_tracks[i]))
    time.sleep(3) # limited to 20 access/s
    if i % 20 == 0 and i > 0:
      print "Completed %d tracks" % i
  return analysis

def set_api_key():
  """
  Description:
    Authenticate access to the Echonest API, using the
    environment variable ECHO_NEST_API_KEY
  """
  config.ECHO_NEST_API_KEY=os.environ['ECHO_NEST_API_KEY']


