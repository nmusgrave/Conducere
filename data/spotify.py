# CSE 481I - Sound Capstone wi16
# Conducere (TM)

# Data collection through the Spotify API. 

import spotipy
import spotipy.util as util

# ==========================================
# TODO: Make sure the following environment variables are set
#       appropriately before trying to run.
#
#       SPOTIPY_CLIENT_ID
#       SPOTIPY_CLIENT_SECRET
#       SPOTIPY_REDIRECT_URI
# ==========================================

 
def get_playlist_tracks(sp, username, pl_id):
  """
  Description:
    Extracts all of the tracks in a user's spotify playlist, stored at
    playlist.tracks.items
  Return:
    An object representing the playlist, as following spotify's
    playlist API.
  """
  pl = sp.user_playlist(username, pl_id)
  tracks = []
  p = pl['tracks']
  while p['next']:
    p = sp.next(p)
    tracks.extend(p['items'])
  pl['tracks']['items'].extend(tracks)
  return pl


def get_playlist_track_uris(playlist):
  """
  Description:
    Extracts track URI data from the specified playlist

    Playlist objects contain a key 'track' which maps to
    an array of playlist-track objects. Each playlist-track object
    contains a key 'track' that maps to a track object with a 
    unique URI.

  Return:
    A list of track URIs for songs in the specified playlist
  """
  track_uris = []
  for pl_track in playlist['tracks']['items']:
    track_uris.append(pl_track['track']['uri'])
  return track_uris


def get_playlist_id(sp, username, playlist_name):
  """
  Description:
    Retrieves the playlist_id associated with the
    given user's playlist.
  """
  playlists = sp.user_playlists(username)
  for pl in playlists['items']:
    if pl['name'] == playlist_name:
      return pl['id']


def collect_playlist_data(username, user_playlists):
  """
  Description:
    Collects and returns a list of track uri's extracted from the 
    given user's playlist. (Duplicates allowed)

  Arguments:
    username: username of the playlist
    playlist_name: name of playlist from which to extract track uri's

    Example: username = 'mlhopp' & user_playlists = ['coffee music', 'running']
  """
  # Getting token to access playlists
  token = util.prompt_for_user_token(username)
  if token:
    sp = spotipy.Spotify(auth=token)
    track_list = []
    for playlist_name in user_playlists:
      pl_id = get_playlist_id(sp, username, playlist_name)
      if not (pl_id == None):
        pl = sp.user_playlist(username, pl_id)
        # get all tracks from playlist
        pl = get_playlist_tracks(sp, username, pl_id)
        track_list.extend(get_playlist_track_uris(pl))
      else:
        print ("WARNING: No playlist by name \'%s \' for user \'%s\'\n" 
                % (playlist_name, username))
    return track_list
  else:
    print "Can't get token for", username 
    return None



