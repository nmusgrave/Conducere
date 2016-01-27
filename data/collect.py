import sys
from spotify import collect_playlist_data
from echonest import get_playlist_track_analysis

# Through command line, accepts a Spotify username and list of 
# playlist names in that user's profile, collects track URI's
# and then prints them out
if __name__ == '__main__':
  if len(sys.argv) > 2:
    username = sys.argv[1]
    playlist_names = sys.argv[2:]
  else:
    print "Usage: %s <username> <playlist name> ..." % (sys.argv[0])
    sys.exit()

  print
  print "=================================================================="

  tracks = collect_playlist_data(username, playlist_names)
  if tracks:
    print "For user: %s" % (username)
    print "Found %s tracks in playlists %s" % (len(tracks), playlist_names)
    analysis = get_playlist_track_analysis(tracks[:5])
    print "Analyzed tracks with Echonest"
    print "=================================================================="
  else:
    print "failed"