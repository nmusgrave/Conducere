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

  print "=================================================================="

  tracks = collect_playlist_data(username, playlist_names)
  if tracks:
    print "For user: %s" % (username)
    print "Found %s tracks in playlists %s" % (len(tracks), playlist_names)

    analysis = get_playlist_track_analysis(tracks)
    
    avg_danceability = 0.0
    avg_energy = 0.0
    avg_liveness = 0.0
    avg_loudness = 0.0
    avg_speechiness = 0.0
    for a in analysis:
      avg_danceability += a.danceability
      avg_energy += a.energy
      avg_liveness += a.liveness
      avg_loudness += a.loudness
      avg_speechiness += a.speechiness
    avg_danceability /= float(len(analysis))
    avg_energy /= float(len(analysis))
    avg_liveness /= float(len(analysis))
    avg_loudness /= float(len(analysis))
    avg_speechiness /= float(len(analysis))

    print "Average danceability (0 - 1): %.2f" % avg_danceability
    print "Average energy (0 - 1): %.2f" % avg_energy 
    print "Average liveness (0 - 1): %.2f" % avg_liveness
    print "Average loudness (dB): %.2f" % avg_loudness
    print "Average speechiness (0 - 1): %.2f" % avg_speechiness

    print "Analyzed tracks with Echonest"
    
    f = open('echonest_' + username + '.txt', 'w')
    for a in analysis:
      print >> f, vars(a)
    f.close()
    print "Saved output to echonest.txt"
    print "=================================================================="
  else:
    print "failed"
