import sys
from spotify import collect_playlist_data
from echonest import get_playlist_track_analysis
from util import ATTRIBUTES

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
    num_tracks = len(tracks)

    print "For user: %s" % (username)
    print "Found %s tracks in playlists %s" % (len(tracks), playlist_names)
    analysis = get_playlist_track_analysis(tracks)

    attribute_sums = {}
    attribute_counts = {}

    for attr in ATTRIBUTES:
      attribute_sums[attr] = 0.0
      attribute_counts[attr] = 0

    for a in analysis:
      # Collect data on each attribute
      for attr in ATTRIBUTES:
        if hasattr(a, attr) and not (getattr(a, attr) == None):
          attribute_sums[attr] += getattr(a, attr)
          attribute_counts[attr] += 1
      
    # Print out results 
    for attr in ATTRIBUTES:
      avg_attr = attribute_sums[attr] / float(attribute_counts[attr])
      print "Average %s: %.2f" % (attr, avg_attr)

    print "Analyzed tracks with Echonest"
    
    f = open('data/echonest_' + username + '.txt', 'w')
    for a in analysis:
      print >> f, vars(a)
    f.close()
    print "Saved output to echonest.txt"
    print "=================================================================="
  else:
    print "failed"
