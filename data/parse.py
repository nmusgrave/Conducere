# CSE 481I: Sound Capstone
# Conducere (TM)

# This file provides parsing functions for echonest 
# music analysis data files. 


# To modify which attributes are evaluated during 
# parsing, go to data/util.py and update ATTRIBUTES

from util import ATTRIBUTES
import re
import os

def parse_track(line, user_id):
  """
  Accepts a line of input data and returns a list of parsed attribute 
  data for a single track with the given source user_id. 
  """
  track_data = []
  # Remove curly braces and split on commas
  tokens = re.split("[\{\}]|, ", line)

  for attr in ATTRIBUTES:
    for tok in tokens:
      tok = tok[2:] # strip (u') from attribute names 
      if tok.startswith(attr):
        attr_data = re.split("\': ", tok)
        val = "0.0"
        if attr_data[1] != "None":
          val = attr_data[1]
        track_data.append(val)    
  return track_data

def parse_echonest_data_file(filepath, user_id):
  """
  Parses a single echonest data file and returns a matrix of parsed
  attribute data, with each row corresponding to a track present in the file.
  """
  f = open(filepath, 'r')

  # printing each track's data to standard out
  for track in f:
    data = user_id + ","
    data += ",".join(parse_track(track, user_id))
    print data
  f.close()


def get_parsed_data():
  """
  Returns a matrix containing the parsed data from the echonest 
  music analysis data. The first row in the matrix is a header 
  describing to which attribute a column's data corresponds.

  Example Header:
  ['user_id', 'danceability', 'tempo', ...]

  Subsequent rows hold data for the specified user and attributes
  for individual tracks.
  """

  echonest_data_files = [f for f in os.listdir('.') if re.match("^echonest_[\w]+.txt$", f)]

  # Setting up header with user id and attributes
  header = ['user_id']
  header.extend(ATTRIBUTES)

  # printing header to standard out
  print ",".join(header) 

  # Processing each file to obtain parsed data
  for data_file in echonest_data_files:
    user_id = data_file[9:-4] # strip file prefix/suffix to get username/id
    parse_echonest_data_file(data_file, user_id)

get_parsed_data()
