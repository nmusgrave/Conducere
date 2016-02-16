# Conducere: An Application of Music Learning

## Use: Learning Module

To make a new learning module, you can put a new python file in the `models` folder.
The only requirement on a module is that it has an execute method, which should have
the following signature:

```
def execute(args)
```

Other than that, run.py does not require anything else of your module. To make sure
that it is added to the `models` import, add `import <my_model>` into `__init.py__`.

To run it from the command line, make sure you're in the `src` folder, and type:

```
python run.py <my_model> [arguments..]
```

And that's it! You should now be able to run your model with any number of arguments.

## Use: Data Collection

To collect a Spotify user's playlist, and analyze the tracks with Echonest, set the following
environment variables: 

* SPOTIPY_CLIENT_ID
* SPOTIPY_CLIENT_SECRET
* SPOTIPY_REDIRECT_URI
* ECHO_NEST_API_KEY

And execute

```
python data/collect.py <username> <playlist name> ...
```

To then parse the collected echonest data, execute

```
python parse.py
```

which prints comma-separated data to standard out (redirect to filepath, if desired)

## Data Sources

| user name       | user id       | playlist name  | playlist id            |
|-----------------|---------------|----------------|------------------------|
| Naomi Musgrave  | naomimusgrave | Conducere      | 5PncMLe2hgXNShCMjTczcJ |
| Megan Hopp      | mlhopp        | conducere      | 7g45qlGsYfZSxIAioYBD8N |
| Connor Moore    | 1260365679    | Capstone       | 1FRHfvYqQBnZfWwZ0aXHFB |
| Svetlana Grabar |               |                |                        |
| Vincent Chan    | 1257662670    | quiet yearning | 1nmlQhiuMGBxOGtH8fz3D2 |
| Mallika Potter  | 1246241522    | vaguely indie  | 3xuiTGv241bH8BER0U9ANo |
| Mallika Potter  | 1246241522    | feminist pop   | 1VnkZa21CrQBG9EGA4Lpxl |
| Becca Saunders  | 1257552049    |                |                        |
| Punya Jain      | 1215184557    |                |                        |


