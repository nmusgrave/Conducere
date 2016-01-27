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
