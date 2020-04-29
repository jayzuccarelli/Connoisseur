import pandas as pd
import pathlib as pl
import urllib as ul


user = 'ezucca'


def get_artists():
    pass


def get_artworks():
    pass


def get_images():
    artworks = load_artworks()
    for i, r in artworks.iterrows():
        try:
            ul.request.urlretrieve(r['image'], '/pool001/' + user + '/Connoisseur/Artworks/' +
                                   str(r['artist']) + '/' + str(r['name']) + r['image'][-4:])
        except:
            continue


def make_directories(artists):
    for artist in artists:
        pl.Path('/pool001/' + user + '/Connoisseur/Artworks/'+str(artist)).mkdir(parents=True, exist_ok=True)


def load_artists():
    return pd.read_csv('/pool001/' + user + '/Connoisseur/Data/artists.csv')


def load_artworks():
    return pd.read_csv('/pool001/' + user + '/Connoisseur/Data/artworks.csv')
