import numpy as np
import pandas as pd
import pathlib as pl
import requests as rq
import string as st
import urllib as ul
import bs4 as bs
import urllib.request


global USER
USER = 'ezucca'


def get_artists():
    url_start = 'https://www.wikiart.org/en/Alphabet/'
    url_end = '/text-list'
    url_class = 'masonry-text-view-all'
    headers = {'User-Agent': 'Mozilla/5.0'}

    artists = pd.DataFrame(columns=['artist', 'url', 'life', 'artworks'])

    for letter in st.ascii_lowercase:

        r = rq.get(url_start + letter + url_end, headers=headers)
        soup = bs.BeautifulSoup(r.text, 'html.parser')

        page = soup.find("div", class_=url_class)

        for l in page.find_all('li'):
            try:
                name = l.a.text
            except:
                name = None
            try:
                link = l.a['href']
            except:
                link = None
            try:
                life = l.span.text
            except:
                life = None
            try:
                artworks = l.find_all('span')[1].text
            except:
                artworks = None

            artist = pd.DataFrame([[name, link, life, artworks]],
                                  columns=['artist', 'url', 'life', 'artworks'])
            artists = artists.append(artist)

    artists = artists.reset_index(drop=True)

    for index, row in artists[artists['artworks'].isna()].iterrows():
        artists.loc[index, 'artworks'] = artists.loc[index, 'life']
        artists.loc[index, 'life'] = None

    return artists


def get_artworks():
    url_start = 'https://www.wikiart.org'
    url_end = '/all-works/text-list'
    url_class = 'painting-list-text-row'
    headers = {'User-Agent': 'Mozilla/5.0'}

    artists = get_artists()

    artworks = pd.DataFrame(columns=['artist', 'name', 'url', 'year'])
    for index, row in artists.iterrows():
        print(index, 'out of', artists.shape[0])

        r = rq.get(url_start + row['url'] + url_end, headers=headers)
        soup = bs.BeautifulSoup(r.text, 'html.parser')

        page = soup.find_all("li", class_=url_class)

        for tag in page:
            try:
                artist = row['artist']
            except:
                artist = None
            try:
                link = tag.a['href']
            except:
                link = None
            try:
                name = tag.a.text
            except:
                name = None
            try:
                year = tag.span
            except:
                year = None

            artwork = pd.DataFrame([[artist, name, link, year]],
                                   columns=['artist', 'name', 'url', 'year'])
            artworks = artworks.append(artwork)

    artworks = artworks.reset_index(drop=True)

    url_start = 'https://www.wikiart.org'
    url_end = '/all-works/text-list'

    url_class = 'ms-zoom-cursor'

    for index, row in artworks.iterrows():
        if index % 100 == 0:
            print(index, 'out of', artworks.shape[0])

        if not (row['url'] is None or str(row['url']) == 'nan'):
            try:
                r = rq.get(url_start + str(row['url']), headers=headers)
                soup = bs.BeautifulSoup(r.text, 'html.parser')
                page = soup.find("img", class_=url_class)
                artworks.loc[index, 'image'] = page['src']
            except:
                continue

    artworks = artworks.reset_index(drop=True)
    return artworks


def make_directories(artists):
    for artist in artists:
        pl.Path('/pool001/' + USER + '/Connoisseur/Artworks/'+str(artist)).mkdir(parents=True, exist_ok=True)


def load_artists():
    return pd.read_csv('/pool001/' + USER + '/Connoisseur/Data/artists.csv')


def load_artworks():
    return pd.read_csv('/pool001/' + USER + '/Connoisseur/Data/artworks.csv')


def get_images():
    artworks = load_artworks()
    log = open('/pool001/' + USER + '/Connoisseur/Logs/images.log', 'w')
    for i, r in artworks.iterrows():
        if (i%100==0) and i!=0:
            print('Completed ', i, ' over ', artworks.shape[0], ' images.')
        if r['image'] != np.nan:
            try:
                ul.request.urlretrieve(r['image'], '/pool001/' + USER + '/Connoisseur/Artworks/' +
                                       str(r['artist']) + '/' + str(r['name']) + str(r['image'])[-4:])
            except Exception as e:  # most generic exception you can catch
                log.write("Failed to download {0}: {1}\n".format(str(r['name']), str(e)))
