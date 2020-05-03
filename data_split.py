import numpy as np
import os
import pathlib as pl
import shutil as su

global USER
USER = os.getcwd().split('/')[2]

def main():
    # get image location
    LOCATION = '/pool001/' + USER + '/Connoisseur/Artworks'
    PATH = pl.Path(LOCATION)

    # define percentage of testing examples
    TEST_SPLIT = 0.1

    # get all classes of artists
    CLASSES = [item.name for item in PATH.glob('*')]

    # create training directory
    if not os.path.exists(LOCATION + '/train'):
        os.makedirs(LOCATION + '/train')

    # create testing directory
    if not os.path.exists(LOCATION + '/test'):
        os.makedirs(LOCATION + '/test')

    # create class directories in test directory
    for cl in CLASSES:
        os.makedirs(LOCATION + '/test/' + cl)

    # randomly select every image with TEST_SPLIT likelihood
    # to be in testing class
    for cl in CLASSES:

        # safety measure
        if cl in ['train', 'test']:
            continue

        # set current class as source location
        source = LOCATION + '/' + str(cl)

        # set test folder class as destination
        destination = LOCATION + '/test/' + str(cl)

        # get all images in destination
        files = os.listdir(source)

        # randomly decide for every image if it should go in testing class
        for f in files:
            if np.random.rand(1) < TEST_SPLIT:
                su.move(source + '/' + f, destination + '/' + f)

    # move training examples to training folder
    for cl in CLASSES:

        # safety measure
        if cl in ['train', 'test']:
            continue

        # same as above, only now in training directory
        # and moving entire folder
        source = LOCATION + '/' + str(cl)
        destination = LOCATION + '/train/' + str(cl)

        # move operation
        su.move(source, destination)

    directories = [item.name for item in PATH.glob('*')]

    print(directories)

if __name__ == '__main__':
    main()
