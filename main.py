import numpy as np
import os as os
import pathlib as pl
import tensorflow as tf
import tensorflow.keras as ks #TODO: Try Keras only


global USER
USER = os.getcwd().split('/')[2]


def main():
    # Locations
    LOCATION = '/pool001/' + USER + '/Connoisseur/Artworks'
    PATH = pl.Path(LOCATION)

    # Data
    IMAGES = PATH
    CLASSES = [item.name for item in PATH.glob('*')]

    # Parameters
    BATCH_SIZE = 32
    EPOCHS = 25
    STEPS_PER_EPOCH = np.ceil(len(CLASSES) / BATCH_SIZE)
    IMG_WIDTH = 256
    IMG_HEIGHT = 256


    # Build Datasets
    ks.backend.common.set_image_dim_ordering('th') #TODO: Isn't th opposite of the order of channels after?
    image_generator = ks.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    train_data_gen = image_generator.flow_from_directory(directory=LOCATION,
                                                         batch_size=BATCH_SIZE,
                                                         shuffle=True,
                                                         target_size=(IMG_HEIGHT, IMG_WIDTH), #TODO: Which order is it?
                                                         classes=CLASSES)
    X, class_label = next(train_data_gen)
    train_X, validation_X, test_X = tf.split(X, [0.7*X.shape[0], 0.2*X.shape[0], 0.1*X.shape[0]], 0)

    # Model
    inputs = ks.layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)) #TODO: Try None, None - Should it be width and height or viceversa?
    step_1 = ks.layers.Convolution2D(filters=8, kernel_size=(2, 2), activation='relu',
                                     padding='same', data_format="channels_last")(inputs)
    step_2 = ks.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(step_1)
    step_3 = ks.layers.Convolution2D(filters=16, kernel_size=(2, 2), activation='relu',
                                     padding='same', data_format="channels_last")(step_2)
    step_4 = ks.layers.MaxPooling2D(pool_size=(2, 2), padding='same', strides=(2, 2))(step_3)
    step_5 = ks.layers.Flatten()(step_4)
    step_6 = ks.layers.Dense(10, activation='relu')(step_5)
    step_7 = ks.layers.Dropout(rate=0.5)(step_6)
    output = ks.layers.Dense(len(CLASSES), activation='softmax')(step_7)

    model = ks.models.Model(inputs=inputs, outputs=output)

    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy']) #TODO: Use Adam too

    model.fit_generator(train_data_gen, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, verbose=1)

    if ks.backend.backend() == 'tensorflow':
        ks.backend.clear_session()


if __name__ == '__main__':
    main()
