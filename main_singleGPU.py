import numpy as np
import os as os
import pathlib as pl
import keras as ks
from PIL import ImageFile


global USER, NUM_GPUS
USER = 'killianf'
NUM_GPUS = 2

def main():
    LOCATION = '/pool001/' + USER + '/Connoisseur/Artworks'
    PATH = pl.Path(LOCATION)
    IMAGES = PATH
    CLASSES = [item.name for item in PATH.glob('*')]
    BATCH_SIZE = 128
    EPOCHS = 10
    STEPS_PER_EPOCH = 1250
    IMG_WIDTH = 200
    IMG_HEIGHT = 200
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    image_generator = ks.preprocessing.image.ImageDataGenerator(rescale=1. / 255,
                                                                width_shift_range=0.2,
                                                                height_shift_range=0.2,
                                                                horizontal_flip=True,
                                                                vertical_flip=True,
                                                                rotation_range=90)
    train_data_gen = image_generator.flow_from_directory(directory=IMAGES,
                                                         batch_size=BATCH_SIZE,
                                                         shuffle=True,
                                                         target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                         classes=CLASSES)
    inputs = ks.layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    in_norm = ks.layers.BatchNormalization()(inputs)
    step_1 = ks.layers.Convolution2D(filters=32, kernel_size=(4, 4), activation='relu',
                                     padding='same')(in_norm)
    step_2 = ks.layers.MaxPooling2D(pool_size=(4, 4), padding='same')(step_1)
    step_3 = ks.layers.Convolution2D(filters=16, kernel_size=(8, 8), activation='relu',
                                     padding='same')(step_2)
    step_4 = ks.layers.MaxPooling2D(pool_size=(4, 4), padding='same')(step_3)
    step_5 = ks.layers.Convolution2D(filters=16, kernel_size=(2, 2), activation='relu',
                                     padding='same')(step_4)
    step_6 = ks.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(step_5)
    step_7 = ks.layers.Flatten()(step_6)
    step_8 = ks.layers.Dense(256, activation='relu')(step_7)
    step_9 = ks.layers.Dropout(rate=0.2)(step_8)
    step_10 = ks.layers.Dense(256, activation='relu')(step_9)
    step_11 = ks.layers.Dropout(rate=0.2)(step_10)
    output = ks.layers.Dense(len(CLASSES), activation='softmax')(step_11)
    model = ks.models.Model(inputs=inputs, outputs=output)
    model = ks.utils.multi_gpu_model(model, gpus=NUM_GPUS)
    model = ks.utils.multi_gpu_model(model, gpus=NUM_GPUS)
    model.compile(loss='categorical_crossentropy',
                  optimizer=ks.optimizers.Adadelta(),
                  metrics=['accuracy'])
    model.fit_generator(train_data_gen, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, verbose=1)
    if ks.backend.backend() == 'tensorflow':
        ks.backend.clear_session()
    print('Done.')

if __name__ == '__main__':
    main()
