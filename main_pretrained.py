import numpy as np
import os as os
import pathlib as pl
import keras as ks
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.optimizers import SGD
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
    EPOCHS = 250
    STEPS_PER_EPOCH = np.ceil(len(CLASSES) / BATCH_SIZE)
    IMG_WIDTH = 256
    IMG_HEIGHT = 256
    ks.backend.set_image_dim_ordering('tf')
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    image_generator = ks.preprocessing.image.ImageDataGenerator(rescale=1. / 255,
                                                                featurewise_center=True,
                                                                featurewise_std_normalization=True)

    image_generator.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 1, 3))  # ordering: [R, G, B]
    image_generator.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 1, 3))

    train_data_gen = image_generator.flow_from_directory(directory=IMAGES,
                                                         batch_size=BATCH_SIZE,
                                                         shuffle=True,
                                                         target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                         color_mode='rgb',
                                                         class_mode='categorical',
                                                         classes=CLASSES)

    input_tensor = ks.layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    base_model = InceptionResNetV2(input_tensor=input_tensor, weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = ks.layers.Flatten()(x)
    # let's add a fully-connected layer
    x = ks.layers.Dense(4096, activation='relu')(x)
    x = ks.layers.Dropout(rate=0.1)(x)
    x = ks.layers.Dense(4096, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = ks.layers.Dense(len(CLASSES), activation='softmax')(x)

    # this is the model we will train
    model = ks.models.Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    model = ks.utils.multi_gpu_model(model, gpus=NUM_GPUS)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy', 'top_k_categorical_accuracy'])
    model.fit_generator(train_data_gen, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, verbose=1)

    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 249 layers and unfreeze the rest:
    for layer in model.layers[:249]:
        layer.trainable = False
    for layer in model.layers[249:]:
        layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', 'top_k_categorical_accuracy'])

    model.fit_generator(train_data_gen, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, verbose=1)

    if ks.backend.backend() == 'tensorflow':
        ks.backend.clear_session()

    model_json = model.to_json()
    with open("model_pretrained.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model_pretrained.h5")

    print('Done.')

if __name__ == '__main__':
    main()