import numpy as np
import os as os
import pathlib as pl
import tensorflow as tf
import keras as ks
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from PIL import ImageFile


global USER, NUM_GPUS
USER = 'killianf'
NUM_GPUS = 3

def main():
    LOCATION = '/pool001/' + USER + '/Connoisseur/Artworks'
    PATH = pl.Path(LOCATION)
    IMAGES = PATH
    CLASSES = [item.name for item in PATH.glob('*')]
    BATCH_SIZE = 256
    EPOCHS = 30
    # STEPS_PER_EPOCH = np.ceil(len(CLASSES) / BATCH_SIZE)
    IMG_WIDTH = 200
    IMG_HEIGHT = 200
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
                                                         classes=CLASSES,
                                                         interpolation='lanczos')

    input_tensor = ks.layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    base_model = InceptionResNetV2(input_tensor=input_tensor, weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = ks.layers.GlobalAveragePooling2D()(x)
    x = ks.layers.BatchNormalization()(x)
    # let's add a fully-connected layer
    x = ks.layers.Dense(4096, activation='relu')(x)
    x = ks.layers.Dropout(rate=0.1)(x)
    x = ks.layers.Dense(4096, activation='relu')(x)
    # and a logistic layer
    predictions = ks.layers.Dense(len(CLASSES), activation='softmax')(x)

    with tf.device('/cpu:0'):
        # this is the model we will train
        model = ks.models.Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all InceptionResNetV2 layers
    for layer in base_model.layers:
        layer.trainable = False

    model = ks.utils.multi_gpu_model(model, gpus=NUM_GPUS)
    model.compile(loss='categorical_crossentropy',
                  optimizer=ks.optimizers.RMSprop(lr=2e-03),
                  metrics=['accuracy', 'top_k_categorical_accuracy'])
    model.fit_generator(train_data_gen, epochs=EPOCHS, verbose=1)

    if ks.backend.backend() == 'tensorflow':
        ks.backend.clear_session()

    model_json = model.to_json()
    with open('/pool001/' + USER + '/Connoisseur/' + "model_pretrained.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights('/pool001/' + USER + '/Connoisseur/' + "model_pretrained.h5")

    print('Done.')

if __name__ == '__main__':
    main()