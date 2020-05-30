import numpy as np
import os as os
import keras as ks
from keras.models import model_from_json
from PIL import ImageFile
import pathlib as pl



USER = 'killianf'
json_file = open('/pool001/killianf/Connoisseur/LoadedModels/model_ResNet.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("/pool001/killianf/Connoisseur/LoadedModels/model_ResNet_weights.h5")

loaded_model.compile(loss='categorical_crossentropy',
                     optimizer=ks.optimizers.SGD(lr=0),
                     metrics=['accuracy', 'top_k_categorical_accuracy'])
loaded_model.training=False


LOCATION = '/pool001/' + USER + '/Connoisseur/Artworks'
PATH = pl.Path(LOCATION)
IMAGES = PATH
CLASSES = [item.name for item in PATH.glob('*')]
BATCH_SIZE = 256
IMG_WIDTH = 200
IMG_HEIGHT = 200
ImageFile.LOAD_TRUNCATED_IMAGES = True
image_generator = ks.preprocessing.image.ImageDataGenerator(rescale=1. / 255,
                                                            featurewise_center=True,
                                                            featurewise_std_normalization=True)
image_generator.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 1, 3))  # ordering: [R, G, B]
image_generator.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 1, 3))
test_data_gen = image_generator.flow_from_directory(directory=IMAGES,
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=False,
                                                    target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                    color_mode='rgb',
                                                    class_mode='categorical',
                                                    classes=CLASSES,
                                                    interpolation='lanczos')
scores = loaded_model.fit(test_data_gen)
print(scores)

