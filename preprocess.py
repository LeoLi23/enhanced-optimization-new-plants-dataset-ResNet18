from tensorflow import keras as keras
from keras.utils import image_dataset_from_directory
from keras.layers.experimental.preprocessing import Rescaling
    
def create_train_valid_test_data():
    train_gen = image_dataset_from_directory(directory="new-plant-diseases-dataset/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/train",
                                            image_size=(256, 256))
    valid_gen = image_dataset_from_directory(directory="new-plant-diseases-dataset/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/valid",
                                            image_size=(256, 256))
    classes_names = train_gen.class_names

    rescale = Rescaling(scale=1.0/255)
    train_gen = train_gen.map(lambda image,label:(rescale(image),label))
    valid_gen  = valid_gen.map(lambda image,label:(rescale(image),label))
    
    return train_gen, valid_gen, classes_names