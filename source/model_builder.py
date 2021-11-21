from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from keras.models import Sequential
from keras import optimizers
import keras
from keras.applications import vgg16, mobilenet_v2
from keras.models import Model
from tensorboard.plugins.hparams import api as hp
import pandas as pd

batch_size = 16
epochs = 15


HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([512, 1024]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.3))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'RMSprop']))

# build the model using the given pre-trained model
def model_builder(pretrained_model, input_shape, hparams):
    model = Sequential()
    new_model = freeze_layers(pretrained_model)
    model.add(new_model)
    model.add(Dense(hparams[HP_NUM_UNITS], activation='relu',kernel_initializer='he_uniform', input_dim=input_shape))
    model.add(Dropout(hparams[HP_DROPOUT]))
    model.add(Dense(32, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(hparams[HP_DROPOUT]))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                    optimizer=hparams[HP_OPTIMIZER],
                    metrics=['accuracy'])
    return model

# Freeze the weights of the first hidden layers 
def freeze_layers(model):
    output = model.layers[-1].output
    output = keras.layers.Flatten()(output)
    model = Model(model.input, output)
    model.trainable = True

    set_trainable = False
    for layer in model.layers:
        if layer.name in ['block5_conv1', 'block4_conv1']:
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False
    return model

# visualize layers
def visualize_model_layers(model):
    pd.set_option('max_colwidth', -1)
    layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
    pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])  