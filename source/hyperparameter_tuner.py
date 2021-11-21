import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from keras.applications import vgg16
from model_builder import model_builder, batch_size, epochs
from data_loader import train_imgs_scaled, train_labels_enc, valid_imgs_scaled, valid_labels_enc, test_imgs_scaled, test_labels_enc

input_shape = (224, 224, 3)
def train_model(hparams):
    vgg = vgg16.VGG16(include_top=False, weights='imagenet', 
                                        input_shape=input_shape)
    model = model_builder(vgg, input_shape, hparams)
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("screw_classifier_model.h5", save_best_only=True, verbose=1) 
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=5)
    model.fit(x=train_imgs_scaled, y=train_labels_enc,
                        validation_data=(valid_imgs_scaled, valid_labels_enc),
                        batch_size=batch_size, epochs=epochs, 
                        callbacks=[checkpoint_cb, early_stopping_cb], verbose=1) 

    test_loss, test_acc = model.evaluate(test_imgs_scaled,  test_labels_enc)
  
    return test_acc

# Run the training model using different hyperparameters to compare
def run(run_dir, hparams):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)
        accuracy = train_model(hparams)
        tf.summary.scalar('accuracy', accuracy, step=1)
