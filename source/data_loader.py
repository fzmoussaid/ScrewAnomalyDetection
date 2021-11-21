import numpy as np
import os
from os import listdir
from os.path import isfile, join
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import load_img, img_to_array, array_to_img

IMG_DIM = (224, 224)
dataset_name = 'screws'
dataset_location = 'archive'
train_dataset = 'train'
test_dataset = 'test'
valid_dataset = 'valid'
dataset_path = os.path.join(os.getcwd(), dataset_name, dataset_location)

# Apply permutation on data to shuffle it
def apply_permutation(data, indexes):
    n = len(data)
    result = np.empty(n, dtype=object)
    for i in range(n):
        result[i] = data[indexes[i]]
    return result

# Load the dataset of type data_type
def load_dataset(data_type):
    path = join(dataset_path, data_type)
    files = []
    labels = []
    f_dir = [f for f in listdir(path) if not isfile(join(path, f))]

    for directory in f_dir:
        new_files = [join(path, directory, f) for f in listdir(join(path, directory)) if isfile(join(path, directory, f))]
        files.extend(new_files)
        for f in new_files:
            labels.append(directory)

    shuffled_indices = np.random.permutation(len(files))
    files = apply_permutation(files, shuffled_indices)
    labels = apply_permutation(labels, shuffled_indices)
    imgs = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in files]
    imgs = np.array(imgs)
    print(data_type + " dataset loaded.")
    return imgs, labels




train_imgs, train_labels = load_dataset('train')
valid_imgs, valid_labels = load_dataset('valid')
test_imgs, test_labels = load_dataset('test')

train_imgs_scaled = train_imgs.astype('float32')
train_imgs_scaled /= 255
valid_imgs_scaled = valid_imgs.astype('float32')
valid_imgs_scaled /= 255
test_imgs_scaled = test_imgs.astype('float32')
test_imgs_scaled /= 255

le = LabelEncoder()
le.fit(train_labels)
train_labels_enc = le.transform(train_labels)
valid_labels_enc = le.transform(valid_labels)
test_labels_enc = le.transform(test_labels)


test_labels_enc = le.transform(test_labels)


