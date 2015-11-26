from PIL import Image
import os
import math
import sys

from sklearn.cross_validation import KFold

import aux_functions
import image_features
import machine_learning_models

import numpy as np

import configparser
import cv2

from datetime import datetime

from sklearn import svm, metrics
from sklearn.feature_extraction import image
from skimage.io import imread

CROSS_TEST = 0
NAIVE_BAYES = 1
SUPPORT_VECTOR_MACHINES = 2
K_NEAREST_NEIGHBORGS = 3
PERCEPTRON = 4
LOGISTICS_REGRESSION = 5
DECISION_TREE = 6
ADABOOST = 7
LINEAR_SVM = 8

RANDOMIZED_PCA = 1
SIFT = 2
SURF = 3
HISTOGRAM_OF_GRADIENTS = 5

def read_settings(file_name='settings.ini'):
    config = configparser.ConfigParser()
    config.read(file_name)
    global settings
    settings = config

def load_train_set(img_dir, cross_validation_classwise=1, percentage=1.0):
    _superclasses = [f for f in os.listdir(img_dir)]
    _images = []
    _images_validation = []
    _labels = []
    _labels_validation = []

    image_x_y_mean = [0, 0]

    for superclass in _superclasses:
        if(superclass == ".DS_Store"): continue
        for subclass in os.listdir(img_dir + superclass):
            if(subclass == ".DS_Store"): continue
            n_images = 0
            n_images_to_load = math.ceil(len([f for f in os.listdir(img_dir + superclass + "/" + subclass) if os.path.isfile(os.path.join(img_dir + superclass + "/" + subclass, f))]) * percentage)
            if cross_validation_classwise > 1:
                n_images_validation = math.ceil(n_images_to_load/cross_validation_classwise)
            else:
                n_images_validation = 0
            for image in os.listdir(img_dir + superclass + "/" + subclass):
                size = aux_functions.get_image_size(img_dir + superclass + "/" + subclass + "/" + image)
                if n_images > n_images_to_load:
                    break

                if n_images > n_images_to_load-n_images_validation:
                    _images_validation.append(img_dir + superclass + "/" + subclass + "/" + image)
                    _labels_validation.append(superclass + "/" + subclass)
                else:
                    image_x_y_mean = [image_x_y_mean[i] + size[i] for i, p in enumerate(size)]
                    _images.append(img_dir + superclass + "/" + subclass + "/" + image)
                    _labels.append(superclass + "/" + subclass)
                n_images += 1

    image_x_y_mean = list(map(lambda x: math.floor(x/(len(_images_validation) + len(_images))), image_x_y_mean))
    global IMG_SIZE
    IMG_SIZE = image_x_y_mean

    return np.array(_images), np.array(_images_validation), np.transpose(np.array(_labels)), np.transpose(np.array(_labels_validation))


def load_test_set(img_dir, percentage=1.0):
    images = [img_dir + f for f in os.listdir(img_dir)]
    data = []
    n_images = 0
    n_images_to_load = math.floor(float(settings['Data']['NImagesTest']) * percentage)
    for image in images:
        #if n_images > n_images_to_load:
            #break
        #img = img_to_matrix(image, IMG_SIZE)
        #img = flatten_image(img)
        data.append(image)
        n_images += 1
    return np.array(data)


def write(class_probabilities, file_name='results/' + (datetime.now()).strftime("%Y.%M.%d_%H.%M.%S.csv")):
    ind = np.transpose(np.matrix(np.arange(1, len(class_probabilities) + 1, 1)))
    ind = np.array(ind,dtype="int32")

    class_probabilities = np.hstack((ind, class_probabilities))
    classes = list(map(lambda x: x.split('/', 1)[-1], model.classes_))
    classes.insert(0, 'Id')

    probabilities_format = "%d"+",%.10f"*(len(classes)-1)

    if not os.path.exists('results'):
        os.makedirs('results')

    file_classes = open(file_name, 'w')
    file_classes.write(','.join(classes) + '\n')
    file_classes.close()

    file_probabilities = open(file_name, 'ab')
    np.savetxt(file_probabilities, class_probabilities, delimiter=",", fmt=probabilities_format)
    file_probabilities.close()

settings = None
read_settings()

(train_data_images, train_data_cross_validation_classwise_images, labels, labels_cross_validation_classwise) = load_train_set('train/', cross_validation_classwise=int(settings['Data']['CrossValidationClasswise']), percentage=float(settings['Data']['TrainPercent']))
test_data_images = load_test_set('test/', percentage=float(settings['Data']['TestPercent']))

using_cross_validation_classwise = False
using_cross_validation2 = False

if(len(train_data_cross_validation_classwise_images) == 0):
    using_cross_validation_classwise = True
    #test_data_images = load_test_set('test/', percentage=float(settings['Data']['TestPercent']))
else:
    using_cross_validation_classwise = False

train_data = []
train_data_cross_validation_classwise = []
test_data = []

train_data_cross_validation_classwise_features = []
train_data_features = []
test_data_features = []

#Choose Image algorithm (Chosen in settings.ini)
if int(settings['ImageFeatureExtraction']['Algorithm']) == RANDOMIZED_PCA:
    train_data_features, train_data_cross_validation_classwise_features, test_data_features = image_features.randomized_pca(train_data_images, train_data_cross_validation_classwise_images, test_data_images, IMG_SIZE)
elif int(settings['ImageFeatureExtraction']['Algorithm']) == SIFT:
    train_data_features, train_data_cross_validation_classwise_features, test_data_features = image_features.sift(train_data_images, train_data_cross_validation_classwise_images,  test_data_images)
elif int(settings['ImageFeatureExtraction']['Algorithm']) == 4:
    n_clusters = 5  # number of regions
    for image in train_data:
        #X = np.reshape(value, (-1, 1) )
        img = imread(image, as_grey=True)
        connectivity = image.grid_to_graph(*img.shape)
        feature = image.AgglomerativeClustering(n_clusters=n_clusters, linkage='ward', connectivity=connectivity).fit(img)
        train_data_features.append(feature)
    for img in test_data:
        X = np.reshape(img, (-1, 1) )
        connectivity = image.grid_to_graph(*img.shape)
        feature = image.AgglomerativeClustering(n_clusters=n_clusters, linkage='ward', connectivity=connectivity).fit(X)
        test_data_features.append(feature)
elif int(settings['ImageFeatureExtraction']['Algorithm']) == HISTOGRAM_OF_GRADIENTS:
    train_data_features, train_data_cross_validation_classwise_features, test_data_features = image_features.hog_features(train_data_images, train_data_cross_validation_classwise_images, test_data_images, IMG_SIZE)


if int(settings['Data']['CrossValidation2']) > 1:
    kf = KFold(len(train_data_features), n_folds=int(settings['Data']['CrossValidation2']), shuffle=True)
    using_cross_validation2 = True

predicted_classes = None
class_probabilities = None
model = None

#Choose ML algorithm (Chosen in settings.ini)
if int(settings['MachineLearningAlgorithm']['Algorithm']) == NAIVE_BAYES:#1
    class_probabilities, predicted_classes, model = machine_learning_models.naive_bayes(train_data_features, train_data_cross_validation_classwise_features, test_data_features, labels, labels_cross_validation_classwise , kf, settings)
elif int(settings['MachineLearningAlgorithm']['Algorithm']) == SUPPORT_VECTOR_MACHINES:#2
    class_probabilities, predicted_classes, model = machine_learning_models.svm_model(train_data_features, train_data_cross_validation_classwise_features, test_data_features, labels, labels_cross_validation_classwise, using_cross_validation2, kf, settings)
elif int(settings['MachineLearningAlgorithm']['Algorithm']) == K_NEAREST_NEIGHBORGS:#3
    class_probabilities, predicted_classes, model = machine_learning_models.k_nearest_neighbors(train_data_features, train_data_cross_validation_classwise_features, test_data_features, labels, labels_cross_validation_classwise, using_cross_validation2, kf, settings)
elif int(settings['MachineLearningAlgorithm']['Algorithm']) == PERCEPTRON:#4
    class_probabilities, predicted_classes, model = machine_learning_models.perc(train_data_features, train_data_cross_validation_classwise_features, test_data_features, labels, labels_cross_validation_classwise, kf, settings)
elif int(settings['MachineLearningAlgorithm']['Algorithm']) == LOGISTICS_REGRESSION:#5
    class_probabilities, predicted_classes, model = machine_learning_models.log_res(train_data_features, train_data_cross_validation_classwise_features, test_data_features, labels, labels_cross_validation_classwise, using_cross_validation2, kf, settings)
elif int(settings['MachineLearningAlgorithm']['Algorithm']) == DECISION_TREE:#6
    class_probabilities, predicted_classes, model = machine_learning_models.des_tree(train_data_features, train_data_cross_validation_classwise_features, test_data_features, labels, labels_cross_validation_classwise, using_cross_validation2, kf, settings)
elif int(settings['MachineLearningAlgorithm']['Algorithm']) == ADABOOST:#7
    class_probabilities, predicted_classes, model = machine_learning_models.adaboost(train_data_features, train_data_cross_validation_classwise_features, test_data_features, labels, labels_cross_validation_classwise, using_cross_validation2, kf, settings)
elif int(settings['MachineLearningAlgorithm']['Algorithm']) == CROSS_TEST:#0
    class_probabilities, predicted_classes, model = machine_learning_models.cross_test(train_data_features, train_data_cross_validation_classwise_features, test_data_features, labels, labels_cross_validation_classwise, using_cross_validation2, kf, settings)
elif int(settings['MachineLearningAlgorithm']['Algorithm']) == LINEAR_SVM:#8
    class_probabilities, predicted_classes, model = machine_learning_models.linear_svm(train_data_features, train_data_cross_validation_classwise_features, test_data_features, labels, labels_cross_validation_classwise, using_cross_validation2, kf, settings)

#print(labels_validation)
#print(predicted_classes)

write(class_probabilities)