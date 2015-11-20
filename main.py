from PIL import Image
import os
import math

from sklearn.cross_validation import KFold

import aux_functions

import numpy as np

import configparser
import cv2


from datetime import datetime

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

from sklearn.decomposition import RandomizedPCA
from sklearn.naive_bayes import GaussianNB
from sklearn import svm, metrics
from sklearn import neighbors
from sklearn.linear_model import Perceptron
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression

NAIVEBAYES = 1
SUPPORTVECTORMACHINES = 2
NEARESTNEIGHBORGS = 3
PERCEPTRON = 4
LOGISTICSREGRESSION = 5


def read_settings(file_name='settings.ini'):
    config = configparser.ConfigParser()
    config.read(file_name)
    global settings
    settings = config


def img_to_matrix(filename, size):
    img = Image.open(filename)
    img = img.resize(size)
    img = list(img.getdata())
    img = np.array(img)
    return img


def flatten_image(img):
    s = img.shape[0] * img.shape[1]
    img_wide = img.reshape(1, s)
    return img_wide[0]


def render3Dplot(data, labels):
    pca = RandomizedPCA(n_components=3)
    X = pca.fit_transform(data)
    df = pd.DataFrame({"x": X[:, 0], "y": X[:, 1], "z": X[:, 2],"label":labels})
    uniquelabels = [labels[i] for i in range(len(labels)) if labels[i] != labels[i-1]]
    N = len(uniquelabels)
    colors = ["red","Aqua","Aquamarine","Bisque","Black","Blue","BlueViolet","Chartreuse","Chocolate","DarkGreen","DeepPink","yellow"]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for label, color in zip(uniquelabels, colors):
        mask = df.loc[df['label']==label]
        ax.scatter(mask.x,mask.y, mask.z, c=color, label=label)
    plt.legend()
    plt.show()


def load_train_set(img_dir, cross_validation=1, percentage=1.0):
    _superclasses = [f for f in os.listdir(img_dir)]
    _images = []
    _images_validation = []
    _labels = []
    _labels_validation = []

    image_x_y_mean = [0, 0]

    for superclass in _superclasses:
        for subclass in os.listdir(img_dir + superclass):
            n_images = 0
            n_images_to_load = math.ceil(len([f for f in os.listdir(img_dir + superclass + "/" + subclass) if os.path.isfile(os.path.join(img_dir + superclass + "/" + subclass, f))]) * percentage)
            if cross_validation > 1:
                n_images_validation = math.floor(n_images_to_load/cross_validation)
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

    data = []
    data_validation = []
    for image in _images:
        img = img_to_matrix(image, IMG_SIZE)
        img = flatten_image(img)
        data.append(img)

    for image2 in _images_validation:
        img2 = img_to_matrix(image2, IMG_SIZE)
        img2 = flatten_image(img2)
        data_validation.append(img2)

    return np.array(data), np.array(data_validation), np.transpose(np.array(_labels)), np.transpose(np.array(_labels_validation))


def load_test_set(img_dir, percentage=1.0):
    images = [img_dir + f for f in os.listdir(img_dir)]
    data = []
    n_images = 0
    n_images_to_load = math.floor(float(settings['Data']['NImagesTest']) * percentage)
    for image in images:
        if n_images > n_images_to_load:
            break
        img = img_to_matrix(image, IMG_SIZE)
        img = flatten_image(img)
        data.append(img)
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

(train_data, validation_data, labels, labels_validation) = load_train_set('train/', cross_validation=1, percentage=float(settings['Data']['TrainPercent']))
using_cross_validation = False
using_cross_validation2 = False


if(len(validation_data) == 0):
    test_data = load_test_set('test/', percentage=float(settings['Data']['TestPercent']))
else:
    using_cross_validation = False
    test_data = validation_data

train_data_features = None
test_data_features = None

#Choose Image algorithm (Chosen in settings.ini)
if int(settings['ImageFeatureExtraction']['Algorithm']) == 1:
    pca = RandomizedPCA(n_components=int(settings['ImageFeatureExtraction']['NumberFeatures']))
    train_data_features = pca.fit_transform(train_data)
    test_data_features = pca.transform(test_data)
elif int(settings['ImageFeatureExtraction']['Algorithm']) == 2:
    surf = cv2.xfeatures2d.SURF_create()
    sd = cv2.FeatureDetector_create("SURF")
    (kps, descs) = surf.detectAndCompute(gray, None)
    kp,des = surf.compute(img, keypoints)
    model = svm.SVC()
    model.fit(des,['type1'])

if int(settings['Data']['CrossValidation2']) > 1:
    kf = KFold(len(train_data_features), n_folds=int(settings['Data']['CrossValidation2']), shuffle=True)
    using_cross_validation2 = True

predicted_classes = None
class_probabilities = None
model = None

#Choose ML algorithm (Chosen in settings.ini)
if int(settings['MachineLearningAlgorithm']['Algorithm']) == NAIVEBAYES:#1
    gnb = GaussianNB()
    model = gnb.fit(train_data_features, labels)
    predicted_classes = model.predict(test_data_features)
    class_probabilities = model.predict_proba(test_data_features)
elif int(settings['MachineLearningAlgorithm']['Algorithm']) == SUPPORTVECTORMACHINES:#2
    if using_cross_validation2:
        svm_results = np.zeros(10)
        #k_neighbors = 2
        #k_neighbors_results = []
        for train, test in kf:
            _svm = svm.SVC(probability=True, kernel='sigmoid')
            model = _svm.fit(train_data_features[train], labels[train])
            predicted_classes = model.predict(train_data_features[test])
            class_probabilities = model.predict_proba(train_data_features[test])
            print("n points:", len(predicted_classes), ", wrong: ", (labels[test] != predicted_classes).sum(), " sum of errors: ", svm_results[0])
            svm_results[0] += (labels[test] != predicted_classes).sum()
        k_neighbors = list(svm_results).index(min(svm_results)) + 2
        _svm = svm.SVC(probability=True)
        model = _svm.fit(train_data_features, labels)
        predicted_classes = model.predict(test_data_features)
        class_probabilities = model.predict_proba(test_data_features)
elif int(settings['MachineLearningAlgorithm']['Algorithm']) == NEARESTNEIGHBORGS:#3
    if using_cross_validation2:
        k_neighbors_results = np.zeros(10)
        #k_neighbors = 2
        #k_neighbors_results = []
        for train, test in kf:
            for k_neighbors in range(2,10):
                clf = neighbors.KNeighborsClassifier(k_neighbors)
                model = clf.fit(train_data_features[train], labels[train])
                predicted_classes = model.predict(train_data_features[test])
                class_probabilities = model.predict_proba(train_data_features[test])
                #print("K result, i - ", k_neighbors, ", n points:", len(predicted_classes), ", wrong: ", (labels[test] != predicted_classes).sum(), " sum of errors: ", k_neighbors_results[k_neighbors])
                k_neighbors_results[k_neighbors] += (labels[test] != predicted_classes).sum()
        k_neighbors = list(k_neighbors_results).index(min(k_neighbors_results)) + 2
        clf = neighbors.KNeighborsClassifier(k_neighbors)
        model = clf.fit(train_data_features, labels)
        predicted_classes = model.predict(test_data_features)
        class_probabilities = model.predict_proba(test_data_features)
    else:
        k_neighbors = 8
        clf = neighbors.KNeighborsClassifier(k_neighbors)
        model = clf.fit(train_data_features, labels)
        predicted_classes = model.predict(test_data_features)
        class_probabilities = model.predict_proba(test_data_features)
elif int(settings['MachineLearningAlgorithm']['Algorithm']) == PERCEPTRON:#4
    prc = Perceptron()
    model = prc.fit(train_data_features, labels)
    predicted_classes = model.predict(test_data_features)
    class_probabilities = model.score(test_data_features,predicted_classes)
elif int(settings['MachineLearningAlgorithm']['Algorithm']) == LOGISTICSREGRESSION:#5
    if using_cross_validation2:
        #print("LOGRES")
        logres_C = 1
        logres_results = []
        for train, test in kf:
            C = logres_C
            p = 'l1'
            clf_l1_LR = LogisticRegression(C=C, penalty=p, tol=0.01)
            model = clf_l1_LR.fit(train_data_features[train], labels[train])
            predicted_classes = model.predict(train_data_features[test])
            class_probabilities = model.predict_proba(train_data_features[test])
            logres_results.append((labels[test] != predicted_classes).sum())
            logres_C += 1
        logres_C = logres_results.index(min(logres_results)) + 1
        #print("Log Res C: ", logres_C)
        clf_l1_LR = LogisticRegression(C=logres_C, penalty=p, tol=0.01)
        model = clf_l1_LR.fit(train_data_features, labels)
        predicted_classes = model.predict(test_data_features)
        class_probabilities = model.predict_proba(test_data_features)
    else:
        C = 1
        p = 'l1'
        clf_l1_LR = LogisticRegression(C=C, penalty=p, tol=0.01)
        model = clf_l1_LR.fit(train_data_features, labels)
        predicted_classes = model.predict(test_data_features)
        class_probabilities = model.predict_proba(test_data_features)


#print(labels_validation)
#print(predicted_classes)
if using_cross_validation:
    #print("Number of mislabeled points out of a total ", len(labels_validation), " points: ", (labels_validation != predicted_classes).sum())
    #print("Classification report for classifier %s:\n%s\n"      % (classifier, metrics.classification_report(expected, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(labels_validation, predicted_classes))
    print("Classification report for classifier %s:\n%s\n" % (model, metrics.classification_report(labels_validation, predicted_classes)))

write(class_probabilities)