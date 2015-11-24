from PIL import Image
import os
import math

from sklearn.cross_validation import KFold

import aux_functions

import numpy as np

import configparser
import cv2


from datetime import datetime

from skimage.feature import hog
from skimage import data, color, exposure


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#import pandas as pd

from sklearn.decomposition import RandomizedPCA
from sklearn.naive_bayes import GaussianNB
from sklearn import svm, metrics
from sklearn import neighbors
from sklearn.linear_model import Perceptron
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import image
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

CROSSTEST = 0
NAIVEBAYES = 1
SUPPORTVECTORMACHINES = 2
NEARESTNEIGHBORGS = 3
PERCEPTRON = 4
LOGISTICSREGRESSION = 5
DECISIONTREE = 6
ADABOOST = 7



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


'''def render3Dplot(data, labels):
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
    plt.show()'''


def load_train_set(img_dir, cross_validation=1, percentage=1.0):
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

    #data = []
    #data_validation = []
    #for image in _images:
        #img = img_to_matrix(image, IMG_SIZE)
        #img = flatten_image(img)
        #data.append(img)

    #for image2 in _images_validation:
        #img2 = img_to_matrix(image2, IMG_SIZE)
        #img2 = flatten_image(img2)
        #data_validation.append(img2)

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

(train_data_images, train_data_split_images, labels, labels_validation) = load_train_set('train/', cross_validation=1, percentage=float(settings['Data']['TrainPercent']))

using_cross_validation = False
using_cross_validation2 = False


if(len(train_data_split_images) == 0):
    test_data_images = load_test_set('test/', percentage=float(settings['Data']['TestPercent']))
else:
    using_cross_validation = False
    test_data_images = train_data_split_images

train_data_features = None
test_data_features = None


train_data = []
train_data_split_crossfold = []
test_data = []
#Choose Image algorithm (Chosen in settings.ini)
if int(settings['ImageFeatureExtraction']['Algorithm']) == 1:
    for image in train_data_images:
        img = img_to_matrix(image, IMG_SIZE)
        img = flatten_image(img)
        train_data.append(img)

    for image in train_data_split_images:
        img = img_to_matrix(image, IMG_SIZE)
        img = flatten_image(img)
        train_data_split_crossfold.append(img)

    for image in test_data_images:
        img = img_to_matrix(image, IMG_SIZE)
        img = flatten_image(img)
        test_data.append(img)

    pca = RandomizedPCA(n_components=int(settings['ImageFeatureExtraction']['NumberFeatures']))
    train_data_features = pca.fit_transform(train_data)
    test_data_features = pca.transform(test_data)
elif int(settings['ImageFeatureExtraction']['Algorithm']) == 2:
    for image in train_data_images:
        img =  cv2.imread(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        train_data.append(gray)

    for image in train_data_split_images:
        img =  cv2.imread(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        train_data_split_crossfold.append(gray)

    for image in test_data_images:
        img =  cv2.imread(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        test_data.append(gray)

    sift = cv2.xfeatures2d.SIFT_create()
    for image in train_data:
        (kps, descs) = sift.detectAndCompute(image, None)
        train_data_features.append(descs)

    for image in train_data_split_crossfold:
        (kps, descs) = sift.detectAndCompute(image, None)
        train_data_features.append(descs)

    for image in test_data:
        (kps, descs) = sift.detectAndCompute(image, None)
        test_data_features.append(descs)

    print("# kps: {}, descriptors: {}".format(len(kps), descs.shape))
    #surf = cv2.xfeatures2d.SURF_create()
    #(kps, descs) = surf.detectAndCompute(gray, None)
    print("# kps: {}, descriptors: {}".format(len(kps), descs.shape))
    #surf = cv2.xfeatures2d.SURF_create()
    #sd = cv2.FeatureDetector_create("SURF")
    #(kps, descs) = surf.detectAndCompute(gray, None)
    #diogokp,des = surf.compute(img, keypoints)
    #model = svm.SVC()
    #diogomodel.fit(des,['type1'])
elif int(settings['ImageFeatureExtraction']['Algorithm']) == 3:
    print(3)
    #sift = cv2.SIFT()
    #train_data_features = sift.detect(train_data)
    #test_data_features = sift.detect(test_data)
elif int(settings['ImageFeatureExtraction']['Algorithm']) == 4:
    n_clusters = 5  # number of regions
    # for img in train_data
        # X = np.reshape(value, (-1, 1) )
        #connectivity = image.grid_to_graph(*img.shape);
        #feature = image.AgglomerativeClustering(n_clusters=n_clusters,
        #                                      linkage='ward', connectivity=connectivity).fit(X)
        # train_data_features.append(feature)
    #for img in test_data
    #    X = np.reshape(img, (-1, 1) )
    #   connectivity = image.grid_to_graph(*img.shape);
    #   feature = image.AgglomerativeClustering(n_clusters=n_clusters,
    #                                           linkage='ward', connectivity=connectivity).fit(X)
    #   test_data_features.append(feature)


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
        #k_neighbors = 1
        #k_neighbors_results = []
        for train, test in kf:
            for k_neighbors in range(2,10):
                clf = neighbors.KNeighborsClassifier(k_neighbors)
                model = clf.fit(train_data_features[train], labels[train])
                predicted_classes = model.predict(train_data_features[test])
                class_probabilities = model.predict_proba(train_data_features[test])
                print("K result, i - ", k_neighbors, ", n points:", len(predicted_classes), ", wrong: ", (labels[test] != predicted_classes).sum(), " sum of errors: ", k_neighbors_results[k_neighbors], " percentage: ",(labels[test] != predicted_classes).sum()*100/len(predicted_classes),"%")
                k_neighbors_results[k_neighbors] += (labels[test] != predicted_classes).sum()
        k_neighbors = list(k_neighbors_results).index(min(k_neighbors_results)) + 2
        print("k = ",k_neighbors)
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
            print(" n points:", len(predicted_classes), ", wrong: ", (labels[test] != predicted_classes).sum(), " percentage: ",(labels[test] != predicted_classes).sum()*100/len(predicted_classes),"%")
            logres_results.append((labels[test] != predicted_classes).sum())
            logres_C += 1
        logres_C = logres_results.index(min(logres_results)) + 1
        print("Log Res C: ", logres_C)
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
elif int(settings['MachineLearningAlgorithm']['Algorithm']) == DECISIONTREE:#6
    if using_cross_validation2:
        _results = []
        base_max_depth = 6
        max_depth = base_max_depth
        step_max_depth = 100
        for train, test in kf:
            clf = DecisionTreeClassifier(max_depth=max_depth)
            model = clf.fit(train_data_features[train], labels[train])
            predicted_classes = model.predict(train_data_features[test])
            class_probabilities = model.predict_proba(train_data_features[test])
            print("maxd ",max_depth," |n points:", len(predicted_classes), ", wrong: ", (labels[test] != predicted_classes).sum(), " percentage: ",(labels[test] != predicted_classes).sum()*100/len(predicted_classes),"%")
            max_depth += step_max_depth
            _results.append((labels[test] != predicted_classes).sum())
        max_depth = 6 + step_max_depth * list(_results).index(min(_results))
        print("opt max depth ",max_depth)
        clf = DecisionTreeClassifier(max_depth = max_depth)
        model = clf.fit(train_data_features, labels)
        predicted_classes = model.predict(test_data_features)
        class_probabilities = model.predict_proba(test_data_features)
    else:
        clf = DecisionTreeClassifier()
        model = clf.fit(train_data_features, labels)
        predicted_classes = model.predict(test_data_features)
        class_probabilities = model.predict_proba(test_data_features)
elif int(settings['MachineLearningAlgorithm']['Algorithm']) == ADABOOST:#7
    if using_cross_validation2:
        _results = np.zeros(10)
        base_n_estimators = 100 # week learners
        step_n_estimators = 100
        ada_results = []
        n_estimators = base_n_estimators
        lr = 1.48
        for train, test in kf:
            #dt = DecisionTreeClassifier(max_depth=26).fit(train_data_features, labels)
            rf = RandomForestClassifier(max_depth=395, n_estimators=80, max_features=7).fit(train_data_features, labels)
            #max_d += 2
            clf = AdaBoostClassifier(algorithm='SAMME.R', base_estimator=rf, n_estimators = n_estimators, learning_rate = lr)
            #lr += 0.01
            model = clf.fit(train_data_features[train], labels[train])
            predicted_classes = model.predict(train_data_features[test])
            class_probabilities = model.predict_proba(train_data_features[test])
            print("ada week learners: ", n_estimators ,"learning rate ",lr," n points:", len(predicted_classes), ", wrong: ", (labels[test] != predicted_classes).sum(), " percentage: ",(labels[test] != predicted_classes).sum()*100/len(predicted_classes),"%"," sum of errors: ", _results[0])
            _results[0] += (labels[test] != predicted_classes).sum()
            ada_results.append((labels[test] != predicted_classes).sum())
            n_estimators += step_n_estimators
        n_estimators = base_n_estimators + step_n_estimators * ada_results.index(min(ada_results))
        print("optimized week learners ", n_estimators)
        #dt = DecisionTreeClassifier(max_depth=26).fit(train_data_features, labels)
        rf = RandomForestClassifier(max_depth=395, n_estimators=80, max_features=7).fit(train_data_features, labels)
        clf = AdaBoostClassifier(base_estimator=rf, n_estimators = n_estimators, learning_rate = lr)
        model = clf.fit(train_data_features, labels)
        predicted_classes = model.predict(test_data_features)
        class_probabilities = model.predict_proba(test_data_features)
    else:
        clf_l1_LR = LogisticRegression(C=1, penalty='l1', tol=0.01)
        lr = clf_l1_LR.fit(train_data_features, labels)
        dt = DecisionTreeClassifier()
        dt = dt.fit(train_data_features, labels)
        clf = AdaBoostClassifier(
            base_estimator=dt,
            learning_rate=1,
            n_estimators=250)
        model = clf.fit(train_data_features, labels)
        predicted_classes = model.predict(test_data_features)
        class_probabilities = model.predict_proba(test_data_features)
elif int(settings['MachineLearningAlgorithm']['Algorithm']) == CROSSTEST:#0
    if using_cross_validation2:

        _results = []
        global_results = []

        names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
         "Random Forest", "AdaBoost", "Naive Bayes", "Linear Discriminant Analysis",
         "Quadratic Discriminant Analysis"]
        classifiers = [
            KNeighborsClassifier(3),
            SVC(kernel="linear", C=0.025, probability=True),
            SVC(gamma=2, C=1, probability=True),
            DecisionTreeClassifier(max_depth=5),
            RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
            AdaBoostClassifier(),
            GaussianNB(),
            LinearDiscriminantAnalysis(),
            QuadraticDiscriminantAnalysis()]

        for name, clf in zip(names, classifiers):
            for train, test in kf:
                model = clf.fit(train_data_features[train], labels[train])
                predicted_classes = model.predict(train_data_features[test])
                class_probabilities = model.predict_proba(train_data_features[test])
                print(name," n points:", len(predicted_classes), ", wrong: ", (labels[test] != predicted_classes).sum(), " percentage: ",(labels[test] != predicted_classes).sum()*100/len(predicted_classes),"%")
                _results.append((labels[test] != predicted_classes).sum())
            result = min(_results)
            global_results.append((name,result))
        print(global_results)

        clf = AdaBoostClassifier()
        model = clf.fit(train_data_features, labels)
        predicted_classes = model.predict(test_data_features)
        class_probabilities = model.predict_proba(test_data_features)
    else:
        clf = AdaBoostClassifier()
        model = clf.fit(train_data_features, labels)
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