# Standard scientific Python imports
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
import imghdr
import struct
import math

from sklearn import datasets, svm, metrics
from sklearn.decomposition import RandomizedPCA
from sklearn.naive_bayes import GaussianNB

IMG_SIZE = (80,80)

def get_image_size(fname):
    with open(fname, 'rb') as fhandle:
        head = fhandle.read(24)
        if len(head) != 24:
            return
        if imghdr.what(fname) == 'png':
            check = struct.unpack('>i', head[4:8])[0]
            if check != 0x0d0a1a0a:
                return
            width, height = struct.unpack('>ii', head[16:24])
        else:
            return
        return width, height

def img_to_matrix(filename, size, verbose=False):
    img = Image.open(filename)
    img = img.resize(size)
    img = list(img.getdata())
    img = np.array(img)
    return img

def flatten_image(img):
    s = img.shape[0] * img.shape[1]
    img_wide = img.reshape(1, s)
    return img_wide[0]

def load_dataset(img_dir):
    superclasses = [f for f in os.listdir(img_dir)]
    images = []
    labels = []

    image_x_y_mean = [0, 0]
    n_images = 0

    for superclass in superclasses:
        for subclass in os.listdir(img_dir + superclass):
            for image in os.listdir(img_dir + superclass + "/" + subclass):
                size = get_image_size(img_dir + superclass + "/" + subclass + "/" + image)
                image_x_y_mean = [image_x_y_mean[i] + size[i] for i, p in enumerate(size)]
                #print("IMAGE: " + superclass + "/" + subclass + "/" + image)
                #print("MEAN: x - ",  image_x_y_mean[0], ", y - ", image_x_y_mean[1])
                #print("SIZE: x - ", size[0], ", y - ", size[1])
                n_images += 1
                images.append(img_dir + superclass + "/" + subclass + "/" + image)
                labels.append([superclass + "/" + subclass])

    image_x_y_mean = list(map(lambda x: math.floor(x/n_images), image_x_y_mean))
    global IMG_SIZE
    IMG_SIZE = image_x_y_mean

    #setup a standard image size; this will distort some images but will get everything into the same shape
    data = []
    for image in images:
        img = img_to_matrix(image, IMG_SIZE)
        img = flatten_image(img)
        data.append(img)

    return (np.array(data), labels)

def load_testset(img_dir):
    images = [img_dir + f for f in os.listdir(img_dir)]
    print(images)

    #setup a standard image size; this will distort some images but will get everything into the same shape
    data = []
    for image in images:
        img = img_to_matrix(image, IMG_SIZE)
        img = flatten_image(img)
        data.append(img)

    return np.array(data)

(data, labels) = load_dataset('train/')
testdata = load_testset('test/')
pca = RandomizedPCA(n_components=5)
train_x = pca.fit_transform(data)
test_x = pca.transform(testdata)


gnb = GaussianNB()
y_pred = gnb.fit(data, labels).predict(data)
print("Number of mislabeled points out of a total 1 points : %d",(labels != y_pred).sum())