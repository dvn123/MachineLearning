# Standard scientific Python imports
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
import imghdr
import struct
import math
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


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

def split_list_percent(a_list, percent):
    half = math.floor(len(a_list)*percent)
    #print(half)
    return a_list[:half]

def flatten_image(img):
    s = img.shape[0] * img.shape[1]
    img_wide = img.reshape(1, s)
    return img_wide[0]

def load_dataset(img_dir, percent=1.0):
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
                labels.append(superclass + "/" + subclass)

    image_x_y_mean = list(map(lambda x: math.floor(x/n_images), image_x_y_mean))
    global IMG_SIZE
    IMG_SIZE = image_x_y_mean

    #setup a standard image size; this will distort some images but will get everything into the same shape
    data = []
    for image in images:
        img = img_to_matrix(image, IMG_SIZE)
        img = flatten_image(img)
        data.append(img)

    return np.array(split_list_percent(data, percent)), np.transpose(split_list_percent(labels, percent))

def load_testset(img_dir, percent=1.0):
    images = [img_dir + f for f in os.listdir(img_dir)]

    data = []
    for image in images:
        img = img_to_matrix(image, IMG_SIZE)
        img = flatten_image(img)
        data.append(img)

    return np.array(split_list_percent(data, percent))

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

(data, labels) = load_dataset('train/')
#render3Dplot(data,labels)
testdata = load_testset('test/', 0.1)
pca = RandomizedPCA(n_components=5)
train_x = pca.fit_transform(data)
test_x = pca.transform(testdata)

#print("TRAIN_X: ", train_x)
#print("TEST_X: ", test_x)
#print("LABELS: ", np.array_repr(labels))

gnb = GaussianNB()
y_pred = gnb.fit(train_x, labels).predict_proba(test_x)

#print(list(map(lambda x: x.split('/', 1)[-1], gnb.classes_)))

##WRITE
#np.set_printoptions(threshold=np.inf)
#y_pred_tmp = np.array([])

#for index, line in enumerate(y_pred):
#print(index)
#print(line)
#print(np.array([index+1]))
#y_pred_tmp = np.append(y_pred_tmp, np.append(np.array([index+1]), y_pred))
#print(y_pred_tmp)
#print(np.append(np.array([index+1]), y_pred))

ind = np.transpose(np.matrix(np.arange(1, len(y_pred) + 1, 1)))
ind = np.array(ind,dtype="int32")
#print(ind)
#print(len(ind))
#print(len(y_pred))
#print(np.hstack((ind, y_pred)))
print(y_pred)
print("______________")
y_pred = np.hstack((ind, y_pred))
print(y_pred)
#y_pred = [np.append((np.array([i+1]), p), y_pred) for i, p in enumerate(y_pred)]
#y_pred = np.array(map(lambda x: np.append((np.array([index+1]), x), y_pred))
#print(y_pred)
time_now = (datetime.now()).strftime("%Y.%M.%d_%H.%M.%S.csv")
if not os.path.exists('results'):
    os.makedirs('results')

f2 = open('results/' + time_now, 'w')

classes = list(map(lambda x: x.split('/', 1)[-1], gnb.classes_))
classes.insert(0, 'Id')
print(classes)
f2.write(','.join(classes) + '\n')
f2.close()
#np.savetxt(f, np.array(','.join(gnb.classes_), fmt="%s", delimiter=',', newline='')
f = open('results/' + str(time_now), 'ab')
sfmt = "%d"+",%.10f"*(len(classes)-1)
np.savetxt(f, y_pred, delimiter=",", fmt=sfmt)
f.close()
#print("Y_PRED:", np.append(labels ,y_pred))
#print("' of mislabeled points out of a total 1 points : %d",(labels != y_pred).sum())