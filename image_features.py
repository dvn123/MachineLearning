import numpy as np
import cv2

from skimage.feature import hog
from skimage import data, color, exposure
from sklearn.decomposition import RandomizedPCA

from PIL import Image

from skimage.io import imread
from skimage.transform import resize

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


def randomized_pca(train_data_images, train_data_split_images, test_data_images, IMG_SIZE):
    train_data_features = []
    test_data_features = []
    train_data = []
    test_data = []
    train_data_split_crossfold = []

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

    pca = RandomizedPCA(50)
    return (pca.fit_transform(train_data), pca.transform(test_data))

def sift(train_data_images, train_data_split_images, test_data_images):
    train_data_features = []
    test_data_features = []
    train_data = []
    test_data = []
    train_data_split_crossfold = []
    print(4)
    bow_train = cv2.BOWKMeansTrainer(8)

    flann_params = dict(algorithm = 1, trees = 5)
    matcher = cv2.FlannBasedMatcher(flann_params, {})

    detect = cv2.xfeatures2d.SIFT_create()
    extract = cv2.xfeatures2d.SIFT_create()

    bow_extract = cv2.BOWImgDescriptorExtractor(extract, matcher)
    #help(bow_train)
    #help(bow_extract)
    for image in train_data_images:
        img =  cv2.imread(image)
        resized_image = cv2.resize(img, (91,92))
        gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        train_data.append(gray)

    for image in train_data_split_images:
        img =  cv2.imread(image)
        resized_image = cv2.resize(img, (91,92))
        gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        train_data_split_crossfold.append(gray)

    for image in test_data_images:
        img =  cv2.imread(image)
        resized_image = cv2.resize(img, (91,92))
        gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        test_data.append(gray)

    print(6)

    for image in train_data:
        descs = extract.compute(image, detect.detect(image))[1]
        if descs is None:
            continue
        #print(descs)
        bow_train.add(descs)

    print(8)
    voc = bow_train.cluster()
    bow_extract.setVocabulary(voc)

    print(7)
    for image in train_data:
        a = bow_extract.compute(image, detect.detect(image))
        if a is None:
            continue
        train_data_features.append(a.flatten())
    print(len(train_data_features))
    #for image in train_data_split_crossfold:
    #bowDes = bow_extract.compute(image, kps, descs)
    #train_data_features.append(bow_extract.compute(image, detect.detect(image)))
    #train_data_features.append(descs)
    for image in test_data:
        #(kps, descs) = detect.detectAndCompute(image, None)
        #bowDes = bow_extract.compute(image, kps, descs)
        a = bow_extract.compute(image, detect.detect(image))
        if a is None:
            continue
        test_data_features.append(a.flatten())

    return np.asarray(train_data_features), np.asarray(test_data_features)

def hog_features(train_data_images, train_data_split_images, test_data_images, IMG_SIZE):
    train_data_features = []
    test_data_features = []
    train_data = []
    test_data = []
    train_data_split_crossfold = []

    IMG_SIZE = (40,40)
    orientations = 8
    pixels_per_cell = (5, 5)
    cells_per_block = (2, 2)
    visualize = False
    normalise = False

    test_data = []

    train_data_split_crossfold_features = []
    train_data_features = []
    test_data_features = []
    for image in train_data_images:
        img = imread(image, as_grey=True)
        img = resize(img, IMG_SIZE)
        train_data.append(img)

    for image in train_data_split_images:
        img = imread(image, as_grey=True)
        img = resize(img, IMG_SIZE)
        train_data_split_crossfold.append(img)

    for image in test_data_images:
        img = imread(image, as_grey=True)
        img = resize(img, IMG_SIZE)
        test_data.append(img)

    for image in train_data:
        fd = hog(image,orientations,pixels_per_cell,cells_per_block,visualize,normalise)
        train_data_features.append(fd)

    for image in train_data_split_crossfold:
        fd = hog(image,orientations,pixels_per_cell,cells_per_block,visualize,normalise)
        train_data_split_crossfold_features.append(fd)

    for image in test_data:
        fd = hog(image,orientations,pixels_per_cell,cells_per_block,visualize,normalise)
        test_data_features.append(fd)

    return np.array(train_data_features), np.array(train_data_split_crossfold_features), np.array(test_data_features)