import numpy as np
import cv2

from skimage.feature import hog
from skimage import data, color, exposure
from sklearn.decomposition import RandomizedPCA
from skimage.feature import daisy
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
from skimage import feature
import matplotlib.pyplot as plt



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
    train_data_split_crossfold_features = []
    train_data = []
    test_data = []
    train_data_split_crossfold = []
    bow_train = cv2.BOWKMeansTrainer(50)

    flann_params = dict(algorithm = 1, trees = 5)
    matcher = cv2.FlannBasedMatcher(flann_params, {})

    detect = cv2.xfeatures2d.SIFT_create(200, 3, 0.003, 30, 1.6)
    extract = cv2.xfeatures2d.SIFT_create()

    bow_extract = cv2.BOWImgDescriptorExtractor(extract, matcher)

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
            print('!!')
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
            print('!!')
            continue
        train_data_features.append(a.flatten())
    print(len(train_data_features))

    for image in train_data_split_crossfold:
        a = bow_extract.compute(image, detect.detect(image))
        if a is None:
            print('!!')
            continue
        train_data_split_crossfold_features.append(a.flatten())

    for image in test_data:
        #(kps, descs) = detect.detectAndCompute(image, None)
        #bowDes = bow_extract.compute(image, kps, descs)
        a = bow_extract.compute(image, detect.detect(image))
        if a is None:
            print('!!')
            continue
        test_data_features.append(a.flatten())

    return np.asarray(train_data_features), np.asarray(train_data_split_crossfold_features), np.asarray(test_data_features)

def bow(des_list, data):
    descriptors = des_list[0]
    for descriptor in des_list[1:]:
        descriptors = np.vstack((descriptors, descriptor))

    print(descriptors)
    voc, variance = kmeans(descriptors, 50)

    im_features = np.zeros((len(data), 50), "float32")
    for i in range(len(data)):
        words, distance = vq(des_list[i][1],voc)
        for w in words:
            im_features[i][w] += 1

    nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
    idf = np.array(np.log((1.0*len(data)+1) / (1.0*nbr_occurences + 1)), 'float32')

    stdSlr = StandardScaler().fit(im_features)
    return stdSlr.transform(im_features)

def daisy_features(train_data_images, train_data_split_images, test_data_images, IMG_SIZE):
    canny(train_data_images, train_data_split_images, test_data_images, IMG_SIZE)
    train_data_features = []
    test_data_features = []
    train_data = []
    test_data = []
    train_data_split_crossfold = []
    print(4)
    #bow_train = cv2.BOWKMeansTrainer(8)

    #flann_params = dict(algorithm = 1, trees = 5)
    #matcher = cv2.FlannBasedMatcher(flann_params, {})

    #detect = cv2.xfeatures2d.SIFT_create()
    #extract = cv2.xfeatures2d.SIFT_create()

    #bow_extract = cv2.BOWImgDescriptorExtractor(extract, matcher)
    #help(bow_train)
    #help(bow_extract)
    for image in train_data_images:
        img =  imread(image, as_grey=True)
        resized_image = resize(img, (40,40))
        train_data.append(resized_image)

    for image in train_data_split_images:
        img =  imread(image, as_grey=True)
        resized_image = resize(img, (40,40))
        train_data_split_crossfold.append(resized_image)

    for image in test_data_images:
        img =  imread(image, as_grey=True)
        resized_image = resize(img, (40,40))
        test_data.append(resized_image)

    print(6)
    des = []
    des_cross = []
    des_test = []

    radius = 5
    for image in train_data:
        descs = daisy(image, radius=radius)
        des.append(descs)

    train_data_features = bow(des, train_data)
    del des

    print('oi1')

    #for image in train_data_split_crossfold:
        #descs = daisy(image, radius=radius)
        #des_cross.append(descs)

    print('oi1')
    #for image in test_data:
        #descs = daisy(image, radius=radius)
        #des_test.append(descs)

    print('oi1')
    #return bow(des, train_data), bow(des_cross, train_data_split_crossfold), bow(des_test, test_data)

def canny(train_data_images, train_data_split_images, test_data_images, IMG_SIZE):
    train_data_features = []
    test_data_features = []
    train_data_split_crossfold_features = []
    train_data = []
    test_data = []
    train_data_split_crossfold = []

    IMG_SIZE = (60, 60)

    for image in train_data_images:
        img =  imread(image, as_grey=True)
        #img = resize(img, IMG_SIZE)
        train_data.append(img)

    for image in train_data_split_images:
        img =  imread(image, as_grey=True)
        #img = resize(img, IMG_SIZE)
        train_data_split_crossfold.append(img)

    for image in test_data_images:
        img =  imread(image, as_grey=True)
        #img = resize(img, IMG_SIZE)
        test_data.append(img)

    """im1 = feature.canny(train_data[0])
    im2 = feature.canny(train_data[0], sigma=3)

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3), sharex=True, sharey=True)

    ax1.imshow(train_data[0], cmap=plt.cm.jet)
    ax1.axis('off')
    ax1.set_title('noisy image', fontsize=20)

    ax2.imshow(im1, cmap=plt.cm.gray)
    ax2.axis('off')
    ax2.set_title('Canny filter, $\sigma=1$', fontsize=20)

    ax3.imshow(im2, cmap=plt.cm.gray)
    ax3.axis('off')
    ax3.set_title('Canny filter, $\sigma=3$', fontsize=20)

    fig.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9,
                        bottom=0.02, left=0.02, right=0.98)

    plt.show()"""

    for image in train_data:
        a = feature.canny(image)
        img = resize(a, IMG_SIZE)
        img = np.array(img)
        b = flatten_image(img)
        train_data_features.append(b)

    for image in train_data_split_crossfold:
        a = feature.canny(image)
        img = resize(a, IMG_SIZE)
        img = np.array(img)
        b = flatten_image(img)
        train_data_split_crossfold_features.append(b)

    for image in test_data:
        a = feature.canny(image)
        img = resize(a, IMG_SIZE)
        img = np.array(img)
        b = flatten_image(img)
        test_data_features.append(b)

    #pca = RandomizedPCA(500)
    #return (pca.fit_transform(train_data_features), pca.transform(train_data_split_crossfold_features), pca.transform(test_data_features))
    return train_data_features, train_data_split_crossfold_features, test_data_features

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