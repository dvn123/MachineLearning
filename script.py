# Standard scientific Python imports
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
import imghdr
import struct
import math

from sklearn import datasets, svm, metrics


def get_image_size(fname):
    '''Determine the image type of fhandle and return its size.
    from draco'''
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
    print(image_x_y_mean)

    #setup a standard image size; this will distort some images but will get everything into the same shape
    STANDARD_SIZE = image_x_y_mean
    def img_to_matrix(filename, verbose=False):
        """
        takes a filename and turns it into a numpy array of RGB pixels
        """
        img = Image.open(filename)
        #if verbose==True:
            #print ("changing size from %s to %s" % (str(img.size), str(STANDARD_SIZE)))
        img = img.resize(STANDARD_SIZE)
        img = list(img.getdata())
        #img = map(list, img)
        img = np.array(img)
        return img

    def flatten_image(img):
        """
        takes in an (m, n) numpy array and flattens it
        into an array of shape (1, m * n)
        """
        s = img.shape[0] * img.shape[1]
        img_wide = img.reshape(1, s)
        return img_wide[0]

    data = []
    for image in images:
        img = img_to_matrix(image)
        img = flatten_image(img)
        data.append(img)

    data = np.array(data)
    print(data[0])


'''
def img_to_matrix(filename, verbose=False):
    """
    takes a filename and turns it into a numpy array of RGB pixels
    """
    img = Image.open(filename)
    if verbose==True:
        print "changing size from %s to %s" % (str(img.size), str(STANDARD_SIZE))
    img = img.resize(STANDARD_SIZE)
    img = list(img.getdata())
    img = map(list, img)
    img = np.array(img)
    return img
'''

load_dataset('train/')
#feature_extraction()
#


'''
# The data that we are interested in is made of 8x8 images of digits, let's
# have a look at the first 3 images, stored in the `images` attribute of the
# dataset.  If we were working from image files, we could load them using
# pylab.imread.  Note that each image must have the same size. For these
# images, we know which digit they represent: it is given in the 'target' of
# the dataset.
images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:4]):
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)

# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001)

# We learn the digits on the first half of the digits
classifier.fit(data[:n_samples / 2], digits.target[:n_samples / 2])

# Now predict the value of the digit on the second half:
expected = digits.target[n_samples / 2:]
predicted = classifier.predict(data[n_samples / 2:])

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

images_and_predictions = list(zip(digits.images[n_samples / 2:], predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)

plt.show()
'''