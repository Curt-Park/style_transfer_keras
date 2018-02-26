import numpy as np
import os

from scipy.misc import imread, imresize, imsave
from keras import backend as K

# mean value of imagenet for each channel
rn_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)

class Evaluator(object):
    '''
    scikit-learn's optimizer require
    '''
    def __init__(self, fn, shp):
        self.loss_val, self.grads_val = None, None
        self.fn, self.shp = fn, shp

    def loss(self, x):
        assert self.loss_val is None
        shp = self.shp
        self.loss_val, self.grads_val = self.fn([x.reshape((1, shp[0], shp[1], shp[2]))])
        return self.loss_val.astype(np.float64)

    def grads(self, x):
        assert self.loss_val is not None
        grads_val = self.grads_val.flatten().astype(np.float64)
        self.loss_val, self.grads_val = None, None
        return grads_val

def preproc(img):
    '''
    data preprocessing
    arguments:
        img: input image
    returns:
        numpy array
    '''
    img = (np.array(img) - rn_mean)[:, :, ::-1] # RGB => BGR
    if K.image_data_format() == 'channels_first':
        img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, 0)
    return img.astype(K.floatx())

def deproc(img, shape):
    '''
    data deprocessing
    arguments:
        img: output image
        shape: output shape (h, w, c)
    returns:
        numpy array
    '''
    if K.image_data_format() == 'channels_first':
        img = img.reshape((shape[2], shape[0], shape[1])).transpose((1, 2, 0))
    else:
        img = img.reshape(shape)

    return np.clip(img[:, :, ::-1] + rn_mean, 0, 255).astype('uint8')

def open_image(fpath, shape=None):
    '''
    open the file if it exists
    arguments:
        fpath: file path
        shape: (h, w, c)
    returns:
        PIL Image object
    '''
    if not os.path.isfile(fpath):
        raise ValueError('No image at %s' % (fpath))

    img = imread(fpath)
    print('opened the image: %s (%d X %d)' % (fpath, img.shape[0], img.shape[1]))
    if shape is not None:
        if (img.shape[0] != shape[0]) or (img.shape[1] != shape[1]):
            print('image resize %s to (%d X %d)' % (fpath, shape[0], shape[1]))
            img = imresize(img, size=shape, interp='lanczos')

    return img

def save_image(img, dpath, file_name):
    '''
    save an image file
    arguments:
        img: an image to save (numpy array)
        dpath: directory path (str)
        file name: file name (str)
    '''
    if not os.path.exists(dpath):
        raise ValueError('No directory: %s' % (dpath))

    imsave(dpath+file_name, img)
    print('saved the image: %s' % (dpath + file_name))

def gram_matrix(features):
    assert K.ndim(features) == 3
    if K.image_data_format() == 'channels_first':
        features = K.batch_flatten(features)
    else:
        features = K.batch_flatten(K.permute_dimensions(features, (2, 0, 1)))

    gram = K.dot(features, K.transpose(features))
    return gram

