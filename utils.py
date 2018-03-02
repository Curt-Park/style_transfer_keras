'''
Title: helper functions for style_transfer
Author: Jinwoo Park (Curt)
Email: www.jwpark.co.kr@gmail.com
'''
import os
import numpy as np

from scipy.misc import imread, imresize, imsave
from keras import backend as K

# mean value of imagenet for each channel
RN_MEAN = np.array([123.68, 116.779, 103.939], dtype=np.float32)

class Evaluator(object):
    '''
    scikit-learn's optimizer require
    '''
    def __init__(self, eval_fn, shp):
        '''
        Arguments:
            eval_fn: returned function of K.function
            shp: image shape
        '''
        self.loss_val, self.grads_val = None, None
        self.content_loss, self.style_loss = None, None
        self.eval_fn, self.shp = eval_fn, shp

    def loss(self, img):
        '''
        stores content_loss and style_loss as its member variables and returns loss
        '''
        assert self.loss_val is None
        self.loss_val, self.content_loss, self.style_loss, self.grads_val = \
                self.eval_fn([np.expand_dims(img.reshape(self.shp), 0)])
        return self.loss_val.astype(np.float64)

    def grads(self, _):
        '''
        returns gradients
        '''
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
    img = (np.array(img) - RN_MEAN)[:, :, ::-1] # RGB => BGR
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

    return np.clip(img[:, :, ::-1] + RN_MEAN, 0, 255).astype('uint8')

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
            img = imresize(img, size=(shape[0], shape[1]), interp='lanczos')

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
        os.makedirs(dpath)
        print('created a directory:', dpath)

    imsave(dpath+file_name, img)
    print('saved the image: %s' % (dpath + file_name))

def gram_matrix(features):
    '''
    returns gram matrix (channel size, channel size)
    '''
    assert K.ndim(features) == 3
    if K.image_data_format() == 'channels_first':
        features = K.batch_flatten(features)
    else:
        features = K.batch_flatten(K.permute_dimensions(features, (2, 0, 1)))

    gram = K.dot(features, K.transpose(features))
    return gram

def get_feat_channel_size(feature):
    '''
    returns feature map size and channel size
    '''
    assert K.ndim(feature) == 3
    if K.image_data_format() == 'channels_first':
        feat_size = feature.shape[1].value * feature.shape[2].value
        ch_size = feature.shape[0].value
    else:
        feat_size = feature.shape[0].value * feature.shape[1].value
        ch_size = feature.shape[2].value
    return (feat_size, ch_size)
