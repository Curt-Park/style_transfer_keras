"""
Title: helper functions for style_transfer
Author: Jinwoo Park (Curt)
Email: www.jwpark.co.kr@gmail.com
"""
import os
import numpy as np

from scipy.misc import imread, imresize, imsave
from keras import backend as K

# mean value of imagenet for each channel
RN_MEAN = np.array([123.68, 116.779, 103.939], dtype=np.float32)

class Evaluator(object):
    """ scikit-learn's optimizer requirement

    it runs the evaluation function created by keras backend.
    """
    def __init__(self, eval_fn, shp):
        """ initialize the parameters which is necessary to run evaluation function.

        Args:
            eval_fn (keras.backend.function): keras function for loss and gradients
            shp (tuple): generated image shape (H, W, 3)
        """
        self.loss_val, self.grads_val = None, None
        self.content_loss, self.style_loss = None, None
        self.eval_fn, self.shp = eval_fn, shp

    def loss(self, img):
        """ stores content_loss and style_loss as its member variables and returns loss

        Args:
            img (np.ndarray): flatten array

        Returns:
            np.ndarray (np.float64): training loss
        """
        assert self.loss_val is None
        self.loss_val, self.content_loss, self.style_loss, self.grads_val = \
                self.eval_fn([np.expand_dims(img.reshape(self.shp), 0)])
        return self.loss_val.astype(np.float64)

    def grads(self, _):
        """ returns gradients and initiate member variables: loss_val and grads_val.

        Returns:
            np.ndarray (np.float64): gradients
        """
        assert self.loss_val is not None
        grads_val = self.grads_val.flatten().astype(np.float64)
        self.loss_val, self.grads_val = None, None
        return grads_val

def preproc(img):
    """ Normalize 'img' and expand dim from 3D to 4D array

    Args:
        img (PIL.Image): 3D RGB array of shape (H, W, 3)

    returns:
        np.ndarray: 4D BGR array of shape (1, H, W, 3) or (1, 3, H, W)
    """
    img = (np.array(img) - RN_MEAN)[:, :, ::-1] # RGB => BGR
    if K.image_data_format() == 'channels_first':
        img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, 0)
    return img.astype(K.floatx())

def deproc(img, shape):
    """ reshape 'img' to 3D RGB array and clip all values between 0 to 255

    Args:
        img (np.ndarray): flatten 1D image array
        shape (tuple): output shape (H, W, C)

    returns:
        np.ndarray (utin8): 3D RGB array of shape (H, W, C)
    """
    if K.image_data_format() == 'channels_first':
        img = img.reshape((shape[2], shape[0], shape[1])).transpose((1, 2, 0))
    else:
        img = img.reshape(shape)

    return np.clip(img[:, :, ::-1] + RN_MEAN, 0, 255).astype('uint8')

def open_image(fpath, shape=None):
    """ open the image file and resize it to 'shape' if it exists

    Args:
        fpath (str): image file path
        shape (tuple): image shape (H, W, C)

    Returns:
        PIL Image object
    """
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
    """ save 'img' in the path: dpath + file_name

    Args:
        img (np.ndarray): 3D RGB array of shape (H, W, 3)
        dpath (str): directory path (ex: './dpath/')
        file_name (str): file name (ex: 'output.jpg')
    """
    if not os.path.exists(dpath):
        os.makedirs(dpath)
        print('created a directory:', dpath)

    imsave(dpath+file_name, img)
    print('saved the image: %s' % (dpath + file_name))

def gram_matrix(features):
    """ make a gramian matrix from the input feature map

    Args:
        features (tensor or variable): feature representations of shape (H, W, C) or (C, H, W)

    Returns:
        a gramian matrix of shape (C, C)
    """
    assert K.ndim(features) == 3
    if K.image_data_format() == 'channels_first':
        features = K.batch_flatten(features)
    else:
        features = K.batch_flatten(K.permute_dimensions(features, (2, 0, 1)))

    gram = K.dot(features, K.transpose(features))
    return gram

def get_feat_channel_size(features):
    """ returns feature map size and channel size from the input feature map

    Args:
        features (tensor or variable): feature representations of shape (H, W, C) or (C, H, W)

    Returns:
        feature map size (int: H x W) and channel size (int: C)
    """
    assert K.ndim(features) == 3
    if K.image_data_format() == 'channels_first':
        feat_size = features.shape[1].value * features.shape[2].value
        ch_size = features.shape[0].value
    else:
        feat_size = features.shape[0].value * features.shape[1].value
        ch_size = features.shape[2].value
    return (feat_size, ch_size)
