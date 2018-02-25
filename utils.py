import numpy as np
import os

from scipy.misc import imread, imresize, imsave

# mean value of imagenet for each channel
rn_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)

def preproc(img):
    '''
    data preprocessing
    arguments:
        img: input image
    returns:
        numpy array
    '''
    img = np.expand_dims(np.array(img), 0)
    return (img - rn_mean)[:, :, :, ::-1] # RGB -> BGR

def deproc(img, shape):
    '''
    data deprocessing
    arguments:
        img: output image
        shape: output shape
    returns:
        numpy array
    '''
    return np.clip(img.reshape(shape)[:, :, :, ::-1] + rn_mean, 0, 255)[0]

def open_image(fpath, shape=None):
    '''
    open the file if it exists
    arguments:
        fpath: file path
        shape: (h, w)
    returns:
        PIL Image object
    '''
    if not os.path.isfile(fpath):
        raise ValueError('No image at %s' % (fpath))

    img = imread(fpath)
    if shape is not None:
        if (img.shape[0] != shape[0]) or (img.shape[1] != shape[1]):
            print('resize %s to %d X %d' % (fpath, shape[0], shape[1]))
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
    print('image saved: %s' % (dpath + file_name))

