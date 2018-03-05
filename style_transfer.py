"""
Title: Keras implementation of Style-Transfer
Paper: https://goo.gl/KDiquz
Author: Jinwoo Park (Curt)
Email: www.jwpark.co.kr@gmail.com
"""
import time
import argparse
import numpy as np

from keras import backend as K
from keras.applications.vgg16 import VGG16
from scipy.optimize import fmin_l_bfgs_b

from utils import Evaluator
import utils


DEFAULT_CONTENT = './images/content/tubingen.jpg'
DEFAULT_STYLE = './images/style/shipwreck.jpg'
DEFAULT_OUTPUT = './outputs/'
DEFAULT_CONTENT_LAYER = 'block4_conv2'
DEFAULT_STYLE_LAYERS = ['block1_conv1',
                        'block2_conv1',
                        'block3_conv1',
                        'block4_conv1',
                        'block5_conv1']

# for reproducibility
np.random.seed(777)

class StyleTransfer(object):
    """Style-Transfer techniques described in the following paper: https://goo.gl/KDiquz

    Differences from the original paper are marked with comments
    which start with "! the original paper ...."

    Attributes:
        output_path (str): the directory path for outcomes.
        save_every_n_iters (int): save images every n iteration.
        initial_canvas (str): the initial canvas for generated images.
                              Choice in {'random', 'content', 'style'}
        verbose (bool): True for print states, False otherwise.
        step (int): store the training step to print out logs.
        img_shape (tuple): 3D array of shape (H, W, 3)
        content_img (np.ndarray): 4D array of shape (1, H, W, 3)
        style_img (np.ndarray): 4D array of shape (1, H, W, 3)
        eval_fn (keras.backend.function): keras function to return training loss and gradients.

    Example:
        style_transfer = StyleTransfer(...) # initialization
        style_transfer.fit(...) # generating images
    """
    def __init__(self, content_path, style_path, output_path,
                 loss_ratio=1e-3, verbose=True,
                 save_every_n_iters=10, initial_canvas='content'):
        """ intialize all things for style transfer

        the overall process is as follows:
            1. image preprocessing
            2. define loss functions for content representations and style representations
            3. define evaluation function for training loss and gradients

        Args:
            content_image_path (str): The path for the content image
            style_image_path (str): The path for the style image
            loss_ratio (float): alpha divided by beta (beta is defined as 1)
            verbose (bool): True for print states, False otherwise
            save_every_n_steps: save images every n steps
            initial_canvas (str): the initial canvas for generated images.
                                  Choice in {'random', 'content', 'style'}
        """
        self.output_path = output_path
        self.save_every_n_iters = save_every_n_iters
        self.initial_canvas = initial_canvas
        self.verbose = verbose
        self.step = 0

        content_layer = DEFAULT_CONTENT_LAYER
        style_layers = DEFAULT_STYLE_LAYERS

        # load the style and content images
        content_img = utils.open_image(content_path)
        self.img_shape = (content_img.shape[0], content_img.shape[1], 3)
        self.content_img = utils.preproc(content_img)
        self.style_img = utils.preproc(utils.open_image(style_path, self.img_shape))
        content_img = K.variable(self.content_img)
        style_img = K.variable(self.style_img)

        # define a placeholder for a generated image
        if K.image_data_format() == 'channels_first':
            generated_img = K.placeholder((1, 3, self.img_shape[0], self.img_shape[1]))
        else:
            generated_img = K.placeholder((1, self.img_shape[0], self.img_shape[1], 3))

        # create a keras tensor for the input
        input_tensor = K.concatenate([content_img, style_img, generated_img], axis=0)

        # load VGG16 with the weights pretrained on imagenet.
        # ! the original paper uses vgg19 and replaces its all max_pooling to avg_pooling.
        vgg16 = VGG16(input_tensor=input_tensor,
                      include_top=False,
                      input_shape=self.img_shape)

        # outputs of each layer
        outputs_dict = {layer.name:layer.output for layer in vgg16.layers}

        # loss for the content image
        content_feat = outputs_dict[content_layer][0]
        generat_feat = outputs_dict[content_layer][2]
        feat_size, ch_size = utils.get_feat_channel_size(generat_feat)

        # ! the original paper suggests 'divided by 2'
        # the following denominator is from:
        # from https://github.com/cysmith/neural-style-tf/blob/master/neural_style.py
        content_loss = K.sum(K.square(content_feat - generat_feat)) \
                        / (2. *  feat_size * ch_size)

        # loss for the style image.
        style_loss = K.variable(0.)
        style_loss_weight = 1. / len(style_layers)

        for style_layer in style_layers:
            style_feat = outputs_dict[style_layer][1]
            generat_feat = outputs_dict[style_layer][2]
            feat_size, ch_size = utils.get_feat_channel_size(generat_feat)

            style_loss += style_loss_weight * \
                          K.sum(K.square(utils.gram_matrix(style_feat) - \
                                         utils.gram_matrix(generat_feat))) / \
                          (4. * feat_size ** 2 * ch_size ** 2)

        # composite loss
        beta = 1
        alpha = loss_ratio * beta
        content_loss = alpha * content_loss
        style_loss = beta * style_loss
        total_loss = content_loss + style_loss

        # gradients
        grads = K.gradients(total_loss, generated_img)

        # evaluation function
        self.eval_fn = K.function([generated_img], [total_loss, content_loss, style_loss]+grads)

    def fit(self, iteration=1000):
        """ generate a style-transfered image.

        The overall process is as follows:
            1. initiating the initial canvas
            2. setting Evaluator
            3. optimizing using L-BFGS method

        Args:
            iteration (int): the total iteration number of optimization
        """

        if self.initial_canvas == 'random':
            generated_img = utils.preproc(np.random.uniform(0, 255, size=self.img_shape)\
                                .astype(K.floatx()))
        elif self.initial_canvas == 'content':
            generated_img = self.content_img.copy()
        else: # style
            generated_img = self.style_img.copy()

        evaluator = Evaluator(self.eval_fn, self.img_shape)
        self.step = 0

        print('Starting optimization with L-BFGS-B')

        for i in range(1, iteration+1):

            if self.verbose:
                print('Starting iteration %d' % (i))

            start_time = time.time()
            generated_img, min_loss, _ = fmin_l_bfgs_b(evaluator.loss,
                                                       generated_img.flatten(),
                                                       fprime=evaluator.grads,
                                                       callback=self._callback,
                                                       maxfun=20)
            generated_img = np.clip(generated_img, -127, 127)
            end_time = time.time()

            if self.verbose:
                print('Total_Loss: %g, Content Loss: %g, Style Loss: %g' % \
                                                                    (min_loss,
                                                                     evaluator.content_loss,
                                                                     evaluator.style_loss))
                print('Iteration %d completed in %d s' % (i, end_time - start_time))

            if i == 1 or (self.save_every_n_iters != 0 and i % self.save_every_n_iters == 0):
                utils.save_image(utils.deproc(generated_img, self.img_shape),
                                 self.output_path, 'generated_%d' % (i) + '.jpg')

    def _callback(self, _):
        """ callback function to print out the each time step in optimization process

        Args:
            _ (list): parameter vector which is not used here
        """
        self.step += 1

        if self.verbose:
            print('step %d is done...' % (self.step))

def get_argument_parser():
    """ Argument parser which returns the options which the user inputted.

    Returns:
        argparse.ArgumentParser().parse_args()
            which contains the following parameters:
                content (str): content image path
                style (str): style image path
                output (str): output directory path
                iteration (int): total iteration number
                loss_ratio (float): loss ratio of content weight divided by style weight
                initialization (string): the inital canvas
                save_image_every_nth (int): save images at every nth the iteration
                verbose (bool): True for printing out the states, False otherwise
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--content',
                        help='The path of the content image',
                        type=str, default=DEFAULT_CONTENT)
    parser.add_argument('--style',
                        help='The path of the style image',
                        type=str, default=DEFAULT_STYLE)
    parser.add_argument('--output',
                        help='The directory path for results',
                        type=str, default=DEFAULT_OUTPUT)
    parser.add_argument('--iteration',
                        help='How many iterations you need to run',
                        type=int, default=1000)
    parser.add_argument('--loss_ratio',
                        help='The ratio between content and style (content/style)',
                        type=float, default=1e-3)
    # ! the original paper uses white noise for an initial canvas
    parser.add_argument('--initialization',
                        help='The initial canvas',
                        type=str, default='content', choices=['random', 'content', 'style'])
    parser.add_argument('--save_image_every_nth',
                        help='Save image every nth iteration',
                        type=int, default=10)
    parser.add_argument('--verbose',
                        help='Print reports',
                        type=int, default=1)
    args = parser.parse_args()

    return args

def main():
    """ this function is called when this script starts

    the overall process is as follows:
        1. parse input parameters through argument paser
        2. initiate a StyleTranfer object
        3. generate images
    """
    args = get_argument_parser()
    style_transfer = StyleTransfer(content_path=args.content,
                                   style_path=args.style,
                                   output_path=args.output,
                                   loss_ratio=args.loss_ratio,
                                   save_every_n_iters=args.save_image_every_nth,
                                   verbose=(args.verbose == 1),
                                   initial_canvas=args.initialization)
    style_transfer.fit(iteration=args.iteration)

if __name__ == '__main__':
    main()
