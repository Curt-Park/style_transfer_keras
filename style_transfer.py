from vgg19_avg import VGG19_Avg
from keras import backend as K

from scipy.optimize import fmin_l_bfgs_b
from utils import Evaluator

import numpy as np
import argparse
import time
import utils

default_content = './images/content/tubingen.jpg'
default_style = './images/style/starry-night.jpg'
default_output = './results/'
default_content_layer = 'block4_conv2'
default_style_layers = ['block1_conv1',
                        'block2_conv1',
                        'block3_conv1',
                        'block4_conv1',
                        'block5_conv1']

# for reproducibility
np.random.seed(777)

class StyleTransfer(object):
    def __init__(self, content_path, style_path, output_path,
                 loss_ratio=1e-3, verbose=True,
                 save_every_n_iters=10, initial_canvas='content',
                 content_layer=default_content_layer, style_layers=default_style_layers):
        '''
        Parameters:
            content_image_path: The path for the content image
            style_image_path: The path for the style image
            iter_n: iteration number for reconstruction
            loss_ratio: alpha / beta (beta = 1)
            verbose: print loss values
            save_every_n_steps: save images at every n steps
            content_layer: name of the layer to reconstruct the content image
            style_layers: name of the layers to reconstruct the style image
        '''
        self.output_path = output_path
        self.save_every_n_iters = save_every_n_iters
        self.initial_canvas = initial_canvas
        self.verbose = verbose
        self.step = 0

        # load the style and content images
        content_img = utils.open_image(content_path)
        self.img_shape = (content_img.shape[0], content_img.shape[1], 3)
        self.content_img = utils.preproc(content_img)
        self.style_img = utils.preproc(utils.open_image(style_path, self.img_shape))
        content_img = K.variable(self.content_img)
        style_img = K.variable(self.style_img)

        # declare a placeholder for a generated image
        if K.image_data_format() == 'channels_first':
            generated_img = K.placeholder((1, 3, self.img_shape[0], self.img_shape[1]))
        else:
            generated_img = K.placeholder((1, self.img_shape[0], self.img_shape[1], 3))

        # create a keras tensor for the input
        input_tensor = K.concatenate([content_img, style_img, generated_img], axis=0)

        # load the pretrained VGG19 with imagenet.
        # the original paper suggests replacing its all max_pooling to avg_pooling.
        vgg19_avg = VGG19_Avg(input_tensor=input_tensor,
                              include_top=False,
                              input_shape=self.img_shape)

        # outputs of each layer
        outputs_dict = {layer.name:layer.output for layer in vgg19_avg.layers}

        # loss for the content image
        content_feat = outputs_dict[content_layer][0]
        generat_feat = outputs_dict[content_layer][2]
        content_loss = 0.5 * K.sum(K.square(content_feat - generat_feat))

        # loss for the style image.
        style_loss = K.variable(0.)
        style_loss_weight = 1. / len(style_layers)

        for style_layer in style_layers:
            style_feat = outputs_dict[style_layer][1]
            generat_feat = outputs_dict[style_layer][2]

            if K.image_data_format() == 'channels_first':
                N = style_feat.shape[0].value
                M = style_feat.shape[1].value * style_feat.shape[2].value
            else:
                N = style_feat.shape[2].value
                M = style_feat.shape[0].value * style_feat.shape[1].value

            style_loss += style_loss_weight * \
                          K.sum(K.square(utils.gram_matrix(style_feat) - \
                                         utils.gram_matrix(generat_feat))) / \
                          (4. * N ** 2 * M ** 2)

        # composite loss
        beta = 1
        alpha = loss_ratio * beta
        loss = alpha * content_loss + beta * style_loss

        # gradients
        grads = K.gradients(loss, generated_img)

        # evaluation function
        self.fn = K.function([generated_img], [loss]+grads)

    def fit(self, iteration=1000):
        '''
        generate a style-transfered image
        '''

        if self.initial_canvas == 'random':
            x = utils.preproc(np.random.uniform(0, 255, size=self.img_shape))
        elif self.initial_canvas == 'content':
            x = self.content_img.copy()
        else: # style
            x = self.style_img.copy()

        evaluator = Evaluator(self.fn, self.img_shape)
        self.step = 0

        print('Starting optimization with L-BFGS-B')

        for i in range(1, iteration+1):

            if self.verbose:
                print('Starting iteration %d' % (i))

            start_time = time.time()
            x, min_loss, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                              fprime=evaluator.grads,
                                              callback=self._callback, maxfun=20)
            x = np.clip(x, -127, 127)

            end_time = time.time()

            if self.verbose:
                print('Loss: ', min_loss)
                print('Iteration %d completed in %d s' % (i, end_time - start_time))

            if i == 1 or (self.save_every_n_iters != 0 and i % self.save_every_n_iters == 0):
                utils.save_image(utils.deproc(x, self.img_shape),
                                 self.output_path, 'generated_%d' % (i) + '.jpg')

    def _callback(self, x):
        '''
        callback function to print out the time step
        '''
        self.step += 1

        if self.verbose:
            print('step %d is done...' % (self.step))

def get_argument_parser():
    '''
    Argument parser which returns the options which the user inputted.

    Args:
        None

    Returns:
        argparse.ArgumentParser().parse_args()
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--content',
                        help='The path of the content image',
                        type=str, default=default_content)
    parser.add_argument('--style',
                        help='The path of the style image',
                        type=str, default=default_style)
    parser.add_argument('--output',
                        help='The directory path for results',
                        type=str, default=default_output)
    parser.add_argument('--iteration',
                        help='How many iterations you need to run',
                        type=int, default=5000)
    parser.add_argument('--loss_ratio',
                        help='The ratio between content and style (content/style)',
                        type=float, default=8e-4)
    parser.add_argument('--initialization',
                        help='The initial canvas',
                        type=str, default='random', choices=['random', 'content', 'style'])
    parser.add_argument('--save_image_every_nth',
                        help='Save image every nth iteration',
                        type=int, default=10)
    parser.add_argument('--verbose',
                        help='Print reports',
                        type=int, default=1)
    args = parser.parse_args()

    return args

def main():
    args = get_argument_parser()
    style_transfer = StyleTransfer(content_path=args.content,
                                   style_path=args.style,
                                   output_path=args.output,
                                   save_every_n_iters=args.save_image_every_nth,
                                   verbose=(args.verbose == 1),
                                   initial_canvas=args.initialization)
    style_transfer.fit(iteration=args.iteration)

if __name__ == '__main__':
    main()
