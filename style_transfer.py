from vgg19_avg import VGG19_Avg
from keras.models import Model
from keras import backend as K

from scipy.optimize import minimize
import numpy as np
import utils

content_path = './images/content.png'
style_path = './images/style.png'
output_path = './results/'
CONTENT_LAYER = ['block4_conv2']
STYLE_LAYERS = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

class StyleTransfer(object):
    def __init__(self, content_image_path, style_image_path,
                 alpha=1e-3, beta=1, verbose=True, save_every_n_iters=10,
                 content_layer=CONTENT_LAYER, style_layers=STYLE_LAYERS):
        '''
        Parameters:
            content_image_path: The path for the content image
            style_image_path: The path for the style image
            iter_n: iteration number for reconstruction
            alpha: content loss function weight
            beta: content loss function weight
            verbose: print loss values
            save_every_n_steps: save images at every n steps
            content_layer: name of the layer to reconstruct the content image
            style_layers: name of the layers to reconstruct the style image
        '''
        self.verbose = verbose
        self.save_every_n_iters = save_every_n_iters

        # load the style and content images
        content_img = utils.preproc(utils.open_image(content_path))
        self.img_shape = content_img.shape
        style_img = utils.preproc(utils.open_image(style_path, self.img_shape[1:3]))

        # load the pretrained VGG19 with imagenet.
        # the original paper suggests replacing its all max_pooling to avg_pooling.
        vgg19_avg = VGG19_Avg(include_top = False,
                              input_shape = self.img_shape[1:])

        # get the target tensor for the content image
        content_output = vgg19_avg.get_layer(name=content_layer[0]).output
        content_model = Model(vgg19_avg.input, content_output)
        content_tensor = K.variable(content_model.predict(content_img))

        # get the target tensors for the style image
        style_outputs = [vgg19_avg.get_layer(name=layer_name).output for layer_name in style_layers]
        style_models = Model(vgg19_avg.input, style_outputs)
        style_tensors = [K.variable(o) for o in style_models.predict(style_img)]

        # loss for the content image
        content_loss = 0.5 * K.sum(K.square(content_tensor - content_output))

        # loss for the style image.
        print('test', style_outputs[0].shape[1])
        style_loss = 0.
        style_loss_weight = 1. / len(style_layers)
        for i, style_tensor in enumerate(style_tensors):
            style_loss += style_loss_weight * \
                          (1. / (4. * (style_outputs[i].shape[1].value ** 2) * (style_outputs[i].shape[2].value ** 2))) * \
                          K.sum(K.square(self._gram_matrix(style_tensor) - self._gram_matrix(style_outputs[i])))

        # composite loss
        loss = alpha * content_loss + beta * style_loss
        loss_gradient = K.gradients(loss=loss, variables=[vgg19_avg.input])

        # bind functions
        self.loss_function = K.function(inputs=[vgg19_avg.input],
                                        outputs=[loss])
        self.gradient_function = K.function(inputs=[vgg19_avg.input],
                                            outputs=loss_gradient)

    def fit(self, iter_n = 100):
        '''
        generate a style-transfered image
        '''
        # generate a random image (range: 0 ~ 255)
        output_img = utils.preproc(np.random.uniform(0, 256, size=self.img_shape))

        # bounds for clipping
        bounds = np.ndarray(shape=(np.prod(output_img.shape), 2))
        bounds[:, 0] = -128.
        bounds[:, 1] = 127.

        print('Starting optimization with L-BFGS-B')

        for i in range(iter_n):
            if self.verbose:
                print('Starting iteration %d' % (i))

            minimize(fun=self._loss, x0=output_img.flatten(),
                     jac=self._loss_gradient,
                     bounds=bounds, method='L-BFGS-B')

            if i % self.save_every_n_iters == 0:
                utils.save_image(utils.deproc(output_img, self.img_shape),
                                 output_path, 'out_%d' % (i) + '.jpg')

    def _gram_matrix(self, features):
        features = K.batch_flatten(K.permute_dimensions(K.squeeze(features, axis=0),
                                                        pattern=(2, 0, 1)))
        return K.dot(features, K.transpose(features))

    def _loss(self, image):
        loss = self.loss_function([image.reshape(self.img_shape).astype(K.floatx())])[0]

        if self.verbose:
            print('Loss: %f' % (loss))

        return loss

    def _loss_gradient(self, image):
        return np.array(self.gradient_function([image.reshape(self.img_shape).astype(K.floatx())])).astype('float64').flatten()

def main():
    style_transfer = StyleTransfer(content_image_path=content_path,
                                   style_image_path=style_path)
    style_transfer.fit()

if __name__ == '__main__':
    main()
