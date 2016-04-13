import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, use_batchnorm=None,weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.use_batchnorm = use_batchnorm
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        F = num_filters
        HH = filter_size
        pad = (filter_size - 1) / 2
        C, H, W = input_dim
        input_dims = F * (1 + (H + 2 * pad - HH) / 1) ** 2 / 4
        self.params['W1'] = weight_scale * np.random.randn(F, C, HH, HH)
        self.params['b1'] = np.zeros(F)
        self.params['W2'] = weight_scale * np.random.randn(input_dims, hidden_dim)
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b3'] = np.zeros(num_classes)
        if self.use_batchnorm:
            self.params['gamma'] = np.ones(hidden_dim)
            self.params['beta'] = np.zeros(hidden_dim)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        mode = 'test' if y is None else 'train'
        if self.use_batchnorm:
            gamma,beta = self.params['gamma'],self.params['beta']
            bn_param = {'mode':mode}

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        conv_out, conv_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        if self.use_batchnorm:
            affine_relu_out, affine_bn_relu_cache = \
                affine_batchnorm_relu_forward(conv_out,W2,b2,gamma,beta,bn_param)
        else:
            affine_relu_out, affine_relu_cache = affine_relu_forward(conv_out, W2, b2)
        scores, affine_cache = affine_forward(affine_relu_out, W3, b3)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        loss, dscores = softmax_loss(scores, y)
        loss += 0.5 * self.reg * np.sum(self.params['W3'] * self.params['W3'])
        loss += 0.5 * self.reg * np.sum(self.params['W2'] * self.params['W2'])
        loss += 0.5 * self.reg * np.sum(self.params['W1'] * self.params['W1'])

        daffine, grads['W3'], grads['b3'] = affine_backward(dscores, affine_cache)
        grads['W3'] += self.reg * self.params['W3']

        if self.use_batchnorm:
            daffine_relu, grads['W2'], grads['b2'], grads['gamma'],grads['beta'] = \
                affine_batchnorm_relu_backward(daffine, affine_bn_relu_cache)
        else:
            daffine_relu, grads['W2'], grads['b2']= \
                affine_relu_backward(daffine, affine_relu_cache)
        grads['W2'] += self.reg * self.params['W2']

        dx, grads['W1'], grads['b1'] = conv_relu_pool_backward(daffine_relu, conv_cache)
        grads['W1'] += self.reg * self.params['W1']
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


def affine_batchnorm_relu_forward(x, w, b, gamma, beta, params):
    affine_out, affine_cache = affine_forward(x, w, b)
    norm_out, batchnorm_cache = batchnorm_forward(affine_out, gamma, beta, params)
    out, relu_cache = relu_forward(norm_out)
    cache = (affine_cache, batchnorm_cache, relu_cache)
    return out, cache


def affine_batchnorm_relu_backward(dout, cache):
    affine_cache, batchnorm_cache, relu_cache = cache
    drelu = relu_backward(dout, relu_cache)
    dnorm, dgamma, dbeta = batchnorm_backward_alt(drelu, batchnorm_cache)
    dout, dw, db = affine_backward(dnorm, affine_cache)
    return dout, dw, db, dgamma, dbeta
