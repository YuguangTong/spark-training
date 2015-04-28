import numpy as np
import cPickle as pickle
from classifier import Classifier
from util.layers import *
from util.dump import *

""" STEP2: Build Two-layer Fully-Connected Neural Network """

class NNClassifier(Classifier):
  def __init__(self, D, H, W, K, iternum):
    Classifier.__init__(self, D, H, W, K, iternum)
    self.L = 100 # size of hidden layer

    """ Layer 1 Parameters """
    # weight matrix: [M * L]
    self.A1 = 0.01 * np.random.randn(self.M, self.L)
    # bias: [1 * L]
    self.b1 = np.zeros((1,self.L))

    """ Layer 3 Parameters """
    # weight matrix: [L * K]
    self.A3 = 0.01 * np.random.randn(self.L, K)
    # bias: [1 * K]
    self.b3 = np.zeros((1,K))

    """ Hyperparams """
    # learning rate
    self.rho = 1e-2
    # momentum
    self.mu = 0.9
    # reg strencth
    self.lam = 0.1
    # velocity for A1: [M * L]
    self.v1 = np.zeros((self.M, self.L))
    # velocity for A3: [L * K] 
    self.v3 = np.zeros((self.L, K))
    return

  def load(self, path):
    data = pickle.load(open(path + "layer1"))
    assert(self.A1.shape == data['w'].shape)
    assert(self.b1.shape == data['b'].shape)
    self.A1 = data['w']
    self.b1 = data['b']
    data = pickle.load(open(path + "layer3"))
    assert(self.A3.shape == data['w'].shape)
    assert(self.b3.shape == data['b'].shape)
    self.A3 = data['w']
    self.b3 = data['b']
    return

  def param(self):
    return [("A1", self.A1), ("b1", self.b1), ("A3", self.A3), ("b3", self.b3)]

  def forward(self, data):
    """
    INPUT:
      - data: RDD[(key, (images, labels)) pairs]
    OUTPUT:
      - RDD[(key, (images, list of layers, labels)) pairs]
    """
    """
    TODO: Implement the forward passes of the following layers
    Layer 1 : linear
    Layer 2 : ReLU
    Layer 3 : linear
    """
    bcA1 = data.context.broadcast(self.A1)
    bcb1 = data.context.broadcast(self.b1)
    bcA3 = data.context.broadcast(self.A3)
    bcb3 = data.context.broadcast(self.b3)

    def helper((k, (x, y))):
      layer1 = linear_forward(x, bcA1.value, bcb1.value)
      layer2 = ReLU_forward(layer1)
      layer3 = linear_forward(layer2, bcA3.value, bcb3.value)
      return (k, (x, [layer1, layer2, layer3], y))
    return data.map(helper) # replace it with your code

  def backward(self, data, count):
    """
    INPUT:
      - data: RDD[(images, list of layers, labels) pairs]
    OUTPUT:
      - loss
    """
    """ 
    TODO: Implement softmax loss layer 
    """
    bcA1 = data.context.broadcast(self.A1)
    bcA3 = data.context.broadcast(self.A3)

    softmax = data.map(lambda (x, layers, y): (x, layers, y, softmax_loss(layers[2], y))) \
                  .map(lambda (x, layers, y, (L, df)): (x, layers, y, (L/count, df/count)))
    """
    TODO: Compute the loss
    """


    #L = 0.0 # replace it with your code
    L = softmax.map(lambda (x, layers, y, (l, dldl3)): l).reduce(lambda a, b: a + b)

    """ regularization """
    L += 0.5 * self.lam * (np.sum(self.A1*self.A1) + np.sum(self.A3*self.A3))

    """ Todo: Implement backpropagation for Layer 3 """
    backward3 = softmax.map(lambda (x, layers, y, (l, dldl3)):
                            (x, layers, y, linear_backward(dldl3, layers[1], bcA3.value)))

    """ Todo: Compute the gradient on A3 and b3 """
    #dLdA3 = np.zeros(self.A3.shape) # replace it with your code
    dLdA3 = backward3.map(lambda (x, layers, y, (dldl2, dlda3, dldb3)): dlda3) \
                     .reduce(lambda a, b: a+b)

    #dLdb3 = np.zeros(self.b3.shape) # replace it with your code
    dLdb3 = backward3.map(lambda (x, layers, y, (dldl2, dlda3, dldb3)): dldb3) \
                     .reduce(lambda a, b: a + b)
    """ Todo: Implement backpropagation for Layer 2 """
    backward2 = backward3.map(lambda (x, layers, y, (dldl2, dlda3, dldb3)):
                              (x, layers, y, ReLU_backward(dldl2, layers[0])))
    """ Todo: Implmenet backpropagation for Layer 1 """
    backward1 = backward2.map(lambda (x, layers, y, dldl1):
                              linear_backward(dldl1, x, bcA1.value))

    """ Todo: Compute the gradient on A1 and b1 """
    #dLdA1 = np.zeros(self.A1.shape) # replace it with your code
    dLdA1 = backward1.map(lambda (dldx, dlda1, dldb1): dlda1) \
                     .reduce(lambda a, b: a + b)
    #dLdb1 = np.zeros(self.b1.shape) # replace it with your code
    dLdb1 = backward1.map(lambda (dldx, dlda1, dldb1): dldb1) \
                     .reduce(lambda a, b: a + b)

    """ regularization gradient """
    dLdA3 = dLdA3.reshape(self.A3.shape)
    dLdA1 = dLdA1.reshape(self.A1.shape)
    dLdA3 += self.lam * self.A3
    dLdA1 += self.lam * self.A1

    """ tune the parameter """
    self.v1 = self.mu * self.v1 - self.rho * dLdA1
    self.v3 = self.mu * self.v3 - self.rho * dLdA3
    self.A1 += self.v1
    self.A3 += self.v3
    self.b1 += - self.rho * dLdb1
    self.b3 += - self.rho * dLdb3

    return L
