import numpy as np
import cPickle as pickle
from time import time
from classifier import Classifier
from util.layers import *
from util.dump import *

""" STEP3: Build Deep Convolutional Neural Network """

class CNNClassifier(Classifier):
  def __init__(self, D, H, W, K, iternum):
    Classifier.__init__(self, D, H, W, K, iternum)

    """ 
    Layer 1 Parameters (Conv 32 x 32 x 16) 
    K = 16, F = 5, S = 1, P = 2
    weight matrix: [K1 * D * F1 * F1]
    bias: [K1 * 1]
    """
    K1, F1, self.S1, self.P1 = 16, 5, 1, 2
    self.A1 = 0.01 * np.random.randn(K1, D, F1, F1)
    self.b1 = np.zeros((K1, 1))
    H1 = (H - F1 + 2*self.P1) / self.S1 + 1
    W1 = (W - F1 + 2*self.P1) / self.S1 + 1

    """ 
    Layer 3 Parameters (Pool 16 x 16 x 16) 
    K = 16, F = 2, S = 2
    """
    K3, self.F3, self.S3 = K1, 2, 2
    H3 = (H1 - self.F3) / self.S3 + 1
    W3 = (W1 - self.F3) / self.S3 + 1
 
    """ 
    Layer 4 Parameters (Conv 16 x 16 x 20) 
    K = 20, F = 5, S = 1, P = 2
    weight matrix: [K4 * K3 * F4 * F4]
    bias: [K4 * 1]
    """
    K4, F4, self.S4, self.P4 = 20, 5, 1, 2
    self.A4 = 0.01 * np.random.randn(K4, K3, F4, F4)
    self.b4 = np.zeros((K4, 1))
    H4 = (H3 - F4 + 2*self.P4) / self.S4 + 1
    W4 = (W3 - F4 + 2*self.P4) / self.S4 + 1

    """ 
    Layer 6 Parameters (Pool 8 x 8 x 20) 
    K = 20, F = 2, S = 2
    """
    K6, self.F6, self.S6 = K4, 2, 2
    H6 = (H4 - self.F6) / self.S6 + 1
    W6 = (W4 - self.F6) / self.S6 + 1

    """ 
    Layer 7 Parameters (Conv 8 x 8 x 20) 
    K = 20, F = 5, S = 1, P = 2
    weight matrix: [K7 * K6 * F7 * F7]
    bias: [K7 * 1]
    """
    K7, F7, self.S7, self.P7 = 20, 5, 1, 2
    self.A7 = 0.01 * np.random.randn(K7, K6, F7, F7)
    self.b7 = np.zeros((K7, 1))
    H7 = (H6 - F7 + 2*self.P7) / self.S7 + 1
    W7 = (W6 - F7 + 2*self.P7) / self.S7 + 1

    """ 
    Layer 9 Parameters (Pool 4 x 4 x 20) 
    K = 20, F = 2, S = 2
    """
    K9, self.F9, self.S9 = K7, 2, 2
    H9 = (H7 - self.F9) / self.S9 + 1
    W9 = (W7 - self.F9) / self.S9 + 1

    """ 
    Layer 10 Parameters (FC 1 x 1 x K)
    weight matrix: [(K6 * H_6 * W_6) * K] 
    bias: [1 * K]
    """
    self.A10 = 0.01 * np.random.randn(K9 * H9 * W9, K)
    self.b10 = np.zeros((1, K))

    """ Hyperparams """
    # learning rate
    self.rho = 1e-2
    # momentum
    self.mu = 0.9
    # reg strength
    self.lam = 0.1
    # velocity for A1: [K1 * D * F1 * F1]
    self.v1 = np.zeros((K1, D, F1, F1))
    # velocity for A4: [K4 * K3 * F4 * F4]
    self.v4 = np.zeros((K4, K3, F4, F4))
    # velocity for A7: [K7 * K6 * F7 * F7]
    self.v7 = np.zeros((K7, K6, F7, F7))
    # velocity for A10: [(K9 * H9 * W9) * K]   
    self.v10 = np.zeros((K9 * H9 * W9, K))
 
    return

  def load(self, path):
    data = pickle.load(open(path + "layer1"))
    assert(self.A1.shape == data['w'].shape)
    assert(self.b1.shape == data['b'].shape)
    self.A1 = data['w']
    self.b1 = data['b'] 
    data = pickle.load(open(path + "layer4"))
    assert(self.A4.shape == data['w'].shape)
    assert(self.b4.shape == data['b'].shape)
    self.A4 = data['w']
    self.b4 = data['b']
    data = pickle.load(open(path + "layer7"))
    assert(self.A7.shape == data['w'].shape)
    assert(self.b7.shape == data['b'].shape)
    self.A7 = data['w']
    self.b7 = data['b']
    data = pickle.load(open(path + "layer10"))
    assert(self.A10.shape == data['w'].shape)
    assert(self.b10.shape == data['b'].shape)
    self.A10 = data['w']
    self.b10 = data['b']
    return 

  def param(self):
    return [
      ("A10", self.A10), ("b10", self.b10),
      ("A7", self.A7), ("b7", self.b7), 
      ("A4", self.A4), ("b4", self.b4), 
      ("A1", self.A1), ("b1", self.b1)] 

  def forward(self, data):
    """
    INPUT:
      - data: RDD[(key, (images, labels)) pairs]
    OUTPUT:
      - RDD[(key, (images, list of layers, labels)) pairs]
    """

    """ TODO: Layer1: Conv (32 x 32 x 16) forward """
    def forwardHelper1((k, (x, y))):
      layer1, X_col1 = conv_forward(x, self.A1, self.b1, self.S1, self.P1)
      return (k, (x, [(layer1, X_col1)], y))
    forward1 = data.map(forwardHelper1)

    """ TODO: Layer2: ReLU (32 x 32 x 16) forward """
    forward2 = forward1.map(lambda (k, (x, layers, y)):
                             (k, (x, layers + [ReLU_forward(layers[0][0])], y)))
    """ DOTO: Layer3: Pool (16 x 16 x 16) forward """
    def forwardHelper3((k, (x, layers, y))):
      layer3, X_idx3 = max_pool_forward(layers[1], self.F3, self.S3)
      return (k, (x, layers + [(layer3, X_idx3)], y))
    forward3 = forward2.map(forwardHelper3)
    """ TODO: Layer4: Conv (16 x 16 x 20) forward """ 
    def forwardHelper4((k, (x, layers, y))):
      layer4, X_col4 = conv_forward(layers[2][0], self.A4, self.b4, self.S4, self.P4)
      return (k, (x, layers + [(layer4, X_col4)], y))
    forward4 = forward3.map(forwardHelper4)
    """ TODO: Layer5: ReLU (16 x 16 x 20) forward """
    forward5 = forward4.map(lambda (k, (x, layers, y)):
                            (k, (x, layers + [ReLU_forward(layers[3][0])], y)))
    """ TODO: Layer6: Pool (8 x 8 x 20) forward """ 
    def forwardHelper6((k, (x, layers, y))):
      layer6, X_idx6 = max_pool_forward(layers[4], self.F6, self.S6)
      return (k, (x, layers + [(layer6, X_idx6)], y))
    forward6 = forward5.map(forwardHelper6)
    """ TODO: Layer7: Conv (8 x 8 x 20) forward """ 
    def forwardHelper7((k, (x, layers, y))):
      layer7, X_col7 = conv_forward(layers[5][0], self.A7, self.b7, self.S7, self.P7)
      return (k, (x, layers + [(layer7, X_col7)], y))
    forward7 = forward6.map(forwardHelper7)
    
    """ TODO: Layer8: ReLU (8 x 8 x 20) forward """ 
    forward8 = forward7.map(lambda (k, (x, layers, y)):
                            (k, (x, layers + [ReLU_forward(layers[6][0])], y)))
    """ TODO: Layer9: Pool (4 x 4 x 20) forward """ 
    def forwardHelper9((k, (x, layers, y))):
      layer9, X_idx9 = max_pool_forward(layers[7], self.F9, self.S9)
      return (k, (x, layers + [(layer9, X_idx9)], y))
    forward9 = forward8.map(forwardHelper9)

    """ TODO: Layer10: FC (1 x 1 x 10) forward """
    forward10 = forward9.map(lambda (k, (x, layers, y)):
                             (k, (x, layers + [linear_forward(layers[8][0], self.A10, self.b10)], y)))
    #return data.map(lambda (k, (x, y)): (k, (x, [(np.array([0]), np.array([0])), np.zeros((x.shape[0], 2))], y))) # replace it with your code
    return forward10

  def backward(self, data, count):
    """
    INPUT:
      - data: RDD[(images, list of layers, labels) pairs]
    OUTPUT:
      - Loss
    """

    """ TODO: Softmax Loss Layer """ 
    softmax = data.map(lambda (x, layers, y):
                       (x, layers, y, softmax_loss(layers[9], y))) \
                  .map(lambda (x, layers, y, (l, dldl10)):
                       (x, layers, y, (l/count, dldl10/count)))
      

    """ TODO: Compute Loss """
    #L = 0.0 # replace it with your code
    L = softmax.map(lambda (x, layers, y, (l, dldl10)): l) \
               .reduce(lambda a, b: a + b)

    """ regularization """
    L += 0.5 * self.lam * np.sum(self.A1*self.A1)
    L += 0.5 * self.lam * np.sum(self.A4*self.A4)
    L += 0.5 * self.lam * np.sum(self.A7*self.A7)
    L += 0.5 * self.lam * np.sum(self.A10*self.A10)

    """ TODO: Layer10: FC (1 x 1 x 10) Backward """
    def  backHelper10((x, layers, y, (l, dldl10))):
      dldl9, dlda10, dldb10 = linear_backward(dldl10, layers[8][0], self.A10)
      return (x, layers, y, (dldl9, dlda10, dldb10))
    backward10 = softmax.map(backHelper10)
    """ TODO: gradients on A10 & b10 """
    #dLdA10 = np.zeros(self.A10.shape) # replace it with your code
    dLdA10 = backward10.map(lambda (x, layers, y, dl): dl[1]) \
                       .reduce(lambda a, b: a + b)
    #dLdb10 = np.zeros(self.b10.shape) # replace it with your code
    dLdb10 = backward10.map(lambda (x, layers, y, dl): dl[2]) \
                       .reduce(lambda a, b: a + b)

    """ TODO: Layer9: Pool (4 x 4 x 20) Backward """
    backward9 = backward10.map(lambda (x, layers, y, dl):
                               (x, layers, y, max_pool_backward(dl[0], layers[7], layers[8][1], self.F9, self.S9)))
    """ TODO: Layer8: ReLU (8 x 8 x 20) Backward """
    backward8 = backward9.map(lambda (x, layers, y, dldl8):
                              (x, layers, y, ReLU_backward(dldl8, layers[6][0])))
    """ TODO: Layer7: Conv (8 x 8 x 20) Backward """
    backward7 = backward8.map(lambda (x, layers, y, dldl7):
                              (x, layers, y, conv_backward(dldl7, layers[5][0], layers[6][1], self.A7, self.S7, self.P7)))
    """ TODO: gradients on A7 & b7 """
    #dLdA7 = np.zeros(self.A7.shape) # replace it with your code
    dLdA7 = backward7.map(lambda (x, layers, y, dl): dl[1]) \
                     .reduce(lambda a, b: a + b)
    #dLdb7 = np.zeros(self.b7.shape) # replace it with your code
    dLdb7 = backward7.map(lambda (x, layers, y, dl): dl[2]) \
                     .reduce(lambda a, b: a + b)
 
    """ TODO: Layer6: Pool (8 x 8 x 20) Backward """
    backward6 = backward7.map(lambda (x, layers, y, dl):
                              (x, layers, y, max_pool_backward(dl[0], layers[4], layers[5][1], self.F6, self.S6)))
    """ TODO: Layer5: ReLU (16 x 16 x 20) Backward """ 
    backward5 = backward6.map(lambda (x, layers, y, dldl5):
                              (x, layers, y, ReLU_backward(dldl5, layers[3][0])))
    """ TODO: Layer4: Conv (16 x 16 x 20) Backward """ 
    backward4 = backward5.map(lambda (x, layers, y, dldl4):
                              (x, layers, y, conv_backward(dldl4, layers[2][0], layers[3][1], self.A4, self.S4, self.P4)))
    """ TODO: gradients on A4 & b4 """
    #dLdA4 = np.zeros(self.A4.shape) # replace it with your code
    dLdA4 = backward4.map(lambda (x, layers, y, dl): dl[1]) \
                     .reduce(lambda a, b: a + b)
    #dLdb4 = np.zeros(self.b4.shape) # replace it with your code
    dLdb4 = backward4.map(lambda (x, layers, y, dl): dl[2]) \
                     .reduce(lambda a, b: a + b)
    """ TODO: Layer3: Pool (16 x 16 x 16) Backward """ 
    backward3 = backward4.map(lambda (x, layers, y, dl):
                              (x, layers, y, max_pool_backward(dl[0], layers[1], layers[2][1], self.F3, self.S3)))
    """ TODO: Layer2: ReLU (32 x 32 x 16) Backward """
    backward2 = backward3.map(lambda (x, layers, y, dldl2):
                              (x, layers, y, ReLU_backward(dldl2, layers[0][0])))
    """ TODO: Layer1: Conv (32 x 32 x 16) Backward """
    backward1 = backward2.map(lambda (x, layers, y, dldl1):
                              (x, layers, y, conv_backward(dldl1, x, layers[0][1], self.A1, self.S1, self.P1)))
    """ TODO: gradients on A1 & b1 """
    #dLdA1 = np.zeros(self.A1.shape) # replace it with your code
    dLdA1 = backward1.map(lambda (x, layers, y, dl): dl[1]) \
                    .reduce(lambda a, b: a + b)
    #dLdb1 = np.zeros(self.b1.shape) # replace it with your code
    dLdb1 = backward1.map(lambda (x, layers, y, dl): dl[2]) \
                    .reduce(lambda a, b: a + b)
    """ regularization gradient """
    dLdA10 = dLdA10.reshape(self.A10.shape)
    dLdA7 = dLdA7.reshape(self.A7.shape)
    dLdA4 = dLdA4.reshape(self.A4.shape)
    dLdA1 = dLdA1.reshape(self.A1.shape)
    dLdA10 += self.lam * self.A10
    dLdA7 += self.lam * self.A7
    dLdA4 += self.lam * self.A4
    dLdA1 += self.lam * self.A1

    """ tune the parameter """
    self.v1 = self.mu * self.v1 - self.rho * dLdA1
    self.v4 = self.mu * self.v4 - self.rho * dLdA4
    self.v7 = self.mu * self.v7 - self.rho * dLdA7
    self.v10 = self.mu * self.v10 - self.rho * dLdA10
    self.A1 += self.v1
    self.A4 += self.v4 
    self.A7 += self.v7
    self.A10 += self.v10
    self.b1 += -self.rho * dLdb1
    self.b4 += -self.rho * dLdb4
    self.b7 += -self.rho * dLdb7
    self.b10 += -self.rho * dLdb10

    return L

