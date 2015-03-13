import numpy as np

def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) where x[i] is the ith input.
  We multiply this against a weight matrix of shape (D, M) where
  D = \prod_i d_i

  Inputs:
  x - Input data, of shape (N, d_1, ..., d_k)
  w - Weights, of shape (D, M)
  b - Biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
      
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  xr = np.reshape(x, (x.shape[0], -1))
  out = np.dot(xr, w) + b
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  xr = np.reshape(x, (x.shape[0], -1))
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  dx = np.reshape(np.dot(dout, w.T), x.shape)
  dw = np.reshape(np.dot(xr.T, dout), w.shape)
  db = np.sum(dout, 0)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  out = np.maximum(x, 0)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = dout, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  dx[x < 0] = 0
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx

  
def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """

  N, C, H, W = x.shape
  F, C, HH, WW = w.shape

  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  pad = conv_param['pad']
  stride = conv_param['stride']
  
  xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')

  H1 = 1 + (H + 2 * pad - HH) / stride
  W1 = 1 + (W + 2 * pad - WW) / stride
  out = np.zeros((N, F, H1, W1))

  for n in range(N):
    for f in xrange(F):
      for i in xrange(H1):
        for j in xrange(W1):
          out[n, f, i, j] = np.sum(xp[n,  :,
                                    stride*i : (HH + stride*i),
                                    stride*j : (WW + stride*j)] * w[f]) + b[f]

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  
  x, w, b, conv_param = cache
  N, C, H, W = x.shape
  F, C, HH, WW = w.shape

  pad = conv_param['pad']
  stride = conv_param['stride']
  H1 = 1 + (H + 2 * pad - HH) / stride
  W1 = 1 + (W + 2 * pad - WW) / stride

  x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
  I_pad = np.pad(np.ones(x.shape), ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
  dx = np.zeros(x_pad.shape)
  dw = np.zeros(w.shape)
  db = np.zeros(b.shape)
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  for n in range(N):
    for f in xrange(F):
      for i in xrange(H1):
        for j in xrange(W1):
          h1, h2 = stride*i, HH + stride*i
          w1, w2 = stride*j, WW + stride*j
          dx[n, :, h1:h2, w1:w2] += I_pad[n,  :, h1:h2, w1:w2] * w[f] * dout[n, f, i, j]
          dw[f] += x_pad[n, :, h1:h2, w1:w2] * dout[n, f, i, j]
          db[f] += dout[n, f, i, j]
          
  dx = dx[:,:,pad:-pad, pad:-pad]
  
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  N, C, H, W = x.shape

  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  stride = pool_param['stride']
  WW = pool_param['pool_width']
  HH = pool_param['pool_height']  
  
  H1 = 1 + (H - HH) / stride
  W1 = 1 + (W - WW) / stride
  out = np.zeros((N, C, H1, W1))

  for n in range(N):
    for c in xrange(C):
      for i in xrange(H1):
        for j in xrange(W1):
          out[n, c, i, j] = np.max(x[n,  c,
                                    stride*i : (HH + stride*i),
                                    stride*j : (WW + stride*j)])

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache

def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """


  x, pool_param = cache
  N, C, H, W = x.shape

  stride = pool_param['stride']
  WW = pool_param['pool_width']
  HH = pool_param['pool_height']  
  
  H1 = 1 + (H - HH) / stride
  W1 = 1 + (W - WW) / stride

  dx = np.zeros(x.shape)

  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  ############################################################################# 
  for n in range(N):
    for c in xrange(C):
      for i in xrange(H1):
        for j in xrange(W1):
          h1, h2 = stride*i, HH + stride*i
          w1, w2 = stride*j, WW + stride*j
          max_ind = np.max(x[n,  c, h1:h2, w1:w2]) == x[n,  c, h1:h2, w1:w2]
          dx[n, c, h1:h2, w1:w2][max_ind] = dout[n, c, i, j]
                                    
  return dx

def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y, beta):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  num_train = x.shape[0]

  #x -= np.max(x, 0)  # normalize scores

  total = np.sum(np.exp(x), 1)
  p =  (np.exp(x).T / total).T

  
  loss1 = -np.sum(np.log(p)[range(x.shape[0]), y])
  loss2 = -np.sum(np.log(p) * p)
  

  loss = beta * loss1 + (1 - beta) * loss2 
  loss /= num_train

  g1 = np.zeros(x.shape)
  for k in xrange(x.shape[1]):
    for i in xrange(x.shape[0]):
      if y[i] == k:
        g1[i, k] = 1 - p[i, k]
      else:
        g1[i, k] = -p[i, k]


  g2 = np.zeros(x.shape)
  for i in xrange(x.shape[0]):
    r = np.log(p[i]) + 1
    for k  in xrange(x.shape[1]):

      q = - p[i, k] * p[i]
      g2[i, k] = r.dot(q) + r[k] * p[i, k]
      
  grad = beta * g1 + (1 - beta) * g2

  grad /= -num_train
        
  return loss, grad
