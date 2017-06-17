import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(X.shape[0]):
      score = X[i].T.dot(W)
      sum_exp_score = np.sum(np.exp(score))
      loss += np.log(sum_exp_score)-score[y[i]] 
      for j in range(W.shape[1]):
          dW[:,j] += X[i]*np.exp(score[j])/sum_exp_score
      dW[:,y[i]] -= X[i]
      
  loss /= X.shape[0]
  loss += reg*np.sum(W * W)
  
  dW /= X.shape[0]
  dW += 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


  
  
def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  score = X.dot(W)
  loss += np.sum(np.log(np.sum(np.exp(score),axis=1)),axis=0)-np.sum(score[range(num_train),y])
  loss /= num_train
  loss += reg*np.sum(W * W)
  
  temp = np.zeros(score.shape)
  temp[range(num_train),y]=1
  
  
  dW += X.T.dot(np.exp(score)/np.sum(np.exp(score),axis=1)[:,None])
  dW -= X.T.dot(temp) 
  dW /= X.shape[0]
  dW += 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

