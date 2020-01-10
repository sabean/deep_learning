"""Linear Softmax Classifier."""
# pylint: disable=invalid-name
import numpy as np

from .linear_classifier import LinearClassifier


def sigmoid(z):
    """
    Returns sigmoid of z
    """
    return 1/(1 + np.exp(-z))


def softmaxA(z):
    exps = np.exp(z - np.max(z))
    return exps / np.sum(exps)

def softmaxB(z):
    exps = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def cross_entropy_loss_naive(W, X, y, reg):
    """
    Cross-entropy loss function, naive implementation (with loops)

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
    # pylint: disable=too-many-locals
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient using explicit     #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################
    
    N = X.shape[0]
    C = W.shape[1]
    D = W.shape[0]
    # predicted = np.zeros((N,C))
    regualization = reg*(1/N)*np.sum(W**2)
    
   
    for i in range(N):
        prediction = X[i] @ W
        current_exp = np.exp(prediction-np.max(prediction))
        current_sum = np.sum(current_exp)
        softmax = current_exp[y[i]] / current_sum
        loss += -np.log(softmax)

        for j in range(C):
            softmax_int = current_exp[j] / current_sum
            if j == y[i]:
                dW[:, j] += (softmax_int - 1) * X[i]
            else:
                dW[:, j] += (softmax_int - 0) * X[i]
    
    loss *= (1/N) 
    loss += regualization
    dW *= (1/N)
    dW += regualization
    
  
    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################
    
    return loss, dW


def cross_entropy_loss_vectorized(W, X, y, reg):
    """
    Cross-entropy loss function, vectorized version.

    Inputs and outputs are the same as in cross_entropy_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient without explicit   #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################
    
    N = X.shape[0]
    predicted = softmaxB(X @ W)

    intermidiate_loss =  (1/N)*np.sum(-np.log(predicted[np.arange(N), y]))
    regualization = reg*(0.1/N)*np.sum(W**2)
    loss = intermidiate_loss + reg*np.sum(W**2)
    
    zero_index = np.zeros_like(predicted)
    zero_index[np.arange(N), y] = 1
    dW = (1/N) * (X.T @ (predicted - zero_index)) +  reg*W

    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


class SoftmaxClassifier(LinearClassifier):
    """The softmax classifier which uses the cross-entropy loss."""

    def loss(self, X_batch, y_batch, reg):
        return cross_entropy_loss_vectorized(self.W, X_batch, y_batch, reg)


def softmax_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    # results is dictionary mapping tuples of the form
    # (learning_rate, regularization_strength) to tuples of the form
    # (training_accuracy, validation_accuracy). The accuracy is simply the
    # fraction of data points that are correctly classified.
    results = {}
    best_val = -1
    best_softmax = None
    all_classifiers = []

    learning_rates = [1e-6, 3e-7, 5e-8]
    regularization_strengths =[1e3, 3e4, 5e5]

    ############################################################################
    # TODO:                                                                    #
    # Write code that chooses the best hyperparameters by tuning on the        #
    # validation set. For each combination of hyperparameters, train a         #
    # classifier on the training set, compute its accuracy on the training and #
    # validation sets, and  store these numbers in the results dictionary.     #
    # In addition, store the best validation accuracy in best_val and the      #
    # Softmax object that achieves this accuracy in best_softmax.              #                                      #
    #                                                                          #
    # Hint: You should use a small value for num_iters as you develop your     #
    # validation code so that the classifiers don't take much time to train;   # 
    # once you are confident that your validation code works, you should rerun #
    # the validation code with a larger value for num_iters.                   #
    ############################################################################
   
    for learn_unit in learning_rates:
        for reg_unit in regularization_strengths:
            softmax = SoftmaxClassifier()
            softmax.train(X_train, y_train, learning_rate=learn_unit, reg=reg_unit, num_iters=1000, verbose=True)
            training_accuracy = np.mean(y_train == softmax.predict(X_train)) 
            validation_accuracy = np.mean(y_val == softmax.predict(X_val))
            
            if validation_accuracy > best_val:
                best_val = validation_accuracy
                best_softmax = softmax
                
            results[(learn_unit, reg_unit)] = (training_accuracy, validation_accuracy)
            all_classifiers.append((softmax, validation_accuracy))
            
    ############################################################################
    #                              END OF YOUR CODE                            #
    ############################################################################
        
    # Print out results.
    for (lr, reg) in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
              lr, reg, train_accuracy, val_accuracy))
        
    print('best validation accuracy achieved during validation: %f' % best_val)

    return best_softmax, results, all_classifiers
