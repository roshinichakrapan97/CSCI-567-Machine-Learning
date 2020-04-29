import numpy as np


def perceptron_loss_func(X, y, w):
    matrix = -y * np.dot(X, w)
    matrix = (matrix>=0).astype(int)
    size = X.shape[0]
    dot_prod = np.dot(X.T, (matrix * y))
    return dot_prod / size


def logistic_loss_func(X, y, w):
    sigmoid_mat = sigmoid(-y * np.dot(X, w))
    size = X.shape[0]
    dot_prod = np.dot(X.T, (sigmoid_mat * y))
    return dot_prod / size


def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
        Inputs:
        - X: training features, a N-by-D numpy array, where N is the
        number of training points and D is the dimensionality of features
        - y: binary training labels, a N dimensional numpy array where
        N is the number of training points, indicating the labels of
        training data
        - loss: loss type, either perceptron or logistic
        - step_size: step size (learning rate)
        - max_iterations: number of iterations to perform gradient descent

        Returns:
        - w: D-dimensional vector, a numpy array which is the weight
        vector of logistic or perceptron regression
        - b: scalar, which is the bias of logistic or perceptron regression
        """
    N, D = X.shape
    assert len(np.unique(y)) == 2
    y[y == 0] = -1

    w = np.zeros(D)
    if w0 is not None:
        w = w0

    b = 0
    if b0 is not None:
        b = b0

    X = np.insert(X, 0, 1, axis=1)
    w = np.insert(w, 0, b)
    i=0
    if loss == "perceptron":
        ############################################
        # TODO 1 : Edit this if part               #
        #          Compute w and b here            #
        ############################################
        while i < max_iterations:
            w = w + (step_size * perceptron_loss_func(X, y, w))
            i+=1


    elif loss == "logistic":
        ############################################
        # TODO 2 : Edit this if part               #
        #          Compute w and b here            #
        ############################################
        while i < max_iterations:
            w = w + (step_size * logistic_loss_func(X, y, w))
            i+=1
    else:
        raise "Loss Function is undefined."

    b = w[0]
    w = w[1:]
    assert w.shape == (D,)
    return w, b


def sigmoid(z):
    """
    Inputs:
    - z: a numpy array or a float number

    Returns:
    - value: a numpy array or a float number after computing sigmoid function value = 1/(1+exp(-z)).
    """

    ############################################
    # TODO 3 : Edit this part to               #
    #          Compute value                   #
    value = (1 + np.exp(-z)) ** -1
    ############################################

    return value


def binary_predict(X, w, b, loss="perceptron"):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - w: D-dimensional vector, a numpy array which is the weight
    vector of your learned model
    - b: scalar, which is the bias of your model
    - loss: loss type, either perceptron or logistic

    Returns:
    - preds: N dimensional vector of binary predictions: {0, 1}
    """

    N, D = X.shape
    X = np.insert(X, 0, 1, axis=1)
    w = np.insert(w, 0, b)

    if loss == "perceptron":
        ############################################
        # TODO 4 : Edit this if part               #
        #          Compute preds                   #
        preds = X.dot(w)
        preds = (preds>=0).astype(int)


        ############################################


    elif loss == "logistic":
        ############################################
        # TODO 5 : Edit this if part               #
        #          Compute preds                   #
        preds = X.dot(w)
        preds = sigmoid(preds)
        preds = (preds >= 0.5).astype(int)

        ############################################


    else:
        raise "Loss Function is undefined."

    assert preds.shape == (N,)
    return preds




def softmax_function(z):
    return (np.exp(z.T) / np.sum(np.exp(z), axis=1)).T


def sgd_loss_func(X, y, w):
    z = w.dot(X)
    z = z - max(z)
    sum_val = np.sum(np.exp(z))
    softmax_val = np.exp(z) / sum_val
    softmax_val[y] -= 1
    return np.outer(softmax_val, X)


def gradient_loss_func(X, yy, w):
    z = X.dot(w.T)
    softmax_val = (softmax_function(z) - yy)
    size = X.shape[0]
    return softmax_val.T.dot(X) / size


def multiclass_train(X, y, C,
                     w0=None,
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5,
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of
    training data
    - C: number of classes in the data
    - gd_type: gradient descent type, either GD or SGD
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: C-by-D weight matrix of multinomial logistic regression, where
    C is the number of classes and D is the dimensionality of features.
    - b: bias vector of length C, where C is the number of classes
    """

    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0

    b = np.zeros(C)
    if b0 is not None:
        b = b0
    w = np.insert(w, w.shape[1], b, axis=1)
    X = np.insert(X, X.shape[1], 1, axis=1)

    np.random.seed(42)
    i =0
    if gd_type == "sgd":
        ############################################
        # TODO 6 : Edit this if part               #
        #          Compute w and b                 #
        while i < max_iterations:
            ind = np.random.choice(X.shape[0])
            w = w - (step_size * sgd_loss_func(X[ind], y[ind], w))
            i+=1
        ############################################


    elif gd_type == "gd":
        ############################################
        # TODO 7 : Edit this if part               #
        #          Compute w and b                 #
        yy = np.zeros((N, C))
        for ind, opt in enumerate(y):
            yy[ind][opt] = 1
        while i < max_iterations:
            w = w - (step_size * gradient_loss_func(X, yy, w))
            i+=1

        ############################################


    else:
        raise "Type of Gradient Descent is undefined."

    b = w[:,w.shape[1] - 1]
    w = w[:,:w.shape[1] - 1]
    assert w.shape == (C, D)
    assert b.shape == (C,)

    return w, b



def multiclass_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - w: weights of the trained multinomial classifier, C-by-D
    - b: bias terms of the trained multinomial classifier, length of C

    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes
    """
    N, D = X.shape
    ############################################
    # TODO 8 : Edit this part to               #
    #          Compute preds                   #
    class_predictions = np.zeros(N)
    ############################################




    X = np.insert(X, 0, 1, axis=1)
    w = np.insert(w, 0, b, axis=1)
    z = X.dot(w.T)
    softmax_value = softmax_function(z)
    class_predictions = softmax_value.argmax(axis=1)
    assert class_predictions.shape == (N,)
    return class_predictions


