import numpy as np
import h5py
from dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward, affine, affine_backward

np.random.seed(1)


def initialize_parameters_deep(layer_dims, sigma=0.01):
    """
    Initialize the weights and biases in the L layers (1~L)
    :param sigma: standard deviation for initialization
    :param layer_dims: dimension of the L+1 layers (0~L layers), layer 0 is input layer, layer L is output layer
    :return: parameters -- a dictionary containing {W1, b1, W1, b2, ..., WL, bL}
    """
    np.random.seed(1)
    L = len(layer_dims) - 1  # number of layers, excluding layer 0
    parameters = {}

    # loop over layer 1 ~ L
    for l in range(1, L + 1):
        # print("/np.sqrt(layer_dims[l-1]) = " + str(1/np.sqrt(layer_dims[l-1])) + ".\n")
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) / np.sqrt(
            layer_dims[l - 1])  # Xavier initialization
        # parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.1
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters


def linear_forward(A_prev, W, b):
    """
    Perform linear mapping from the previous layer output A[l-1] to Z[l] for one layer
    :param A_prev: previous layer activation output
    :param W: weights of layer l
    :param b: bias of layer l
    :return: Z, linear_cache -- Z is the linear output, linear_cache contains the input of linear function
    """
    Z = np.dot(W, A_prev) + b
    linear_cache = (A_prev, W, b)

    return Z, linear_cache


def linear_activation_forward(A_prev, W, b, activation):
    """
    Perform nonlinear activation mapping from Z[l] to A[l] for layer l
    :param A_prev: previous layer activation output
    :param W: weights of layer l
    :param b: bias of layer l
    :param activation: a string specifying the activation function, "relu" or "sigmoid" or "affine"
    :return: A, cache -- activation output A[l], and cache = {linear_cache, activation_cache}
    """
    Z, linear_cache = linear_forward(A_prev, W, b)
    if activation == "relu":
        A, activation_cache = relu(Z)  # relu(Z) return A, cache, where cache = Z
    elif activation == "sigmoid":
        A, activation_cache = sigmoid(Z)  # sigmoid(Z) returns A, cache, where cache = Z
    elif activation == "affine":
        A, activation_cache = affine(Z)  # linear(Z) returns A, cache, where cache = Z
    else:
        print("Warning: The activation function is not recognized, the activation output is set to zero")
        A = np.zeros(Z.shape)  # when input is a non-exist activation function, set A[l] to zero
        activation_cache = Z

    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X, parameters):
    """
    Perform forward propagation for the entire network
    :param X: Input data
    :param parameters:  weights and biases for all the layers
    :return: AL, caches -- AL is the output of the last layer, also referred to as Y_hat, caches stores all the caches
    """
    L = len(parameters) // 2  # number of layers (excluding layer 0)
    A = X
    caches = []

    # loop over layer 1 ~ L-1 for the L-1 relu layers
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], 'relu')
        caches.append(cache)

    # for the output layer
    A_prev = A
    # the final layer is linear layer
    AL, cache = linear_activation_forward(A_prev, parameters["W" + str(L)], parameters["b" + str(L)], 'affine')
    caches.append(cache)

    return AL, caches


def compute_cost(AL, S, F, R, gamma):
    """
    Compute the cost with self-defined function
    :param F:       receive sample matrix NrxNxLs
    :param gamma:   nuclear norm constant
    :param S:   complex pilot symbols of shape Nt x Ls
    :param R:   complex received symbols of shape N x Ls x m
    :param AL:  output of the last layer, i.e., h_hat = vec( [Re(H_tilde) Im(H_tilde)] ), of shape (2*Nr*Nt) x m
    :return:    cost -- the cross-entropy
    """
    m = R.shape[2]
    Nr = R.shape[0]  # Nr = N here
    Nt = S.shape[0]
    Ls = R.shape[1]
    cost = 0
    H_hat, _ = channel_vec2mat(AL, Nr, Nt)  # Nr x Nt x m channel samples
    for i in range(m):
        LS = 0
        Ri = R[:, :, i]     # ith received matrix Y
        H_hat_i = H_hat[:, :, i]
        for t in range(Ls):
            Ft = F[:, :, t]
            st = S[:, t]
            rt = Ri[:, t]
            LS = LS + np.power(np.linalg.norm(rt - np.dot(Ft.conj().T, np.dot(H_hat_i, st))), 2)

        cost += 1/2 * LS + gamma * np.linalg.norm(H_hat_i, ord='nuc')
    cost = 1 / m * cost
    cost = float(np.squeeze(cost))
    assert (isinstance(cost, float))

    return cost


def linear_backward(dZ, linear_cache):
    """
    Compute the gradient dW, db, dA for the linear unit in layer l, assume dZ is computed
    :param dZ:  dZ[l]
    :param linear_cache: A[l-1], W[l]
    :return: dW, dB, dA_prev
    """
    A_prev, W, b = linear_cache
    m = A_prev.shape[1]

    dW = 1 / m * np.dot(dZ, A_prev.T)
    db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    """
    Compute the back propagation for one layer
    :param dA:  output dA[l] of layer l+1
    :param cache: (linear_cache, activation_cache) of layer l in forward propagation
    :param activation:  activation function, "relu" or "sigmoid"
    :return: dW, db, dA_prev
    """
    linear_cache, activation_cache = cache
    Z = activation_cache
    if activation == "relu":
        # gZ_derivative = np.ones(Z.shape)
        # gZ_derivative[Z < 0] = 0
        # dZ = dA * gZ_derivative
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "sigmoid":
        # A = 1 / (1 + np.exp(-Z))
        # dZ = dA * A * (1-A)
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "affine":
        dZ = affine_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    else:
        dW = np.zeros((dA.shape[0], linear_cache(0).shape[0]))
        db = np.zeros((dA.shape[0], 1))
        dA_prev = np.zeros(linear_cache(0).shape)

    return dA_prev, dW, db


def L_model_backward(AL, S, F, Y, gamma, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR

    Arguments:
    S -- complex pilot symbols of shape Nt x Ls
    F -- complex receive sample matrices of shape Nr x N x Ls
    Y -- complex received symbols of shape Nr x Ls x m
    AL -- output vector, output of the forward propagation (L_model_forward()), of shape (2*Nr*Nt) x m
    gamma -- nuclear norm constant
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ...
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ...
    """
    grads = {}
    L = len(caches)  # the number of layers
    m = AL.shape[1]
    Nr = Y.shape[0]
    Nt = S.shape[0]
    Ls = S.shape[1]

    # Initializing the backpropagation for the last layer
    dAL = np.zeros((2 * Nr * Nt, m))  # dAL = vec(real(dH) imag(dH))
    # Y_tilde = np.concatenate((np.real(Y), np.imag(Y)), axis=1)
    H_hat, _ = channel_vec2mat(AL, Nr, Nt)
    # S_tilde_fist_row = np.concatenate((np.real(S), np.imag(S)), axis=1)
    # S_tilde_sec_row = np.concatenate((-np.imag(S), np.real(S)), axis=1)
    # S_tilde = np.concatenate((S_tilde_fist_row, S_tilde_sec_row), axis=0)
    for i in range(m):
        # Y_tilde_i = Y_tilde[:, :, i]
        # H_tilde_i = H_tilde[:, :, i]
        # S_tilde_i = S_tilde
        Yi = Y[:, :, i]
        # sub-gradient for nuclear norm
        H_hat_i = H_hat[:, :, i]
        u, s, vh = np.linalg.svd(H_hat_i, full_matrices=False)
        nuc_grad = u.dot(vh)
        nuc_grad_tilde = np.concatenate((np.real(nuc_grad), np.imag(nuc_grad)), axis=1)

        dLS = 0
        for t in range(Ls):
            Ft = F[:, :, t]
            st = S[:, t]
            yt = Yi[:, t]
            dLS = dLS + np.dot(np.reshape(-1 * np.dot(Ft, yt - np.dot(Ft.conj().T, np.dot(H_hat_i, st))), (-1, 1)), np.reshape(st.conj().T, (1, -1))  )
        dLS_tilde = np.concatenate((np.real(dLS), np.imag(dLS)), axis=1)
        dH_i = dLS_tilde + gamma * nuc_grad_tilde
        dAL[:, i] = np.squeeze(np.reshape(dH_i, (2 * Nr * Nt, -1), 'F'))

    # Lth layer (affine -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"],
    # grads["dbL"] ## START CODE HERE ### (approx. 2 lines)
    current_cache = caches[L - 1]
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = \
        linear_activation_backward(dAL, current_cache, activation="affine")

    # Loop from l=L-2 to l=0
    for l in reversed(range(L - 1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 1)], current_cache".
        # Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)]
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = \
            linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation="relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def channel_vec2mat(h_tilde, Nr, Nt):
    """
    change the output of the DNN to complex and real channel matrix
    :param Nt:  # of transmitters
    :param Nr:  # of receivers
    :param h_tilde:     (2*Nr*Nt) x m samples, looks like h_tilde = vec([Re(H) Im(H)])
    :return:            H_hat -- Nr x Nt x m channel samples
                        H_tilde -- Nr x 2*Nt x m channel samples
    """
    m = h_tilde.shape[1]
    mid = Nr * Nt

    h_tilde_real = h_tilde[0:mid, :]
    h_tilde_imag = h_tilde[mid:, :]

    H_real = np.reshape(h_tilde_real, (Nr, Nt, -1), 'F')
    H_imag = np.reshape(h_tilde_imag, (Nr, Nt, -1), 'F')
    H_hat = H_real + 1j * H_imag
    H_tilde = np.concatenate((H_real, H_imag), axis=1)
    # print(H_real.shape, H_imag.shape, H_tilde.shape)

    return H_hat, H_tilde


def channel_mat2vec(H):
    """
    convert the complex channel matrix H to the format of output of DNN
    :param H:   complex channel matrix of size Nr x Nt x m
    :return:    (2*Nr*Nt) x m samples, looks like h_tilde = vec([Re(H) Im(H)])
    """
    Nr = H.shape[0]
    Nt = H.shape[1]
    m = H.shape[2]
    h_vec = np.reshape(H, (Nr * Nt, -1), 'F')  # (Nr*Nt) x test_mum
    h_vec_real = np.real(h_vec)
    h_vec_imag = np.imag(h_vec)
    h_tilde = np.concatenate((h_vec_real, h_vec_imag), axis=0)

    return h_tilde


def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent

    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients, output of L_model_backward

    Returns:
    parameters -- python dictionary containing your updated parameters
                  parameters["W" + str(l)] = ...
                  parameters["b" + str(l)] = ...
    """

    L = len(parameters) // 2  # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters


def update_parameters_with_gd(parameters, grads, learning_rate):
    """
    Update parameters using one step of gradient descent

    Arguments:
    parameters -- python dictionary containing your parameters to be updated:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients to update each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    learning_rate -- the learning rate, scalar.

    Returns:
    parameters -- python dictionary containing your updated parameters
    """

    L = len(parameters) // 2  # number of layers in the neural networks

    # Update rule for each parameter
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads['dW' + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads['db' + str(l + 1)]

    return parameters


def initialize_velocity(parameters):
    """
    Initializes the velocity as a python dictionary with:
                - keys: "dW1", "db1", ..., "dWL", "dbL"
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
    Arguments:
    parameters -- python dictionary containing your parameters.
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl

    Returns:
    v -- python dictionary containing the current velocity.
                    v['dW' + str(l)] = velocity of dWl
                    v['db' + str(l)] = velocity of dbl
    """

    L = len(parameters) // 2  # number of layers in the neural networks
    v = {}

    # Initialize velocity
    for l in range(L):
        v["dW" + str(l + 1)] = np.zeros(parameters['W' + str(l + 1)].shape)
        v["db" + str(l + 1)] = np.zeros(parameters['b' + str(l + 1)].shape)

    return v


def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    """
    Update parameters using Momentum

    Arguments:
    parameters -- python dictionary containing your parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v -- python dictionary containing the current velocity:
                    v['dW' + str(l)] = ...
                    v['db' + str(l)] = ...
    beta -- the momentum hyperparameter, scalar
    learning_rate -- the learning rate, scalar

    Returns:
    parameters -- python dictionary containing your updated parameters
    v -- python dictionary containing your updated velocities
    """

    L = len(parameters) // 2  # number of layers in the neural networks

    # Momentum update for each parameter
    for l in range(L):
        # compute velocities
        v["dW" + str(l + 1)] = beta * v["dW" + str(l + 1)] + (1 - beta) * grads["dW" + str(l + 1)]
        v["db" + str(l + 1)] = beta * v["db" + str(l + 1)] + (1 - beta) * grads["db" + str(l + 1)]
        # update parameters
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * v["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * v["db" + str(l + 1)]

    return parameters, v


def initialize_adam(parameters):
    """
    Initializes v and s as two python dictionaries with:
                - keys: "dW1", "db1", ..., "dWL", "dbL"
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.

    Arguments:
    parameters -- python dictionary containing your parameters.
                    parameters["W" + str(l)] = Wl
                    parameters["b" + str(l)] = bl

    Returns:
    v -- python dictionary that will contain the exponentially weighted average of the gradient.
                    v["dW" + str(l)] = ...
                    v["db" + str(l)] = ...
    s -- python dictionary that will contain the exponentially weighted average of the squared gradient.
                    s["dW" + str(l)] = ...
                    s["db" + str(l)] = ...

    """

    L = len(parameters) // 2  # number of layers in the neural networks
    v = {}
    s = {}

    # Initialize v, s. Input: "parameters". Outputs: "v, s".
    for l in range(L):
        v["dW" + str(l + 1)] = np.zeros(parameters["W" + str(l + 1)].shape)
        v["db" + str(l + 1)] = np.zeros(parameters["b" + str(l + 1)].shape)
        s["dW" + str(l + 1)] = np.zeros(parameters["W" + str(l + 1)].shape)
        s["db" + str(l + 1)] = np.zeros(parameters["b" + str(l + 1)].shape)

    return v, s


def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate=0.01,
                                beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Update parameters using Adam

    Arguments:
    parameters -- python dictionary containing your parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    learning_rate -- the learning rate, scalar.
    beta1 -- Exponential decay hyperparameter for the first moment estimates
    beta2 -- Exponential decay hyperparameter for the second moment estimates
    epsilon -- hyperparameter preventing division by zero in Adam updates

    Returns:
    parameters -- python dictionary containing your updated parameters
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    """

    L = len(parameters) // 2  # number of layers in the neural networks
    v_corrected = {}  # Initializing first moment estimate, python dictionary
    s_corrected = {}  # Initializing second moment estimate, python dictionary

    # Perform Adam update on all parameters
    for l in range(L):
        # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
        v["dW" + str(l + 1)] = beta1 * v["dW" + str(l + 1)] + (1 - beta1) * grads["dW" + str(l + 1)]
        v["db" + str(l + 1)] = beta1 * v["db" + str(l + 1)] + (1 - beta1) * grads["db" + str(l + 1)]

        # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
        v_corrected["dW" + str(l + 1)] = v["dW" + str(l + 1)] / (1 - np.power(beta1, t))
        v_corrected["db" + str(l + 1)] = v["db" + str(l + 1)] / (1 - np.power(beta1, t))

        # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
        s["dW" + str(l + 1)] = beta2 * s["dW" + str(l + 1)] + (1 - beta2) * np.power(grads["dW" + str(l + 1)], 2)
        s["db" + str(l + 1)] = beta2 * s["db" + str(l + 1)] + (1 - beta2) * np.power(grads["db" + str(l + 1)], 2)

        # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
        s_corrected["dW" + str(l + 1)] = s["dW" + str(l + 1)] / (1 - np.power(beta2, t))
        s_corrected["db" + str(l + 1)] = s["db" + str(l + 1)] / (1 - np.power(beta2, t))

        # Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon".
        # Output: "parameters".
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * v_corrected["dW" + str(l + 1)] / (
                    np.sqrt(s_corrected["dW" + str(l + 1)]) + epsilon)
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * v_corrected["db" + str(l + 1)] / (
                    np.sqrt(s_corrected["db" + str(l + 1)]) + epsilon)

    return parameters, v, s


def gradients_to_vector(gradients, layers_dims):
    """
    Roll all our gradients dictionary into a single vector satisfying our specific required shape.
    """
    L = len(layers_dims) - 1
    for l in range(1, L + 1):
        key_W = "dW" + str(l)
        key_b = "db" + str(l)
        new_vec_W = np.reshape(gradients[key_W], (-1, 1))
        new_vec_b = np.reshape(gradients[key_b], (-1, 1))

        if l == 1:
            theta = new_vec_W
            theta = np.concatenate((theta, new_vec_b), axis=0)
        else:
            theta = np.concatenate((theta, new_vec_W, new_vec_b), axis=0)

    return theta


def dictionary_to_vector(parameters, layers_dims):
    """
    Roll all our parameters dictionary into a single vector satisfying our specific required shape.
    """
    L = len(layers_dims) - 1
    keys = []
    for l in range(1, L+1):
        key_W = "W" + str(l)
        key_b = "b" + str(l)
        new_vec_W = np.reshape(parameters[key_W], (-1, 1))
        new_vec_b = np.reshape(parameters[key_b], (-1, 1))
        keys = keys + [key_W] * new_vec_W.shape[0] + [key_b] * new_vec_b.shape[0]

        if l == 1:
            theta = new_vec_W
            theta = np.concatenate((theta, new_vec_b), axis=0)
        else:
            theta = np.concatenate((theta, new_vec_W, new_vec_b), axis=0)

    return theta, keys


def vector_to_dictionary(theta, layers_dims):
    """
    Unroll all our parameters dictionary from a single vector satisfying our specific required shape.
    """
    parameters = {}
    L = len(layers_dims) - 1
    index = 0
    for l in range(1, L+1):
        parameters["W"+str(l)] = theta[index:(index+layers_dims[l]*layers_dims[l-1])].reshape((layers_dims[l], layers_dims[l - 1]))
        index = index+layers_dims[l]*layers_dims[l-1]
        parameters["b"+str(l)] = theta[index:index + layers_dims[l]].reshape((layers_dims[l], 1))
        index = index + layers_dims[l]

    return parameters


def gradient_check(parameters, gradients, layers_dims, X, S, Y, epsilon=1e-7):
    """
    Checks if backward_propagation_n computes correctly the gradient of the cost output by forward_propagation_n

    Arguments:
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
    grad -- output of backward_propagation_n, contains gradients of the cost with respect to the parameters.
    X input data
    S -- pilots
    Y -- received signals
    epsilon -- tiny shift to the input to compute approximated gradient with formula(1)

    Returns:
    difference -- difference (2) between the approximated gradient and the backward propagation gradient
    """

    # Set-up variables
    parameters_values, keys = dictionary_to_vector(parameters, layers_dims)
    grad = gradients_to_vector(gradients, layers_dims)
    num_parameters = parameters_values.shape[0]
    print("num_parameters = ", num_parameters)
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))

    # Compute gradapprox
    for i in range(num_parameters):
        print("parameter_num = ", i)
        # Compute J_plus[i]. Inputs: "parameters_values, epsilon". Output = "J_plus[i]".
        # "_" is used because the function you have to outputs two parameters but we only care about the first one
        thetaplus = np.copy(parameters_values)  # Step 1
        thetaplus[i][0] = thetaplus[i][0] + epsilon  # Step 2
        AL_plus, _ = L_model_forward(X, vector_to_dictionary(thetaplus, layers_dims))
        J_plus[i] = compute_cost(AL_plus, S, Y)

        # Compute J_minus[i]. Inputs: "parameters_values, epsilon". Output = "J_minus[i]".
        thetaminus = np.copy(parameters_values)  # Step 1
        thetaminus[i][0] = thetaminus[i][0] - epsilon  # Step 2
        AL_minus, _ = L_model_forward(X, vector_to_dictionary(thetaminus, layers_dims))
        J_minus[i] = compute_cost(AL_minus, S, Y)

        # Compute gradapprox[i]
        gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)

    # Compare gradapprox to backward propagation gradients by computing difference.
    numerator = np.linalg.norm(gradapprox - grad)  # Step 1'
    denominator = np.linalg.norm(gradapprox) + np.linalg.norm(grad)  # Step 2'
    difference = numerator / denominator  # Step 3'

    if difference > 2e-7:
        print(
            "\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
    else:
        print(
            "\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")

    return difference, gradapprox, grad, keys


def predict(parameters, X, Nr, Nt):
    AL, _ = L_model_forward(X, parameters)
    H_hat, _ = channel_vec2mat(AL, Nr, Nt)
    return H_hat


def compute_metric(H_hat, H_true):
    Nr = H_hat.shape[0]
    Nt = H_hat.shape[1]
    m = H_hat.shape[2]
    metric = 0
    for i in range(m):
        # metric += np.power(np.linalg.norm(H_hat[:, :, i] - H_true[:, :, i], 'fro') /
        #                    (np.linalg.norm(H_true[:, :, i], 'fro')), 2)
        metric += 10*np.log10(1/(Nr*Nt)*np.power(np.linalg.norm(H_hat[:, :, i] - H_true[:, :, i], 'fro'), 2))
    metric = metric/m
    return metric


# test
# H_real = np.random.randn(2, 2, 3)
# H_imag = np.random.randn(2, 2, 3)
# H = H_real + 1j*H_imag
#
# h_tilde = channel_mat2vec(H)
# # print("h_tilde.shape = ", h_tilde.shape)
# H_hat, H_tilde = channel_vec2mat(h_tilde, 2, 2)
#
# S = np.random.randn(2, 4, 3) + 1j * np.random.randn(2, 4, 3)
# Y = np.zeros((2, 4, 3), dtype=complex)      # dtype = complex if assign complex number to the holder
# for i in range(3):
#     noise = 0.01*(np.random.randn(2, 4) + 1j*np.random.randn(2, 4))
#     Y[:, :, i] = np.dot(H_hat[:, :, i], S[:, :, i]) + noise
#
# cost = compute_cost(h_tilde, S, Y)
# print("cost = ", cost)
#
# print("Finish test")

# gradient check
# def cost_function(Y, h, S, gamma):
#     Nr = Y.shape[0]
#     Nt = S.shape[0]
#     H_hat, _ = channel_vec2mat(h, Nr, Nt)
#     H_hat = np.squeeze(H_hat)
#     cost = np.power(np.linalg.norm(Y - np.dot(H_hat, S), 'fro'), 2) + gamma * np.linalg.norm(H_hat, ord='nuc')
#     return cost
#
#
# def num_gradient_h(Y, h, S, gamma):
#     dim = len(h)
#     eps = 0.1
#     dh_num = np.zeros(h.shape)
#     for i in range(dim):
#         h_plus = np.copy(h)
#         h_plus[i] = h_plus[i] + eps
#         F_plus = cost_function(Y, h_plus, S, gamma)
#
#         h_minus = np.copy(h)
#         h_minus[i] = h_minus[i]
#         F_minus = cost_function(Y, h_minus, S, gamma)
#
#         dh_num[i] = (F_plus - F_minus) / (eps)
#
#     return dh_num


# Nr = 3
# Nt = 3
# Ls = 4
# h_hat = np.random.randn(2*Nr*Nt, 1)
# Y = np.random.randn(Nr, Ls) + 1j * np.random.randn(Nr, Ls)
# S = np.random.randn(Nt, Ls) + 1j * np.random.randn(Nt, Ls)
#
# dh_num = num_gradient_h(Y, h_hat, S)
#
# Y_tilde = np.concatenate((np.real(Y), np.imag(Y)), axis=1)
# _, H_tilde = channel_vec2mat(h_hat, Nr, Nt)
# H_tilde = np.squeeze(H_tilde)
# S_tilde_fist_row = np.concatenate((np.real(S), np.imag(S)), axis=1)
# S_tilde_sec_row = np.concatenate((-np.imag(S), np.real(S)), axis=1)
# S_tilde = np.concatenate((S_tilde_fist_row, S_tilde_sec_row), axis=0)
#
# dH = -2 * np.dot((Y_tilde - np.dot(H_tilde, S_tilde)), S_tilde.T)
# dh = np.reshape(dH, (2 * Nr * Nt, -1), 'F')
# # dh = np.zeros(h_hat.shape)
#
# print("dh = ", dh)
# print("Next")
# print("dh_num = ", dh_num)
# print("next")
# print(dh-dh_num)
#
# print("OK")
