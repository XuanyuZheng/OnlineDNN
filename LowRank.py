import scipy.io
import matplotlib.pyplot as plt
from DNNLowRankBlks import *
from six.moves import cPickle as pickle  # for performance
import time


def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)


def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di


def load_dataset_SNR(dirct):
    """
    Load Y, S, H in of specific SNR, or random SNR
    :return:    S, Y, H
    """
    data_S = scipy.io.loadmat(dirct + '/S.mat')
    data_F = scipy.io.loadmat(dirct + '/F.mat')
    data_Y = scipy.io.loadmat(dirct + '/Y_test_SNR1.mat')
    data_R = scipy.io.loadmat(dirct + '/R_test_SNR1.mat')
    data_H = scipy.io.loadmat(dirct + '/H_test_SNR1.mat')
    S = data_S['S']
    F = data_F['F']
    Y = data_Y['Y_test_SNR']
    R = data_R['R_test_SNR']
    H = data_H['H_test_SNR']

    return S, F, Y, R, H


def load_dataset_SNRs(dirct='Data_test_multi_gamma_randArray1'):
    """
    dir: test data directory
    Load cost and metric for different algorithms in of different SNRs for test & comparism
    :return:    cost_LS_SNRs,
    """
    data_cost_PG_SNRs = scipy.io.loadmat(dirct + '/1cost_PG_SNRs.mat')
    cost_PG_SNRs = data_cost_PG_SNRs['cost_PG_SNRs']

    data_err_PG_SNRs = scipy.io.loadmat(dirct + '/1err_PG_SNRs.mat')
    err_PG_SNRs = data_err_PG_SNRs['err_PG_SNRs']

    data_S = scipy.io.loadmat(dirct + '/S.mat')
    data_F = scipy.io.loadmat(dirct + '/F.mat')
    data_Y = scipy.io.loadmat(dirct + '/R_test_SNR1.mat')
    data_H = scipy.io.loadmat(dirct + '/H_test_SNR1.mat')
    S = data_S['S']
    F = data_F['F']
    Y = data_Y['R_test_SNR']
    H = data_H['H_test_SNR']

    return cost_PG_SNRs, err_PG_SNRs, S, F, Y, H


def L_layer_model(X, S, F, Y, R, H, X_val, S_val, F_val, Y_val, R_val, H_val,
                  layers_dims, gamma, learning_rate=0.003, num_iterations=1, print_cost=False,
                  optimizer="adam", beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8
                  ):
    """
    Optimize the parameters given the training data X and Y
    :param R_val:           received sample signal of validation set
    :param R:               received sample signal of training set
    :param F_val:           receive sample matrices for validation set
    :param F:               receive sample matrices
    :param gamma:           nuclear norm constant
    :param X_val:           input for validation set
    :param S_val:           pilots for validation set
    :param Y_val:           received signal for validation set
    :param H_val:           True channel for validation set
    :param H:               true channel for training data set
    :param epsilon:         hyperparameter preventing division by zero in Adam updates
    :param beta2:           Exponential decay hyperparameter for the past squared gradients estimates
    :param beta1:           Exponential decay hyperparameter for the past gradients estimates
    :param beta:            Momentum hyperparameter
    :param optimizer:       choose optimizer
    :param S:               complex pilot symbols of shape Nt x Ls x m
    :param print_cost:      flag for printing cost
    :param X:               input training data of real pilots and received signals
    :param Y:               complex received signals of shape Nr x Ls x m
    :param layers_dims:      dimension of each layer (a list containing L + 1 layers: layer 0, 1, ..., L)
    :param learning_rate:   step size
    :param num_iterations:   max iteration number
    :return:                optimized parameters, costs during training
    """
    # np.random.seed(1)
    costs = []  # keep track of cost during training
    metrics = []  # keep track of metric during training
    costs_val = []  # keep track of cost during training for validation set
    metrics_val = []  # keep track of metric during training for validation set
    t = 0
    Nr = Y.shape[0]
    Nt = S.shape[0]
    # Ls = Y.shape[1]

    # Parameters initialization. (≈ 1 line of code)
    parameters = initialize_parameters_deep(layers_dims)

    # Initialize the optimizer
    if optimizer == "gd":
        pass  # no initialization required for gradient descent
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)

    # Loop (gradient descent)
    for i in range(1, num_iterations + 1):
        # print("Iteration = ", i)

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)

        if i == 1:
            # Compute cost on training set.
            cost = compute_cost(AL, S, F, R, gamma)
            costs.append(cost)

            # compute cost on validation set
            AL_val, _ = L_model_forward(X_val, parameters)
            cost_val = compute_cost(AL_val, S_val, F_val, R_val, gamma)
            costs_val.append(cost_val)

            print("0 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            print("Cost after iteration %i:       %f" % (0, cost))
            print("Cost_val after iteration %i:   %f" % (0, cost_val))

        # Backward propagation.
        grads = L_model_backward(AL, S, F, R, gamma, caches)

        # Update parameters
        if optimizer == "gd":
            parameters = update_parameters_with_gd(parameters, grads, learning_rate)
        elif optimizer == "momentum":
            parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
        elif optimizer == "adam":
            t = t + 1  # Adam counter
            parameters, v, s = update_parameters_with_adam(parameters, grads, v, s,
                                                           t, learning_rate, beta1, beta2, epsilon)
        # Print the cost every 100 training example
        if print_cost and (i % 10 == 0 or i == 1):
            # for training data
            AL, _ = L_model_forward(X, parameters)
            cost = compute_cost(AL, S, F, R, gamma)
            H_hat = predict(parameters, X, Nr, Nt)
            metric = compute_metric(H_hat, H)
            costs.append(cost)
            metrics.append(metric)

            # for validation set, note that in practice, online training will only work on training set.
            # i.e., there is no validation/test set
            AL_val, _ = L_model_forward(X_val, parameters)
            cost_val = compute_cost(AL_val, S_val, F_val, R_val, gamma)
            H_hat_val = predict(parameters, X_val, Nr, Nt)
            metric_val = compute_metric(H_hat_val, H_val)
            costs_val.append(cost_val)
            metrics_val.append(metric_val)
            print("%i <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" % i)
            print("Cost after iteration %i:       %f" % (i, cost))
            print("metric after iteration %i:     %f" % (i, metric))
            print("Cost_val after iteration %i:   %f" % (i, cost_val))
            print("metric_val after iteration %i: %f" % (i, metric_val))

    # plot the cost
    plt.figure(1)
    plt.plot(np.squeeze(costs), 'bo', label='cost on training data set')
    plt.plot(np.squeeze(costs_val), 'b', label='cost on validation data set')
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title("Cost during training, Learning rate =" + str(learning_rate))
    plt.legend()
    plt.grid(b=True, color='k', linestyle='--', linewidth=0.3)
    plt.autoscale()

    # plot the metrics
    plt.figure(2)
    plt.plot(np.squeeze(metrics), 'bo', label='metric on training data set')
    plt.plot(np.squeeze(metrics_val), 'b', label='metric on validation data set')
    plt.ylabel('metric')
    plt.xlabel('iterations')
    plt.title("Metric during training, Learning rate =" + str(learning_rate))
    plt.legend()
    plt.grid(b=True, color='k', linestyle='--', linewidth=0.3)
    plt.autoscale()

    # plt.show()

    return parameters, costs, costs_val, metrics, metrics_val


def prediction(parameters, X_test, S_test, F_test, Y_test, H_test, gamma):
    """
    predict and metric on test data
    :param F_test:      receive sample matrix for test set
    :param gamma:       nuclear norm constant
    :param H_test:      test set for channel
    :param parameters: learned parameters
    :param X_test:  input for test set
    :param S_test:  pilots for test set
    :param Y_test:  received signals for test set
    :return: H_test_hat -- the estimated channel, cost_test -- the cost on test set
            metric_test -- the metric on test set
    """
    Nr = Y_test.shape[0]
    Nt = S_test.shape[0]
    AL_test, _ = L_model_forward(X_test, parameters)
    cost_test = compute_cost(AL_test, S_test, F_test, Y_test, gamma)
    H_test_hat = predict(parameters, X_test, Nr, Nt)
    metric_test = compute_metric(H_test_hat, H_test)

    return H_test_hat, cost_test, metric_test


t = time.time()
# load the data
# S is Nt * Ls, H is Nr * Nt, Y is Nr * Ls
S, F, Y, R, H = load_dataset_SNR('Data')


# look at the dimension of the original data
print("S.shape = ", S.shape)
print("F.shape = ", F.shape)
print("Y.shape = ", Y.shape)
print("R.shape = ", R.shape)
print("H.shape = ", H.shape)

Nr = H.shape[0]
Nt = H.shape[1]
N = F.shape[1]
Ls = Y.shape[1]
test_num = Y.shape[2]
print("Nr = ", Nr)
print("Nt = ", Nt)
print("N = ", N)
print("Ls = ", Ls)
print("<<<<<<<<<<<<<<<<<<<<<<")

# vectorize the reveived signal Y
Y_vec = np.reshape(Y, (Nr * Ls, -1), 'F')  # (Nt*Ls) x test_mum
print("Y_vec.shape = ", Y_vec.shape)

# concatenate real and imaginary of S and Y
Y_vec_real = np.real(Y_vec)
Y_vec_imag = np.imag(Y_vec)

X_org = np.concatenate((Y_vec_real, Y_vec_imag), axis=0)  # input containing Y only
print("X_org.shape = ", X_org.shape)

train_index = 90000  # end index. Note that in online training, the data sample comes in real-time (infinite samples)
val_index = 9500  # end index
test_index = 10000  # end inedx

X_org_train = X_org[:, 0:train_index]
X_org_val = X_org[:, train_index:val_index]
X_org_test = X_org[:, val_index:test_num]

# normalize input data
X_mean = np.mean(X_org_train, axis=1, keepdims=True)
X_std = np.std(X_org_train, axis=1, keepdims=True)


X_train = (X_org_train - X_mean) / X_std
X_val = (X_org_val - X_mean) / X_std
X_test = (X_org_test - X_mean) / X_std
print("X_train.shape = ", X_train.shape)
print("X_val.shape = ", X_val.shape)
print("X_test.shape = ", X_test.shape)
print("<<<<<<<<<<<<<<<<<<<<<<")

# get the online training, validation and test matrices, pilots and F are the same for all examples
S_train = S
F_train = F
Y_train = Y[:, :, 0:train_index]
R_train = R[:, :, 0:train_index]
H_train = H[:, :, 0:train_index]

S_val = S
F_val = F
Y_val = Y[:, :, train_index:val_index]
R_val = R[:, :, train_index:val_index]
H_val = H[:, :, train_index:val_index]

S_test = S
F_test = F
Y_test = Y[:, :, val_index:]
R_test = R[:, :, val_index:]
H_test = H[:, :, val_index:]

input_dim = X_train.shape[0]
output_dim = 2 * Nr * Nt
print("input_dim = ", input_dim)
print("output_dim = ", output_dim)
print("<<<<<<<<<<<<<<<<<<<<<<")

# define DNN structure
layers_dims = [input_dim, 500, 400, 300, output_dim]
rho = 1
SNR = 30
noise_var = rho / (10 ** (SNR / 10))

gamma = np.sqrt(max(Nr, Nt) * noise_var/rho)
learning_rate = 0.001
print("gamma = ", gamma)


# train the model
parameters, costs, costs_val, metrics, metrics_val = \
    L_layer_model(X_train, S_train, F_train, Y_train, R_train, H_train, X_val, S_val, F_val, Y_val, R_val, H_val,
                  layers_dims, gamma, learning_rate=learning_rate, num_iterations=1,
                  print_cost=True, optimizer="adam")

elapsed = time.time() - t
print("Elapsed time after training = ", elapsed)

# save results, costs and metrics
save_dict(parameters, './parameters.pkl')  # save parameters
np.save('costs.npy', costs)
np.save('costs_val.npy', costs_val)
np.save('metrics.npy', metrics)
np.save('metrics_val.npy', metrics_val)

# load parameters data
parameters = load_dict('./parameters.pkl')
costs = np.load('costs.npy')
costs_val = np.load('costs_val.npy')
metrics = np.load('metrics.npy')
metrics_val = np.load('metrics_val.npy')

# performance on test set
H_test_hat, cost_test, metric_test = prediction(parameters, X_test, S_test, F_test, Y_test, H_test, gamma)
print("cost_test = ", cost_test)
print("metric_test = ", metric_test)




