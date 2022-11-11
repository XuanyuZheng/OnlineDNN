import numpy as np


def steering_vector(AntSYS, phi):
    """
    Construct one basic steering vector
    :param AntSYS: a dictionary containing antenna system parameters, D, lamda and Nr
    :param phi: angle of arrival
    :return: Nrx1 complex steering vector a(phi)
    """
    Nr = AntSYS["Nr"]
    D = AntSYS["D"]
    lamda = AntSYS["lamda"]
    m = np.arange(0, Nr)
    a = np.exp(-1j * 2 * np.pi * D / lamda * np.cos(phi) * m)
    a = np.reshape(a, (Nr, 1))
    return a


def generate_one_channel(AntSYS, phis):
    """
    generate one channel instance
    :param AntSYS: a dictionary containing antenna system parameters, D, lamda and Nr,...
    :param phis:    angle of arrivals, same for each transmitter
    :return:        one channel realization
                    and channel gains g (for each path and each tx, Nt x P complex matrix, iid unit Gaussian)
    """
    P = AntSYS["P"]  # number of AoAs
    Nr = AntSYS["Nr"]
    Nt = AntSYS["Nt"]  # tx antenna number

    g = 1 / np.sqrt(2) * (np.random.randn(Nt, P) + 1j * np.random.randn(Nt, P))
    # initialize the channel to all zero matrix
    H = np.zeros((Nr, Nt), dtype=complex)
    for nt in range(Nt):
        a_temp = np.zeros((Nr, 1))
        for p in range(P):
            a = steering_vector(AntSYS, phis[p])
            a_temp = a_temp + g[nt, p] * a
        a_temp = a_temp / np.sqrt(P)
        H[:, nt] = np.squeeze(a_temp)

    return H, g


def generate_pilot(Nt, L, pilot_type="Gaussian"):
    """
    generate pilot matrix
    :param pilot_type: pilot type -- Gaussian
    :param Nt:  transmit number
    :param L:   pilot length
    :return:    pilot matrix S of shape (Nt x L)
    """
    if pilot_type == "Gaussian":
        S = (np.random.randn(Nt, L) + 1j*np.random.randn(Nt, L)) / np.sqrt(2)   # tx power = 1 for each tx and ch use
    else:
        S = (np.random.randn(Nt, L) + 1j*np.random.randn(Nt, L)) / np.sqrt(2)   # tx power = 1 for each tx and ch use

    return S
