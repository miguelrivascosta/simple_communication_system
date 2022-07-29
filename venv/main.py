import numpy as np
import matplotlib.pyplot as plt
import math


def modulation(constellation):
    if constellation == 'BPSK':
        constellation = None
    elif constellation == 'QAM':
        constellation = None
    elif constellation == 'QPSK':
        constellation = None
    else:
        print(f"The constellation: {constellation} is not implemented. Closing the script")
        exit()


def rrcDesign(beta, span, fs):
    T = 1
    Ts = 1 / fs
    t = np.arange(-span, span + Ts, Ts)
    h = np.zeros((2, len(t)))

    for n in range(0, len(t)):
        print(n)
    return h


def srrcDesign(beta, span, fs):
    # Assuming T = 1 (1 second/symb)
    T = 1
    Ts = 1 / fs
    t = np.arange(-span, span + Ts, Ts)
    h = np.zeros((2, len(t)))
    h[0, :] = t
    for n in range(0, len(t)):
        if t[n] == 0:
            h[1, n] = 1 / T * (1 + beta * (4 / math.pi - 1))
        elif t[n] == T / (4 * beta) or t[n] == -T / (4 * beta):
            h[1, n] = beta / (T * math.sqrt(2)) * (
                    (1 + 2 / math.pi) * math.sin(math.pi / (4 * beta)) + (1 - 2 / math.pi) * math.cos(
                math.pi / (4 * beta)))
        else:
            h[1, n] = 1 / T * (math.sin(math.pi * t[n] / T * (1 - beta)) + 4 * beta * t[n] / T * math.cos(
                math.pi * t[n] / T * (1 + beta))) / (math.pi * t[n] / T * (1 - (4 * beta * t[n] / T) ** 2))
    return h / math.sqrt(fs)


if __name__ == '__main__':
    fs = 3
    srrc = srrcDesign(beta=0.5, span=10, fs=fs)
    rrc = np.convolve(srrc[1, :], srrc[1, :])
    # plt.stem(srrc[0, :], srrc[1, :])

    n_plots = 6
    plt.subplot(n_plots, 1, 1)
    plt.stem(srrc[0, :], srrc[1, :])

    plt.subplot(n_plots, 1, 2)
    plt.stem(rrc)

    # plt.show()

    symbols = np.sign(np.random.uniform(-1, 1, 10))
    symbols_ext = np.zeros(len(symbols) * (fs) - fs + 1)
    symbols_ext[::fs] = symbols

    plt.subplot(n_plots, 1, 3)
    plt.stem(symbols_ext)

    s = np.convolve(srrc[1, :], symbols_ext)
    plt.subplot(n_plots, 1, 4)
    plt.plot(s)

    n = np.random.normal(0, 1, len(s))

    y = s + n
    y_filt = np.convolve(srrc[1, :], y)

    plt.subplot(n_plots, 1, 4)
    plt.plot(y)

    plt.subplot(n_plots, 1, 5)
    plt.plot(y_filt)

    plt.show()
