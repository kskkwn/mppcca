import numpy as np
import random

import sys
from os.path import dirname
from os.path import join as path_join
sys.path.append(path_join(dirname(__file__), "../../"))

import mppcca
from example.toy_scatter_data.params import params1, params2, params3


def generate_data():

    def generate_data_in_cluster(params, nb_data):
        data = []

        dimension_1 = params.mu1.shape[0]
        dimension_2 = params.mu2.shape[0]
        dimension_t = min(dimension_1, dimension_2)

        for i in range(nb_data):
            t = np.matrix([np.random.normal(0, 1) for i in range(dimension_t)]).T
            x = np.matrix(np.random.normal(params.mean_x, params.sdev_x))

            mean_1 = list(np.array(params.Wx1 * x + params.Wt1 * t + params.mu1).reshape(-1,))
            mean_2 = list(np.array(params.Wx2 * x + params.Wt2 * t + params.mu2).reshape(-1,))
            y1 = np.matrix(np.random.multivariate_normal(mean_1, params.Psi1)).T
            y2 = np.matrix(np.random.multivariate_normal(mean_2, params.Psi2)).T

            x = np.array(x).reshape(-1)
            y1 = np.array(y1).reshape(-1)
            y2 = np.array(y2).reshape(-1)

            data.append([y1, y2, x])

        return data

    nb_data = 3000
    π = [0.3, 0.2, 0.5]
    list_nb_data = [nb_data * π_k for π_k in π]
    list_params = [params1, params2, params3]

    data = []
    labels = []
    for i, (params_k, nb_data_k) in enumerate(zip(list_params, list_nb_data)):
        data.extend(generate_data_in_cluster(params_k, int(nb_data_k)))
        labels.extend([i for _ in range(int(nb_data_k))])

    data_labels = list(zip(data, labels))
    random.shuffle(data_labels)

    y1_N = []
    y2_N = []
    x_N = []
    labels_N = []
    for data_label in data_labels:
        [y1, y2, x], label = data_label
        y1_N.append(y1)
        y2_N.append(y2)
        x_N.append(x)
        labels_N.append(label)
    return np.array(y1_N), np.array(y2_N), np.array(x_N), labels_N


def main():
    K = 3
    y1_N, y2_N, x_N, labels_N = generate_data()

    import time
    start = time.time()
    params, predicted_labels = mppcca.mppcca(y1_N, y2_N, x_N, K)
    print(time.time() - start)

    for params_k in params:
        print(params_k["μ"])

    print(params1.mu1, params1.mu2)
    print(params2.mu1, params2.mu2)
    print(params3.mu1, params3.mu2)

    from utils import calc_misallocation_rate
    print(calc_misallocation_rate(predicted_labels, labels_N, K))

if __name__ == '__main__':
    main()
