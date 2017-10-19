import numpy as np
import pylab as plt
from pylab import cm
import sys
from sklearn.cluster import KMeans
from os.path import dirname
from os.path import join as path_join
sys.path.append(path_join(dirname(__file__), "../../"))
from example.time_series_exp.params import params1, params2, params3
import mppcca
import utils


def causal(x, y, param):
    Xtkm = np.matrix(x).T
    Ytkm = np.matrix(y).T
    mean_Yt = float(param.a.T * Ytkm + param.b.T * Xtkm)
    Yt = np.random.normal(mean_Yt, float(param.e_Yt))
    Xt = np.random.normal(param.mean_Xt, float(param.e_Xt))
    return Xt, Yt


def non_causal(x, y, param):
    Yt = np.random.normal(param.mean_Yt, float(param.e_Yt))
    Xt = np.random.normal(param.mean_Xt, float(param.e_Xt))
    return Xt, Yt


def generate_data(list_k, list_nb_data, list_is_causal, list_params):
    x = [0]
    y = [0]
    labels = []

    for k, nb_data, is_causal, params in zip(
            list_k, list_nb_data, list_is_causal, list_params):

        generate = causal if is_causal else non_causal

        for t in range(nb_data):
            xt, yt = generate(x[-1], y[-1], params)
            x.append(xt)
            y.append(yt)
            labels.append(k)
    return np.array(x).reshape(-1, 1)[1:], np.array(y).reshape(-1, 1)[1:], np.array(labels)


def generate_data_exp1():
    return generate_data(list_k=[0, 1, 2],
                         list_nb_data=[1000, 1000, 1000],
                         list_is_causal=[True, True, True],
                         list_params=[params1, params2, params3])


def generate_data_exp2():
    return generate_data(list_k=[0, 1, 0],
                         list_nb_data=[1300, 400, 1300],
                         list_is_causal=[False, True, False],
                         list_params=[params3, params1, params3])


def plot(x, y, labels, K, title):
    fig = plt.figure(figsize=(6, 4))
    subplot1 = fig.add_subplot(211)
    subplot1.plot(x)
    subplot1.set_xlabel("Time Steps", size=7)
    subplot1.set_ylabel("Node 1 that sends\nsignals to Node 2", size=7)
    subplot1.set_xlim(0, len(x))
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.title(title, size=7)

    subplot2 = fig.add_subplot(212)
    subplot2.plot(y, c="g")
    subplot2.set_xlabel("Time Steps", size=7)
    subplot2.set_xlim(0, len(x))
    subplot2.set_ylabel("Node 2 that recieves\nsignals from Node 1", size=7)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)

    plt.subplots_adjust(hspace=0.5)

    for i, color_i in enumerate(labels):
        subplot1.axvspan(i, i + 1, facecolor=cm.hsv(float(color_i) / K), alpha=0.2, lw=0)
        subplot2.axvspan(i, i + 1, facecolor=cm.hsv(float(color_i) / K), alpha=0.2, lw=0)
    plt.show()


def exp1():
    K = 3
    x, y, labels = generate_data_exp1()
    d = 1  # delay time
    τ = 1  # embedded time
    xt_1 = utils.embed(x, d, τ)
    yt_1 = utils.embed(y, d, τ)
    yt = y[d + τ:]
    labels = labels[d + τ:]

    predicted_params, predicted_labels = mppcca.mppcca(yt, xt_1, yt_1, K)

    kmeans_labels = KMeans(n_clusters=K).fit(np.c_[yt, xt_1, yt_1]).labels_

    plot(x, y, labels, K, "true")
    plot(x, y, predicted_labels, K, "predicted")
    plot(x, y, kmeans_labels, K, "kmeans")

    utils.pca_scatter(yt_1, yt, xt_1, labels, "true")
    utils.pca_scatter(yt_1, yt, xt_1, predicted_labels, "predicted")
    utils.pca_scatter(yt_1, yt, xt_1, kmeans_labels, "kmeans")


def exp2():
    K = 3
    x, y, labels = generate_data_exp2()
    d = 1  # delay time
    τ = 1  # embedded time
    xt_1 = utils.embed(x, d, τ)
    yt_1 = utils.embed(y, d, τ)
    yt = y[d + τ:]
    labels = labels[d + τ:]

    predicted_params, predicted_labels = mppcca.mppcca(yt, xt_1, yt_1, K)

    kmeans_labels = KMeans(n_clusters=K).fit(np.c_[yt, xt_1, yt_1]).labels_

    plot(x, y, labels, K, "true")
    plot(x, y, predicted_labels, K, "predicted")
    plot(x, y, kmeans_labels, K, "kmeans")

    utils.pca_scatter(yt_1, yt, xt_1, labels, "true")
    utils.pca_scatter(yt_1, yt, xt_1, predicted_labels, "predicted")
    utils.pca_scatter(yt_1, yt, xt_1, kmeans_labels, "kmeans")


if __name__ == '__main__':
    exp1()
    exp2()
