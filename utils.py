from itertools import permutations
from sklearn.decomposition import PCA
import numpy as np
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from pylab import cm
# import seaborn as sns


def calc_misallocation_rate(predicted_labels, true_labels, K):
    def _calc(predicted_labels, true_labels):
        return sum(map(lambda predict, true: predict == true, predicted_labels, true_labels))

    associate_dicts = [{k: cluster for k, cluster in zip(range(K), cluster_set)} for cluster_set in permutations(range(K))]

    list_nb_correct_allocation = []
    for associate_dict in associate_dicts:
        nb_correct_allocation = _calc([associate_dict[pre] for pre in predicted_labels], true_labels)
        list_nb_correct_allocation.append(nb_correct_allocation)
    misallocation_rate = 1 - max(list_nb_correct_allocation) / len(predicted_labels)
    return misallocation_rate


def pca_scatter(y1, y2, x, labels, plot_title="", view=(60, 60)):
    pca = PCA(n_components=3)
    pcaed = pca.fit_transform(np.c_[y1, y2, x])
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pcaed[:, 0], pcaed[:, 1], pcaed[:, 2],
               c=cm.brg(labels / np.max(labels) + 0.1), lw=0, marker=".", alpha=0.3)
    ax.set_title(plot_title)
    ax.set_xlabel("pca axis 1")
    ax.set_ylabel("pca axis 2")
    ax.set_zlabel("pca axis 3")
    ax.view_init(elev=view[0], azim=view[1])
    plt.show()


def embed(x, delay, nb_embedding_frames):
    embedded_x = []
    for t in range(nb_embedding_frames + delay, len(x)):
        embedded_x_t = [x[t - i - delay] for i in range(0, nb_embedding_frames)]
        embedded_x_t = np.array(embedded_x_t).flatten()
        embedded_x.append(embedded_x_t)
    return np.array(embedded_x)


if __name__ == '__main__':
    from example.toy_scatter_data import generate_data
    y1_N, y2_N, x_N, labels_N = generate_data()
    pca_scatter(y1_N, y2_N, x_N, labels_N)
