import numpy as np
from numpy.linalg import inv
from scipy.misc import logsumexp


def init_params(nb_K, dim_y1, dim_y2, dim_x):
    list_params = [{} for k in range(nb_K)]

    dim_t = min(dim_y1, dim_y2)

    Σπ = 0
    for params in list_params:
        params["μ"] = np.transpose(np.random.randn(dim_y1 + dim_y2))
        params["Wx"] = np.random.randn(dim_x, dim_y1 + dim_y2)
        params["μx"] = np.transpose(np.zeros(dim_x))
        params["Ψx"] = np.zeros((dim_x, dim_x))
        params["π"] = np.random.randn() ** 2
        Σπ += params["π"]

        Wt = np.matrix(np.random.randn(dim_y1 + dim_y2, dim_t))
        Ψ1 = np.random.randn(dim_y1, dim_y1)
        Ψ2 = np.random.randn(dim_y2, dim_y2)

        Ψ1 = Ψ1 * Ψ1
        Ψ2 = Ψ2 * Ψ2
        temp_zero_mat1 = np.zeros((dim_y1, dim_y2))
        temp_zero_mat2 = np.zeros((dim_y2, dim_y1))

        Ψ = np.r_[np.c_[Ψ1, temp_zero_mat1], np.c_[temp_zero_mat2, Ψ2]]
        params["C"] = Ψ + Wt * Wt.T

        for key, value in params.items():
            temp = np.matrix(value)
            params[key] = temp

    for params in list_params:
        params["π"] = params["π"] / Σπ

    return list_params


def calc_μ(params_k, γ_N, y_N, x_N):
    # Σₙ(γₙ(yₙ-Wₓxₙ)) / Σₙγₙ
    return np.dot(γ_N, (y_N - np.einsum("jk,ij->ik", params_k["Wx"], x_N))) / np.sum(γ_N)


def calc_π(γ_N):
    return np.sum(γ_N) / len(γ_N)


def calc_Wx(y_tilde_N, x_tilde_N, γ_N):
    temp1 = np.einsum("ijk,i->jk",
                      np.einsum("ij,ik->ijk", y_tilde_N, x_tilde_N),
                      γ_N)
    temp2 = np.einsum("ijk,i->jk",
                      np.einsum("ij,ik->ijk", x_tilde_N, x_tilde_N),
                      γ_N)
    return np.dot(temp1, inv(temp2)).transpose()


def calc_C(params_k, y_tilde_N, x_tilde_N, γ_N):
    temp = y_tilde_N - np.einsum("jk,ij->ik", params_k["Wx"], x_tilde_N)
    return np.einsum("i,ijk->jk", γ_N,
                     np.einsum("ij,ik->ijk", temp, temp)) / np.sum(γ_N)


def calc_lpdf_norm(y_N, x_N, params_k):
    sign, logdet = np.linalg.slogdet(2 * np.pi * params_k["C"])

    mean = np.einsum("jk,ij->ik", params_k["Wx"], x_N) + params_k["μ"]
    covariance_inv = inv(params_k["C"])

    # temp = (y-mean).T * C.I * (y-mean)
    temp_N = np.einsum("ij,ij->i",
                       np.einsum("ij,jk->ik", y_N - mean, covariance_inv),
                       y_N - mean)
    return np.array(-0.5 * logdet - 0.5 * temp_N + np.log(params_k["π"])).reshape(len(y_N))


def E_step(y_N, x_N, params, K):
    lpdf_K_N = [calc_lpdf_norm(y_N, x_N, params[k]) for k in range(K)]
    lpdf_N = logsumexp(lpdf_K_N, axis=0)
    lpdf_K_N -= lpdf_N

    γ_K_N = np.exp(lpdf_K_N)

    return γ_K_N


def M_step(γ_K_N, y_N, x_N, params):
    for k, (γ_N, params_k) in enumerate(zip(γ_K_N, params)):
        μ_k = calc_μ(params_k, γ_N, y_N, x_N)
        y_tilde_N = y_N - np.dot(γ_N, y_N) / np.sum(γ_N)
        x_tilde_N = x_N - np.dot(γ_N, x_N) / np.sum(γ_N)
        Wx_k = calc_Wx(y_tilde_N, x_tilde_N, γ_N)
        C_k = calc_C(params_k, y_tilde_N, x_tilde_N, γ_N)
        π_k = calc_π(γ_N)

        params[k]["μ"] = μ_k
        params[k]["Wx"] = Wx_k
        params[k]["C"] = C_k
        params[k]["π"] = π_k


def mppcca(y1_N, y2_N, x_N, nb_K):
    params = init_params(nb_K,
                         len(y1_N[0]),
                         len(y2_N[0]),
                         len(x_N[0]))

    y_N = np.concatenate([y1_N, y2_N], axis=1)

    history_labels = []
    while True:
        log_γ = E_step(y_N, x_N, params, nb_K)
        M_step(log_γ, y_N, x_N, params)

        history_labels.append(np.argmax(log_γ, axis=0))
        if len(history_labels) < 2:
            continue

        if np.array_equal(history_labels[-2], history_labels[-1]):
            break
        print("%d step - updated %d labels" % (len(history_labels), (np.count_nonzero(history_labels[-1] - history_labels[-2]))))

    return params, history_labels[-1]
