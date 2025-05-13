import numpy as np


def compute_D(N):
    D = np.zeros((N, N))
    alpha = np.ones(N) * np.sqrt(2 / N)
    alpha[0] = 1 / np.sqrt(N)

    for k in range(N):
        for i in range(N):
            D[k, i] = alpha[k] * np.cos((k) * np.pi * (2 * i + 1) / (2 * N))
    return D

def dct_2D(f_mat):
    N = f_mat.shape[0]
    D = compute_D(N)

    # Applica DCT per colonne
    c_mat = D @ f_mat

    # Applica DCT per righe
    c_mat = (D @ c_mat.T).T

    return c_mat

def idct_2D(c_mat):
    N = len(c_mat)
    D = compute_D(N)

    # Applica IDCT per colonne
    f_mat = D.T @ c_mat

    # Applica IDCT per righe
    f_mat = (D.T @ f_mat.T).T

    return f_mat

