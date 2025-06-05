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


# 8x8 blocco di test
block = np.array([
    [231, 32, 233, 161, 24, 71, 140, 245],
    [247, 40, 248, 245, 124, 204, 36, 107],
    [234, 202, 245, 167, 9, 217, 239, 173],
    [193, 190, 100, 167, 43, 180, 8, 70],
    [11, 24, 210, 177, 81, 243, 8, 112],
    [97, 195, 203, 47, 125, 114, 165, 181],
    [193, 70, 174, 167, 41, 30, 127, 245],
    [87, 149, 57, 192, 65, 129, 178, 228]
], dtype=float)

# Calcolo DCT2
dct_custom = dct_2D(block)

# Calcolo DCT1D prima riga
D1 = compute_D(8)
dct1_custom = D1 @ block[0, :]

print("DCT2 calcolata:\n", np.round(dct_custom, 2))
print("DCT1D prima riga:\n", np.round(dct1_custom, 2))