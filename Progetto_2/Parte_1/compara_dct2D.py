import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import time
import matplotlib.pyplot as plt

from scipy.fftpack import dct, idct
from dct_2D import dct_2D, idct_2D


# -------------------------
# DCT 2D fast usando scipy
# -------------------------
def dct_2D_fast(matrix):
    return dct(dct(matrix.T, norm='ortho').T, norm='ortho')


# -------------------------
# Misura tempi e traccia grafico
# -------------------------
def benchmark_dct2(N_values, repeats=10):
    custom_times = []
    fast_times = []

    for N in N_values:
        A = np.random.rand(N, N)

        # Tempo per DCT custom
        start = time.perf_counter()
        for _ in range(repeats):
            dct_2D(A)
        end = time.perf_counter()
        custom_time = (end - start) / repeats
        custom_times.append(custom_time)

        # Tempo per DCT fast (usiamo dct(dct(x.T, norm='ortho').T, norm='ortho') per 2D)
        start = time.perf_counter()
        for _ in range(repeats):
            dct(dct(A.T, norm='ortho').T, norm='ortho')
        end = time.perf_counter()
        fast_time = (end - start) / repeats
        fast_times.append(fast_time)

        print(f"N = {N:3d} -> Custom: {custom_time:.4e}s | Fast: {fast_time:.4e}s")

    # Plot in scala semilogaritmica (solo asse Y)
    plt.figure(figsize=(8, 5))
    plt.semilogy(N_values, custom_times, label="Custom DCT2", marker='o')
    plt.semilogy(N_values, fast_times, label="Fast DCT2 (scipy)", marker='x')
    plt.xlabel("Matrix size N")
    plt.ylabel("Time (s)")
    plt.title("DCT2 Execution Time Comparison")
    plt.grid(True, which='both', linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    N_values = list(range(4, 132, 12))  # oppure range(4, 128+1, 16)
    benchmark_dct2(N_values, repeats=10)  # puoi aumentare a 100 per medie più stabili
