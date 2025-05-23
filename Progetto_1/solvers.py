import numpy as np
import time


def jacobi(A, b, x0, tol, nmax):
    M, N = A.shape
    L = len(x0)

    if M != N:
        print('Matrix A is not a square matrix')
        return None, 0, 0, 0
    elif L != M:
        print('Dimensions of matrix A do not match dimensions of initial guess x0')
        return None, 0, 0, 0
    if np.any(np.diag(A) == 0):
        print('At least one diagonal entry of A is zero. The method cannot proceed.')
        return None, 0, 0, 0

    D = np.diag(np.diag(A))
    B = D - A

    xold = x0
    xnew = xold + 1
    nit = 0

    start_time = time.time()
    while np.linalg.norm(xnew - xold, np.inf) > tol and nit < nmax:
        xold = xnew
        xnew = np.linalg.solve(D, B @ xold + b)
        nit += 1

    elapsed_time = time.time() - start_time
    err = np.linalg.norm(xnew - xold, np.inf)

    return xnew, nit, elapsed_time, err


def triang_inf(L, b):
    M, N = L.shape

    if M != N:
        print('Matrix L is not a square matrix')
        return None
    if not np.allclose(L, np.tril(L), atol=1e-15):
        print('Matrix L is not a lower triangular matrix')
        return None

    x = np.zeros(M, dtype=float)

    x[0] = b[0] / L[0, 0]
    for i in range(1, N):
        x[i] = (b[i] - np.dot(L[i, :i], x[:i])) / L[i, i]

    return x


def gauss_seidel(A, b, x0, tol, nmax):
    M, N = A.shape

    if M != N:
        print('Matrix A is not a square matrix')
        return None, 0, 0
    if len(x0) != M:
        print('Dimensions of matrix A do not match dimension of initial guess x0')
        return None, 0, 0

    L = np.tril(A)
    B = A - L

    xOld = x0
    xNew = xOld.copy() + 1
    nit = 0
    start_time = time.time()

    while np.linalg.norm(xNew - xOld, np.inf) > tol and nit < nmax:
        xOld = xNew.copy()
        xNew = triang_inf(L, (b - B @ xOld))
        nit += 1

    elapsed_time = time.time() - start_time
    err = np.linalg.norm(xNew - xOld, np.inf)

    return xNew, nit, elapsed_time, err


def gradiente(A, b, x0, tol, nmax):
    M, N = A.shape
    if M != N:
        raise ValueError("La matrice A deve essere quadrata")
    if len(x0) != M:
        raise ValueError("Le dimensioni della matrice A non corrispondono alla dimensione di x0")

    try:
        np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        raise ValueError("La matrice A non è definita positiva")

    nit = 0
    err = 1
    xold = np.array(x0, dtype=float)
    start_time = time.time()
    while nit < nmax and err > tol:
        r = b - A @ xold
        alpha = (r.T @ r) / (r.T @ A @ r)
        xnew = xold + alpha * r
        err = np.linalg.norm(b - A @ xnew) / np.linalg.norm(xnew)
        xold = xnew
        nit += 1
    elapsed_time = time.time() - start_time

    return xold, nit, elapsed_time, err


def gradiente_coniugato(A, b, x0, tol, nmax):
    M, N = A.shape
    if M != N:
        raise ValueError("La matrice A deve essere quadrata")
    if len(x0) != M:
        raise ValueError("Le dimensioni della matrice A non corrispondono alla dimensione di x0")

    try:
        np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        raise ValueError("La matrice A non è definita positiva")

    x = np.array(x0, dtype=float)
    r = b - A @ x
    p = r.copy()
    rs_old = r @ r
    start_time = time.time()

    for k in range(nmax):
        Ap = A @ p
        alpha = rs_old / (p @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = r @ r
        if np.sqrt(rs_new) / np.linalg.norm(b) < tol:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    elapsed_time = time.time() - start_time
    err = np.linalg.norm(A @ x - b, np.inf)

    return x, k+1, elapsed_time, err



if __name__ == "__main__":
    A = np.array([[4, -1, 0, 0],
                  [-1, 4, -1, 0],
                  [0, -1, 4, -1],
                  [0, 0, -1, 3]], dtype=float)

    b = np.array([15, 10, 10, 10], dtype=float)
    x0 = np.zeros_like(b)
    tol = 1e-6
    nmax = 1000

    for method in [jacobi, gauss_seidel, gradiente, gradiente_coniugato]:
        print(f"\nMetodo: {method.__name__}")
        sol, nit, tempo, err = method(A, b, x0, tol, nmax)
        print(f"Soluzione: {sol}")
        print(f"Iterazioni: {nit}")
        print(f"Tempo: {tempo:.6f} s")
        print(f"Errore: {err}")
