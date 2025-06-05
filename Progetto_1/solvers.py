import numpy as np
import time


def jacobi(A, b, x0, tol, nmax):
    # Controlli preliminari
    n, m = A.shape
    if n != m:
        print("La matrice A non è quadrata")
        return None, 0, 0, 0
    if len(x0) != n:
        print("Le dimensioni della matrice A non corrispondono a quelle della stima iniziale x0")
        return None, 0, 0, 0

    # Blocca se c'è uno zero sulla diagonale
    if np.any(np.diag(A) == 0):
        print("Almeno un elemento diagonale di A è zero. Il metodo non può procedere.")
        return None, 0, 0, 0

    # Controllo di dominanza diagonale stretta per righe (non bloccante)
    dominante = True
    for i in range(n):
        somma_fuori_diag = np.sum(np.abs(A[i, :])) - abs(A[i, i])
        if abs(A[i, i]) <= somma_fuori_diag:
            dominante = False
            break
    if not dominante:
        print("[Avviso] La matrice non è a dominanza diagonale stretta. La convergenza non è garantita.")

    # Estraggo D e B
    D = np.diag(np.diag(A))
    D_inv = np.diag(1 / np.diag(D))
    B = D - A

    # Inizializzazione
    xold = x0.astype(float)
    xnew = xold + 1.0
    nit = 0

    start_time = time.time()

    # Iterazione: while ||xnew - xold|| > tol e nit < nmax
    while np.linalg.norm(xnew - xold, ord=np.inf) > tol and nit < nmax:
        xold = xnew.copy()
        xnew = D_inv @ (B @ xold + b)
        nit += 1

    tempo = time.time() - start_time

    x = xnew
    err = np.linalg.norm(xnew - xold, ord=np.inf)
    return x, nit, tempo, err


def triang_inf(L, b):
    M, N = L.shape
    if M != N:
        print("Errore in triang_inf: L non è quadrata.")
        return None
    if not np.allclose(L, np.tril(L), atol=1e-15):
        print("Errore in triang_inf: L non è triangolare inferiore.")
        return None

    y = np.zeros(M, dtype=float)
    y[0] = b[0] / L[0, 0]
    for i in range(1, N):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
    return y


def gauss_seidel(A, b, x0, tol, nmax):
    # Controlli preliminari
    n, m = A.shape
    if n != m:
        print("La matrice A non è quadrata")
        return None, 0, 0, 0
    if len(x0) != n:
        print("Le dimensioni della matrice A non corrispondono a quelle del vettore iniziale x0")
        return None, 0, 0, 0

    # Controllo di dominanza diagonale stretta per righe (non bloccante)
    dominante = True
    for i in range(n):
        somma_fuori_diag = np.sum(np.abs(A[i, :])) - abs(A[i, i])
        if abs(A[i, i]) <= somma_fuori_diag:
            dominante = False
            break
    if not dominante:
        print("[Avviso] La matrice non è a dominanza diagonale stretta. La convergenza non è garantita.")

    # Estraggo L e B
    L = np.tril(A)   # triangolare inferiore
    B = A - L        # rimanente

    xold = x0.astype(float)
    xnew = xold + 1.0
    nit = 0

    start_time = time.time()

    while np.linalg.norm(xnew - xold, ord=np.inf) > tol and nit < nmax:
        xold = xnew.copy()
        rhs = b - B @ xold
        xnew = triang_inf(L, rhs)
        nit += 1

    tempo = time.time() - start_time

    x = xnew
    err = np.linalg.norm(xnew - xold, ord=np.inf)
    return x, nit, tempo, err


def gradiente(A, b, x0, tol, nmax):
    n, m = A.shape
    if n != m:
        print("La matrice A non è quadrata")
        return None, 0, 0, 0
    if len(x0) != n:
        print("Le dimensioni della matrice A non corrispondono a quelle del vettore iniziale x0")
        return None, 0, 0, 0

    try:
        np.linalg.cholesky(A)      # Verifica SPD
    except np.linalg.LinAlgError:
        raise ValueError("A non è SPD")

    xk = x0.astype(float)
    nit = 0
    err = 1
    start_time = time.time()

    while nit < nmax and err > tol:
        r = b - A @ xk                # r^(k) = b – A x^(k)
        r_dot = r @ r
        Ar = A @ r
        denom = r @ Ar

        if denom == 0 or np.isnan(denom) or np.isinf(denom):
            print(f"[Stop] Iterazione {nit}: denominatore non valido (denom = {denom})")
            break

        alpha = r_dot / denom
        if np.isnan(alpha) or np.isinf(alpha):
            print(f"[Stop] Iterazione {nit}: alpha non valido (alpha = {alpha})")
            break

        xkp1 = xk + alpha * r         # x^(k+1) = x^(k) + α_k r^(k)

        # Errore relativo = ||b – A x^(k+1)|| / ||x^(k+1)||
        err = np.linalg.norm(b - A @ xkp1) / np.linalg.norm(xkp1)

        xk = xkp1
        nit += 1

    tempo = time.time() - start_time
    return xk, nit, tempo, err


def gradiente_coniugato(A, b, x0, tol, nmax):
    n, m = A.shape
    # Controlli preliminari
    if n != m:
        print("La matrice A non è quadrata")
        return None, 0, 0, 0
    if len(b) != n or len(x0) != n:
        print("Le dimensioni di A, b e x0 non corrispondono")
        return None, 0, 0, 0

    # Verifica SPD bloccante
    try:
        np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        raise ValueError("A non è SPD")


    # Inizializzazione delle variabili
    xk = x0.astype(float)
    r = b - A @ xk                # r^(0) = b - A x^(0)
    d = r.copy()
    # Errore relativo iniziale = ||b - A x^(0)|| / ||x^(0)||
    norm_xk = np.linalg.norm(xk)
    if norm_xk == 0:
        err = np.linalg.norm(b - A @ xk)
    else:
        err = np.linalg.norm(b - A @ xk) / norm_xk
    nit = 0
    start_time = time.time()

    # Ciclo principale
    while err > tol and nit < nmax:
        y = A @ d                  # y^(k) = A d^(k)

        num = d @ r                # d^(k)^T r^(k)
        denom = d @ y              # d^(k)^T A d^(k)
        if denom == 0 or np.isnan(denom) or np.isinf(denom):
            print(f"[Stop] Iterazione {nit}: denominatore per alpha non valido (denom = {denom})")
            break
        alpha = num / denom

        xk = xk + alpha * d        # x^(k+1) = x^(k) + α_k d^(k)

        r = b - A @ xk             # r^(k+1) = b - A x^(k+1)
        w = A @ r                  # w^(k) = A r^(k+1)

        num_beta = d @ w
        denom_beta = denom
        if denom_beta == 0 or np.isnan(denom_beta) or np.isinf(denom_beta):
            print(f"[Stop] Iterazione {nit}: denominatore per beta non valido (denom_beta = {denom_beta})")
            break
        beta = num_beta / denom_beta

        d = r - beta * d           # d^(k+1) = r^(k+1) - β_k d^(k)

        # Errore relativo = ||b - A x^(k+1)|| / ||x^(k+1)||
        err = np.linalg.norm(b - A @ xk) / np.linalg.norm(xk)
        nit += 1

    tempo = time.time() - start_time
    return xk, nit, tempo, err


# Test rapido sui metodi presenti
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
