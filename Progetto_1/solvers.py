import numpy as np
import time


def jacobi(A, b, x0, tol, nmax):

    # Controlli preliminari
    n, m = A.shape
    if n != m:
        print("Matrix A non è una matrice quadrata")
        return None, 0, 0, 0
    if len(x0) != n:
        print("Le dimensioni della matrice A non corrispondono a quelle della stima iniziale x0")
        return None, 0, 0, 0
    # Controllo su eventuali zeri sulla diagonale
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

    # Iterazione: while ||xnew - xold||_inf > tol e nit < nmax
    while np.linalg.norm(xnew - xold, ord=np.inf) > tol and nit < nmax:
        xold = xnew.copy()
        # xnew = inv(D) * (B*xold + b)
        xnew = D_inv @ (B @ xold + b)
        nit += 1

    tempo = time.time() - start_time

    x = xnew
    # Errore finale = ||xnew - xold||_inf
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

    x = np.zeros(M, dtype=float)

    x[0] = b[0] / L[0, 0]
    for i in range(1, N):
        x[i] = (b[i] - np.dot(L[i, :i], x[:i])) / L[i, i]

    return x


def gauss_seidel(A, b, x0, tol, nmax):

    # Controlli preliminari
    n, m = A.shape
    if n != m:
        print("La matrice A non è quadrata")
        return None, 0, 0, 0
    if len(x0) != n:
        print("Le dimensioni della matrice A non corrispondono a quelle del vettore iniziale x0")
        return None, 0, 0, 0

    # Controllo di dominanza diagonale
    dominante = True
    for i in range(n):
        somma_fuori_diag = np.sum(np.abs(A[i, :])) - abs(A[i, i])
        if abs(A[i, i]) <= somma_fuori_diag:
            dominante = False
            break
    if not dominante:
        print("[Avviso] A non è a dominanza diagonale. La convergenza non è garantita.")

    # Estraggo L e B
    L = np.tril(A)          # triangolare inferiore
    B = A - L               # rimanente

    xold = x0.astype(float)
    xnew = xold + 1.0       # serve per entrare nel while
    nit = 0

    start_time = time.time()

    while np.linalg.norm(xnew - xold, ord=np.inf) > tol and nit < nmax:
        xold = xnew.copy()
        # Risolvo il sottosistema triangolare L * xnew = b - B*xold
        rhs = b - B @ xold
        xnew = triang_inf(L, rhs)
        nit += 1

    tempo = time.time() - start_time

    x = xnew
    # Errore finale = ||xnew - xold||_inf
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

    # Verifica SPD bloccante
    try:
        np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        print("La matrice A non è definita positiva")
        return None, 0, 0, 0

    xk = x0.astype(float)
    nit = 0
    err = 1
    start_time = time.time()

    while nit < nmax and err > tol:
        r = b - A @ xk              # r^(k) = b – A x^(k)
        # Calcolo del passo α_k = (r^T r) / (r^T A r)
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

    M, N = A.shape
    # controlli preliminari
    if M != N:
        raise ValueError("La matrice non è quadrata")
    if len(b) != M or len(x0) != M:
        raise ValueError("Le dimensioni di A, b e x0 non corrispondono")

    # verifica SPD bloccante
    try:
        np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        raise ValueError("La matrice A non è definita positiva")

    # inizializzazione delle variabili
    xk = x0.astype(float)        # x^(0)
    r = b - A @ xk                # r^(0) = b - A x^(0)
    d = r.copy()                  # d^(0) = r^(0)
    err = np.linalg.norm(b - A @ xk) / np.linalg.norm(xk)  # errore relativo iniziale
    nit = 0
    start_time = time.time()

    # ciclo principale
    while err > tol and nit < nmax:
        # y^(k) = A d^(k)
        y = A @ d

        # α_k = (d^T r) / (d^T y)
        num = d @ r
        denom = d @ y
        if denom == 0 or np.isnan(denom) or np.isinf(denom):
            raise RuntimeError(f"Denominatore per alpha non valido: {denom}")
        alpha = num / denom

        # x^(k+1) = x^(k) + α_k d^(k)
        xk = xk + alpha * d

        # r^(k+1) = b - A x^(k+1)
        r = b - A @ xk

        # w^(k) = A r^(k+1)
        w = A @ r

        # β_k = (d^T w) / (d^T y)
        num_beta = d @ w
        denom_beta = denom  # = d^T y
        if denom_beta == 0 or np.isnan(denom_beta) or np.isinf(denom_beta):
            raise RuntimeError(f"Denominatore per beta non valido: {denom_beta}")
        beta = num_beta / denom_beta

        # d^(k+1) = r^(k+1) - β_k d^(k)
        d = r - beta * d

        # aggiorna errore relativo e contatore
        err = np.linalg.norm(b - A @ xk) / np.linalg.norm(xk)
        nit += 1

    tempo = time.time() - start_time
    return xk, nit, tempo, err




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
