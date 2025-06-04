import numpy as np
import matplotlib.pyplot as plt
import csv
from collections import defaultdict
from scipy.io import mmread
from solvers import jacobi, gauss_seidel, gradiente, gradiente_coniugato
import os


def carica_matrice(nome_file):
    A = mmread(nome_file)
    return A.toarray() if hasattr(A, 'toarray') else A


def confronta_solver(A, tol, nmax):
    x_exact = np.ones(A.shape[0])
    b = A @ x_exact
    x0 = np.zeros_like(b)

    metodi = {
        'Jacobi': jacobi,
        'Gauss-Seidel': gauss_seidel,
        'Gradiente': gradiente,
        'Gradiente Coniugato': gradiente_coniugato
    }

    risultati = {}

    for nome, metodo in metodi.items():
        try:
            x, nit, tempo, err = metodo(A, b, x0, tol, nmax)
            rel_err = np.linalg.norm(x - x_exact) / np.linalg.norm(x_exact)
            risultati[nome] = {
                'iterazioni': nit,
                'tempo': tempo,
                'errore': rel_err
            }
        except Exception as e:
            print(f"{nome} fallito: {e}")
            risultati[nome] = {
                'iterazioni': None,
                'tempo': None,
                'errore': None
            }

    return risultati


def plot_risultati(nome_matrice, tol, risultati):
    metodi = list(risultati.keys())
    iterazioni = [risultati[m]['iterazioni'] or 0 for m in metodi]
    tempi = [risultati[m]['tempo'] or 0 for m in metodi]
    errori = [risultati[m]['errore'] or 0 for m in metodi]

    x = np.arange(len(metodi))
    width = 0.25

    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle(f'{nome_matrice} – tol = {tol}')

    ax[0].bar(x, iterazioni, width, color='skyblue')
    ax[0].set_title('Iterazioni')
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(metodi)

    ax[1].bar(x, tempi, width, color='lightgreen')
    ax[1].set_title('Tempo (s)')
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(metodi)

    ax[2].bar(x, errori, width, color='salmon')
    ax[2].set_title('Errore relativo')
    ax[2].set_xticks(x)
    ax[2].set_xticklabels(metodi)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    output_name = f'confronto_{nome_matrice}_tol{str(tol).replace("-", "")}.png'
    plt.savefig(output_name)
    plt.close()

def salva_csv(risultati_totali, nome_file="risultati_aggregati.csv"):
    intestazioni = ["Matrice", "Tolleranza", "Metodo", "Iterazioni", "Tempo (s)", "Errore Relativo"]
    with open(nome_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(intestazioni)
        for r in risultati_totali:
            writer.writerow([r["matrice"], r["tol"], r["metodo"],
                             r["iterazioni"], r["tempo"], r["errore"]])
    print(f"\n✅ Risultati salvati in '{nome_file}'")


def plot_tempi_medi(risultati_totali):
    media_per_metodo = defaultdict(list)

    for r in risultati_totali:
        if r["tempo"] is not None:
            media_per_metodo[r["metodo"]].append(r["tempo"])

    metodi = list(media_per_metodo.keys())
    medie = [np.mean(media_per_metodo[m]) for m in metodi]

    plt.figure(figsize=(8, 4))
    plt.bar(metodi, medie, color='orchid')
    plt.ylabel("Tempo medio (s)")
    plt.title("Tempi medi per metodo (su tutte le matrici/tolleranze)")
    plt.tight_layout()
    plt.savefig("grafico_tempi_medi.png")
    plt.show()



if __name__ == "__main__":
    file_matrici = ["spa1.mtx", "spa2.mtx", "vem1.mtx", "vem2.mtx"]
    tolleranze = [1e-4, 1e-6, 1e-8, 1e-10]
    nmax = 20000 # maxIter

    risultati_totali = []

    for file in file_matrici:
        nome_matrice = os.path.splitext(file)[0]
        print(f"\n== Test sulla matrice: {nome_matrice} ==")
        A = carica_matrice(file)

        for tol in tolleranze:
            print(f"\n--- Tolleranza: {tol} ---")
            risultati = confronta_solver(A, tol, nmax)
            for metodo, res in risultati.items():
                print(f"{metodo:20s} | iter: {res['iterazioni']}, tempo: {res['tempo']:.4f}s, errore: {res['errore']:.2e}")
                risultati_totali.append({
                    "matrice": nome_matrice,
                    "tol": tol,
                    "metodo": metodo,
                    "iterazioni": res["iterazioni"],
                    "tempo": res["tempo"],
                    "errore": res["errore"]
                })
            plot_risultati(nome_matrice, tol, risultati)

    salva_csv(risultati_totali)
    plot_tempi_medi(risultati_totali)
