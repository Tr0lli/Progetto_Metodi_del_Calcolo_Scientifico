import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from scipy.fftpack import dct, idct
import os

class JPEGGui:
    def __init__(self, root):
        self.root = root
        self.root.title("Compressione JPEG DCT2 - Progetto MCS")
        self.root.configure(bg="#f0f0f0")
        self.root.geometry("400x500")

        # Percorso e immagini (originale vs compressa)
        self.image_path = None
        self.original_image = None
        self.compressed_image = None

        # Canvas in alto per mostrare l'immagine caricata (256×256px, griglia)
        self.canvas = tk.Canvas(
            root, width=256, height=256,
            bg="white", bd=2, relief="sunken"
        )
        self.canvas.pack(pady=(20, 10))

        # Label che mostra il nome del file caricato
        self.image_label = tk.Label(
            root, text="Nessuna immagine caricata",
            bg="#f0f0f0", fg="#555", font=("Arial", 10)
        )
        self.image_label.pack()

        # Crea i controlli (bottoni e campi di inserimento) nella finestra
        self._create_main_controls()

    def _create_main_controls(self):
        btn_frame = tk.Frame(self.root, bg="#f0f0f0")
        btn_frame.pack(pady=10)

        # Bottone per caricare l'immagine
        tk.Button(
            btn_frame,
            text="Carica immagine",
            command=self.load_image,
            width=20
        ).pack()

        entry_frame = tk.Frame(self.root, bg="#f0f0f0")
        entry_frame.pack(pady=5)

        # Entry per F (dimensione blocco) e d (soglia frequenze)
        self.f_entry = self._create_labeled_entry(
            entry_frame, "Dimensione blocco (F):", default="8"
        )
        self.d_entry = self._create_labeled_entry(
            entry_frame, "Soglia frequenze (d):", default="8"
        )

        # Bottone per avviare la compressione
        tk.Button(
            self.root,
            text="Comprimi immagine",
            command=self.process_image,
            width=20,
            bg="#007acc",
            fg="white"
        ).pack(pady=15)

    def _create_labeled_entry(self, parent, label, default=""):
        frame = tk.Frame(parent, bg="#f0f0f0")
        frame.pack(pady=5)
        tk.Label(frame, text=label, bg="#f0f0f0", fg="#333").pack(side=tk.LEFT, padx=5)
        entry = tk.Entry(frame, width=5)
        entry.insert(0, default)
        entry.pack(side=tk.LEFT)
        return entry

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("BMP files", "*.bmp"), ("Tutti i file", "*.*")]
        )
        if file_path:
            try:
                # Caricamento e conversione in scala di grigi
                img = Image.open(file_path).convert("L")
                self.original_image = img
                self.image_path = file_path
                # Mostra l'immagine nel canvas principale
                self.display_original_image(img)
                # Aggiorna il testo della label
                self.image_label.config(text=os.path.basename(file_path))
            except Exception as e:
                messagebox.showerror("Errore", f"Errore nel caricamento immagine: {e}")

    def dct2(self, block):
        # DCT sulle colonne e sulle righe
        return dct(dct(block.T, norm='ortho').T, norm='ortho')

    def idct2(self, block):
        return idct(idct(block.T, norm='ortho').T, norm='ortho')

    def apply_frequency_cut(self, c, d):
        F = c.shape[0]
        for k in range(F):
            for l in range(F):
                if k + l >= d:
                    c[k, l] = 0
        return c

    def process_image(self):
        if self.original_image is None:
            messagebox.showerror("Errore", "Nessuna immagine caricata.")
            return

        # Verifica validità parametri F e d
        try:
            F = int(self.f_entry.get())
            d = int(self.d_entry.get())
            if F <= 0 or d < 0 or d > (2 * F - 2):
                raise ValueError
        except ValueError:
            messagebox.showerror("Errore", "Inserisci valori validi per F (maggiore di 0) e d (tra 0 e 2F-2).")
            return

        # Converte l'immagine in array NumPy
        img_array = np.array(self.original_image, dtype=float)
        height, width = img_array.shape

        # Numero di blocchi che ricoprono interamente l'immagine
        h_blocks = height // F
        w_blocks = width // F

        # Matrice di destinazione (scartiamo gli avanzi)
        compressed_img = np.zeros((h_blocks * F, w_blocks * F))

        # Ciclo su ogni blocco i,j
        for i in range(h_blocks):
            for j in range(w_blocks):
                # Estrae il blocco F×F
                block = img_array[i*F:(i+1)*F, j*F:(j+1)*F]

                # Applica DCT2 (float)
                c = self.dct2(block)

                # Taglia le frequenze alte
                c = self.apply_frequency_cut(c, d)

                # Ricostruisce il blocco via IDCT2
                ff = self.idct2(c)

                # Clippa e arrotonda i valori in [0,255]
                ff = np.clip(np.round(ff), 0, 255)

                # Inserisce il blocco ricostruito nell'array di destinazione
                compressed_img[i*F:(i+1)*F, j*F:(j+1)*F] = ff

        # Ricampionamento finale: converti in uint8
        original_crop = img_array[:h_blocks * F, :w_blocks * F]
        img_original = Image.fromarray(original_crop.astype(np.uint8))
        self.compressed_image = Image.fromarray(compressed_img.astype(np.uint8))

        # Mostra la finestra di confronto
        self.show_compressed_window(img_original, self.compressed_image)

    def display_original_image(self, image):
        self.canvas.delete("all")
        resized = image.resize((256, 256))
        self.tk_image = ImageTk.PhotoImage(resized)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

    def show_compressed_window(self, original, compressed):
        win = tk.Toplevel(self.root)
        win.title("Confronto: Originale vs Compressa")
        win.configure(bg="#f9f9f9")
        win.geometry("550x350")

        frame = tk.Frame(win, bg="#f9f9f9")
        frame.pack(pady=10)

        # Ridimensionamento per la visualizzazione (256×256 ciascuna)
        orig_resized = original.resize((256, 256))
        comp_resized = compressed.resize((256, 256))

        tk_orig = ImageTk.PhotoImage(orig_resized)
        tk_comp = ImageTk.PhotoImage(comp_resized)

        # Label di intestazione per i due canvas
        label_orig = tk.Label(
            frame, text="Originale", bg="#f9f9f9",
            font=("Arial", 10, "bold")
        )
        label_comp = tk.Label(
            frame, text="Compressa", bg="#f9f9f9",
            font=("Arial", 10, "bold")
        )
        label_orig.grid(row=0, column=0, padx=10)
        label_comp.grid(row=0, column=1, padx=10)

        # Canvas contenenti le due immagini
        canvas_orig = tk.Canvas(
            frame, width=256, height=256,
            bg="white", bd=1, relief="sunken"
        )
        canvas_orig.grid(row=1, column=0, padx=10)
        canvas_orig.create_image(0, 0, anchor="nw", image=tk_orig)
        canvas_orig.image = tk_orig  # evita il garbage-collection

        canvas_comp = tk.Canvas(
            frame, width=256, height=256,
            bg="white", bd=1, relief="sunken"
        )
        canvas_comp.grid(row=1, column=1, padx=10)
        canvas_comp.create_image(0, 0, anchor="nw", image=tk_comp)
        canvas_comp.image = tk_comp  # evita il garbage-collection

        # Bottone per salvare l'immagine compressa
        def salva_immagine():
            path = filedialog.asksaveasfilename(
                defaultextension=".bmp",
                filetypes=[("Bitmap files", "*.bmp")]
            )
            if path:
                self.compressed_image.save(path)
                messagebox.showinfo(
                    "Salvataggio",
                    "Immagine compressa salvata."
                )

        tk.Button(
            win,
            text="Salva immagine compressa",
            command=salva_immagine,
            bg="#007acc",
            fg="white"
        ).pack(pady=10)


if __name__ == "__main__":
    root = tk.Tk()
    app = JPEGGui(root)
    root.mainloop()
