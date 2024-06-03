import tkinter as tk
from tkinter import messagebox
import os
import pickle

# Ścieżka do pliku z wynikami
scores_file = 'scores.pkl'


# Funkcja do wczytywania wyników z pliku
def load_scores():
    if os.path.exists(scores_file):
        with open(scores_file, 'rb') as f:
            return pickle.load(f)
    return []


# Funkcja do zapisywania wyników do pliku
def save_scores(scores):
    with open(scores_file, 'wb') as f:
        pickle.dump(scores, f)


# Funkcja uruchamiana po kliknięciu przycisku Start
def start_game():
    name = name_entry.get()
    if not name:
        messagebox.showwarning("Błąd", "Proszę wpisać imię!")
        return
    # Tutaj kod Twojej gry
    # Na potrzeby przykładu dodajemy losowy wynik
    import random
    score = random.randint(0, 100)
    messagebox.showinfo("Wynik", f"{name}, Twój wynik to: {score}")

    # Zaktualizuj listę wyników i zapisz do pliku
    scores = load_scores()
    scores.append((name, score))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[:5]
    save_scores(scores)
    update_scores_display()


# Funkcja do aktualizacji wyświetlania top 5 wyników
def update_scores_display():
    scores = load_scores()
    scores_text = "\n".join([f"{name}: {score}" for name, score in scores])
    scores_label.config(text=scores_text)


# Tworzenie głównego okna
root = tk.Tk()
root.title("Gra")

# Pole do wpisania imienia
tk.Label(root, text="Wpisz swoje imię:").pack()
name_entry = tk.Entry(root)
name_entry.pack()

# Przycisk startu
start_button = tk.Button(root, text="Start", command=start_game)
start_button.pack()

# Pole wyświetlające top 5 wyników
tk.Label(root, text="Top 5 wyników:").pack()
scores_label = tk.Label(root, text="", justify="left")
scores_label.pack()

# Zaktualizuj wyświetlanie wyników przy starcie aplikacji
update_scores_display()

# Uruchomienie głównej pętli aplikacji
root.mainloop()
