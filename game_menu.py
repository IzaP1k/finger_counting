import streamlit as st
import random
import os
import pickle
from camera_snake import snake_game

scores_file = 'scores.pkl'

WINDOW = 1000
TITLE_SIZE = 50
length = 1
time_step = 200

def load_scores():
    if os.path.exists(scores_file):
        with open(scores_file, 'rb') as f:
            return pickle.load(f)
    return []

def save_scores(scores):
    with open(scores_file, 'wb') as f:
        pickle.dump(scores, f)

def update_scores(name, score):
    scores = load_scores()
    scores.append((name, score))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[:5]
    save_scores(scores)
    return scores

def start_game(name):
    scores = snake_game(WINDOW, TITLE_SIZE, length, time_step)
    score = max(scores)
    st.success(f"{name}, Twój wynik to: {score}")
    return score

st.title("Gra")
name = st.text_input("Wpisz swoje imię:")

if st.button("Start"):
    if name:
        score = start_game(name)
        scores = update_scores(name, score)
    else:
        st.warning("Proszę wpisać imię!")

st.header("Top 5 wyników:")
scores = load_scores()
for i, (name, score) in enumerate(scores, 1):
    st.write(f"{i}. {name}: {score}")

