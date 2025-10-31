# — nai © 2025

"""
text_predictor.py
---------------------------------
A small next-word prediction project that I built for fun and learning.

This script can:
  - train a next-word model using text data
  - use POS tagging to improve predictions
  - optionally load GloVe embeddings
  - predict interactively from the terminal

Run `python text_predictor.py --help` for options.
"""

import os
import re
import random
import pickle
import heapq
import argparse
import logging
import collections
import numpy as np
import nltk
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Embedding, Dense, Dropout, Concatenate, Bidirectional, GRU
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import mixed_precision


# ---- setup ----
mixed_precision.set_global_policy("float32")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)

# Logging setup
logger = logging.getLogger("text_predictor")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S")
handler.setFormatter(formatter)
logger.addHandler(handler)

# ---- default config ----
CONFIG = {
    "dataset": "datasetsh.txt",
    "glove_path": "glove.6B.200d.txt",
    "glove_dim": 200,
    "seq_len": 20,
    "epochs": 8,
    "batch": 128,
    "lr": 1e-4,
    "use_glove": True,
    "sample_size": 600_000,
    "max_vocab": 20000,
    "oov_token": "<OOV>",
    "model": "text_predictor_model.keras",
    "best_model": "text_predictor_best.keras",
    "tokenizer": "text_predictor_tokenizer.pkl",
    "pos_dim": 16,
    "top_n": 5
}


# ---- small helpers ----
def clean_sentence(sentence: str):
    """Simple text cleaner."""
    s = re.sub(r"[^a-z\s]", "", sentence.lower())
    return [w for w in s.split() if w]


def build_tokenizer(tokens, path, max_vocab, oov_token):
    """Loads tokenizer if found, otherwise builds one from scratch."""
    if os.path.exists(path):
        logger.info("Loading tokenizer from %s", path)
        with open(path, "rb") as f:
            token_index = pickle.load(f)
    else:
        logger.info("Building new tokenizer...")
        counter = collections.Counter(tokens)
        common = [t for t, _ in counter.most_common(max_vocab - 1)]
        token_index = {oov_token: 0, **{t: i + 1 for i, t in enumerate(common)}}
        with open(path, "wb") as f:
            pickle.dump(token_index, f)
    index_token = {i: w for w, i in token_index.items()}
    return token_index, index_token


def make_pos_map(words):
    """Maps POS tags to integers."""
    tags = [tag for _, tag in nltk.pos_tag(words)]
    tagset = sorted(set(tags))
    pos_map = {t: i for i, t in enumerate(tagset)}
    return pos_map


def pos_to_idx(words, pos_map):
    """Converts list of words to list of POS tag indices."""
    tagged = nltk.pos_tag(words)
    return [pos_map.get(tag, 0) for _, tag in tagged]


def prepare_sequences(sentences, token_index, pos_map, seq_len):
    """Builds the X and y training sets."""
    prev, nxt, pos_inputs = [], [], []
    for s in sentences:
        words = clean_sentence(s)
        if len(words) <= seq_len:
            continue
        tagged = nltk.pos_tag(words)
        pos_tags = [pos_map.get(tag, 0) for _, tag in tagged]
        for i in range(len(words) - seq_len):
            prev.append(words[i:i+seq_len])
            nxt.append(words[i+seq_len])
            pos_inputs.append(pos_tags[i:i+seq_len])

    X_words = np.array([[token_index.get(w, 0) for w in seq] for seq in prev], dtype=np.int32)
    X_pos = np.array(pos_inputs, dtype=np.int32)
    y = np.array([token_index.get(w, 0) for w in nxt], dtype=np.int32)
    logger.info("Sequences prepared: %d samples", len(X_words))
    return X_words, X_pos, y


def load_glove(path, token_index, dim):
    """Loads GloVe embeddings into a matrix aligned with token_index."""
    if not os.path.exists(path):
        logger.warning("GloVe file not found, skipping.")
        return None
    logger.info("Loading GloVe vectors...")
    emb_index = {}
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            vals = line.strip().split()
            if len(vals) < dim + 1:
                continue
            word = vals[0]
            vec = np.asarray(vals[1:], dtype="float32")
            emb_index[word] = vec
    matrix = np.zeros((len(token_index), dim))
    hits = 0
    for w, i in token_index.items():
        if w in emb_index:
            matrix[i] = emb_index[w]
            hits += 1
    logger.info("Loaded %d/%d vectors from GloVe.", hits, len(token_index))
    return matrix


def build_model(seq_len, vocab_size, glove_dim, emb_matrix, pos_tags, pos_dim, lr):
    """Defines and compiles the model."""
    word_in = Input(shape=(seq_len,), name="word_input")
    pos_in = Input(shape=(seq_len,), name="pos_input")

    if emb_matrix is not None:
        word_emb = Embedding(vocab_size, glove_dim, weights=[emb_matrix], trainable=False)(word_in)
    else:
        word_emb = Embedding(vocab_size, glove_dim)(word_in)

    pos_emb = Embedding(pos_tags, pos_dim)(pos_in)
    merged = Concatenate()([word_emb, pos_emb])

    x = Bidirectional(GRU(256, dropout=0.3, recurrent_dropout=0.0))(merged)
    x = Dropout(0.3)(x)
    out = Dense(vocab_size, activation="softmax")(x)

    model = Model([word_in, pos_in], out)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=Adam(learning_rate=lr, clipnorm=1.0),
        metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)]
    )
    model.summary(print_fn=logger.info)
    return model


# ---- main training ----
def train(cfg):
    logger.info("Loading dataset: %s", cfg["dataset"])
    with open(cfg["dataset"], "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    sentences = nltk.sent_tokenize(text)

    tokens = []
    for s in sentences:
        tokens.extend(clean_sentence(s))

    if len(tokens) > cfg["sample_size"]:
        tokens = random.sample(tokens, cfg["sample_size"])

    token_index, index_token = build_tokenizer(tokens, cfg["tokenizer"], cfg["max_vocab"], cfg["oov_token"])
    pos_map = make_pos_map(list(token_index.keys())[:8000])
    Xw, Xp, y = prepare_sequences(sentences, token_index, pos_map, cfg["seq_len"])

    emb_matrix = load_glove(cfg["glove_path"], token_index, cfg["glove_dim"]) if cfg["use_glove"] else None
    model = build_model(cfg["seq_len"], len(token_index), cfg["glove_dim"], emb_matrix, len(pos_map), cfg["pos_dim"], cfg["lr"])

    callbacks = [
        ModelCheckpoint(cfg["best_model"], monitor="val_loss", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
    ]

    model.fit([Xw, Xp], y, batch_size=cfg["batch"], epochs=cfg["epochs"], validation_split=0.1, shuffle=True, callbacks=callbacks)

    logger.info("Saving final model to %s", cfg["model"])
    model.save(cfg["model"])
    with open(cfg["tokenizer"], "wb") as f:
        pickle.dump(token_index, f)
    with open(cfg["tokenizer"].replace(".pkl", "_pos.pkl"), "wb") as f:
        pickle.dump(pos_map, f)
    logger.info("Training done.")


# ---- prediction helpers ----
def top_n_sample(preds, index_token, top_n=5, temperature=0.9):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-12) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    idxs = heapq.nlargest(top_n * 3, range(len(preds)), preds.take)
    idxs = [i for i in idxs if index_token.get(i) != "<OOV>"][:top_n]
    probs = preds[idxs] / np.sum(preds[idxs])
    return np.random.choice(idxs, p=probs)


def predict_next(model, token_index, index_token, pos_map, text, seq_len, top_n):
    words = clean_sentence(text)[-seq_len:]
    if len(words) < seq_len:
        words = [""] * (seq_len - len(words)) + words
    Xw = np.array([[token_index.get(w, 0) for w in words]])
    Xp = np.array([pos_to_idx(words, pos_map)])
    preds = model.predict([Xw, Xp], verbose=0)[0]
    return [index_token.get(top_n_sample(preds, index_token, top_n=top_n), "<OOV>") for _ in range(top_n)]


def run_predict(cfg):
    if not os.path.exists(cfg["model"]):
        logger.error("No model found. Train first.")
        return
    model = load_model(cfg["model"], compile=True)
    with open(cfg["tokenizer"], "rb") as f:
        token_index = pickle.load(f)
    with open(cfg["tokenizer"].replace(".pkl", "_pos.pkl"), "rb") as f:
        pos_map = pickle.load(f)
    index_token = {i: w for w, i in token_index.items()}

    logger.info("Model ready. Type something below (or 'exit').")
    while True:
        text = input("Your text: ").strip()
        if text.lower() in {"exit", "quit"}:
            break
        preds = predict_next(model, token_index, index_token, pos_map, text, cfg["seq_len"], cfg["top_n"])
        print("Next word predictions:", preds)


# ---- CLI ----
def main():
    parser = argparse.ArgumentParser(description="Next-word predictor (by nai)")
    parser.add_argument("--mode", choices=["train", "predict"], default="predict")
    args = parser.parse_args()

    if args.mode == "train":
        train(CONFIG)
    else:
        run_predict(CONFIG)


if __name__ == "__main__":
    main()
