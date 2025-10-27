# word_predictor_tokens.py

import os
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.optimizers import RMSprop
import pickle
import heapq

# --------------------------
# Settings
# --------------------------
TOKEN_FILE = r"C:\ai proj\dataset.tokens"  # path to your .token file
SEQ_LEN = 5
EMBED_DIM = 100
LSTM_UNITS = 128
EPOCHS = 5
MODEL_SAVE_PATH = r"C:\ai proj\next_word_model_tokens.h5"
TOKENIZER_SAVE_PATH = r"C:\ai proj\tokenizer_tokens.pickle"
TOP_N = 3

# --------------------------
# Load tokens
# --------------------------
with open(TOKEN_FILE, "r", encoding="utf-8") as f:
    tokens = f.read().split()  # assuming space-separated tokens

print("Number of tokens:", len(tokens))
unique_tokens = sorted(list(set(tokens)))
token_index = {token: i for i, token in enumerate(unique_tokens)}
index_token = {i: token for token, i in token_index.items()}
vocab_size = len(unique_tokens)
print("Vocabulary size:", vocab_size)

# --------------------------
# Create sequences
# --------------------------
prev_tokens = []
next_tokens = []
for i in range(len(tokens) - SEQ_LEN):
    prev_tokens.append(tokens[i:i+SEQ_LEN])
    next_tokens.append(tokens[i+SEQ_LEN])

# Encode sequences as integers
X = np.array([[token_index[token] for token in seq] for seq in prev_tokens])
y = np.array([token_index[token] for token in next_tokens])

print("Number of sequences:", len(X))

# --------------------------
# Build model
# --------------------------
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=EMBED_DIM, input_length=SEQ_LEN),
    LSTM(LSTM_UNITS),
    Dense(vocab_size, activation='softmax')
])

optimizer = RMSprop(learning_rate=0.01)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.summary()

# --------------------------
# Train or load model
# --------------------------
if os.path.exists(MODEL_SAVE_PATH) and os.path.exists(TOKENIZER_SAVE_PATH):
    print("Loading existing model and tokenizer...")
    model = load_model(MODEL_SAVE_PATH)
    with open(TOKENIZER_SAVE_PATH, "rb") as f:
        token_index = pickle.load(f)
        index_token = {i: t for t, i in token_index.items()}
else:
    print("Training model...")
    model.fit(X, y, batch_size=128, epochs=EPOCHS, validation_split=0.05, shuffle=True)
    model.save(MODEL_SAVE_PATH)
    with open(TOKENIZER_SAVE_PATH, "wb") as f:
        pickle.dump(token_index, f)
    print("Model and tokenizer saved!")

# --------------------------
# Prediction helpers
# --------------------------
def top_n_sampling(preds, top_n=3, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    if temperature != 1.0:
        preds = np.log(preds + 1e-12) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
    top_indices = heapq.nlargest(top_n, range(len(preds)), preds.take)
    top_probs = preds[top_indices] / np.sum(preds[top_indices])
    return np.random.choice(top_indices, p=top_probs)

def predict_next_words(seed_tokens, n_predictions=5, top_n=TOP_N, temperature=1.0):
    seed_tokens = seed_tokens.lower().split()  # convert to list of tokens
    if len(seed_tokens) < SEQ_LEN:
        seed_tokens = [""]*(SEQ_LEN - len(seed_tokens)) + seed_tokens
    else:
        seed_tokens = seed_tokens[-SEQ_LEN:]
    x = np.array([[token_index.get(tok, 0) for tok in seed_tokens]])
    preds = model.predict(x, verbose=0)[0]
    candidates = []
    for _ in range(n_predictions):
        idx = top_n_sampling(preds, top_n=top_n, temperature=temperature)
        candidates.append(index_token.get(idx, "<OOV>"))
    return candidates

# --------------------------
# Interactive mode
# --------------------------
print("\n=== Next-Word Predictor (Interactive with .token dataset) ===")
print("Type a phrase and press Enter to predict the next word(s).")
print("Type 'exit' to quit.\n")

while True:
    seed_text = input("Your text: ")
    if seed_text.lower() in ["exit", "quit"]:
        print("Have a nice day, Good Bye!")
        break

    predictions = predict_next_words(seed_text, n_predictions=5)
    print("Predicted next words:", predictions)
    print("-"*40)
