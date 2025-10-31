text predictor

heyyy, 
this is a small side project i built to explore how a language model can learn to predict the next word in a sentence.

it’s a single python file (`text_predictor.py`) that can train its own model and then let you test it interactively right in your terminal.

---

what does it do?

- learns to predict the next word from a text dataset  
- uses **word embeddings** + **POS tags** as inputs  
- optionally supports **GloVe** pretrained vectors  
- built with TensorFlow and NLTK  
- runs fine on **CPU-only** setups  

---

setup!!

you’ll need Python 3.8+ and a few dependencies:

    pip install tensorflow numpy nltk

the script will automatically download a few NLTK resources the first time it runs:
- punkt  
- stopwords  
- averaged_perceptron_tagger  

make sure your dataset is a plain-text file (like `datasetsh.txt`) that contains enough English sentences for training.

---

how to use?

### train the model

    python text_predictor.py --mode train

this will:
- read your dataset  
- build a tokenizer and POS-tag map  
- train a GRU-based language model  
- save it as `text_predictor_model.keras`

you can tweak things like epochs, batch size, or embeddings inside the config at the top of the file.

### predict interactively

once trained, you can run:

    python text_predictor.py --mode predict

then just type something like:

    Your text: the world is
    Next word predictions: ['beautiful', 'changing', 'mine', 'huge', 'vast']

type `exit` or `quit` to leave.

---

pls note :
- the default setup works on CPU.  
- if you have GloVe embeddings, place them in the project folder and keep the same path as in the config.  
- you can expand the script later for sentence-level prediction or text generation.  

---

**— nai © 2025**
