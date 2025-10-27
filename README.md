# 🧠 Next Word Prediction using LSTM-Based Neural Networks

This project implements a **Next Word Prediction** model using a **Long Short-Term Memory (LSTM)** neural network.  
It learns to predict the next word in a sequence based on previous context, enabling applications such as predictive typing and text completion.

---

## 🚀 Overview

This project demonstrates how **LSTM models** can learn long-term dependencies in natural language data.  
The model is trained on an English text corpus and evaluated using **accuracy, loss**, and **perplexity** metrics.

Key highlights:
- Built using **Python, TensorFlow, and Keras**
- Trained in **Google Colab** with GPU acceleration
- Token-based dataset for efficient training
- Visualized training performance (accuracy & loss)
- Predicts contextually relevant next words

---

## 📁 Repository Contents

```
next-word-prediction-lstm/
│
├── word_predictor.py                # Main training & prediction script
├── dataset.tokens                   # Tokenized English text corpus (Git LFS)
├── report.pdf                       # IEEE-format project report
├── presentation.pdf                 # Presentation slides
├── .gitattributes                   # Git LFS configuration
└── README.md                        # Project documentation
```

---

## 🧠 Model Architecture

| Layer | Description |
|-------|--------------|
| **Embedding** | Converts tokens into dense 100D vectors |
| **LSTM (128 units)** | Captures sequential dependencies |
| **Dense Softmax** | Outputs probability distribution for next word |

**Optimizer:** RMSProp  
**Loss:** Sparse Categorical Crossentropy  
**Batch Size:** 128  
**Epochs:** 5  

---

## 📊 Training Results

| Metric | Description |
|---------|--------------|
| **Training Accuracy** | Increases steadily per epoch |
| **Validation Accuracy** | Follows training closely |
| **Loss** | Decreases smoothly, minimal overfitting |
| **Perplexity** | Confirms effective language modeling |

---

### 📈 Example Graphs

Add your plots here:

- `accuracy_plot.png`
- `loss_plot.png`
- `perplexity_plot.png`

---

## 💬 Sample Predictions

| Input | Predicted Next Words |
|--------|----------------------|
| `the quick brown fox` | `jumps`, `runs`, `leaps` |
| `deep learning models are` | `powerful`, `complex`, `effective` |

---

## 💻 Run the Project in Google Colab

You can run the full workflow in Colab.

1. Open a new Colab notebook  
2. Upload:
   - `word_predictor.py`
   - `dataset.tokens`
3. Paste and run this setup:

```python
!pip install tensorflow numpy matplotlib
from google.colab import files
uploaded = files.upload()
!python word_predictor.py
```

👉 **Open in Colab:**  
[![Open In Colab]([https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abinairv/next-word-prediction-lstm/blob/main/colab_notebook.ipynb](https://colab.research.google.com/drive/1PeiOxja3ESl_KANscnQZoBs7W3ztCbw5?usp=sharing))

---

## ⚙️ Dependencies

Install locally:
```bash
pip install tensorflow numpy matplotlib
```

---

## 🧮 Evaluation Metric — Perplexity

Perplexity measures prediction uncertainty:

\[
Perplexity = e^{H(p)}
\]

Lower values = better next-word prediction.

---

## 📦 Output Files

| File | Purpose |
|------|----------|
| `next_word_model_tokens.h5` | Trained model weights |
| `tokenizer_tokens.pickle` | Vocabulary tokenizer |
| `accuracy_plot.png` | Accuracy curve |
| `loss_plot.png` | Loss curve |
| `perplexity_plot.png` | Perplexity chart |

---

## 🏫 Author Information

**Author:** Abhinay R V  
**Department:** Computer Science  
**Institution:** [Your University Name]  
**Email:** abhinayyy.rv@gmail.com  

---

## 🙏 Acknowledgments

- **Google Colab** — GPU compute resources  
- **TensorFlow/Keras** — Deep learning framework  
- **Faculty Supervisor** — Academic guidance  
- **Open English Text Corpora** — Dataset sources  

---

## 🧾 License

Licensed under the **MIT License** — free to use, modify, and distribute with attribution.

---

## ⭐ Final Thoughts

This project highlights how **LSTM networks** remain effective for sequence modeling and next-word prediction, even with moderate datasets and limited computational resources.

> “Predicting the next word isn’t guessing — it’s understanding context.”

---

📍 **GitHub Repository:** [https://github.com/abinairv/next-word-prediction-lstm](https://github.com/abinairv/next-word-prediction-lstm)  
🕒 **Last Updated:** October 2025
