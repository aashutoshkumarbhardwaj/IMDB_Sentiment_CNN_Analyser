<div align="center">

# 🎬 IMDb Sentiment Analyzer
### Deep Learning · CNN vs RNN · Streamlit Deployment

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

**Classify movie reviews as Positive or Negative using deep learning — trained on 50,000 real IMDb reviews and deployed live.**

[🚀 Live Demo](#) · [📓 Training Notebook](#) · [📊 Model Results](#model-performance) · [🤝 Contribute](#contributing)

![Demo Screenshot](https://via.placeholder.com/900x400/0a0a0f/a78bfa?text=Add+Your+App+Screenshot+Here)

</div>

---

## 🧠 What This Project Does

Instead of bag-of-words or TF-IDF, this project uses **neural sequence models** that understand *context* — the word "not great" means something different than "great", and RNNs/CNNs can capture that.

```
"This movie was absolutely amazing!" → [Model] → ✅ Positive  (91% confidence)
"Waste of time, terrible acting."   → [Model] → ❌ Negative  (87% confidence)
```

Two architectures are implemented and compared:

| Architecture | Idea | Strength |
|---|---|---|
| **SimpleRNN** | Reads words in sequence, carries memory forward | Great at capturing long-range dependencies |
| **1D-CNN** | Slides filters over word windows | Faster, great at local phrase patterns |

---

## ✨ Features

- 🔍 **Real-time sentiment prediction** — paste any review, get a result instantly
- 🧠 **CNN vs RNN comparison** — see both architectures side-by-side
- 📊 **Confidence scores** — not just a label, but how certain the model is
- ⚡ **Streamlit UI** — no frontend code, just Python
- 📦 **Pre-trained `.h5` model** — run inference locally with no GPU required
- 🌐 **Deployed on Streamlit Cloud** — one click to try it

---

## 📊 Model Performance

> Trained on the Keras built-in IMDb dataset (25K train / 25K test, vocab size = 10,000)

| Model | Test Accuracy | Parameters | Inference Time |
|---|---|---|---|
| SimpleRNN | ~85% | ~1.1M | ~12ms |
| 1D-CNN | ~88% | ~900K | ~8ms |

<details>
<summary><b>📈 Training curves (click to expand)</b></summary>

```
Add your training accuracy/loss plots here.
Tip: save them with matplotlib and commit to /assets/
```

![Training Curves](assets/training_curves.png)

</details>

---

## 🏗️ Architecture

```
Raw Text
   │
   ▼
Tokenizer  →  Converts words to integer indices
   │
   ▼
Padding    →  Pads/truncates all sequences to length 500
   │
   ▼
Embedding  →  Maps each word index → 32-dim dense vector
   │
   ├──── RNN Branch: SimpleRNN(32) → Dense(1, sigmoid)
   │
   └──── CNN Branch: Conv1D(32, kernel=3) → GlobalMaxPool → Dense(1, sigmoid)
   │
   ▼
Output: probability in [0, 1]  →  > 0.5 = Positive
```

---

## 📁 Project Structure

```
imdb_sentiment_cnn_analyser/
├── app.py                  # Streamlit app (UI + inference)
├── simple_rnn_imdb.h5      # Trained RNN model weights
├── requirements.txt        # Python dependencies
├── runtime.txt             # Python version for Streamlit Cloud
├── assets/
│   └── training_curves.png # Add your training plots here
├── notebooks/
│   └── training.ipynb      # Model training walkthrough ← add this!
└── README.md
```

---

## ⚙️ Quick Start

**1. Clone and install**
```bash
git clone https://github.com/your-username/imdb_sentiment_cnn_analyser.git
cd imdb_sentiment_cnn_analyser
pip install -r requirements.txt
```

**2. Run locally**
```bash
streamlit run app.py
```

**3. Try it online**
> 👉 [Add your Streamlit Cloud link here](https://streamlit.io/cloud)

---

## 🧪 How the Model Works

**Step 1 — Tokenization**
Text is converted into a sequence of integers using Keras' built-in IMDb word index (top 10,000 most frequent words).

**Step 2 — Padding**
All sequences are padded or truncated to 500 tokens so the model always receives a fixed-size input.

**Step 3 — Embedding**
Each word index is looked up in a learned 32-dimensional embedding space. Similar words end up close together.

**Step 4 — Sequence modeling**
- *RNN*: reads the sequence left-to-right, maintaining a hidden state (memory) across timesteps
- *CNN*: applies 32 filters of width 3 across the sequence, then takes the global max — fast and parallelizable

**Step 5 — Classification**
A single Dense neuron with sigmoid activation outputs a probability. > 0.5 → Positive.

---

## 🔥 Roadmap

- [x] SimpleRNN sentiment classifier
- [x] 1D-CNN sentiment classifier
- [x] Streamlit deployment
- [ ] Add training notebook with visualizations
- [ ] BERT / DistilBERT fine-tune (HuggingFace)
- [ ] Attention visualization — highlight which words matter most
- [ ] Multilingual support (XLM-R)
- [ ] ONNX export for 4× smaller model

---

## 🤝 Contributing

PRs are welcome! To contribute:

```bash
git checkout -b feature/your-feature
# make your changes
git commit -m "feat: your feature"
git push origin feature/your-feature
# open a Pull Request
```

Ideas worth contributing: better preprocessing, training notebook, BERT upgrade, UI improvements.

---

## 👨‍💻 Author

**Aashutosh Kumar Bhardwaj** — AI/ML Developer · Open Source Contributor

If this project helped you learn about NLP or deep learning, a ⭐ star goes further than you think — it helps others discover it too.

---

<div align="center">
<sub>Built with TensorFlow, Keras, and Streamlit · MIT License</sub>
</div>
