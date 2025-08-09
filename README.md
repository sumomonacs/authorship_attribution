# Authorship Attribution with Sentence-BERT and Siamese Network

This project implements an **authorship attribution** system using a **Sentence-BERT** encoder within a **Siamese neural network** trained with contrastive loss.  
The model takes two text excerpts and predicts whether they were written by the same author. It can also identify the most likely author for an unseen excerpt by comparing it against a reference set.

---
## 📥 Download Pretrained Model

You can download the pretrained **`best.pt`** checkpoint from the **[Releases](https://github.com/sumomonacs/authorship_attribution/releases)** page.  
After downloading, place it in the `models/` directory:
```bash
mkdir -p models
mv path/to/best.pt models/best.pt
```

## 🚀 Quick Start

### 1️⃣ Install dependencies
```bash
pip install torch sentence-transformers scikit-learn tqdm matplotlib numpy pandas
```

### 2️⃣ Train the model
```bash
python -m scripts.train_model \
    --pairs data/pairs.json \
    --val-ratio 0.2 \
    --epochs 15 \
    --batch 64 \
    --freeze 0 \
    --lr 5e-6 \
    --grad-accum 1 \
    --out models
```

### 3️⃣  Evaluate the model
```bash
python -m scripts.evaluate_model \
```

### 4️⃣  Run Inference
```bash
python -m scripts.infer_author \
    --ckpt models/best.pt \
    --refs data/author_refs.json \
    --inputs data/batch_quotes.json \
    --out results/batch_results.json
```

## 💻 Environment

- **Python:** 3.10+ recommended  
- **Required packages:**  
  `torch`, `sentence-transformers`, `scikit-learn`, `tqdm`, `matplotlib`, `numpy`, `pandas`

---

## 📂 Project Structure
## 📦 authorship_attribution
 ├── 📁 data/               # 📄 Input datasets and references
 ├── 📁 models/             # 💾 Saved model checkpoints (not tracked in repo)
 ├── 📁 results/            # 📊 Inference outputs
 ├── 📁 scripts/            # 🛠️ Training, inference, and evaluation scripts
 ├── 📁 src/                # 🧠 Core model and utility modules
 └── README.md              # 📜 Project documentation


## 📂 Dataset

The dataset consists of excerpts from five canonical Anglophone authors (Charlotte Brontë, Edith Wharton, George Eliot, Henry James, and Virginia Woolf).  
All raw texts were sourced from **[Project Gutenberg](https://www.gutenberg.org/)**, which provides works in the public domain.  

The excerpts are split into:
- **Raw texts** — full books in plain text format (public domain).
- **Cleaned excerpts** — extracted passages of 300–500 words, balanced across authors, used for training and evaluation.

### 🔹 Using the dataset
- **Option 1:** Download the compressed dataset  

  Unzip into the project root so it becomes `data/`.
- **Option 2:** Rebuild from scratch using the provided preprocessing scripts in `src/preprocess.py`.

### 🔹 Training
Train the model on author pairs:
```bash
python -m scripts.train_model \
    --pairs data/pairs.json \
    --val-ratio 0.2 \
    --epochs 15 \
    --batch 64 \
    --freeze 0 \
    --lr 5e-6 \
    --grad-accum 1 \
    --out models
```

### 🔹 Inference
Run author prediction for a batch of unseen excerpts:
```bash
python -m scripts.infer_author \
    --ckpt models/best.pt \
    --refs data/author_refs.json \
    --inputs data/batch_quotes.json \
    --out results/batch_results.json
```

### 🔹 Evaluation
Evaluate model accuracy and generate plots:
```bash
python -m scripts.evaluate_model
```
