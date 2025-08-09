Authorship Attribution with Sentence-BERT and Siamese Network
This project implements an authorship attribution system using a Sentence-BERT encoder within a Siamese neural network architecture trained with contrastive loss.
Given two text excerpts, the model predicts whether they were written by the same author, and can also identify the most likely author from a reference set.

Quick Start
1. Install dependencies:
pip install torch sentence-transformers scikit-learn tqdm matplotlib numpy pandas

2. Train the model:
python -m scripts.train_model \
    --pairs data/pairs.json \
    --val-ratio 0.2 \
    --epochs 15 \
    --batch 64 \
    --freeze 0 \
    --lr 5e-6 \
    --grad-accum 1 \
    --out models

3. Evaluate the model
python -m scripts.evaluate_model \
    --ckpt models/best.pt \
    --pairs data/pairs_eval.json \
    --plots

4. Run inference
python -m scripts.infer_author \
    --ckpt models/best.pt \
    --refs data/author_refs.json \
    --inputs data/batch_quotes.json \
    --out results/batch_results.json

Environment:
Python: 3.10+ recommended

Required packages:
torch, sentence-transformers, scikit-learn, tqdm, matplotlib, numpy, pandas

Project Structure

data/               # Input datasets and references
models/             # Saved model checkpoints
results/            # Inference outputs
scripts/            # Training, inference, and evaluation scripts
src/                # Core model and utility modules

Training
Train the model on author pairs:
python -m scripts.train_model \
    --pairs data/pairs.json \
    --val-ratio 0.2 \
    --epochs 15 \
    --batch 64 \
    --freeze 0 \
    --lr 5e-6 \
    --grad-accum 1 \
    --out models

Inference
Run author prediction for a batch of unseen excerpts:
python -m scripts.infer_author \
    --ckpt models/best.pt \
    --refs data/author_refs.json \
    --inputs data/batch_quotes.json \
    --out results/batch_results.json

Evaluation
Evaluate model accuracy and generate plots:
python -m scripts.evaluate_model \
    --ckpt models/best.pt \
    --pairs data/pairs_eval.json \
    --plots

Notes
For out-of-domain generalization checks, see Appendix of the accompanying report.# authorship_attribution
