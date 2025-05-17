# Deep Learning for Sequential Analysis
## DeepLearning-Approaches-for-Binary-Sentiment-Classification-on-IMDB

A PyTorch/TensorFlow repository for sentiment analysis on the [IMDb Movie Reviews dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).  
Training logs, metrics, and visualizations are available on [Weights & Biases](https://wandb.ai/anndischeh-univ-/Deep%20Learning%20for%20Sequential%20Analysis?nw=nwuseranndischeh).

---

## 🚀 Quick Start



## 📈 Results & Logs

All experiment metrics, loss curves, and confusion matrices are logged to W\&B and can be explored here:
[WandB project for deep learning for sequence analysis coursework](https://wandb.ai/anndischeh-univ-/Deep%20Learning%20for%20Sequential%20Analysis?nw=nwuseranndischeh)

----

## 🛠️ Configuration

----

## 📂 Repository Structure

```

├── data/
│   ├── raw/                   # original IMDb dataset files
│   └── processed/             # tokenized, padded, and split train/val/test sets
├── notebooks/                 # Jupyter notebooks for EDA & prototyping
├── src/
│   ├── datasets.py            # Dataset loading & preprocessing
│   ├── models.py              # Model definitions (RNN, LSTM, Transformer…)
│   ├── train.py               # Training loop, logging to W\&B
│   ├── evaluate.py            # Evaluation & metrics
│   └── utils.py               # helper functions
├── configs/
│   └── default.yaml           # hyperparameters & paths
├── scripts/
│   ├── prepare\_data.sh        # download & preprocess data
│   └── run\_experiment.sh      # example command-line launch
├── requirements.txt           # Python dependencies
├── README.md                  # this file
└── .gitignore

````


