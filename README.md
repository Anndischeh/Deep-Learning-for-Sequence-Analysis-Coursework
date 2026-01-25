# Deep Learning for Sequential Analysis

![Python](https://img.shields.io/badge/python-3.8+-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange)
![NLP](https://img.shields.io/badge/task-NLP-blueviolet)
![Text Classification](https://img.shields.io/badge/task-Text%20Classification-informational)
![Weights & Biases](https://img.shields.io/badge/experiment_tracking-W%26B-yellow)
![Reproducible](https://img.shields.io/badge/reproducible-experiments-success)
![Dataset](https://img.shields.io/badge/dataset-IMDb%2050K-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![CNN](https://img.shields.io/badge/model-CNN-lightgrey)
![RNN](https://img.shields.io/badge/model-RNN-lightgrey)
![LSTM](https://img.shields.io/badge/model-LSTM-lightgrey)
![GRU](https://img.shields.io/badge/model-GRU-lightgrey)
![Transformer](https://img.shields.io/badge/model-Transformer-lightgrey)



## 🌟 Project Highlights
- DistilBERT achieved **91.6% test accuracy** and **0.92 F1-score**
- RNN/LSTM/GRU/CNN models implemented to compare efficiency vs performance
- Preprocessing pipeline included tokenization, stopword removal, stemming, and attention masks for Transformers
- All experiments logged and visualized in [Weights & Biases](https://wandb.ai/anndischeh-univ-/Deep%20Learning%20for%20Sequential%20Analysis?nw=nwuseranndischeh)

| Model      | Accuracy | F1-score |
|------------|---------|----------|
| CNN        | 0.804   | 0.813    |
| DCNN       | 0.799   | 0.807    |
| RNN        | 0.504   | 0.091    |
| LSTM       | 0.859   | 0.857    |
| GRU        | 0.849   | 0.846    |
| DistilBERT | 0.916   | 0.920    |

---

## DeepLearning-Approaches-for-Binary-Sentiment-Classification-on-IMDB

A PyTorch/TensorFlow repository for sentiment analysis on the [IMDb Movie Reviews dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).  
Training logs, metrics, and visualizations are available on [Weights & Biases](https://wandb.ai/anndischeh-univ-/Deep%20Learning%20for%20Sequential%20Analysis?nw=nwuseranndischeh).

---

## 🛠 Prerequisites
- Python >= 3.8
- PyTorch >= 2.0
- TensorFlow (optional, for comparison)
- NLTK, Transformers, WandB
- Linux / Colab recommended for GPU support

## 🚀 Quick Start

You can run the project in **two** main ways:

---

### ✅ **Option 1: Run Locally (Linux-based System)**

This method uses a virtual environment to isolate dependencies.

#### **1. Clone and Enter the Project Directory:**

```bash
git clone https://github.com/Anndischeh/Deep-Learning-for-Sequence-Analysis-Coursework.git
cd <your-project-folder>
```

#### **2. Run the Setup Script:**

This script will create and activate a virtual environment and install all required packages.

```bash
./setup.sh
```

#### **3. Run Inference with `main.py`:**

```bash
python main.py --model_type cnn --mode all --text "I fell asleep halfway through."
```

> 📌 You can replace `cnn` with other models like `dcnn`, `rnn`, `lstm`, `gru`, `distilbert`, and change the input text as needed.

---

### 📒 **Option 2: Run in Google Colab**

If you prefer not to set up a local environment, run everything in a Colab notebook.

#### **1. Upload Files:**

Upload all necessary files and folders (including `Deep_Learning_for_Sequence_Analysis`) to your Colab session, except for the main notebook `App.ipynb`. The instructions and code below are already included in that notebook.

If you upload a zip file, unzip it first:

```python
!unzip Deep_Learning_for_Sequence_Analysis.zip
```

#### **2. Install Requirements:**

```python
!pip install -r requirements.txt
```

#### **3. Login to Weights & Biases (wandb):**

Replace with your actual WandB API key:

```python
import wandb
wandb.login(key="YOUR_WANDB_API_KEY")
```

#### **4. Run Inference in Colab:**

```python
%run main.py --model_type cnn --mode all --text "I fell asleep halfway through."
```

---



## 📈 Results & Logs

All experiment metrics, loss curves, and confusion matrices are logged to W\&B and can be explored here:
[WandB project for deep learning for sequence analysis coursework](https://wandb.ai/anndischeh-univ-/Deep%20Learning%20for%20Sequential%20Analysis?nw=nwuseranndischeh)

----

## 📂 Repository Structure

```

├── data/
│   └── IMDB_Dataset.csv              # IMDB reviews dataset with corresponding sentiment labels
│
├── models/                           # Contains different model architectures for sentiment classification
│   ├── cnn_model.py                  # CNN-based models, including Dynamic MaxPool variations
│   ├── rnn_model.py                  # RNN-based models, including LSTM and GRU variants
│   └── transformer_model.py          # DistilBERT-based model (requires GPU for efficient execution)
│
├── preprocessing/                    # Text preprocessing scripts
│   └── text_processor.py             # Handles data cleaning (removing URLs, stopwords, etc.) and data visualization
│
├── training/                         # Training and evaluation utilities
│   ├── evaluator.py                  # Class for evaluating trained models (accuracy, F1 score, etc.)
│   └── trainer.py                    # Class for training models with given parameters
│
├── utils/                            # Utility scripts for data handling and model management
│   ├── dataset.py                    # Dataset loader and data splitter
│   ├── helpers.py                    # Helper functions for plotting, model saving/loading, etc.
│   └── predictors.py                 # Utilities for making sentiment predictions
│
├── App.ipynb                         # Interactive notebook for demonstrating sentiment prediction, which helps run all other scripts.
├── config.py                         # Centralized configuration file for setting hyperparameters and paths
└── main.py                           # Main execution script (can be renamed to train.py if desired, but kept as main.py to avoid confusion with trainer.py) 

````


