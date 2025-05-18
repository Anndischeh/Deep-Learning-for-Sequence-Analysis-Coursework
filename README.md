# Deep Learning for Sequential Analysis
## DeepLearning-Approaches-for-Binary-Sentiment-Classification-on-IMDB

A PyTorch/TensorFlow repository for sentiment analysis on the [IMDb Movie Reviews dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).  
Training logs, metrics, and visualizations are available on [Weights & Biases](https://wandb.ai/anndischeh-univ-/Deep%20Learning%20for%20Sequential%20Analysis?nw=nwuseranndischeh).

---


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

> 📌 You can replace `cnn` with other models like `dcnn`, `rnn`, `lstm`, `gru`, `distilbert` and change the input text as needed.

---

### 📒 **Option 2: Run in Google Colab**

If you prefer not to set up a local environment, run everything in a Colab notebook.

#### **1. Upload Files:**

Upload all necessary files and folders (including `Deep_Learning_for_Sequence_Analysis`) to your Colab session. If you upload a zip file, unzip it first:

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

## 🛠️ Configuration

----

## 📂 Repository Structure

```

├── data/
│   └── IMDB_Dataset.csv        
├── models/
│   ├── cnn_model.py           
│   ├── rnn_model.py           
│   └── transformer_model.py                
├── preprocessing/
│   └── text_processor.py          
├── training/
│   ├── evaluator.py        
│   └── trainer.py
├── utils/
│   ├── dataset.py
│   ├── helpers.py     
│   └── predictors.py     
├── App.ipynb           
├── config.py          
├── main.py
└── README.md                  

````


