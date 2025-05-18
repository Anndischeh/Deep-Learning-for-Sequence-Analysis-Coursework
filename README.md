# Deep Learning for Sequential Analysis
## DeepLearning-Approaches-for-Binary-Sentiment-Classification-on-IMDB

A PyTorch/TensorFlow repository for sentiment analysis on the [IMDb Movie Reviews dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).  
Training logs, metrics, and visualizations are available on [Weights & Biases](https://wandb.ai/anndischeh-univ-/Deep%20Learning%20for%20Sequential%20Analysis?nw=nwuseranndischeh).

---

 ## 🚀Quick Start

The `App.ipynb` notebook provides a streamlined interface for running inference without needing to delve into the underlying code:

**1. Setup in Google Colab:**

Upload: Upload all files and folders (including the `Deep_Learning_for_sequential_Analysis` directory) to your Google Colab environment. If you are using a zip file, unzip it first.

Install Requirements: Install the necessary Python packages by running the following commands:

```python
 !unzip Deep_Learning_for_sequential_Analysis.zip  # Only if uploading a zip file
 !pip install -r requirements.txt
 ```


Wandb Login: Authenticate with Weights & Biases (wandb) using your API key. Replace wandb_key with your actual API key:

```python
 import wandb
 wandb.login(key="YOUR_WANDB_API_KEY")
 ```

**⚠️2. Running Inference with `main.py`:**

📌 The `main.py` script allows you to specify the model and mode of operation. The following commands demonstrate running inference with various models and a sample input text.

Model Selection and Inference: Choose the model you want to use (e.g., cnn, dcnn, rnn, lstm, gru, distilbert). The --mode all argument likely runs all available processes or functions associated with the model (e.g., training, testing, prediction). The --text argument takes a string to be used as input for testing (e.g., "I fell asleep halfway through."). Change the text and file paths as required.

```python
 %run main.py --model_type cnn --mode all --text "I fell asleep halfway through."
 ```


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


