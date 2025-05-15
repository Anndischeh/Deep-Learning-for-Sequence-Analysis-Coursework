import torch
import torch.nn as nn
from transformers import DistilBertModel
import config

class DistilBERTClassifier(nn.Module):
    def __init__(self, output_dim=config.OUTPUT_DIM, dropout=config.DROPOUT_RATE,
                 model_name=config.DISTILBERT_MODEL_NAME):
        super().__init__()
        print(f"Loading pretrained DistilBERT model: {model_name}")
        self.bert = DistilBertModel.from_pretrained(model_name)
        # Freeze BERT parameters initially if desired (optional, for faster fine-tuning)
        # for param in self.bert.parameters():
        #     param.requires_grad = False

        # Get the hidden size of the DistilBERT model (usually 768)
        bert_hidden_size = self.bert.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        # Classifier layer on top of the [CLS] token's output
        self.fc = nn.Linear(bert_hidden_size, output_dim)

    def forward(self, input_ids, attention_mask):
        # input_ids = [batch size, seq len]
        # attention_mask = [batch size, seq len]

        # Pass input through DistilBERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Get the hidden state of the [CLS] token (first token)
        # outputs.last_hidden_state shape: [batch size, seq len, hidden size]
        cls_hidden_state = outputs.last_hidden_state[:, 0, :]
        # cls_hidden_state shape: [batch size, hidden size]

        # Apply dropout and the final linear layer
        pooled_output = self.dropout(cls_hidden_state)
        logits = self.fc(pooled_output)
        # logits = [batch size, output_dim]

        return logits

