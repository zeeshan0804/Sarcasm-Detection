import torch
import torch.nn as nn
from transformers import BertModel

class SarcasmDetectionModel(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased'):
        super(SarcasmDetectionModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.cnn = nn.Conv1d(in_channels=768, out_channels=128, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(128, 1)
        self.fc = nn.Linear(128, 2)  # Output layer for binary classification (sarcasm or not)

    def forward(self, x_batch, attention_mask=None):
        # BERT embeddings
        with torch.no_grad():
            outputs = self.bert(x_batch, attention_mask=attention_mask)
            E = outputs.last_hidden_state  # [batch_size, seq_len, 768]

        # CNN feature extraction
        E = E.permute(0, 2, 1)  # [batch_size, 768, seq_len]
        U = self.cnn(E)  # [batch_size, 128, seq_len]
        U = U.permute(0, 2, 1)  # [batch_size, seq_len, 128]

        # LSTM for sequential modeling
        U, _ = self.lstm(U)  # [batch_size, seq_len, 128]

        # Simple attention mechanism
        attn_weights = torch.softmax(self.attention(U), dim=1)  # [batch_size, seq_len, 1]
        attn_output = torch.sum(U * attn_weights, dim=1)  # [batch_size, 128]

        # Output layer
        logits = self.fc(attn_output)  # [batch_size, 2]
        return logits

    def to(self, device):
        super(SarcasmDetectionModel, self).to(device)
        self.bert = self.bert.to(device)
        self.cnn = self.cnn.to(device)
        self.lstm = self.lstm.to(device)
        self.attention = self.attention.to(device)
        self.fc = self.fc.to(device)
        return self