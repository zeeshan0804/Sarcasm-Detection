import torch
import torch.nn as nn
from transformers import BertModel

class SarcasmDetectionModel(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', num_heads=8, d_model=128):
        super(SarcasmDetectionModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.cnn = nn.Conv1d(in_channels=768, out_channels=d_model, kernel_size=3, padding=1)
        self.bigru = nn.GRU(input_size=d_model, hidden_size=d_model//2, num_layers=1, batch_first=True, bidirectional=True)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.fc = nn.Linear(d_model, 2)  # Output layer for binary classification (sarcasm or not)

    def forward(self, x_batch, attention_mask=None):
        # BERT embeddings
        with torch.no_grad():
            outputs = self.bert(x_batch, attention_mask=attention_mask)
            E = outputs.last_hidden_state  # [batch_size, seq_len, 768]

        # CNN feature extraction
        E = E.permute(0, 2, 1)  # [batch_size, 768, seq_len]
        U = self.cnn(E)  # [batch_size, d_model, seq_len]
        U = U.permute(0, 2, 1)  # [batch_size, seq_len, d_model]

        # BiGRU for sequential modeling
        U, _ = self.bigru(U)  # [batch_size, seq_len, d_model]

        # Multi-head self-attention
        attn_output, _ = self.multihead_attn(U, U, U)  # [batch_size, seq_len, d_model]

        # Pooling (mean pooling)
        attn_output = attn_output.mean(dim=1)  # [batch_size, d_model]

        # Feed-Forward Neural Network
        ffn_output = self.ffn(attn_output)  # [batch_size, d_model]

        # Output layer
        logits = self.fc(ffn_output)  # [batch_size, 2]
        return logits

    def to(self, device):
        super(SarcasmDetectionModel, self).to(device)
        self.bert = self.bert.to(device)
        self.cnn = self.cnn.to(device)
        self.bigru = self.bigru.to(device)
        self.multihead_attn = self.multihead_attn.to(device)
        self.ffn = self.ffn.to(device)
        self.fc = self.fc.to(device)
        return self
