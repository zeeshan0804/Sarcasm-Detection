import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from config import Config

torch.manual_seed(123)
config = Config()

class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(CapsuleLayer, self).__init__()
        self.num_capsules = num_capsules
        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding) 
            for _ in range(num_capsules)
        ])

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):
        outputs = [capsule(x) for capsule in self.capsules]
        outputs = torch.stack(outputs, dim=4)
        return self.squash(outputs)

class CapsNet(nn.Module):
    def __init__(self):
        super(CapsNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 256, kernel_size=(config.n_gram, 768), stride=1)
        self.primary_capsules = CapsuleLayer(num_capsules=8, in_channels=256, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.digit_capsules = CapsuleLayer(num_capsules=10, in_channels=32*8, out_channels=16, kernel_size=1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, embedding):
        embedding = embedding.unsqueeze(1)
        x = F.relu(self.conv1(embedding))
        x = self.primary_capsules(x)
        x = x.view(x.size(0), -1, x.size(2), x.size(3))
        x = self.digit_capsules(x)
        x = self.dropout(x)
        x = x.squeeze().transpose(2, 1)
        return x

class Model(nn.Module):
    def __init__(self, bert):
        super(Model, self).__init__()
        self.bert = bert.train()
        self.fc0 = nn.Linear(768, 100)

        self.embed_size = config.embed_size
        self.capsnet = CapsNet()
        self.phrase_attention = Phrase_attention()
        self.self_attention = Self_Attention()
        self.batch_size = config.batch_size
        self.embed_size = config.embed_size
        self.linear = nn.Linear(160, 2)  # Changed from 768 to 160 (16 * 10)
        self.use_glove = config.use_glove
        self.uw = nn.Parameter(torch.FloatTensor(torch.randn(100)))

    def forward(self, x_batch):
        with torch.no_grad():
            E, _ = self.bert(x_batch)

        E = torch.stack(E[-4:]).sum(0)
        U = self.capsnet(E)
        a = self.phrase_attention(U).unsqueeze(2)
        f_a = self.self_attention(a * U)
        result = self.linear(f_a)
        return result

class Phrase_attention(nn.Module):
    def __init__(self):
        super(Phrase_attention, self).__init__()
        self.linear = nn.Linear(160, config.max_sen_len - config.n_gram + 1)  # Changed from 768 to 160
        self.tanh = nn.Tanh()
        self.u_w = nn.Parameter(nn.init.xavier_uniform_(torch.FloatTensor(config.max_sen_len - config.n_gram + 1, 1)))

    def forward(self, embedding):
        u_t = self.tanh(self.linear(embedding))
        a = torch.matmul(u_t, self.u_w).squeeze(2)
        a = F.log_softmax(a, dim=1)
        return a

class Self_Attention(nn.Module):
    def __init__(self):
        super(Self_Attention, self).__init__()
        self.w1 = nn.Parameter(nn.init.xavier_uniform_(torch.FloatTensor(160, 1)))  # Changed from 768 to 160
        self.w2 = nn.Parameter(nn.init.xavier_uniform_(torch.FloatTensor(160, 1)))  # Changed from 768 to 160
        self.b = nn.Parameter(torch.FloatTensor(torch.randn(1)))

    def forward(self, embedding):
        f1 = torch.matmul(embedding, self.w1)
        f2 = torch.matmul(embedding, self.w2)
        f1 = f1.repeat(1, 1, embedding.size(1))
        f2 = f2.repeat(1, 1, embedding.size(1)).transpose(1, 2)
        S = f1 + f2 + self.b
        mask = torch.eye(embedding.size(1), embedding.size(1)).type(torch.ByteTensor)
        S = S.masked_fill(mask.bool().cuda(), -float('inf'))
        max_row = F.max_pool1d(S, kernel_size=embedding.size(1), stride=1)
        a = F.softmax(max_row, dim=1)
        v_a = torch.matmul(a.transpose(1, 2), embedding)
        return v_a.squeeze(1)
