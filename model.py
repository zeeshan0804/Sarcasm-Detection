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

class Phrase_attention(nn.Module):
    def __init__(self, input_dim=160, output_dim=38):
        super(Phrase_attention, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()
        self.u_w = nn.Parameter(nn.init.xavier_uniform_(torch.FloatTensor(output_dim, 1)))

    def forward(self, embedding):
        u_t = self.tanh(self.linear(embedding))
        a = torch.matmul(u_t, self.u_w.to(embedding.device)).squeeze(1)
        return a

class Self_Attention(nn.Module):
    def __init__(self, input_dim=160, hidden_dim=38):
        super(Self_Attention, self).__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.scale = nn.Parameter(torch.sqrt(torch.FloatTensor([hidden_dim])))

    def forward(self, embedding):
        Q = self.query(embedding)
        K = self.key(embedding)
        V = self.value(embedding)
        
        attention = torch.matmul(Q, K.transpose(-2, -1)) / self.scale.to(embedding.device)
        attention = F.softmax(attention, dim=-1)
        x = torch.matmul(attention, V)
        
        return x


class Model(nn.Module):
    def __init__(self, bert):
        super(Model, self).__init__()
        self.bert = bert.train()
        self.fc0 = nn.Linear(768, 100)
        self.embed_size = config.embed_size
        self.capsnet = CapsNet()
        self.phrase_attention = Phrase_attention(input_dim=160, output_dim=38)
        self.self_attention = Self_Attention(input_dim=160, hidden_dim=38)
        self.batch_size = config.batch_size
        self.embed_size = config.embed_size
        self.linear = nn.Linear(38, 2)
        self.use_glove = config.use_glove
        self.uw = nn.Parameter(torch.FloatTensor(torch.randn(100)))

    def forward(self, x_batch):
        device = x_batch.device
        with torch.no_grad():
            E, _ = self.bert(x_batch)
        E = torch.stack(E[-4:]).sum(0)
        U = self.capsnet(E)
        original_shape = U.shape
        U = U.reshape(-1, 160)  # Flatten to [total_elements, 160]
        
        a = self.phrase_attention(U)
        f_sa = self.self_attention(U)
        
        f_a = f_sa * a.unsqueeze(1)
        
        # Reshape f_a to match the original batch size
        f_a = f_a.view(original_shape[0], -1, f_a.size(-1))
        
        # Sum over the time steps (dimension 1)
        f_a = f_a.sum(1)
        
        result = self.linear(f_a)
        return result

    def to(self, device):
        super(Model, self).to(device)
        self.bert = self.bert.to(device)
        self.capsnet = self.capsnet.to(device)
        self.phrase_attention = self.phrase_attention.to(device)
        self.self_attention = self.self_attention.to(device)
        return self
