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

class CRF(nn.Module):
    def __init__(self, num_tags: int):
        super(CRF, self).__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))

    def _forward_algorithm(self, emissions, mask):
        batch_size, seq_length, num_tags = emissions.size()
        
        # Initialize the forward variables with start_transitions
        alphas = self.start_transitions.view(1, num_tags) + emissions[:, 0]
        
        for i in range(1, seq_length):
            alpha_t = []
            
            for next_tag in range(num_tags):
                # Broadcast alpha for current tag
                emit_score = emissions[:, i, next_tag].view(batch_size, 1)
                trans_score = self.transitions[:, next_tag].view(1, num_tags)
                next_tag_var = alphas + trans_score + emit_score
                alpha_t.append(torch.logsumexp(next_tag_var, dim=1))
            
            new_alphas = torch.stack(alpha_t, dim=1)
            
            # Apply mask
            mask_i = mask[:, i].view(batch_size, 1)
            alphas = torch.where(mask_i, new_alphas, alphas)
        
        # Add end_transitions score
        alphas += self.end_transitions.view(1, num_tags)
        
        # Return log sum of exp over all the paths
        return torch.logsumexp(alphas, dim=1)

    def _score_sentence(self, emissions, tags, mask):
        batch_size, seq_length, num_tags = emissions.size()
        scores = torch.zeros(batch_size, device=emissions.device)
        
        # Start transition score
        scores += self.start_transitions[tags[:, 0]]
        scores += emissions[torch.arange(batch_size), 0, tags[:, 0]]

        # Transition scores & emission scores
        for i in range(1, seq_length):
            mask_i = mask[:, i]
            current_tag = tags[:, i]
            previous_tag = tags[:, i-1]
            
            # Transition score from previous to current tag
            trans_score = self.transitions[previous_tag, current_tag]
            # Emission score for current tag
            emit_score = emissions[torch.arange(batch_size), i, current_tag]
            # Add scores where mask is valid
            scores += (trans_score + emit_score) * mask_i

        # End transition score
        last_valid_idx = mask.sum(1) - 1
        last_tag = tags.gather(1, last_valid_idx.unsqueeze(1)).squeeze()
        scores += self.end_transitions[last_tag]

        return scores

    def forward(self, emissions, tags=None, mask=None):
        if mask is None:
            mask = torch.ones_like(emissions[:,:,0], dtype=torch.bool)
            
        if tags is not None:
            return self._compute_neg_log_likelihood(emissions, tags, mask)
        else:
            return self._viterbi_decode(emissions, mask)

    def _compute_neg_log_likelihood(self, emissions, tags, mask):
        # Forward score
        forward_score = self._forward_algorithm(emissions, mask)
        # Gold score
        gold_score = self._score_sentence(emissions, tags, mask)
        return (forward_score - gold_score).mean()

    def _viterbi_decode(self, emissions, mask):
        batch_size, seq_length, num_tags = emissions.size()
        
        # Initialize score and history
        scores = self.start_transitions + emissions[:,0]
        history = []

        for t in range(1, seq_length):
            broadcast_scores = scores.unsqueeze(2)
            broadcast_emissions = emissions[:,t].unsqueeze(1)
            
            # Calculate next scores
            next_scores = broadcast_scores + self.transitions + broadcast_emissions
            
            # Get maximum scores and their indices
            scores, indices = next_scores.max(dim=1)
            history.append(indices)
            
            # Apply mask
            mask_t = mask[:,t].unsqueeze(1)
            scores = torch.where(mask_t, scores, broadcast_scores.squeeze(2))

        # Add end transition scores
        scores += self.end_transitions
        
        # Get best last tags
        _, best_last_tags = scores.max(dim=1)
        best_tags = [best_last_tags]
        
        # Follow the back pointers to get the best path
        for hist in reversed(history):
            best_last_tags = hist.gather(1, best_last_tags.unsqueeze(1)).squeeze()
            best_tags.append(best_last_tags)
            
        best_tags.reverse()
        return torch.stack(best_tags, dim=1)

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
        self.linear = nn.Linear(38, 2)  # Emissions for CRF
        self.use_glove = config.use_glove
        self.uw = nn.Parameter(torch.FloatTensor(torch.randn(100)))
        self.crf = CRF(num_tags=2)

    def forward(self, x_batch, tags=None):
      device = x_batch.device
    
    # BERT embeddings
      with torch.no_grad():
          E, _ = self.bert(x_batch)
      E = torch.stack(E[-4:]).sum(0)
      
      # CapsNet processing
      U = self.capsnet(E)
      original_shape = U.shape
      U = U.reshape(-1, 160)
      
      # Attention mechanisms
      a = self.phrase_attention(U)
      f_sa = self.self_attention(U)
      f_a = f_sa * a.unsqueeze(1)
      
      # Reshape attention outputs
      f_a = f_a.view(original_shape[0], -1, f_a.size(-1))
      f_a = f_a.sum(1)
      
      # Get emissions for CRF
      emissions = self.linear(f_a)  # [batch_size, 2]
      emissions = emissions.unsqueeze(1)  # [batch_size, 1, 2]
      
      if tags is not None:
          # Training mode - return loss
          tags = tags.unsqueeze(1)  # [batch_size, 1]
          mask = torch.ones_like(tags, dtype=torch.bool, device=device)
          loss = self.crf(emissions, tags, mask)
          return loss  # This is a scalar value
      else:
          # Inference mode - return predictions
          mask = torch.ones((emissions.size(0), emissions.size(1)), 
                          dtype=torch.bool, device=device)
          predictions = self.crf(emissions, mask=mask)
          return predictions.squeeze(1)  # [batch_size]

    def to(self, device):
        super(Model, self).to(device)
        self.bert = self.bert.to(device)
        self.capsnet = self.capsnet.to(device)
        self.phrase_attention = self.phrase_attention.to(device)
        self.self_attention = self.self_attention.to(device)
        self.crf = self.crf.to(device)
        return self
