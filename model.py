import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from config import Config

torch.manual_seed(123)
config = Config()


class Model(nn.Module):
    # def __init__(self, word2id):
    def __init__(self, bert):
        super(Model, self).__init__()
        # self.word2id = word2id
        # self.vocab_size = len(self.word2id) #句子长度

        self.bert = bert.train()
        self.fc0 = nn.Linear(768, 100)

        self.embed_size = config.embed_size #词向量维数
        self.cnn = CNN_layer()
        self.phrase_attention = Phrase_attention()
        self.self_attention = Self_Attention()
        self.batch_size = config.batch_size
        self.embed_size = config.embed_size
        self.linear = nn.Linear(768, 2) #矩阵size（100,2）
        self.use_glove = config.use_glove #调用glove模型
        self.uw = nn.Parameter(torch.FloatTensor(torch.randn(100)))
        # if self.use_glove:
        #     self.weight = utils.load_glove(self.word2id)
        #     self.embedding = nn.Embedding.from_pretrained(self.weight)
        #     self.embedding.weight.requires_grad = True
        # else:
        #     self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
            # 用bert

    def forward(self, x_batch):
        # print(x_batch)
        # print(x_batch.shape)
        # # exit()
        # x_batch = torch.tensor(x_batch)
        with torch.no_grad():
            E, _ = self.bert(x_batch)

        E = torch.stack(E[-4:]).sum(0)
        # print(E)
        # print(E.size())
        # E=self.fc0(E)
        # print(E)
        # print(E.size())
        # exit()

        # exit()
        # E = self.embedding(x_batch)#torch.Size([32, 40, 100])
        # print(E)
        # print(E.shape)
        # exit()
        U = self.cnn(E)#torch.Size([32, 38, 100])
        # print(U)
        # print(U.shape)
        # exit()
        a = self.phrase_attention(U).unsqueeze(2)#torch.Size([32, 38, 1])
        # print(a)
        # print(a.shape)
        # exit()
        f_a = self.self_attention(a * U)#torch.Size([32, 100])
        # print(f_a)
        # print(f_a.shape)
        # exit()
        result = self.linear(f_a)#torch.Size([32, 2])
        # print(result)
        # print(result.shape)
        # exit()
        return result



#卷积层
class CNN_layer(nn.Module):
    def __init__(self):
        super(CNN_layer, self).__init__()
        self.conv = nn.Conv2d(1, 768, (config.n_gram, 768))  #通道数是1，输出深度是100（用100个卷积核分别卷积），卷积核尺寸为（3*100）
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, embedding):
        embedding = embedding.unsqueeze(1) #在维度序号为1的地方增加1维
        embedding = self.conv(embedding)
        embedding = F.relu(embedding.squeeze(3)) #使用relu函数激活特征向量
        embedding = self.dropout(embedding)
        embedding = embedding.transpose(2, 1) #交换tensor的两个维度
        # print(embedding)
        # print(embedding.size())
        # exit()
        return embedding

#注意力权重层
class Phrase_attention(nn.Module):
    def __init__(self):
        super(Phrase_attention, self).__init__()
        self.linear = nn.Linear(768, config.max_sen_len - config.n_gram + 1)
        self.tanh = nn.Tanh()
        self.u_w = nn.Parameter(nn.init.xavier_uniform_(torch.FloatTensor(config.max_sen_len - config.n_gram + 1, 1)))

    def forward(self, embedding):
        u_t = self.tanh(self.linear(embedding))
        a = torch.matmul(u_t, self.u_w).squeeze(2)
        a = F.log_softmax(a, dim=1)
        # print(a)
        # print(a.size())
        # exit()
        return a

#自注意力层
class Self_Attention(nn.Module):
    def __init__(self):
        super(Self_Attention, self).__init__()
        self.w1 = nn.Parameter(nn.init.xavier_uniform_(torch.FloatTensor(768, 1)))
        self.w2 = nn.Parameter(nn.init.xavier_uniform_(torch.FloatTensor(768, 1)))
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
        # print(v_a)
        # print(v_a.size())
        # exit()
        return v_a.squeeze(1)