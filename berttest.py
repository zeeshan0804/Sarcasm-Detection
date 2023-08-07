
from pytorch_pretrained_bert import BertTokenizer, BertModel
import torch
import numpy as np

tokenizer = BertTokenizer.from_pretrained('bert/bert-base-uncased')
model = BertModel.from_pretrained('bert/bert-base-uncased')


text = " the man went to the store "
tokenized_text = tokenizer.tokenize(text) #token初始化
# print(tokenized_text)
# exit()
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text) #获取词汇表索引
# print(indexed_tokens)
# exit()
tokens_tensor = torch.tensor([indexed_tokens]) #将输入转化为torch的tensor
# print(tokens_tensor)
# exit()


tokens_tensor = torch.from_numpy(np.asarray(tokens_tensor)).type(torch.LongTensor)
# print(tokens_tensor)
# exit()





with torch.no_grad():
    text, _ = model(tokens_tensor)
    # token_embeddings = []


# concatenated_last_4_layers = [torch.cat((layer[-1], layer[-2], layer[-3], layer[-4]), 0) for layer in token_embeddings] # [number_of_tokens, 3072]
# summed_last_4_layers = [torch.sum(torch.stack(layer)[-4:], 0) for layer in token_embeddings] # [number_of_tokens, 768]
summed_last_4_layerss = torch.stack(text[-4:]).sum(0)
# print(summed_last_4_layerss)
# print(summed_last_4_layerss.size())



# from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
# import torch
# import numpy as np
#
# text = ' the man went to the store '
# tokenizer = BertTokenizer.from_pretrained('bert/uncased_L-12_H-768_A-12')
# tokenized_text = tokenizer.tokenize(text) #token初始化
# # print(tokenized_text)
#
# indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text) #获取词汇表索引
# # print(indexed_tokens)
# tokens_tensor = torch.tensor([indexed_tokens]) #将输入转化为torch的tensor
# # print(tokens_tensor)
# # print(tokens_tensor.size())
#
#
# model = BertModel.from_pretrained('bert/uncased_L-12_H-768_A-12')
# model.eval()
#
#
#
#
#
#
#
#
# # tokens_tensor = torch.from_numpy(np.asarray(tokens_tensor)).type(torch.FloatTensor)
# # print(tokens_tensor)
# # print(tokens_tensor.type())
#
#
#
#
#
# with torch.no_grad():
#     text, _ = model(tokens_tensor)
#     token_embeddings = []
# # print(text)
# # print(token_embeddings)
#
# concatenated_last_4_layers = [torch.cat((layer[-1], layer[-2], layer[-3], layer[-4]), 0) for layer in token_embeddings] # [number_of_tokens, 3072]
# # print(concatenated_last_4_layers)
# # exit()
# # summed_last_4_layers = [torch.sum(torch.stack(layer)[-4:], 0) for layer in token_embeddings] # [number_of_tokens, 768]
# summed_last_4_layerss =torch.stack(text[-4:]).sum(0)
# print(summed_last_4_layerss)
# print(summed_last_4_layerss.size())
# exit()


