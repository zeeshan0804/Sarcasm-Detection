import _pickle
import gensim
import os
from config import Config
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile

config = Config()
# read word embeddings and save as pickle
glove_file = datapath(os.getcwd() + "/" + config.glove_path)  #读入 下载好的glove文件
tmp_file = get_tmpfile(os.getcwd() + "/" + config.word2vec)   #输出 希望转换到的目标文件
glove2word2vec(glove_file, tmp_file)    #开始转换
model = gensim.models.KeyedVectors.load_word2vec_format(
    config.word2vec)  #读取新的模型文件
with open(config.glove_model_100d, "wb") as f:
    _pickle.dump(model, f)

# test
# f = open(config.glove_model_100d, 'rb')
# model = _pickle.load(f)
# print(model.wv.similarity('have', 'has'))    #查看两个词的相似度
# print(model.most_similar("china"))  #查看与其相似的词
# result = model.most_similar(positive=["women", "king"], negative=["man"])  #寻找性质相同的匹配关系
# print("{}:{:.4f}".format(*result[0]))
