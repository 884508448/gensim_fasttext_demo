from pprint import pprint
from gensim.models import word2vec

MIN_COUNT = 2
CPU_NUM = 8
CONTEXT_WINDOW = 5  # 提取目标词上下文距离最长5个词
EPOCH = 20  # 训练轮数


# 用生成器的方式读取文件里的句子
# 适合读取大容量文件，而不用加载到内存
class Sentences(object):
    def __init__(self, fname):
        self.fname = fname

    def __iter__(self):
        for line in open(self.fname, 'r', encoding="utf-8"):
            yield line.split()


# 模型训练函数
def w2vTrain(f_input, model_output, sg):
    sentences = Sentences(f_input)
    w2v_model = word2vec.Word2Vec(sentences,
                                  sg=sg,
                                  min_count=MIN_COUNT,
                                  workers=CPU_NUM,
                                  window=CONTEXT_WINDOW,
                                  epochs=EPOCH
                                  )
    w2v_model.save(model_output)


if __name__ == '__main__':
    input = "data/data.txt"

    # CBOW
    output = "model/CBOW"
    w2vTrain(f_input=input, model_output=output, sg=0)
    w2v_model = word2vec.Word2Vec.load(output)
    # 相似词查询
    pprint(w2v_model.wv.most_similar("中国"))

    # Skip-gram
    output="model/Skip-gram"
    w2vTrain(f_input=input, model_output=output, sg=1)
    w2v_model = word2vec.Word2Vec.load(output)
    # 相似词查询
    pprint(w2v_model.wv.most_similar("中国"))
