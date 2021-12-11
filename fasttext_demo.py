import jieba
import fasttext


# 数据清洗，只需要标签和文本，按照fasttext指定格式
def preprocess_data(f_input, f_output):
    with open(f_input, "r", encoding="utf-8") as raw_file:
        with open(f_output, "w", encoding="utf-8") as out_file:
            for line in raw_file:
                line = line.split("|,|")
                if len(line) < 3:
                    continue
                label = line[1]
                content = jieba.lcut(line[2])
                content = filter(lambda x: len(x) > 1, content)  # 过滤掉单个字或标点符号
                line = "__label__" + label + "\t" + " ".join(content)
                out_file.write(line + "\n")


def divide_data(f_input, f_out1, f_out2, divi_num):
    with open(f_input, "r", encoding="utf-8") as origin_file:
        with open(f_out1, "w", encoding="utf-8") as sink1:
            with open(f_out2, "w", encoding="utf-8") as sink2:
                lines = origin_file.readlines()
                sink1.writelines(lines[:divi_num])
                sink2.writelines(lines[divi_num:])


if __name__ == '__main__':
    f_input = "data/fasttext_raw_data"
    f_output = "data/classify_data"
    sink1 = "data/train_data"
    sink2 = "data/test_data"
    preprocess_data(f_input, f_output)
    divide_data(f_output, sink1, sink2, 50000)

    classifier = fasttext.train_supervised(sink1, label='__label__', wordNgrams=2, epoch=20, lr=0.1, dim=100, thread=16)
    """
      训练一个监督模型, 返回一个模型对象

      @param input: 训练数据文件路径
      @param lr:              学习率
      @param dim:             向量维度
      @param ws:              cbow模型时使用
      @param epoch:           次数
      @param minCount:        词频阈值, 小于该值在初始化时会过滤掉
      @param minCountLabel:   类别阈值，类别小于该值初始化时会过滤掉
      @param minn:            构造subword时最小char个数
      @param maxn:            构造subword时最大char个数
      @param neg:             负采样
      @param wordNgrams:      n-gram个数
      @param loss:            损失函数类型, softmax, ns: 负采样, hs: 分层softmax
      @param bucket:          词扩充大小, [A, B]: A语料中包含的词向量, B不在语料中的词向量
      @param thread:          线程个数, 每个线程处理输入数据的一段, 0号线程负责loss输出
      @param lrUpdateRate:    学习率更新
      @param t:               负采样阈值
      @param label:           类别前缀
      @param verbose:         显示详细信息
      @param pretrainedVectors: 预训练的词向量文件路径, 如果word出现在文件夹中初始化不再随机
      @return model object
      """

    # 测试
    train_result = classifier.test(sink2)
    print('train_precision:', train_result[1])
    print('train_recall:', train_result[2])
    print('Number of train examples:', train_result[0])

