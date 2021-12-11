from word2vec import Sentences
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from pprint import pprint


def lda_train(f_input, f_output):
    sentences = Sentences(f_input)
    dic = Dictionary(sentences)
    dic.filter_extremes()  # 过滤掉一些极端词汇
    dic.compactify()  # 修正id，避免过滤后造成id gap
    dic.save(f_output + "dic")

    lda_model = LdaModel(corpus=[dic.doc2bow(sen) for sen in sentences], id2word=dic, num_topics=5, iterations=400)
    lda_model.save(f_output)


if __name__ == '__main__':
    input = "data/data.txt"
    output = "model/lda"
    lda_train(input, output)
    lda_model = LdaModel.load(output)
    topics = lda_model.print_topics(5)
    pprint(topics)
