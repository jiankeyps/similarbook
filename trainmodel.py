#coding:utf-8
from gensim import models,corpora,similarities
import jieba.posseg as pseg
from gensim.models.doc2vec import TaggedDocument,Doc2Vec
import os

def a_sub_b(a,b):
    ret = []
    for el in a:
        if el not in b:
            ret.append(el)
    return ret
stop = [line.strip().decode('utf-8') for line in open('stopword.txt').readlines() ]

#读取文件
raw_documents=[]
walk = os.walk(os.path.realpath("/Users/liuke/dev/kenbook/text"))
for root, dirs, files in walk:
    for name in files:
        f = open(os.path.join(root, name), 'r')
        raw={"book":'',"content":''}
        raw['book'] = name
        raw['content'] += f.read()
        raw_documents.append(raw)

#构建语料库
corpora_documents = []
doc=[]        
for i, item_text in enumerate(raw_documents):
    words_list=[]
    item=(pseg.cut(item_text['content']))
    for j in list(item):
        words_list.append(j.word)
    words_list=a_sub_b(words_list,list(stop))
    document = TaggedDocument(words=words_list, tags=[item_text['book'].replace(".txt","").decode('utf-8')])
    corpora_documents.append(document)
    doc.append(words_list)
#创建model
model = Doc2Vec(size=50, min_count=1, iter=10)
model.build_vocab(corpora_documents)
model.train(corpora_documents)
model.save("model/book.model")
