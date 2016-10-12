#coding:utf-8
#使用doc2vec 判断文档相似性
from gensim import models,corpora,similarities
import jieba.posseg as pseg
from gensim.models.doc2vec import TaggedDocument,Doc2Vec
import os
import sys

book=sys.argv[1]

model = Doc2Vec.load("model/book.model")

for i in model.docvecs.most_similar(book.decode('utf-8'),topn=50):
    print i[0],"相似度:",i[1]
