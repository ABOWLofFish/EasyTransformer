import math
import pandas as pd
from nltk import word_tokenize
import random

stopword_list = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']

if __name__ == '__main__':
    df = pd.read_table("spa.txt", names=['eng', 'spa'])
    # print(df.head)
    eng = []
    spa = []
    '''清洗数据集'''
    for tup in zip(df['eng'], df['spa']):
        words = word_tokenize(tup[0])
        eng.append([word.lower() for word in words if word not in stopword_list])
        words = word_tokenize(tup[1])
        spa.append([word.lower() for word in words if word not in stopword_list])

    '''划分数据集'''
    num = len(eng)

    random.shuffle(eng)
    random.shuffle(spa)
    train, dev = math.ceil(num * 0.7), math.ceil(num * 0.8)
    train_src, dev_src, test_src = eng[:train], eng[train:dev], eng[dev:num]
    train_tgt, dev_tgt, test_tgt = spa[:train], spa[train:dev], spa[dev:num]

    '''写入文件'''
    train_file = open("D:\\PycharmProjects\\transformerdemo\\datas\\data\\spa-eng\\en\\train.en", 'w', encoding='utf-8')
    dev_file = open("D:\\PycharmProjects\\transformerdemo\\datas\\data\\spa-eng\\en\\dev.en", 'w', encoding='utf-8')
    test_file = open("D:\\PycharmProjects\\transformerdemo\\datas\\data\\spa-eng\\en\\test.en", 'w', encoding='utf-8')

    train_file.writelines(" ".join(line)+"\n" for line in train_src)
    dev_file.writelines(" ".join(line)+"\n" for line in dev_src)
    test_file.writelines(" ".join(line)+"\n" for line in test_src)
    train_file.close()
    dev_file.close()
    test_file.close()

    train_file = open("D:\\PycharmProjects\\transformerdemo\\datas\\data\\spa-eng\\spa\\train.spa", 'w', encoding='utf-8')
    dev_file = open("D:\\PycharmProjects\\transformerdemo\\datas\\data\\spa-eng\\spa\\dev.spa", 'w', encoding='utf-8')
    test_file = open("D:\\PycharmProjects\\transformerdemo\\datas\\data\\spa-eng\\spa\\test.spa", 'w', encoding='utf-8')
    train_file.writelines(" ".join(line)+"\n" for line in train_tgt)
    dev_file.writelines(" ".join(line)+"\n" for line in dev_tgt)
    test_file.writelines(" ".join(line)+"\n" for line in test_tgt)
    train_file.close()
    dev_file.close()
    test_file.close()
