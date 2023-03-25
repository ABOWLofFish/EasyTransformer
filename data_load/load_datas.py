import data_load.vocab
'''
return 
src_vocab
tgt_vocab
sentence[['input','output','tgt'],]
'''
def organize(tgt, src):
    dec_input = ['<sos> ' + item[:-1] for item in tgt]
    target = [item[:-1] + ' <eos>' for item in tgt]
    enc_input = [line[:-1] for line in src]
    data_set = []

    for tup in zip(enc_input, dec_input, target):
        data_set += [[tup[0], tup[1], tup[2]]]
    return data_set

def load_corpus(path):
    ''' 读取文件'''
    train_src = open(path+"en\\train.en", 'r', encoding='utf-8')
    train_tgt = open(path+"spa\\train.spa", 'r', encoding='utf-8')
    dev_src = open(path+"en\\dev.en", 'r', encoding='utf-8')
    dev_tgt = open(path+"spa\\dev.spa", 'r', encoding='utf-8')
    test_src = open(path+"en\\test.en", 'r', encoding='utf-8')
    test_tgt = open(path+"spa\\test.spa", 'r', encoding='utf-8')

    ''' 加载语料，构建词典'''
    train_en = train_src.readlines()
    train_en = train_en[:128]
    train_spa = train_tgt.readlines()
    train_spa = train_spa[:128]

    dev_en = dev_src.readlines()
    dev_en = dev_en[:128]
    dev_spa = dev_tgt.readlines()
    dev_spa = dev_spa[:128]

    test_en = test_src.readlines()
    test_en = test_en[:128]
    test_spa = test_tgt.readlines()
    test_spa = test_spa[:128]

    vocab_en = data_load.vocab.Vocab(train_en + dev_en + test_en, 1, [])
    vocab_spa = data_load.vocab.Vocab(train_spa + dev_spa + test_spa, 1, ['<eos>', '<sos>'])

    ''' 句子组织成model读取的pattern'''
    train_set = organize(train_spa, train_en)
    dev_set = organize(dev_spa, dev_en)
    test_set = organize(test_spa, test_en)

    return vocab_en, vocab_spa, train_set, dev_set, test_set

