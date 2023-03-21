import data_load.vocab
'''
return 
src_vocab
tgt_vocab
sentence[['input','output','tgt'],]
'''
def load_corpus(path):
    ''' 读取文件'''
    train_src = open(path+"\\en\\train.en", 'r', encoding='utf-8')
    train_tgt = open(path+"\\spa\\train.spa", 'r', encoding='utf-8')

    ''' 加载词典'''
    train_en = train_src.readlines()
    vocab_en = data_load.vocab.Vocab(train_en, 1, [])
    train_spa = train_tgt.readlines()
    vocab_spa = data_load.vocab.Vocab(train_spa, 0, ['<eos>', '<sos>'])

    ''' 句子组织成model读取的pattern'''
    outputs = ['<sos> '+item[:-1] for item in train_spa]
    tgt = [item[:-1]+' <eos>' for item in train_spa]
    inputs = [line[:-1] for line in train_en]
    sentences = []

    for tup in zip(inputs, outputs, tgt):
        sentences += [[tup[0], tup[1], tup[2]]]

    return vocab_en, vocab_spa, sentences

