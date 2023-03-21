import random

from EasyTransformer import EasyTransformer
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import data_load.load_datas

def set_vocab_size():
    return len(src_vocab), len(tgt_vocab)

def make_batch(corpus, batch_size):
    # random.shuffle(corpus)
    input_batch, output_batch, tgt_batch = [], [], []
    for i in range(batch_size):
        input_batch.append([src_vocab.__getitem__(corpus[i][0])])
        output_batch.append([tgt_vocab.__getitem__(corpus[i][1])])
        tgt_batch.append([tgt_vocab.__getitem__(corpus[i][2])])
    return torch.LongTensor(input_batch), torch.LongTensor(output_batch), torch.LongTensor(tgt_batch)

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    set_seed(0)

    src_vocab, tgt_vocab, sentences = data_load.load_datas.load_corpus("D:\\PycharmProjects\\transformerdemo\\datas\\data\\spa-eng\\")    # src_path, tgt_path
    ''' def model,criterion,optimizer '''
    model = EasyTransformer()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    ''' begin training '''
    print('=======training begin=======')
    for epoch in range(30):
        optimizer.zero_grad()
        enc_inputs, dec_inputs, target_batch = make_batch(sentences, 100)    # pack_batch
        outputs = model(enc_inputs, dec_inputs).view(-1, len(tgt_vocab))
        loss = criterion(outputs, target_batch.contiguous().view(-1))
        print('epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        loss.backward()
        optimizer.step()
