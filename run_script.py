import random

from EasyTransformer import EasyTransformer
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def set_vocab_size():
    return len(src_vocab), len(tgt_vocab)

def make_batch(sentences):
    input_batch = [[src_vocab[word] for word in sentences[0].split()]]
    output_batch = [[tgt_vocab[word] for word in sentences[1].split()]]
    target_batch = [[tgt_vocab[word] for word in sentences[2].split()]]
    return torch.LongTensor(input_batch), torch.LongTensor(output_batch), torch.LongTensor(target_batch)

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    set_seed(0)
    # build vocabulary，
    sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']

    src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4}

    tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'S': 5, 'E': 6}

    src_len = 5  # length of source
    tgt_len = 5  # length of target

    # def model,criterion,optimizer
    model = EasyTransformer()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0004)

    # pack_batch
    # [batch_size,seq_len]
    enc_inputs, dec_inputs, target_batch = make_batch(sentences)

    # begin training
    print('=======training begin=======')
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = model(enc_inputs, dec_inputs).view(-1, len(tgt_vocab))
        loss = criterion(outputs, target_batch.contiguous().view(-1))
        print('epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        loss.backward()
        optimizer.step()
