import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import data_load.load_datas
from models.transformer.EasyTransformer import EasyTransformer
from evalutaion import bleu


def set_vocab_size():
    return len(src_vocab), len(tgt_vocab)


def make_batch(corpus: list, batch_size: int):
    inputs, outputs, tgt = shuffle_and_padding(sentence=corpus)
    input_batch, output_batch, tgt_batch = [], [], []
    corpus = np.array(corpus)
    # print("inputs[0][0]: ", inputs[0])
    # print("shape: ", inputs.shape)
    idx_list = np.arange(len(corpus))  # 0~sentence_num-1
    while len(idx_list) > batch_size:
        idx = np.random.choice(idx_list, size=batch_size)
        # append batch
        input_batch.append(inputs[idx])
        output_batch.append(outputs[idx])
        tgt_batch.append(tgt[idx])
        # del batch from idx_list to avoid duplicate choice
        idx_list = np.setdiff1d(idx_list, idx)
    enc_inputs, dec_inputs, target = torch.LongTensor(input_batch), torch.LongTensor(output_batch), torch.LongTensor(
        tgt_batch)
    # append final batch with size < batch_size
    # input_batch.append(inputs[idx_list])
    # output_batch.append(outputs[idx_list])
    # tgt_batch.append(tgt[idx_list])
    # enc_inputs = torch.cat((enc_inputs, torch.unsqueeze(torch.LongTensor(inputs[idx_list]), dim=0)), dim=1)
    # dec_inputs = torch.cat((dec_inputs, torch.unsqueeze(torch.LongTensor(outputs[idx_list]), dim=0)), dim=1)
    # target = torch.cat((target, torch.unsqueeze(torch.LongTensor(tgt[idx_list]), dim=0)), dim=1)
    return enc_inputs, dec_inputs, target


def shuffle_and_padding(sentence):
    maxLen = 40
    random.shuffle(sentence)
    # print("shuffle corpus[0][0].split", sentence[0][0].split(" "))
    enc_input_batch, dec_input_batch, target_batch = [], [], []
    for i in range(len(sentence)):
        enc_input_batch.append(src_vocab.to_idx(words=sentence[i][0].split(" "), maxLen=maxLen))
        dec_input_batch.append(tgt_vocab.to_idx(words=sentence[i][1].split(" "), maxLen=maxLen))
        target_batch.append(tgt_vocab.to_idx(words=sentence[i][2].split(" "), maxLen=maxLen))
    return np.array(enc_input_batch), np.array(dec_input_batch), np.array(target_batch)


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    set_seed(0)
    src_vocab, tgt_vocab, train, dev, test = data_load.load_datas.load_corpus(
        "D:\\PycharmProjects\\transformerdemo\\datas\\data\\spa-eng\\")  # src_path, tgt_path

    ''' def model,criterion,optimizer '''
    model = EasyTransformer()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.0004)

    ''' begin training '''
    for epoch in range(30):
        optimizer.zero_grad()
        print("making batch...")
        # [batch_size, ]
        enc_inputs, dec_inputs, target_batch = make_batch(train, 64)  # pack_batch
        print("start training")
        Loss = 0
        for i, (enc_input, dec_input, target) in enumerate(zip(enc_inputs, dec_inputs, target_batch)):
            outputs = model(enc_input, dec_input).view(-1, len(tgt_vocab))
            loss = criterion(outputs, target.contiguous().view(-1))
            loss.backward()
            print('batch:', '%04d' % i, 'cost =', '{:.6f}'.format(loss))
            Loss += loss
        print('epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(Loss / enc_inputs.shape[0]))

        # '''evaluate on dev set'''
        # enc_inputs, dec_inputs, target_batch = make_batch(dev, 64)  # pack_batch
        # print("evaluate on dev set")
        # for (enc_input, dec_input, target) in zip(enc_inputs, dec_inputs, target_batch):
        #     outputs = model(enc_input, dec_input)  # [batch_size, seq_len, tgt_vocab_size]
        #     outputs = torch.max(outputs, dim=-1)  # indices.shape = [batch_size, seq_len]
        #     candidates = []
        #     for line in range(outputs.indices.shape[0]):
        #         print(candidates[0,:].type)
        #         candidates.append(src_vocab.to_tokens(outputs.indices[line, :]))
        #     score = bleu.corpus_bleu(target, candidates)
        #     print("BLEU:", '{:.6f}'.format(score))

        optimizer.step()
