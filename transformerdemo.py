import torch
import torch.nn as nn
import torch.optim as optim
import math

'''
Multi-head Attention
softmax((K V^t)/sqrt(dk))Q -> concat -> linear ==>[output]
'''
class MultiHead_Attention(nn.Module):
    def __init__(self):
        super(MultiHead_Attention, self).__init__()
        # 参数矩阵Wq, Wk, Wv,并行计算多头后拆分
        self.W_Q = nn.Linear(embedding_size, d_q * n_heads)
        self.W_K = nn.Linear(embedding_size, d_k * n_heads)
        self.W_V = nn.Linear(embedding_size, d_v * n_heads)

    def forward(self, embedding):
        batch_size = embedding.shape()[0]
        #   1.计算多头Q K V
        #   E[embedding,seq] W[seq,d*n_heads] = Q,K,V[embedding,d*n_heads]
        q = self.W_Q(embedding).view(batch_size, -1, n_heads, d_q).transpose(1,
                                                                             2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k = self.W_K(embedding).view(batch_size, -1, n_heads, d_k).transpose(1,
                                                                             2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v = self.W_V(embedding).view(batch_size, -1, n_heads, d_v).transpose(1,
                                                                             2)  # v_s: [batch_size x n_heads x len_k x d_v]
        # print(q.size())
        #   2.计算 attention_score


class Mask_MultiHead_Attention(nn.Module):
    def __init__(self):
        super(Mask_MultiHead_Attention, self).__init__()
        # 输入进来的QKV是相等的，我们会使用映射linear做一个映射得到参数矩阵Wq, Wk,Wv
        self.W_Q = nn.Linear(embedding_size, d_k * n_heads)
        self.W_K = nn.Linear(embedding_size, d_k * n_heads)
        self.W_V = nn.Linear(embedding_size, d_v * n_heads)
        self.linear = nn.Linear(n_heads * d_v, embedding_size)
        self.layer_norm = nn.LayerNorm(embedding_size)


'''
embedding = X_embedding + PE
return size : [batch_size,max_len,embedding_size]
'''
class PositionalEncoding(nn.Module):
    def __init__(self, embedding_size, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        """
        PE = sin((pos*d_model)/1e^{5+2i}) /cos
        可以推出三角函数内部共同计算部分
        common_div  ==> pos * e^{-(2i/d_model * log(1e5)}
                    ==> torch.exp(-log(10000.0)/d_model * pos||pos-1)
        PE.size() = [max_len ,embedding_size]
        """
        pe = torch.zeros(max_len, embedding_size)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(-math.log(1e5 * 1.0) / embedding_size * torch.arange(0, embedding_size, 2).float())

        # pe分别从0，1取步长为2计算每个位置的positional embedding
        pe[:, 1::2] = torch.sin(pos * div)
        pe[:, 0::2] = torch.cos(pos * div)

        # [batch_size,max_len,embedding_size]
        pe = pe.unsqueeze(0).transpose(0, 1)
        # 参数放入缓冲区，无需更新
        self.register_buffer('pe', pe)

    def forward(self, x):
        x += self.pe
        return x



class FeedForward(nn.Module):
    def __init__(self):
        super(FeedForward, self).__init__()


class Add_Norm(nn.Module):
    def __init__(self):
        super(Add_Norm, self).__init__()


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.multi_head_attention = Mask_MultiHead_Attention
        self.add_norm = Add_Norm
        self.feedforward = FeedForward


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()


class Prediction(nn.Module):
    def __init__(self):
        super(Prediction, self).__init__()


class EasyTransformer(nn.Module):
    def __init__(self):
        super(EasyTransformer, self).__init__()
        self.encoder = Encoder
        self.encoder = Decoder
        self.prediction = Prediction


def make_batch(sentences):
    input_batch = [[src_vocab[word] for word in sentences[0].split()]]
    output_batch = [[tgt_vocab[word] for word in sentences[1].split()]]
    target_batch = [[tgt_vocab[word] for word in sentences[2].split()]]
    return torch.LongTensor(input_batch), torch.LongTensor(output_batch), torch.LongTensor(target_batch)


if __name__ == '__main__':
    # model param
    embedding_size = 512
    hidden_size = 256
    n_heads = 6
    Nx = 6  # number of layer(Encoder.Decoder)
    d_q = 128 * Nx
    d_k = d_v = 64 * Nx

    # build vocabulary，
    sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']

    src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4}
    src_vocab_size = len(src_vocab)

    tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'S': 5, 'E': 6}
    tgt_vocab_size = len(tgt_vocab)

    src_len = 5  # length of source
    tgt_len = 5  # length of target

    # def model,criterion,optimizer
    model = EasyTransformer()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # pack_batch
    # [batch_size,seq_len]
    enc_inputs, dec_inputs, target_batch = make_batch(sentences)

    # begin training
    print('=======training begin=======')
    for epoch in range(20):
        optimizer.zero_grad()
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
        loss = criterion(outputs, target_batch.contiguous().view(-1))
        print('epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        loss.backward()
        optimizer.step()
