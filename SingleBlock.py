import torch
import torch.nn as nn
import math
from transformer_hyper_param_def import *
'''
Multi-head Attention
softmax((K V^t)/sqrt(dk))Q -> concat -> linear ==>[output]
    (K V^t)/sqrt(dk))Q size:[batch_size, n_heads, seq,dq]
    concat size :[batch_size, seq, dq]
    output size :[batch_size, max_len, embedding_size]
'''


class MultiHead_Attention(nn.Module):
    def __init__(self):
        super(MultiHead_Attention, self).__init__()
        # 参数矩阵Wq, Wk, Wv
        self.W_Q = nn.Linear(embedding_size, d_q * n_heads)
        self.W_K = nn.Linear(embedding_size, d_k * n_heads)
        self.W_V = nn.Linear(embedding_size, d_v * n_heads)
        self.Linear = nn.Linear(embedding_size, d_v * n_heads)

    def forward(self, inputs, mask=False, enc_to_dec=None):
        # embedding size : [batch_size,seq_len,embedding_size]
        batch_size = inputs.size()[0]

        #   计算多头的Qi Ki Vi矩阵
        #   [batch_size, n_heads, seq, d_i]
        q = self.W_Q(inputs).view(batch_size, -1, n_heads, d_q).transpose(1, 2)
        if enc_to_dec is not None:
            k = self.W_K(enc_to_dec).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
            v = self.W_V(enc_to_dec).view(batch_size, -1, n_heads, d_v).transpose(1, 2)
        else:
            k = self.W_K(inputs).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
            v = self.W_V(inputs).view(batch_size, -1, n_heads, d_v).transpose(1, 2)

        # 计算 attention_score 注意这边不要直接矩阵转置v,需要转置的维度仅有dv,max_len！
        kv = k * v.transpose(-1, -2) / math.sqrt(d_k)  # [batch_size, n_heads, seq_len, seq_len]
        if mask:
            kv += torch.tril(torch.ones_like(kv), -1e9)

        # [batch_size, n_heads, seq_len, dq] => [batch_size, seq_len, embedding_size]
        return self.Linear(torch.sum(torch.softmax(kv, 2) * q, 2))  # concat&linear

'''
input size: [batch_size, seq_len, embedding_size]
output size: [batch_size, seq_len, embedding_size]
将 Multi-Head Attention 得到的向量再投影到一个更大的空间（论文里将空间放大了 4 倍）
在那个大空间里可以更方便地提取需要的信息（使用 Relu 激活函数），最后再投影回 token 向量原来的空间
'''


class FeedForward(nn.Module):
    def __init__(self):
        super(FeedForward, self).__init__()
        self.W_1 = nn.Linear(embedding_size, embedding_size * 4, True)
        self.W_2 = nn.Linear(embedding_size * 4, embedding_size, True)

    def forward(self, pre_out):
        return self.W_2(nn.ReLU(self.W_1(pre_out)))


'''
input size: [batch_size, seq_len, embedding_size]
output size: [batch_size, seq_len, embedding_size]
残差链接 + Layer Normalization
'''


class Add_Norm(nn.Module):
    def __init__(self):
        super(Add_Norm, self).__init__()
        self.layer_norm = nn.LayerNorm(embedding_size)

    def forward(self, residual, pre_out):
        return self.layer_norm(pre_out + residual)



