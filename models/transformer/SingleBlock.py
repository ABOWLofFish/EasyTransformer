import torch
import torch.nn as nn
import math
from transformer_hyper_param_def import *
'''
Transformer中可直接组装的基本模块
'''
class MultiHead_Attention(nn.Module):
    """
    多头注意力机制 steps：
        -> {Q, K, V}
        -> softmax((K*V^t)/sqrt(dk))Q  <-----(*equ)
        -> concat
        -> linear
        ==>[output]

    *equ size:[batch_size, n_heads, seq,dq]
    concat size :[batch_size, seq, dq]
    output size :[batch_size, seq, embedding_size]
    """
    def __init__(self):
        super(MultiHead_Attention, self).__init__()
        # 参数矩阵Wq, Wk, Wv
        self.W_Q = nn.Linear(embedding_size, d_q * n_heads)
        self.W_K = nn.Linear(embedding_size, d_k * n_heads)
        self.W_V = nn.Linear(embedding_size, d_v * n_heads)
        self.Linear = nn.Linear(d_q, embedding_size)

    def forward(self, inputs, mask=False, enc_to_dec=None):
        # embedding size : [batch_size,seq_len,embedding_size]
        batch_size = inputs.size()[0]

        #   计算多头的Qi Ki Vi矩阵 [batch_size, n_heads, seq, d_i]
        q = self.W_Q(inputs).view(batch_size, -1, n_heads, d_q).transpose(1, 2)
        if enc_to_dec is not None:
            k = self.W_K(enc_to_dec).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
            v = self.W_V(enc_to_dec).view(batch_size, -1, n_heads, d_v).transpose(1, 2)
        else:
            k = self.W_K(inputs).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
            v = self.W_V(inputs).view(batch_size, -1, n_heads, d_v).transpose(1, 2)

        # 计算 attention_score 注意不要直接矩阵转置v,需要转置的维度仅有dv,max_len
        kv = torch.matmul(k, v.transpose(-1, -2)) / math.sqrt(d_k)  # [batch_size, n_heads, seq_len, seq_len]
        if mask:
            masked = torch.triu(torch.ones(kv.shape[2], kv.shape[3], dtype=bool), diagonal=1)  # masked必须指定bool值才能覆盖
            kv = torch.masked_fill(kv, mask=masked, value=float('-inf'))

        # [batch_size, n_heads, seq_len, seq_len] => [batch_size, seq_len, embedding_size]
        concat_n = torch.sum(torch.matmul(torch.softmax(kv, dim=2), q), dim=1)  # [batch_size, seq_len, d_q]
        return self.Linear(concat_n)  # concat&linear


class FeedForward(nn.Module):
    """
    FF层 steps:
        ->Linear 投影到更大空间（paper中 dim*4）
        ->Relu
        ->Linear (dim//4)
        ==>[output]

    input size: [batch_size, seq_len, embedding_size]
    output size: [batch_size, seq_len, embedding_size]
    """
    def __init__(self):
        super(FeedForward, self).__init__()
        self.W_1 = nn.Linear(embedding_size, embedding_size * 4, True)
        self.W_2 = nn.Linear(embedding_size * 4, embedding_size, True)
        self.act_F = nn.ReLU()

    def forward(self, pre_out):
        act_out = self.act_F(self.W_1(pre_out))
        return self.W_2(act_out)


class Add_Norm(nn.Module):
    """
    残差连接+LN层 steps:
        ->add residual
        ->layer norm

    input size: [batch_size, seq_len, embedding_size]
    output size: [batch_size, seq_len, embedding_size]
    """
    def __init__(self):
        super(Add_Norm, self).__init__()
        self.layer_norm = nn.LayerNorm(embedding_size)

    def forward(self, residual, pre_out):
        return self.layer_norm(pre_out + residual)



