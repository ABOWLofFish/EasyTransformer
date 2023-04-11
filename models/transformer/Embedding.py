import torch
import torch.nn as nn
from transformer_hyper_param_def import *
import math

class PositionalEmbedding(nn.Module):
    """
    Positional Embedding steps:
        -> embedding = X_embedding + PE

    output size : [batch_size,max_len,embedding_size]

    PE = sin((pos/1e^{5+(2i/d_model)}) /cos
        可以推出三角函数内部共同计算部分
        common_div  ==> pos * e^{-(2i/d_model * log(1e5)}
                    ==> torch.exp(-log(10000.0)/d_model * pos||pos-1)
        PE.size() = [max_len ,embedding_size]
    """
    def __init__(self, max_len=5000):
        super(PositionalEmbedding, self).__init__()

        pe = torch.zeros(max_len, embedding_size)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(-math.log(1e5 * 1.0) / embedding_size * torch.arange(0, embedding_size, 2).float())

        # pe分别从0，1取步长为2计算每个位置的positional embedding
        pe[:, 1::2] = torch.sin(pos * div)
        pe[:, 0::2] = torch.cos(pos * div)

        # [batch_size,max_len,embedding_size]
        pe = pe.unsqueeze(0)
        # 参数放入缓冲区，无需更新
        self.register_buffer('pe', pe)

    def forward(self, x):
        x.transpose(1, 2)
        x += self.pe[:, 0:x.size(1), :]
        return x  # [batch_size, seq_len, embedding_size]
