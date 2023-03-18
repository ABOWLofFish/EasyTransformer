import torch
import torch.nn as nn
import math


'''
embedding = X_embedding + PE
return size : [batch_size,max_len,embedding_size]
'''
class PositionalEmbedding(nn.Module):
    def __init__(self, embedding_size, max_len=5000):
        super(PositionalEmbedding, self).__init__()

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