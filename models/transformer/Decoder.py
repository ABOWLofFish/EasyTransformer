import torch
import torch.nn as nn
from SingleBlock import *


class DecoderLayer(nn.Module):
    """
        Decoder层步骤
    """
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.multi_head_attention = MultiHead_Attention()
        self.add_norm = Add_Norm()
        self.feedforward = FeedForward()
        self.dropout = nn.Dropout()

    '''
    Decoder layer steps
    1.masked-multi-head-attention => dropout->add&norm
    2.enc_to_dec -> multi-head-attention => dropout->add&norm
    3.FF => dropout->add&norm
    '''
    def forward(self, inputs, enc_to_dec):
        # 1
        masked_attn_out = self.dropout(self.multi_head_attention.forward(inputs, True))
        add_norm_out = self.add_norm.forward(inputs, masked_attn_out)

        # 2
        attn_out = self.dropout(self.multi_head_attention.forward(add_norm_out, False, enc_to_dec))
        add_norm_out = self.add_norm.forward(add_norm_out, attn_out)

        # 3
        ff_out = self.dropout(self.feedforward(add_norm_out))
        add_norm_out = self.add_norm.forward(add_norm_out, ff_out)

        return add_norm_out
