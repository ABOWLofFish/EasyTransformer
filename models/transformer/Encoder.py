from SingleBlock import *


class EncoderLayer(nn.Module):
    """
    Encoder层步骤
    """
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = MultiHead_Attention()
        self.add_norm = Add_Norm()
        self.feedforward = FeedForward()
        self.dropout = nn.Dropout()

    '''
    Encoder layer steps
    1.multi-head-attention => dropout->add&norm
    2.FF => dropout->add&norm
    '''
    def forward(self, inputs):
        # 1
        multi_attn = self.dropout(self.multi_head_attention.forward(inputs))
        add_norm_out = self.add_norm.forward(inputs, multi_attn)

        # 2
        ff_out = self.dropout(self.feedforward.forward(add_norm_out))
        return self.add_norm(ff_out, add_norm_out)
