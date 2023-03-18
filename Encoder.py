from SingleBlock import *


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = MultiHead_Attention()
        self.add_norm = Add_Norm()
        self.feedforward = FeedForward()

    '''
    Encoder layer steps
    1.multi-head-attention => add&norm
    2.FF => add&norm
    '''
    def forward(self, inputs):
        # 1
        multi_attn = self.multi_head_attention.forward(inputs)
        add_norm_out = self.add_norm.forward(inputs, multi_attn)

        # 2
        ff_out = self.feedforward.forward(add_norm_out)
        return self.add_norm(ff_out, add_norm_out)
