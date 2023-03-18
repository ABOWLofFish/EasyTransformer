import torch
import torch.nn as nn
from transformer_hyper_param_def import *
from Encoder import EncoderLayer
from Decoder import DecoderLayer
from Embedding import PositionalEmbedding


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.pos_embedding = PositionalEmbedding()
        self.layers = nn.ModuleList([EncoderLayer() for layer in range(Nx)])
    '''
    Encoder steps
    1.positional embedding
    2.Nx层循环 (Nx=6) 
    3.产生到Decoder的输入(k,v)
    '''
    def forward(self, src_embedding):
        inputs = self.pos_embedding.forward(src_embedding)
        for layer in self.layers:
            output = layer.forward(inputs)
            inputs = output
        return output

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.pos_embedding = PositionalEmbedding()
        self.encoder_out = Encoder()
        self.layers = nn.ModuleList([DecoderLayer() for layer in range(Nx)])
    '''
    1.positional embedding
    2.Nx=6 6层循环
    3.产生到Decoder的输入
    '''
    def forward(self, pre_embedding, enc_to_dec):
        inputs = self.pos_embedding.forward(pre_embedding)
        for layer in self.layers:
            output = layer.forward(inputs, enc_to_dec)
            inputs = output
        return output



class EasyTransformer(nn.Module):
    def __init__(self):
        super(EasyTransformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.prediction = nn.Linear(embedding_size, tgt_vocab_size, False)  # output_size: [batch_size, seq_len, tgt_vocab_size]

    def forward(self, raw_embedding):
        # [batch_size, seq_len, embedding]
        enc_to_dec = self.encoder.forward(raw_embedding)
        dec_out = self.decoder.forward(raw_embedding, enc_to_dec)

        # [batch_size, seq_len, tgt_vocab_size]
        return torch.softmax(self.prediction(dec_out), 2)

