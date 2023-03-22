import torch
import torch.nn as nn
from transformer_hyper_param_def import *
from Encoder import EncoderLayer
from Decoder import DecoderLayer
from Embedding import PositionalEmbedding


class Encoder(nn.Module):
    """
    Encoder steps:
        1.enc_tokens->positional embedding
        2.进行Nx层迭代 (Nx=6)
        (opt)最后一层产生到Decoder的输入(k,v)
    """
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, embedding_size, padding_idx=1)
        self.pos_embedding = PositionalEmbedding()
        self.layers = nn.ModuleList([EncoderLayer() for layer in range(Nx)])

    def forward(self, enc_input):
        nn.init.normal_(self.src_emb.weight, mean=0, std=embedding_size ** -0.5)
        nn.init.constant_(self.src_emb.weight[1], 0)
        raw = self.src_emb(enc_input)
        inputs = self.pos_embedding.forward(raw)
        for layer in self.layers:
            output = layer.forward(inputs)
            inputs = output
        return output

class Decoder(nn.Module):
    """
    Decoder steps:
        1.dec_tokens->positional embedding
        2.进行Nx层迭代 (Nx=6) { Encoder的K V作为multi-head-attn的一部分输入 }
        3.输出
    """
    def __init__(self):
        super(Decoder, self).__init__()
        self.src_emb = nn.Embedding(tgt_vocab_size, embedding_size)
        self.pos_embedding = PositionalEmbedding()
        self.encoder_out = Encoder()
        self.layers = nn.ModuleList([DecoderLayer() for layer in range(Nx)])

    def forward(self, dec_input, enc_to_dec):
        raw = self.src_emb(dec_input)
        inputs = self.pos_embedding.forward(raw)
        for layer in self.layers:
            output = layer.forward(inputs, enc_to_dec)
            inputs = output
        return output



class EasyTransformer(nn.Module):
    """
    Transformer:
        task:Translation
    """
    def __init__(self):
        super(EasyTransformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.prediction = nn.Linear(embedding_size, tgt_vocab_size, False)  # output_size: [batch_size, seq_len, tgt_vocab_size]

    def forward(self, enc_input, dec_input):
        # [batch_size, seq_len, embedding]
        enc_to_dec = self.encoder.forward(enc_input)
        dec_out = self.decoder.forward(dec_input, enc_to_dec)

        # [batch_size, seq_len, tgt_vocab_size] ==> [seq_len,tgt_vocab_size]
        proj_out = self.prediction(dec_out)

        return torch.softmax(proj_out.view(-1, tgt_vocab_size), 1)

