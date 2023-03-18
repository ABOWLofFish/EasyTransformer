import run_script
# model param
embedding_size = 512
hidden_size = 256
n_heads = 6
Nx = 6  # number of layer(Encoder.Decoder)
d_q = 128 * Nx
d_k = d_v = 64 * Nx
src_vocab_size, tgt_vocab_size = run_script.set_vocab_size()
