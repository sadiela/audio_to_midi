import os
import torch
from torch import nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module): # probably need to change this!
    def __init__(self, emb_size: int,
                 dropout: float = 0.0,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding): # dim
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

class FeedForwardWithGEGLU(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super(FeedForwardWithGEGLU).__init__()
        self.W = nn.Linear(embed_dim, hidden_dim) 
        self.V = nn.Linear(embed_dim, hidden_dim) 
        self.W2= nn.Linear(hidden_dim, embed_dim)
        self.gelu = F.gelu()

    def forward(self, x):
        geglu = torch.mul(self.gelu(self.W(x)),self.V(x))
        out = self.W2(geglu)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.0):
        super(EncoderLayer, self).__init__()
        self.self_attn = nn.MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForwardWithGEGLU(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.0):
        super(DecoderLayer, self).__init__()
        self.self_attn = nn.MultiHeadAttention(d_model, num_heads)
        self.cross_attn = nn.MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForwardWithGEGLU(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x
    
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, emb_size, num_heads, num_layers, d_ff, max_seq_length, dropout=0.0):
        super(Transformer, self).__init__()

        self.feedforward_src_emb = nn.Linear(emb_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)

        #self.encoder_embedding = nn.Embedding(src_vocab_size, emb_size)
        #self.decoder_embedding = nn.Embedding(tgt_vocab_size, emb_size)

        self.positional_encoding = PositionalEncoding(emb_size, max_seq_length)


        # actual transformer layers
        self.encoder_layers = nn.ModuleList([EncoderLayer(emb_size, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(emb_size, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(emb_size, tgt_vocab_size) # generator
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)

        # embedd the data
        src_embedded = self.dropout(self.positional_encoding(self.feedforward_src_emb(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.tgt_tok_emb(tgt)))

        # encoder layers
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        # decoder layers
        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output