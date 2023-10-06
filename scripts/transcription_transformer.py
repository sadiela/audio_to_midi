import torch
from torch import nn
import torch.nn.functional as F
import sys
import math
import random
import copy 
import os
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EOS_IDX = 0
PAD_IDX = 1
BOS_IDX = 2

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
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=PAD_IDX)
        self.emb_size = emb_size

    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

class FeedForwardWithGEGLU(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super(FeedForwardWithGEGLU, self).__init__()
        self.W = nn.Linear(embed_dim, hidden_dim) 
        self.V = nn.Linear(embed_dim, hidden_dim) 
        self.W2= nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        geglu = torch.mul(F.gelu(self.W(x)),self.V(x))
        out = self.W2(geglu)
        return out

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout=0.0):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(DEVICE)
        
    def forward(self, query, key, value, mask = None):        
        batch_size = query.shape[0]
        
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
                
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        #Q = [batch size, query len, hid dim]
        #K = [batch size, key len, hid dim]
        #V = [batch size, value len, hid dim]
                
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim]
                
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
        #energy = [batch size, n heads, query len, key len]
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -float('inf'))
        
        attention = torch.softmax(energy, dim = -1)
                
        #attention = [batch size, n heads, query len, key len]
                
        x = torch.matmul(self.dropout(attention), V)
        
        #x = [batch size, n heads, query len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        #x = [batch size, query len, n heads, head dim]
        
        x = x.view(batch_size, -1, self.hid_dim) # concatenate heads
        
        #x = [batch size, query len, hid dim]
        x = self.fc_o(x)
        
        #x = [batch size, query len, hid dim]
        return x, attention

class EncoderLayer(nn.Module):
    def __init__(self, emb_dim, num_heads, d_ff, dropout=0.0):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttentionLayer(emb_dim, num_heads)
        self.feed_forward = FeedForwardWithGEGLU(emb_dim, d_ff)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x): #, mask):
        #print("ATTN INPUT DIMS:", x.shape)
        # x = [batch_size, src_len, embed_dim]
        attn_output = self.self_attn(x, x, x) #, mask) # for the forward encoder layer, we do not need a mask
                                               # The model is allowed to see the full input and there is no padding in the spectrograms   
        #print("ATTN OUTPUT DIMS:", attn_output[0].shape)    
        x = self.norm1(x + self.dropout(attn_output[0]))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        # x = [batch_size, src_len, hid_dim]
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, emb_dim, num_heads, d_ff, dropout=0.0):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttentionLayer(emb_dim, num_heads)
        self.cross_attn = MultiHeadAttentionLayer(emb_dim, num_heads)
        self.feed_forward = FeedForwardWithGEGLU(emb_dim, d_ff)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.norm3 = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, tgt_mask):
        # trg = [batch_size, tgt_len, hid_dim]
        # enc_output = [batch_size, src_len, hid_dim]
        # tgt_mask = [batch_size, 1, tgt_len, tgt_len]
        #print("DECODER LAYER:")
        #print("trg, enc_out, mask size:", x.shape, enc_output.shape, tgt_mask.shape)

        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output[0]))
        attn_output = self.cross_attn(x, enc_output, enc_output) #WOULD BE SOURCE PADDING BUT DONT NEED, key_padding_mask=src_padding_mask, attn=cross_lookahead_mask)
        x = self.norm2(x + self.dropout(attn_output[0]))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x
    
class TranscriptionTransformer(nn.Module):
    def __init__(self, n_enc, n_dec, emb_dim, nhead, vocab_size, ffn_hidden, dropout=0.0):
        super(TranscriptionTransformer, self).__init__()

        self.feedforward_src_emb = nn.Linear(emb_dim, emb_dim)
        self.tgt_emb = TokenEmbedding(vocab_size, emb_dim)

        self.positional_encoding = PositionalEncoding(emb_dim)

        # actual transformer layers
        self.encoder_layers = nn.ModuleList([EncoderLayer(emb_dim, nhead, ffn_hidden, dropout) for _ in range(n_enc)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(emb_dim, nhead, ffn_hidden, dropout) for _ in range(n_dec)])

        self.generator = nn.Linear(emb_dim, vocab_size) # generator
        self.dropout = nn.Dropout(dropout)

    def make_trg_mask(self, trg):
        #trg = [batch size, trg len]
        trg_pad_mask = (trg != PAD_IDX).unsqueeze(1).unsqueeze(2)
        
        #trg_pad_mask = [batch size, 1, 1, trg len]
        trg_len = trg.shape[1]
        
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = DEVICE)).bool()
        #trg_sub_mask = [trg len, trg len]
            
        trg_mask = trg_pad_mask & trg_sub_mask
        #trg_mask = [batch size, 1, trg len, trg len]
        
        return trg_mask

    def forward(self, src, tgt):
        #src = [batch size, src len, embed_dim]
        #trg = [batch size, trg len]

        tgt_mask = self.make_trg_mask(tgt).to(DEVICE) # no padding mask needed for input

        # embedd the data
        src_embedded = self.dropout(self.positional_encoding(self.feedforward_src_emb(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.tgt_emb(tgt)))

        # src_embedded = [batch size, src len, embed_dim]
        # tgt_embedded = [batch size, tgt len, embed_dim]

        # encoder layers
        enc_output = src_embedded
        for enc_layer in self.encoder_layers: # run encoder layers!
            enc_output = enc_layer(enc_output) #, src_mask)

        # decoder layers
        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, tgt_mask)

        output = self.generator(dec_output)
        return output
    
    def encode(self, src):
        src_emb = self.positional_encoding(self.feedforward_src_emb(src)) # no dropout i assumed

        enc_output = src_emb
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output) #, src_mask)
    
        return enc_output
    
    def decode(self, tgt, enc_output, trg_mask): #, tgt_padding_mask, self_lookahead_mask):
        tgt_embedded = self.positional_encoding(self.tgt_emb(tgt))
        
        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, trg_mask) # takes 4 args, 5 given
        
        #output = self.generator(dec_output)
        print(dec_output.shape, dec_output[:,-1].shape)
        last_output = self.generator(dec_output[:, -1])

        #out = out.transpose(0, 1)
        #print("OUTPUT SHAPE:", out.shape)
        #prob =  # last word?
        
        return last_output #output
    
    def greedy_decode(self, src, max_len, start_symbol):
        src = src.to(DEVICE).to(torch.float32)
        memory = self.encode(src)
        ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
        #print("ys:", ys.shape, ys)
        for i in range(max_len-1):
            memory = memory.to(DEVICE)
            tgt_mask = self.make_trg_mask(ys)
            #tgt_padding_mask = (ys == PAD_IDX).to(DEVICE)
            #self_lookahead_mask = (self.lookahead_mask(ys.size(0),ys.size(0)))#.type(torch.bool)).to(DEVICE)
            #print(ys.shape, memory.shape, tgt_mask.shape)
            prob = self.decode(ys, memory, tgt_mask) #, tgt_padding_mask, self_lookahead_mask) #k.shape[0], bsz * num_heads, head_dim
            _, next_word = torch.max(prob, axis=1)
            next_word = next_word.item()

            ys = torch.cat([ys,
                            torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
            if next_word == EOS_IDX:
                break

            print(ys)
        return ys
    
    def translate(self, src):
        self.eval()
        #src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
        tgt_tokens = self.greedy_decode(
            src, max_len=1024, start_symbol=BOS_IDX).flatten()
        return tgt_tokens