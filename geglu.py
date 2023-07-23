import torch
from torch import nn
import torch.nn.functional as F
import math
#from midi_vocabulary import *
#from audio_midi_dataset import *
import random
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
        self.embedding = nn.Embedding(vocab_size, emb_size,padding_idx=PAD_IDX)
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
        return out

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.0):
        super(EncoderLayer, self).__init__()
        self.self_attn = nn.MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForwardWithGEGLU(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x): #, mask):
        attn_output = self.self_attn(x, x, x) #, mask) # for the forward encoder layer, we do not need a mask
                                                    # The model is allowed to see the full input and there is no padding in the spectrograms       
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
        
    def forward(self, x, enc_output, src_padding_mask, tgt_padding_mask, self_lookahead_mask, cross_lookahead_mask):
        attn_output = self.self_attn(x, x, x, key_padding_mask=tgt_padding_mask, attn=self_lookahead_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, key_padding_mask=src_padding_mask, attn=cross_lookahead_mask)
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

    def forward(self, src, tgt):
        print("SRC AND TGT SHAPES:", src.shape, tgt.shape)
        S = src.shape[0]
        T = tgt.shape[0] # seq length is first

        src_padding_mask = torch.ones(S,S)
        tgt_padding_mask = (tgt == PAD_IDX).to(DEVICE) # no padding mask needed for input
        self_lookahead_mask = lookahead_mask(T,T)
        cross_lookahead_mask = lookahead_mask(T,S)

        # embedd the data
        src_embedded = self.dropout(self.positional_encoding(self.feedforward_src_emb(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.tgt_tok_emb(tgt)))

        # encoder layers
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output) #, src_mask)

        # decoder layers
        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_padding_mask, tgt_padding_mask, self_lookahead_mask, cross_lookahead_mask)

        output = self.fc(dec_output)
        return output
    
    def encode(self, src):
        src_emb = self.positional_encoding(self.feedforward_src_emb(src)) # no dropout i assumed

        enc_output = src_emb
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output) #, src_mask)

def lookahead_mask(sz1, sz2): # sz1 always T, sz2 S or T
    look_ahead_mask = torch.triu(torch.ones(sz1, sz2), diagonal=1)
    look_ahead_mask[look_ahead_mask.bool()] = -float('inf')
    return look_ahead_mask

if __name__ == '__main__':
    #transformer = Seq2SeqTransformer(n_enc=2, n_dec=2, emb_dim=512, nhead=2, vocab_size=6013, ffn_hidden=512)

    '''training_data = AudioMidiDataset(audio_file_dir=audio_dir, midi_file_dir=midi_dir)
    train_dataloader = DataLoader(training_data, batch_size=1, collate_fn=collate_fn)

    #logging.log("HOW MUCH DATA: %d", len(train_dataloader))
    for i, data in enumerate(train_dataloader):
        src = data[0].to(DEVICE).to(torch.float32)
        tgt = data[1].to(DEVICE).to(torch.float32)'''
    
    B=5 # batch size
    V=100
    N,E,S,T = 5,16,10,20 # E embedding dim, S max source sequence length, T max target sequence length
    nhead=2
    attn = nn.MultiheadAttention(embed_dim=E, num_heads=nhead)
    emb = nn.Embedding(num_embeddings=V,embedding_dim=E,padding_idx=0) # V is vocab size
    seq = torch.LongTensor([[random.randint(1,V-1) for _ in range(S)] for _ in range(N)]) # s is max sequence length
    for b in range(N):
        seq[b][random.randint(S//5, S-5):] = 0 # PADDING AT THE END
    print(seq)
    print(seq.shape)

    seq_padding_mask = (seq == 0).to(DEVICE) #.transpose(0, 1).to(DEVICE)
    print(seq_padding_mask)

    tgt_len = T
    src_len = S
    look_ahead_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1)
    look_ahead_mask[look_ahead_mask.bool()] = -float('inf')
    print(look_ahead_mask.shape, look_ahead_mask) # TxS

    mask = (torch.triu(torch.ones((tgt_len, tgt_len), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    print(mask.shape, mask)
    print(look_ahead_mask==mask)