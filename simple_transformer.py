from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math
from midi_vocabulary import *
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EOS_IDX = 0
PAD_IDX = 1
BOS_IDX = 2

# DO NOT USE DROPOUT DURING PRETRAINING!!!
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module): # probably need to change this!
    def __init__(self,
                 emb_size: int,
                 dropout: float,
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

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,

                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size) # I will skip this because we are passing in spectrograms
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size) # will still need this 
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self, src: Tensor, trg: Tensor, src_mask: Tensor, tgt_mask: Tensor,
                src_padding_mask: Tensor, tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        # SRC_MASK SHAPE: SRC_MAX_BATCH_SEQ_LEN x SRC_MAX_BATCH_SEQ_LEN
        # TGT_MASK SHAPE: TGT_MAX_BATCH_SEQ_LEN x TGT_MAX_BATCH_SEQ_LEN
        # SRC_PADDING_MASK SHAPE: BATCH_SIZE x SRC_MAX_BATCH_SEQ_LEN
        # TGT_PADDING_MASK SHAPE: BATCH_SIZE x TGT_MAX_BATCH_SEQ_LEN
        src_emb = self.positional_encoding(self.src_tok_emb(src)) # won't need this step
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        # SRC SHAPE: SRC_MAX_BATCH_SEQ_LEN x BATCH_SIZE x EMBED_DIM
        # TRG SHAPE: TGT_MAX_BATCH_SEQ_LEN x BATCH_SIZE x EMBED_DIM
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        # OUT SHAPE: TGT_MAX_BATCH_SEQ_LEN x BATCH_SIZE x EMBED_DIM pre-generator
        #            TGT_MAX_BATCH_SEQ_LEN x BATCH_SIZE x TGT_VOCAB_SIZE post-generator
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)
    
    def greedy_decode(self, src, src_mask, max_len, start_symbol):
        src = src.to(DEVICE)
        src_mask = src_mask.to(DEVICE)

        memory = self.encode(src, src_mask)
        ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
        for i in range(max_len-1):
            memory = memory.to(DEVICE)
            tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                        .type(torch.bool)).to(DEVICE)
            out = self.decode(ys, memory, tgt_mask)
            out = out.transpose(0, 1)
            prob = self.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()

            ys = torch.cat([ys,
                            torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
            if next_word == EOS_IDX:
                break
        return ys

    # actual function to translate input sentence into target language
    def translate(self, src_spec):
        self.eval()
        src = src_spec
        num_tokens = src.shape[0]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
        tgt_tokens = self.greedy_decode(
            src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
        print("TARGET TOKENS:", tgt_tokens, tgt_tokens.shape)
        pretty_obj = seq_chunks_to_pretty_midi(seq_chunks, target_dir)
        return pretty_obj
    
