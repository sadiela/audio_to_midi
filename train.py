import torch 
from simple_transformer import *
from audio_midi_dataset import *
from timeit import default_timer as timer

torch.manual_seed(0)


midi_dir = './small_matched_data/midi/'
audio_dir = './small_matched_data/raw_audio/'
SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = 630
EMB_SIZE = 512
NHEAD = 4
FFN_HID_DIM = 512
BATCH_SIZE = 32 # 256
NUM_ENCODER_LAYERS = 3 #8
NUM_DECODER_LAYERS = 3 #8
NUM_EPOCHS = 5

def train_epoch(model, optimizer):
    model.train()
    losses = 0
    training_data = AudioMidiDataset(audio_file_dir=audio_dir, midi_file_dir=midi_dir)
    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for src, tgt in train_dataloader:
        print("Source:", src[:,0], src[0,:])
        print("Target:", tgt[:,0], tgt[0,:])
        print("Source:", src.shape)
        print("Target:", tgt.shape)
        input("Continue...")
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
        print("Source mask:", src_mask.shape)
        print("Target mask:", tgt_mask.shape)
        print("Source padding mask:", src_padding_mask.shape)
        print("Target padding mask:", tgt_padding_mask.shape)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)
        print("outputs post softmax:", logits.shape)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()
        input("Continue...")

def evaluate(model):
    model.eval()
    losses = 0

    val_iter = AudioMidiDataset(audio_file_dir=audio_dir, midi_file_dir=midi_dir)
    val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for src, tgt in val_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(list(val_dataloader))


if __name__ == '__main__':

    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                    NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    transformer = transformer.to(DEVICE)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    for epoch in range(1, NUM_EPOCHS+1):
        start_time = timer()
        train_loss = train_epoch(transformer, optimizer)
        end_time = timer()
        val_loss = evaluate(transformer)
        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))

