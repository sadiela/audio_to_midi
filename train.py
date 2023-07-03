import torch 
from simple_transformer import *
from audio_midi_dataset import *
from timeit import default_timer as timer
import torch_optimizer as optim

torch.manual_seed(0)

midi_dir = './small_matched_data/midi/'
audio_dir = './small_matched_data/raw_audio/'
SRC_VOCAB_SIZE = 512 #len(vocab_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = 631
EMB_SIZE = 512
NHEAD = 4
FFN_HID_DIM = 512
BATCH_SIZE = 2 # 256
NUM_ENCODER_LAYERS = 2 #8
NUM_DECODER_LAYERS = 2 #8
NUM_EPOCHS = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_DIR = './models'

def train_epoch(model, optimizer, loss_fn):
    model.train()
    losses = 0
    training_data = AudioMidiDataset(audio_file_dir=audio_dir, midi_file_dir=midi_dir)
    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    print("BEGINNING TRAINING LOOP")
    for src, tgt in train_dataloader:
        src = src.to(DEVICE).to(torch.float32)
        tgt = tgt.to(DEVICE).to(torch.float32)

        tgt_input = tgt[:-1, :]

        #print("INPUT SIZE FOR CREATE_MASK FUNC:", src.shape, tgt.shape)
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
        #print("outputs post softmax:", logits.shape)

        optimizer.zero_grad()

        logits = logits.reshape(-1, logits.shape[-1])
        tgt_out = tgt[1:, :].reshape(-1).to(torch.long)
        #print(logits.shape, tgt_out.shape)
        # 631 logits
        loss = loss_fn(logits, tgt_out)
        print("LOSS:", loss)
        loss.backward()

        optimizer.step()
        losses += loss.item()
        #nput("Continue...")

def evaluate(model, loss_fn):
    model.eval()
    losses = 0

    val_iter = AudioMidiDataset(audio_file_dir=audio_dir, midi_file_dir=midi_dir)
    val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for src, tgt in val_dataloader:
        src = src.to(DEVICE).to(torch.float32)
        tgt = tgt.to(DEVICE).to(torch.long)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(list(val_dataloader))


def training_setup():
    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                    NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    transformer = transformer.to(DEVICE)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    #optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    optimizer = optim.Adafactor(transformer.parameters(), lr=1e-3)

    for epoch in range(1, NUM_EPOCHS+1):
        start_time = timer()
        train_loss = train_epoch(transformer, optimizer, loss_fn)
        end_time = timer()
        val_loss = evaluate(transformer, loss_fn)
        print((f"Epoch: {epoch}, Train loss: {train_loss}, Val loss: {val_loss}, "f"Epoch time = {(end_time - start_time)}s"))
    
    torch.save(transformer.state_dict(), MODEL_DIR + '/model1.pt')


if __name__ == '__main__':
    # save model
    # LOAD
    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)
    transformer.load_state_dict(torch.load(MODEL_DIR + '/model1.pt'))
    transformer.to(DEVICE).eval()

    test_audio_file = './small_matched_data/raw_audio/23ae70e204549444ec91c9ee77c3523a_6.wav'
    y, sr = librosa.load(test_audio_file, sr=SAMPLE_RATE)
    y = split_audio(y,SEG_LENGTH)
    # convert to melspectrograms
    M =  librosa.feature.melspectrogram(y=y, sr=sr, 
              hop_length=HOP_WIDTH, 
              n_fft=FFT_SIZE, 
              n_mels=NUM_MEL_BINS, 
              fmin=MEL_LO_HZ, fmax=7600.0)
    # transpose to be SEQ_LEN x BATCH_SIZE x EMBED_DIM
    M_transposed = np.transpose(M, (2, 0, 1)) # append EOS TO THE END OF EACH SEQUENCE!
    eos_block = LEARNABLE_EOS * np.ones((1, M_transposed.shape[1], NUM_MEL_BINS))
    M_transposed = np.append(M_transposed, np.atleast_3d(eos_block), axis=0)
    # TARGET SIZE: 512x6x512
    # logscale magnitudes
    M_db = librosa.power_to_db(M_transposed, ref=np.max)

    print(M_db.shape)
    print("READY TO TRANSLATE")
    
    transformer.translate(torch.tensor(M_db))

    input("Continue...")
