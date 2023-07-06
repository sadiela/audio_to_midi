import torch 
from simple_transformer import *
from audio_midi_dataset import *
from timeit import default_timer as timer
import torch_optimizer as optim
import argparse
import yaml


torch.manual_seed(0)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_DIR = './models/'

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
    return losses / len(list(train_dataloader))

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


def train(n_enc, n_dec, emb_dim, nhead, vocab_size, ffn_hidden, n_epoch, lr):
    transformer = Seq2SeqTransformer(n_enc, n_dec, emb_dim, nhead, vocab_size, ffn_hidden)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    transformer = transformer.to(DEVICE)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    #optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    optimizer = optim.Adafactor(transformer.parameters(), lr=lr)

    for epoch in range(1, n_epoch+1):
        start_time = timer()
        train_loss = train_epoch(transformer, optimizer, loss_fn)
        end_time = timer()
        val_loss = evaluate(transformer, loss_fn)
        print((f"Epoch: {epoch}, Train loss: {train_loss}, Val loss: {val_loss}, "f"Epoch time = {(end_time - start_time)}s"))
    
    return transformer

def transcribe_midi(model, audio_file): 
    y, sr = librosa.load(audio_file, sr=SAMPLE_RATE)
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

    # translate one chunk at a time    
    seq_chunks = [[] for _ in range(M_db.shape[1])] 
    for i in range(M_db.shape[1]):   
        cur_translation = model.translate(torch.tensor(M_db[:,[0],:]))
        seq_chunks[i] += (cur_translation.int().tolist())

    # convert sequence chunks to a pretty_midi object
    pretty_obj = seq_chunks_to_pretty_midi(seq_chunks)

    

    print("CONVERT TO MIDI")
    if not os.path.exists(results_dir + 'translation.mid'):
        pretty_obj.write(results_dir + 'translation.mid')

    input("Continue to fs...")
    output_path = results_dir + 'translation.wav'
    midi_path = results_dir + 'translation.mid'
    cmd = "fluidsynth -F " + output_path + ' ' + SOUNDFONT_PATH + ' ' + midi_path + ' -r 16000 -i'
    ret_status = os.system(cmd)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Arguments for training')
    parser.add_argument('-m', '--modeldir', help='desired model subdirectory name', required=True) # default=modeldir)
    parser.add_argument('-v', '--verbosity', dest='loglevel', action='store_const', const='DEBUG',
                        default='INFO', help='specify level of detail for log file')

    # qrsh -l gpus=1 -l gpu_c=6
    # cd /projectnb/textconv/sadiela/midi_generation/scripts
    # CONTINUED TRAINING

    args = vars(parser.parse_args())
    modelsubdir = args['modeldir']
    ### create directory for models and results ###
    modeldir = MODEL_DIR + modelsubdir
    if not os.path.isdir(modeldir):
        #print("DIRECTORY DOES NOT EXIST:", modeldir)
        sys.exit(1)

    ### save hyperparameters to YAML file in folder ###
    param_file = modeldir + "/MODEL_PARAMS.yaml"
    results_file = modeldir + "/results.yaml"

    # Read params from this file!
    # Make sure there is a parameter file! Need one to continue
    try: 
        with open(str(param_file)) as file: 
            model_hyperparams = yaml.load(file, Loader=yaml.FullLoader)
    except Exception as e: 
        print(e)
        sys.exit(1)

    model_hyperparams['modeldir']=modelsubdir
    n_enc = int(model_hyperparams['num_enc'])
    n_dec = int(model_hyperparams['num_dec'])
    nhead = int(model_hyperparams['num_heads'])
    num_epochs = int(model_hyperparams['num_epochs'])
    soundfont = model_hyperparams['soundfont']
    batch_size = int(model_hyperparams['batch_size'])
    ffn_hidden = int(model_hyperparams['ffn_hidden'])
    emb_dim = int(model_hyperparams['emb_dim'])
    vocab_size = int(model_hyperparams['vocab_size'])
    learning_rate = float(model_hyperparams['learningrate'])
    midi_dir = int(model_hyperparams['midi_dir'])
    audio_dir = int(model_hyperparams['audio_dir'])

    # save param file again

    transformer = train(n_enc, n_dec, emb_dim, nhead, vocab_size, 
                        ffn_hidden, num_epochs, learning_rate)
    
    # SAVE MODEL
    torch.save(transformer.state_dict(), MODEL_DIR + '/model2.pt')


    #transformer = Seq2SeqTransformer(n_enc, n_dec, emb_dim, nhead, vocab_size, ffn_hidden)
    #transformer.load_state_dict(torch.load(MODEL_DIR + '/model1.pt'))
    #transformer.to(DEVICE).eval()