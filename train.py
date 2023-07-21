import torch 
from simple_transformer import *
from audio_midi_dataset import *
from utility import *
from timeit import default_timer as timer
import torch_optimizer as optim
import argparse
import yaml
import logging


torch.manual_seed(0)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_DIR = './models/'

def train_epoch(model, optimizer, loss_fn, batch_size):
    logging.info("TRAINING BATCH SIZE: %d", batch_size)
    model.train().to(DEVICE)
    losses = 0
    training_data = AudioMidiDataset(audio_file_dir=audio_dir, midi_file_dir=midi_dir)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, collate_fn=collate_fn)

    start_time = timer()
    #logging.log("HOW MUCH DATA: %d", len(train_dataloader))
    for i, data in enumerate(train_dataloader):
        src = data[0].to(DEVICE).to(torch.float32)
        tgt = data[1].to(DEVICE).to(torch.float32)

        tgt_input = tgt[:-1, :]

        #print("INPUT SIZE FOR CREATE_MASK FUNC:", src.shape, tgt.shape)
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        logits = logits.reshape(-1, logits.shape[-1])
        tgt_out = tgt[1:, :].reshape(-1).to(torch.long)
        #print(logits.shape, tgt_out.shape)
        # 631 logits
        loss = loss_fn(logits, tgt_out)
        loss.backward()

        optimizer.step()
        losses += loss.item()
        if i%10 ==0:
            logging.info("ITERATION: %d, LOSS: %f", i, loss.item())
            end_time = timer()
            logging.info("Time: %f", (end_time-start_time))
            start_time = timer()
        #nput("Continue...")
    return losses / len(list(train_dataloader))

def evaluate(model, loss_fn, batch_size):
    model.eval()
    losses = 0

    val_iter = AudioMidiDataset(audio_file_dir=audio_dir, midi_file_dir=midi_dir)
    val_dataloader = DataLoader(val_iter, batch_size=batch_size, collate_fn=collate_fn)

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

def prepare_model(modeldir, n_enc, n_dec, emb_dim, nhead, vocab_size, ffn_hidden, learning_rate, num_epochs):
    transformer = Seq2SeqTransformer(n_enc, n_dec, emb_dim, nhead, vocab_size, ffn_hidden)
    #optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    optimizer = optim.Adafactor(transformer.parameters(), lr=learning_rate)

    # check for previously trained models: 
    previous_models = [f for f in os.listdir(modeldir) if f[-2:] == 'pt' ]

    if len(previous_models) == 0:
        for p in transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    else: 
        most_recent_model = get_newest_file(previous_models)
        stat_dictionary = torch.load(modeldir + '/' + most_recent_model, map_location=torch.device(DEVICE))
        model_params = stat_dictionary["model_state_dict"]
        transformer.load_state_dict(model_params)
        optimizer.load_state_dict(stat_dictionary['optimizer_state_dict']) # load optimizer back as well!

    transformer = transformer.to(DEVICE)

    return transformer, optimizer, (num_epochs - len(previous_models))

def train(transformer, optimizer, n_epoch, batch_size, modeldir):
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    for epoch in range(1, n_epoch+1):
        start_time = timer()
        train_loss = train_epoch(transformer, optimizer, loss_fn, batch_size)
        logging.info("Finished training epoch %d", int(epoch))
        end_time = timer()
        # SAVE INTERMEDIATE MODEL
        cur_model_file = get_free_filename('model-'+str(e), modeldir, suffix='.pt')
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': transformer.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, cur_model_file) # incremental saves
        logging.info("Saved model at epoch %d to %s", epoch, cur_model_file)
        val_loss = evaluate(transformer, loss_fn, batch_size)
        logging.info("Epoch: %d, Train loss: %f, Val loss: %f, Epoch time: %f", epoch, train_loss, val_loss, (end_time-start_time))
    return transformer

def transcribe_midi(model, audio_file): 
    M_db = calc_mel_spec(audio_file=audio_file)

    # translate one chunk at a time    
    seq_chunks = [[] for _ in range(M_db.shape[1])] 
    for i in range(M_db.shape[1]):   
        cur_translation = model.translate(torch.tensor(M_db[:,[i],:]))
        seq_chunks[i] += (cur_translation.int().tolist())

    # convert sequence chunks to a pretty_midi object
    pretty_obj = seq_chunks_to_pretty_midi(seq_chunks)
    return pretty_obj

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for training')
    parser.add_argument('-m', '--modeldir', help='desired model subdirectory name', required=True) # default=modeldir)
    parser.add_argument('-v', '--verbosity', dest='loglevel', action='store_const', const='DEBUG',
                        default='INFO', help='specify level of detail for log file')
    args = vars(parser.parse_args())

    modelsubdir = args['modeldir']
    ### create directory for models and results ###
    modeldir = MODEL_DIR + modelsubdir
    if not os.path.isdir(modeldir):
        #print("DIRECTORY DOES NOT EXIST:", modeldir)
        sys.exit(1)
    param_file = modeldir + "/MODEL_PARAMS.yaml"
    results_file = modeldir + "/results.yaml"
    
    '''LOGGING STUFF'''
    loglevel= args['loglevel']
    numeric_level = getattr(logging, loglevel.upper(), None) # put it into uppercase
    logfile = get_free_filename('train', modeldir, suffix='.log', date=False)
    logging.basicConfig(filename=logfile, level=numeric_level)

    # qrsh -l gpus=1 -l gpu_c=6
    # cd /projectnb/textconv/sadiela/midi_generation/scripts
    # CONTINUED TRAINING

    ### save hyperparameters to YAML file in folder ###
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
    midi_dir = model_hyperparams['midi_dir']
    audio_dir = model_hyperparams['audio_dir']
    val_midi_dir = model_hyperparams['midi_dir']
    val_audio_dir = model_hyperparams['audio_dir']

    # save param file again
    transformer, optimizer, num_epochs = prepare_model(modeldir, n_enc, n_dec, emb_dim, nhead, vocab_size, ffn_hidden, learning_rate, num_epochs)

    logging.info("Training transformer model")
    print("DEVICE:", DEVICE)
    transformer = train(transformer, optimizer, num_epochs, batch_size, modeldir)
    
    '''print("TRANSCRIBING MIDI")
    midi_data = transcribe_midi(transformer, './small_matched_data/raw_audio/23ae70e204549444ec91c9ee77c3523a_6.wav')
    print("PLOTTING MIDI")
    img = custom_plot_pianoroll(midi_data)'''

    #transformer = Seq2SeqTransformer(n_enc, n_dec, emb_dim, nhead, vocab_size, ffn_hidden)
    #transformer.load_state_dict(torch.load(MODEL_DIR + '/model1.pt'))
    #transformer.to(DEVICE).eval()