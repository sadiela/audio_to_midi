import torch 
from transcription_transformer import *
from audio_midi_dataset import *
from timeit import default_timer as timer
import torch_optimizer as optim
import argparse
import yaml
import logging
from tqdm import tqdm
import sys
import glob
from datetime import datetime
import re

torch.manual_seed(0)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_DIR = './models/'

def get_free_filename(stub, directory, suffix='', date=False):
    # Create unique file/directory 
    counter = 0
    while True:
        if date:
            file_candidate = '{}/{}-{}-{}{}'.format(str(directory), stub, datetime.today().strftime('%Y-%m-%d'), counter, suffix)
        else: 
            file_candidate = '{}/{}-{}{}'.format(str(directory), stub, counter, suffix)
        if Path(file_candidate).exists():
            #print("file exists")
            counter += 1
        else:  # No match found
            print("Counter:", counter)
            if suffix=='.p':
                print("will create pickle file")
            elif suffix:
                Path(file_candidate).touch()
            else:
                Path(file_candidate).mkdir()
            return file_candidate

def train_epoch(model, optimizer, loss_fn, train_dataloader):
    model.train()
    losses = 0

    start_time = timer()
    for i, data in enumerate(train_dataloader):
        if data is None: 
            logging.info("NO DATA, passing")
            pass
        try:
            src = data[0].to(DEVICE).to(torch.float32) # 512 x 16 x 512 (seq_len x batch_size x spec_bins)
            tgt = data[1].to(DEVICE).to(torch.long) # 1024 x 16 (seq_len x batch_size) # I think i want batch size first
            tgt_input = tgt[:,:-1] # slice off EOS token

            logits = model(src, tgt_input)
            optimizer.zero_grad()

            logits = logits.reshape(-1, logits.shape[-1])
            tgt_out = tgt[:,1:].reshape(-1) # slice EOS token

            loss = loss_fn(logits, tgt_out)
            loss.backward()

            optimizer.step()
            losses += loss.item()
            if i%30000 == 0 and i != 0:
                logging.info("ITERATION: %d, LOSS: %f", i, loss.item())
                end_time = timer()
                logging.info("Time: %f", (end_time-start_time))
                cur_model_file = get_free_filename('model-' + str(i) + 'iters', modeldir, suffix='.pt') #modeldir + '/model-' + str(epoch + num_previous_models) + '.pt' 
                torch.save({
                            'iters': i,
                            'model_state_dict': transformer.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            }, cur_model_file) # incremental saves
                logging.info("Saved model file %s", cur_model_file)
                start_time = timer()
        except RuntimeError as e:
            logging.info("RUNTIME ERROR IN TRAINING LOOP: %s", str(e))
            print("RUNTIME ERROR", e)
        except Exception as ex:
            logging.info("ERROR IN TRAINING LOOP: %s", str(ex))
            print("ERROR", ex)
    return losses / len(list(train_dataloader))

def evaluate(model, loss_fn, eval_dataloader):
    losses = 0
    model.eval()

    with torch.no_grad(): # do not need to save gradients since we will not be running a backwards pass
        for i, data in enumerate(eval_dataloader):
            if data is None: 
                logging.log("NO DATA, passing")
                pass
            try: 
                src = data[0].to(DEVICE).to(torch.float32)
                tgt = data[1].to(DEVICE).to(torch.long)

                tgt_input = tgt[:-1, :]

                logits = model(src, tgt_input)

                tgt_out = tgt[1:, :]
                loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
                losses += loss.item()
            except Exception as e:
                logging.info("ERROR: %s", str(e))

    return losses / len(list(eval_dataloader))

def prepare_model(modeldir, n_enc, n_dec, emb_dim, nhead, vocab_size, ffn_hidden, learning_rate, num_epochs):
    transformer = TranscriptionTransformer(n_enc, n_dec, emb_dim, nhead, vocab_size, ffn_hidden)
    #optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    # check for previously trained models: 
    #previous_models = [f for f in os.listdir(modeldir) if f[-2:] == 'pt' ]
    previous_models = glob.glob(modeldir + '/*.pt') # * means all if need specific format then *.csv
    print("PREVIOUS MODELS:", previous_models)

    if len(previous_models) == 0:
        logging.info("training new transformer from scratch")
        for p in transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) # Why are we using this? 
        optimizer = optim.Adafactor(transformer.parameters(), lr=learning_rate)
    else: 
        most_recent_model = max(previous_models, key=os.path.getctime)
        logging.info("loading parameters from most recent model file: %s", most_recent_model)
        stat_dictionary = torch.load(most_recent_model, map_location=torch.device(DEVICE))
        model_params = stat_dictionary["model_state_dict"]
        transformer.load_state_dict(model_params)
        transformer.to(DEVICE) # have to do this before constructing optimizers...
        optimizer = optim.Adafactor(transformer.parameters(), lr=learning_rate)
        optimizer.load_state_dict(stat_dictionary['optimizer_state_dict']) # load optimizer back as well!

    param_size = 0
    for param in transformer.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in transformer.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))
    logging.info('model size: {:.3f}MB'.format(size_all_mb))

    return transformer, optimizer, (num_epochs - len(previous_models))

def train(transformer, optimizer, n_epoch, batch_size, modeldir, audio_dir, midi_dir, train_paths, eval_paths):
    transformer.to(DEVICE)
    train_losses = []
    eval_losses = []
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    num_previous_models = len([f for f in os.listdir(modeldir) if f[-2:] == 'pt' ])
    
    logging.info("TRAINING BATCH SIZE: %d", batch_size)
    training_data = AudioMidiDataset(train_paths, audio_file_dir=audio_dir, midi_file_dir=midi_dir)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, collate_fn=collate_fn)
    
    eval_data = AudioMidiDataset(eval_paths, audio_file_dir=audio_dir, midi_file_dir=midi_dir)
    eval_dataloader = DataLoader(eval_data, batch_size=batch_size, collate_fn=collate_fn)

    for epoch in range(1, n_epoch+1):
        start_time = timer()
        train_loss = train_epoch(transformer, optimizer, loss_fn, train_dataloader)
        logging.info("Finished training epoch %d", int(epoch))
        end_time = timer()
        train_losses.append(train_loss)
        # SAVE INTERMEDIATE MODEL
        cur_model_file = get_free_filename('model-full-epoch', modeldir, suffix='.pt') #modeldir + '/model-' + str(epoch + num_previous_models) + '.pt' 
        torch.save({
                    'epoch': num_previous_models + epoch,
                    'model_state_dict': transformer.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, cur_model_file) # incremental saves
        logging.info("Saved model at epoch %d to %s", num_previous_models + epoch, cur_model_file)
        eval_loss = evaluate(transformer, loss_fn, eval_dataloader)
        eval_losses.append(eval_loss)
        logging.info("Epoch: %d, Train loss: %f, Val loss: %f, Epoch time: %f", epoch+num_previous_models, train_loss, eval_loss, (end_time-start_time))
    return transformer, train_losses, eval_losses

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
    
    loglevel= args['loglevel']
    numeric_level = getattr(logging, loglevel.upper(), None) # put it into uppercase
    logfile = get_free_filename('train', modeldir, suffix='.log', date=False)
    logging.basicConfig(filename=logfile, level=logging.DEBUG)
    logging.getLogger('numba').setLevel(logging.WARNING)


    ### save hyperparameters to YAML file in folder ###
    try: 
        with open(str(param_file)) as file: 
            model_hyperparams = yaml.load(file, Loader=yaml.FullLoader)
    except Exception as e: 
        print(e)
        sys.exit(1)

    # load hyperparameters from MODEL_PARAMS.yaml file
    model_hyperparams['modeldir']=modelsubdir
    n_enc = int(model_hyperparams['num_enc'])
    n_dec = int(model_hyperparams['num_dec'])
    nhead = int(model_hyperparams['num_heads'])
    num_epochs = int(model_hyperparams['num_epochs'])
    batch_size = int(model_hyperparams['batch_size'])
    ffn_hidden = int(model_hyperparams['ffn_hidden'])
    emb_dim = int(model_hyperparams['emb_dim'])
    vocab_size = int(model_hyperparams['vocab_size'])
    learning_rate = float(model_hyperparams['learningrate'])
    midi_dir = model_hyperparams['midi_dir']
    audio_dir = model_hyperparams['audio_dir']
    train_midi_pickle = model_hyperparams['train_paths']
    eval_midi_pickle = model_hyperparams['eval_paths']

    with open(train_midi_pickle, 'rb') as fp:
        train_midi_paths = pickle.load(fp)
    with open(eval_midi_pickle, 'rb') as fp:
        eval_midi_paths = pickle.load(fp)

    # save param file again
    transformer, optimizer, num_epochs = prepare_model(modeldir, n_enc, n_dec, emb_dim, nhead, vocab_size, ffn_hidden, learning_rate, num_epochs)

    logging.info("Training transformer model")
    print("DEVICE:", DEVICE)
    transformer, train_losses, eval_losses = train(transformer, optimizer, num_epochs, batch_size, modeldir, audio_dir, midi_dir, train_paths=train_midi_paths, eval_paths=eval_midi_paths)
    
    results = {}
    results["trainloss"] = train_losses
    results["evalloss"] = eval_losses

    with open(results_file, 'w') as fp:
        yaml.dump(results, fp)
    print("SAVED RESULTS")
