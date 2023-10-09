import torch
import yaml
import argparse

from spectrograms import *
from midi_vocabulary import *
from transcription_transformer import TranscriptionTransformer
from utility import *

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_DIR = '../models/'

def transcribe_wavs(transformer, wavdir, transcription_folder, plot=False, synth=True):
    transformer.eval()
    wav_files = [wav for wav in os.listdir(wavdir) if wav[-3:]=='wav']
    for wav in wav_files: 
        midi_data = transcribe_wav(transformer, wavdir + wav) # this is a pretty_midi
        midi_data.write(transcription_folder + wav[:-3] + 'mid')

        if plot: 
            _, ax = plt.subplots()
            ax  = custom_plot_pianoroll(midi_data)
            ax.figure.set_size_inches(20, 5)
            plt.title(file[:-4])
            plt.savefig( transcription_folder +  str(wav.split('.')[0] + '.png') ,bbox_inches="tight")
            plt.clf()
            plt.close()

        if synth:
            midi_to_wav(transcription_folder + wav[:-3] + 'mid',transcription_folder + wav[:-3] + 'wav')

def transcribe_wav(model, audio_file): 
    M_db = calc_mel_spec(audio_file=audio_file) # get spectrogram....will be all chunks in order

    # translate one chunk at a time    
    seq_chunks = [[] for _ in range(M_db.shape[0])] 
    for i in range(M_db.shape[0]):   
        cur_translation = model.translate(torch.tensor(M_db[[i],:,:]))
        seq_chunks[i] += (cur_translation.int().tolist())

    # convert sequence chunks to a pretty_midi object
    pretty_obj = seq_chunks_to_pretty_midi(seq_chunks)
    return pretty_obj

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for training')
    parser.add_argument('-m', '--modeldir', help='desired model subdirectory name', required=True) # default=modeldir)
    parser.add_argument('-w', '--wav', help="directory with files you want to transcribe", required=True)
    parser.add_argument('-n', '--mname', help="filename for model you want to use", required=True)

    args = vars(parser.parse_args())
    print(args)

    modelsubdir = args['modeldir']
    wavdir = args['wav']
    modelname = args['mname']
    ### create directory for models and results ###
    modeldir = MODEL_DIR + modelsubdir
    if not os.path.isdir(modeldir):
        print("NO MODEL DIR")
        sys.exit(1)
    param_file = modeldir + "/MODEL_PARAMS.yaml"
    transcription_folder = modeldir + 'transcriptions/'
    os.makedirs(transcription_folder, mode=0o777, exist_ok=True)
    modelpath = modeldir + modelname

    try: 
        with open(str(param_file)) as file: 
            mod_hyper = yaml.load(file, Loader=yaml.FullLoader)
    except Exception as e: 
        print("COULDNT LOAD HYPERPARAMS")
        print(e)
        sys.exit(1)

    # load state dict into model
    transformer = TranscriptionTransformer(int(mod_hyper['num_enc']), int(mod_hyper['num_dec']), 
                                        int(mod_hyper['emb_dim']), int(mod_hyper['num_heads']),
                                        int(mod_hyper['vocab_size']),int(mod_hyper['ffn_hidden']))
    
    print("instantiated transformer")
    
    state_dictionary = torch.load(modelpath, map_location=torch.device(DEVICE))
    model_params = state_dictionary["model_state_dict"]
    transformer.load_state_dict(model_params)
    transformer.to(DEVICE) # have to do this before constructing optimizers...

    print("loaded parameters")

    transcribe_wavs(transformer, wavdir, transcription_folder)