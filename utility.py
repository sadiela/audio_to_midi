import re  
import os
from datetime import datetime
from pathlib import Path

def get_newest_file(fnames, prefix="model-"):
    newest = fnames[0]
    max_num = int(fnames[0][6:].split('.')[0])
    for f in fnames: 
        number = int(f[6:].split('.')[0])
        print(f, number)
        if number > max_num:
            newest = f 
            max_num = number
    return newest

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
        
if __name__ == '__main__':
    fnames = ["model-0.pt","model-1.pt","model-2.pt","model-3.pt",
              "model-4.pt","model-5.pt","model-6.pt","model-7.pt",
              "model-8.pt","model-9.pt","model-10.pt"]
    newest = get_newest_file(fnames)
    print(newest)