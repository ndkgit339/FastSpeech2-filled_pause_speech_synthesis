import glob
import pandas as pd
import os
import argparse
from pathlib import Path
import shutil
# %%
def retrieve_jsut_files(path):
    transcript_files = glob.glob(path + "/*/*/transcript_utf8.txt")
    if not os.path.exists("raw_data/JSUT/JSUT"):
        os.makedirs("raw_data/JSUT/JSUT")
    for transcript in transcript_files:
        with open(transcript, mode='r') as f:
            lines = f.readlines()
        for line in lines:
            filename, text = line.split(':')
            with open('raw_data/JSUT/JSUT/' + filename + '.lab', mode='w') as f:
                f.write(text.strip('\n'))
def retrieve_bc2013_files(path):
    bc2013_path = Path(path)
    bc2013_path.glob('**/*.wav')
    raw_data_path = Path('raw_data/BC2013/BC2013')
    raw_data_path.mkdir(parents=True, exist_ok=True)
    for file in bc2013_path.glob('**/*.wav'):
        shutil.copy(file, raw_data_path / file.name)
        shutil.copy(file.with_suffix('.txt'), raw_data_path / file.with_suffix('.lab').name)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str,default='jsut')
    parser.add_argument('path_to_corpus',type=str, default="/media/ssd/corpus/jsut_ver1.1/*/transcript_utf8.txt")
    args = parser.parse_args()
    if args.type == 'jsut':
        retrieve_jsut_files(args.path_to_corpus)
    elif args.type == 'bc2013':
        retrieve_bc2013_files(args.path_to_corpus)
    else:
        print("Invalid type")
        exit(1)
    print("Done")
    exit(0)


# %%
