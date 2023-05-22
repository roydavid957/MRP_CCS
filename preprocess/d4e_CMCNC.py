import pandas as pd
from tqdm import tqdm
import sys

def get_data(src_path):
    samples = []
    sample = []
    with open(src_path,'r') as f:
        num_lines = sum(1 for line in f)
        for idx,line in tqdm(f,total=num_lines):
            line = line.strip('\n')
            if line == 'DOC_SEP':
                target_idx = sample.index('HOLDOUT_SEP')
                neg_sep_idx = sample.index('NEG_SEP')
                samples.append({'StoryID':str(idx),'input':('|').join(sample[:target_idx]),'target':sample[target_idx+1],'neg':('|').join(sample[neg_sep_idx+1:])})
                sample = []
            elif line:
                sample.append((' ').join(line.split('|')))
    return samples

if __name__ == '__main__':
    df = pd.DataFrame(get_data(sys.argv[1]))
    df.to_csv(sys.argv[2],index=False,sep='\t')