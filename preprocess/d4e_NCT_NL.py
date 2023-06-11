from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer, MarianMTModel
import torch
import sys

def get_translated_sentence(sample_text, tokenizer, model, device):
    '''
    Translate sentence
    to target language
    '''

    batch = tokenizer([sample_text], return_tensors="pt").to(device)
    generated_ids = model.generate(**batch)
    translated_batch = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return translated_batch

def get_data(src_path, model_name, device):
    model = MarianMTModel.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    df = pd.read_csv(src_path,sep='\t')
    translated_stories = []
    for _,row in tqdm(df.iterrows(),total=df.shape[0]):
      translated_story = {'StoryID': row['StoryID'], 'Sentence1': '', 'Sentence2': '', 'Sentence3': '', 'Sentence4': '', 'Continuation1': '',
                          'Continuation2': '', 'Label': row['Label']}
      sentences = [row['Sentence1'],row['Sentence2'],row['Sentence3'],row['Sentence4'],row['Continuation1'],row['Continuation2']]
      for idx,sample_text in enumerate(sentences):
        translated_batch = get_translated_sentence(sample_text, tokenizer, model, device)
        if idx+1 <= 4:
          translated_story[f'Sentence{idx+1}'] = translated_batch
        elif idx+1 == 5:
          translated_story['Continuation1'] = translated_batch
        elif idx+1 == 6:
          translated_story['Continuation2'] = translated_batch
      translated_stories.append(translated_story)
    return translated_stories

if __name__ == '__main__':
    model_name = 'Helsinki-NLP/opus-mt-en-nl'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.DataFrame(get_data(sys.argv[1], model_name, device))
    df.to_csv(sys.argv[2],index=False)