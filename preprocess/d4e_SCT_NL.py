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
    df = pd.read_csv(src_path)
    translated_stories = []
    for _,row in tqdm(df.iterrows(),total=df.shape[0]):
      translated_story = {'InputStoryid': row['InputStoryid'], 'InputSentence1': '', 'InputSentence2': '', 'InputSentence3': '', 'InputSentence4': '', 'RandomFifthSentenceQuiz1': '',
                          'RandomFifthSentenceQuiz2': '', 'AnswerRightEnding': row['AnswerRightEnding']}
      sentences = [row['InputSentence1'],row['InputSentence2'],row['InputSentence3'],row['InputSentence4'],row['RandomFifthSentenceQuiz1'],row['RandomFifthSentenceQuiz2']]
      for idx,sample_text in enumerate(sentences):
        translated_batch = get_translated_sentence(sample_text, tokenizer, model, device)
        if idx+1 <= 4:
          translated_story[f'InputSentence{idx+1}'] = translated_batch
        elif idx+1 == 5:
          translated_story['RandomFifthSentenceQuiz1'] = translated_batch
        elif idx+1 == 6:
          translated_story['RandomFifthSentenceQuiz2'] = translated_batch
      translated_stories.append(translated_story)
    return translated_stories

if __name__ == '__main__':
    model_name = 'Helsinki-NLP/opus-mt-en-nl'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.DataFrame(get_data(sys.argv[1], model_name, device))
    df.to_csv(sys.argv[2],index=False)