import sys
import json
import numpy as np
import pandas as pd
import nltk.data

def get_data(src_path,n=6,spacy_model = "nl_core_news_lg"):
    np.random.seed(2023)
    
    story_list_dict = []
    tokenizer = nltk.data.load('tokenizers/punkt/dutch.pickle')
    
    with open(src_path,'r') as f:
        
        for line in f:
            line = json.loads(line)
            if line['text'] != '':
                sents = tokenizer.tokenize(line['text'])
                if len(sents) >= n:
                    
                    story_id = line['id']

                    last_sents = sents[n-1:]
                    rand_sent = np.random.choice(last_sents)
                    while len(rand_sent.split(' ')) < 1:
                        rand_sent = np.random.choice(last_sents)
                    label = np.random.choice([1,2])
                    if label == 1:
                        opt1 = sents[n-2]
                        opt2 = rand_sent
                    elif label == 2:
                        opt2 = sents[n-2]
                        opt1 = rand_sent

                    story_list_dict.append({'StoryID':story_id,
                                            'Sentence1':sents[0],
                                            'Sentence2':sents[1],
                                            'Sentence3':sents[2],
                                            'Sentence4':sents[3],
                                            'Continuation1':opt1,
                                            'Continuation2':opt2,
                                            'Label':label
                                            })
                            # print(story_list_dict)
    return story_list_dict


if __name__ == '__main__':
    data = get_data(sys.argv[1])
    df = pd.DataFrame(data)
    df.to_csv(sys.argv[2],sep='\t', index=False)