import sys
import json
import spacy
import numpy as np
import pandas as pd


def get_data(src_path,n=6,spacy_model = "nl_core_news_lg"):
    
    story_list_dict = []
    spacy.prefer_gpu()
    nlp = spacy.load(spacy_model)
    
    with open(src_path,'r') as f:
        
        for line in f:
            line = json.loads(line)
            if line['text'] != '':
                try:
                    doc = nlp(line['text'])
                except IndexError as e:
                    print(f'\n{e}\n{line}')
                    next(f)
                if len(list(doc.sents)) >= n:
                    doc_sents = [str(sent) for sent in doc.sents]
                    
                    story_id = line['id']

                    last_sents = doc_sents[n-1:]
                    rand_sent = np.random.choice(last_sents)
                    while len(rand_sent.split(' ')) < 1:
                        rand_sent = np.random.choice(last_sents)
                    label = np.random.choice([1,2])
                    if label == 1:
                        opt1 = doc_sents[n-2]
                        opt2 = rand_sent
                    elif label == 2:
                        opt2 = doc_sents[n-2]
                        opt1 = rand_sent

                    story_list_dict.append({'StoryID':story_id,
                                            'Sentence1':doc_sents[0],
                                            'Sentence2':doc_sents[1],
                                            'Sentence3':doc_sents[2],
                                            'Sentence4':doc_sents[3],
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