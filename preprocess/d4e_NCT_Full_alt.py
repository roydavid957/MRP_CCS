import numpy as np
import pandas as pd
from tqdm import tqdm
import sys

def get_data(src_path:str,n=6,s=2023):
    np.random.seed(s)

    with open(src_path,'r') as f:
        story_list_dict = []

        for line in f:
            if line.count('|SENT|') >= n-1:
                # story_dict = dict()
                sentences = []
                story_id = line.split('|')[0]
                # story_dict['StoryID'] = story_id
                raw_story = [l for l in line.strip('\n').split('|SENT|')]

                for idx, sentence in enumerate(raw_story):
                    sent_id = sentence.split('|')[1]
                    sent = sentence.split('|')[2]
                    # tup = (',').join(sentence.split(sent)[-1].split('|TUP|')[1:])
                    # sentence_dict = {'SentID':sent_id,'sentence':sent,'tup':tup}
                    sentence_dict = {'SentID':sent_id,'sentence':sent}
                    sentences.append(sentence_dict)

                    # story_dict[f'Sentence{idx+1}'] = sent

                # story_list_dict.append(story_dict)

                last_sents = sentences[n-1:]
                rand_sent = np.random.choice(last_sents)
                label = np.random.choice([1,2])
                if label == 1:
                    opt1 = sentences[4]
                    opt2 = rand_sent
                elif label == 2:
                    opt2 = sentences[4]
                    opt1 = rand_sent

                story_list_dict.append({'StoryID':story_id,
                                        'Sentence1':sentences[0]['sentence'],#'TUP1':sentences[0]['tup'],
                                        'Sentence2':sentences[1]['sentence'],#'TUP2':sentences[1]['tup'],
                                        'Sentence3':sentences[2]['sentence'],#'TUP3':sentences[2]['tup'],
                                        'Sentence4':sentences[3]['sentence'],#'TUP4':sentences[3]['tup'],
                                        'Continuation1':opt1['sentence'],#'TUP5':opt1['tup']
                                        'Continuation2':opt2['sentence'],#'TUP5':opt2['tup']
                                        'Label': label
                                        })

    return story_list_dict

if __name__ == '__main__':
    data = get_data(sys.argv[1])
    df = pd.DataFrame(data)
    df.to_csv(sys.argv[2],index=False,sep='\t')