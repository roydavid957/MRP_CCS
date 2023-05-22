import numpy as np
import pandas as pd
from tqdm import tqdm
import sys

def get_data(path:str,n=5,s=2023) -> list[dict]:
    np.random.seed(s)

    with open(path,'r') as f:
        story_list_dict = []
        num_lines = sum(1 for line in f)

        for line in tqdm(f,total=num_lines):
            if line.count('|SENT|') >= n-1:
                sentences = []
                story_id = line.split('|')[0]
                raw_story = [l for l in line.strip('\n').split('|SENT|')]

                for sentence in raw_story:
                    sent_id = sentence.split('|')[1]
                    sent = sentence.split('|')[2]
                    tup = (',').join(sentence.split(sent)[-1].split('|TUP|')[1:])
                    sentence_dict = {'SentID':sent_id,'sentence':sent,'tup':tup}
                    sentences.append(sentence_dict)

                    story_list_dict.append({'StoryID':story_id,
                                            'Sentence1':sentences[0]['sentence'],'TUP1':sentences[0]['tup'],
                                            'Sentence2':sentences[1]['sentence'],'TUP2':sentences[1]['tup'],
                                            'Sentence3':sentences[2]['sentence'],'TUP3':sentences[2]['tup'],
                                            'Sentence4':sentences[3]['sentence'],'TUP4':sentences[3]['tup'],
                                            'Sentence5':sentences[4]['sentence'],'TUP5':sentences[4]['tup']})
    
    # add random final sentences
    picked_id_list = []             # include only 1 sentence per story
    for story in story_list_dict:   # to avoid possible final sentence bias
        add=True
        while add: 
            rand_sent = np.random.choice(story_list_dict)
            if rand_sent['StoryID'] not in picked_id_list and rand_sent['StoryID'] != story['StoryID']:
                n = np.random.choice([i+1 for i in range(n)])
                # story['RandomFinalStoryID'] = rand_sent['StoryID']
                story['RandomFinalSentence'] = rand_sent[f"Sentence{n}"]
                story['RandomFinalTUP'] = rand_sent[f"TUP{n}"]
                picked_id_list.append(rand_sent['StoryID'])
                add=False

    return story_list_dict

if __name__ == '__main__':
    df = pd.DataFrame(get_data(sys.argv[1]))
    df.to_csv(sys.argv[2],index=False,sep='\t')